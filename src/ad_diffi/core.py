from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest


FeatureTypes = Dict[int, str]
NoiseBaselines = Dict[str, Dict[str, float]]

_CONTINUOUS = "cont"
_BINARY = "bin"
_VALID_FEATURE_TYPES = {_CONTINUOUS, _BINARY}


def _validate_feature_types(
    feature_types: FeatureTypes,
    n_features: int,
) -> None:
    """Validate complete feature-type metadata."""
    expected_indices = set(range(n_features))
    observed_indices = set(feature_types)

    missing_indices = expected_indices - observed_indices
    extra_indices = observed_indices - expected_indices

    if missing_indices:
        raise ValueError(
            f"Missing feature types: {sorted(missing_indices)}"
        )

    if extra_indices:
        raise ValueError(
            f"Invalid feature indices: {sorted(extra_indices)}"
        )

    invalid_types = {
        feature: feature_types[feature]
        for feature in range(n_features)
        if feature_types[feature] not in _VALID_FEATURE_TYPES
    }

    if invalid_types:
        raise ValueError(
            "Feature types must be 'cont' or 'bin': "
            f"{invalid_types}"
        )


def _validate_anomaly_mask(
    anomaly_mask: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Validate and return a Boolean outlier mask."""
    anomaly_mask = np.asarray(
        anomaly_mask,
        dtype=bool,
    )

    if anomaly_mask.shape != (n_samples,):
        raise ValueError(
            "anomaly_mask must have shape (n_samples,)."
        )

    if anomaly_mask.all() or (~anomaly_mask).all():
        raise ValueError(
            "Both outliers and inliers are required."
        )

    return anomaly_mask


def _validate_inputs(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Validate common Isolation Forest attribution inputs."""
    X_data = np.asarray(
        X_data,
        dtype=float,
    )

    if X_data.ndim != 2:
        raise ValueError(
            "X_data must be a two-dimensional array."
        )

    if (
        not hasattr(iforest, "estimators_")
        or not iforest.estimators_
    ):
        raise ValueError(
            "iforest must be a fitted IsolationForest."
        )

    n_samples, n_features = X_data.shape

    if (
        hasattr(iforest, "n_features_in_")
        and iforest.n_features_in_ != n_features
    ):
        raise ValueError(
            "X_data and iforest have different feature counts."
        )

    _validate_feature_types(
        feature_types,
        n_features,
    )

    if anomaly_mask is None:
        return X_data, None

    return X_data, _validate_anomaly_mask(
        anomaly_mask,
        n_samples,
    )


def _node_lambdas(tree) -> np.ndarray:
    """Return raw split-imbalance coefficients for all tree nodes.

    For an internal node v,

        lambda(v) = 1 - min(n_left, n_right) / n_parent.

    Leaf nodes receive zero.
    """
    lambdas = np.zeros(
        tree.node_count,
        dtype=float,
    )

    for node in range(tree.node_count):
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left == right:
            continue

        n_parent = float(tree.n_node_samples[node])

        if n_parent <= 0:
            continue

        n_left = tree.n_node_samples[left]
        n_right = tree.n_node_samples[right]

        lambdas[node] = 1.0 - (
            min(n_left, n_right)
            / n_parent
        )

    return lambdas


def _parent_array(tree) -> np.ndarray:
    """Return the parent node index for each node."""
    parent = np.full(
        tree.node_count,
        -1,
        dtype=int,
    )

    for node in range(tree.node_count):
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left != right:
            parent[left] = node
            parent[right] = node

    return parent


def _paths_for_rows(
    estimator,
    X_data: np.ndarray,
) -> list[list[int]]:
    """Return root-to-terminal paths for every row in X_data."""
    tree = estimator.tree_
    parent = _parent_array(tree)

    terminal_nodes = estimator.apply(X_data).astype(int)

    cache: dict[int, list[int]] = {}
    paths: list[list[int]] = []

    for terminal_node in terminal_nodes:
        terminal_node = int(terminal_node)

        if terminal_node not in cache:
            nodes = []
            node = terminal_node

            while node >= 0:
                nodes.append(node)
                node = parent[node]

            cache[terminal_node] = nodes[::-1]

        paths.append(cache[terminal_node])

    return paths


def _node_membership(
    estimator,
    X_data: np.ndarray,
) -> np.ndarray:
    """Return a dense Boolean node-membership matrix.

    Rows correspond to observations and columns correspond to nodes.
    Entry [i, v] is True when observation i passes through node v.
    """
    return estimator.decision_path(
        X_data
    ).toarray().astype(bool)


def _node_outlier_enrichment_from_membership(
    tree,
    node_membership: np.ndarray,
    anomaly_mask: np.ndarray,
) -> np.ndarray:
    """Return child-level outlier-prevalence contrasts for all nodes.

    For every internal node v,

        E(v) = | P(outlier | left child of v)
             - P(outlier | right child of v) |.

    Leaf nodes and nodes with an empty child in X_data receive zero.
    """
    enrichment = np.zeros(
        tree.node_count,
        dtype=float,
    )

    for node in range(tree.node_count):
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left == right:
            continue

        in_left = node_membership[:, left]
        in_right = node_membership[:, right]

        n_left = int(in_left.sum())
        n_right = int(in_right.sum())

        if n_left == 0 or n_right == 0:
            continue

        outlier_rate_left = float(
            anomaly_mask[in_left].mean()
        )

        outlier_rate_right = float(
            anomaly_mask[in_right].mean()
        )

        enrichment[node] = abs(
            outlier_rate_left
            - outlier_rate_right
        )

    return enrichment


def _eligible_node(
    node: int,
    feature: int,
    feature_types: FeatureTypes,
) -> bool:
    """Return whether a split node contributes under AD-DIFFI rules."""
    feature_type = feature_types[feature]

    return (
        feature_type == _CONTINUOUS
        or (
            feature_type == _BINARY
            and node == 0
        )
    )


def _feature_enrichment_terms(
    estimator,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-row AD-DIFFI numerators and denominators.

    The primary enrichment-based AD-DIFFI contribution for an eligible node v
    on the path of observation x is

        lambda(v) * E(v) / h_t(x),

    where:
    - lambda(v) is the raw child-size imbalance;
    - E(v) is the absolute difference in model-defined outlier prevalence
      between the two child nodes;
    - h_t(x) is the number of internal split nodes on x's path.

    Continuous features contribute at every internal node on which they are
    selected. Binary features contribute only if they are selected at the root
    node under the Root-Split-Only rule.
    """
    n_samples, n_features = X_data.shape

    tree = estimator.tree_
    lambdas = _node_lambdas(tree)
    paths = _paths_for_rows(
        estimator,
        X_data,
    )

    node_membership = _node_membership(
        estimator,
        X_data,
    )

    enrichment = _node_outlier_enrichment_from_membership(
        tree,
        node_membership,
        anomaly_mask,
    )

    numerator = np.zeros(
        (n_samples, n_features),
        dtype=float,
    )

    denominator = np.zeros(
        (n_samples, n_features),
        dtype=float,
    )

    for row, path in enumerate(paths):
        internal_path = [
            node
            for node in path
            if tree.feature[node] >= 0
        ]

        path_length = max(
            len(internal_path),
            1,
        )

        for node in internal_path:
            feature = int(tree.feature[node])

            if not _eligible_node(
                node,
                feature,
                feature_types,
            ):
                continue

            numerator[row, feature] += (
                lambdas[node]
                * enrichment[node]
                / path_length
            )

            denominator[row, feature] += 1.0

    return numerator, denominator


def compute_enrichment_ad_diffi(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: np.ndarray,
) -> np.ndarray:
    """Compute raw enrichment-based AD-DIFFI feature rankings.

    This is the primary AD-DIFFI score.

    For every fitted tree, the score averages feature-specific eligible-node
    contributions:

        lambda(v) * E(v) / h_t(x),

    over the feature's path occurrences. Continuous features are eligible at
    all internal split nodes. Binary features are eligible only when selected
    at the root node.

    The final score is the average of tree-specific feature means across the
    forest. Larger values indicate stronger model-based outlier enrichment
    attributed to the feature.

    Parameters
    ----------
    iforest:
        A fitted sklearn IsolationForest.

    X_data:
        Two-dimensional data array used for attribution.

    feature_types:
        Mapping from feature index to "cont" or "bin".

    anomaly_mask:
        Boolean array identifying model-defined outliers. In a standard
        workflow, use ``iforest.predict(X_data) == -1``.

    Returns
    -------
    np.ndarray
        One raw enrichment-based AD-DIFFI score per feature.

    Notes
    -----
    Raw enrichment scores are the primary mixed-type ranking measure. Their
    empirical ranking performance should be evaluated in the intended data
    setting. Type-specific null calibration is available separately through
    ``get_enrichment_noise_baselines`` and ``calculate_ad_diffi_zscore``.
    """
    X_data, anomaly_mask = _validate_inputs(
        iforest,
        X_data,
        feature_types,
        anomaly_mask,
    )

    if anomaly_mask is None:
        raise ValueError(
            "anomaly_mask is required."
        )

    n_features = X_data.shape[1]

    forest_importance = np.zeros(
        n_features,
        dtype=float,
    )

    for estimator in iforest.estimators_:
        numerator, denominator = _feature_enrichment_terms(
            estimator,
            X_data,
            feature_types,
            anomaly_mask,
        )

        tree_numerator = numerator.sum(axis=0)
        tree_denominator = denominator.sum(axis=0)

        forest_importance += np.divide(
            tree_numerator,
            tree_denominator,
            out=np.zeros_like(tree_numerator),
            where=tree_denominator > 0,
        )

    return forest_importance / len(
        iforest.estimators_
    )


def _resolve_max_samples(
    max_samples,
    n_samples: int,
) -> int:
    """Resolve sklearn max_samples into a valid integer."""
    if max_samples == "auto":
        return min(256, n_samples)

    if isinstance(max_samples, (float, np.floating)):
        if not 0 < max_samples <= 1:
            raise ValueError(
                "Float max_samples must be in (0, 1]."
            )

        return max(
            2,
            int(np.ceil(max_samples * n_samples)),
        )

    resolved = int(max_samples)

    if resolved < 1:
        raise ValueError(
            "Integer max_samples must be at least 1."
        )

    return min(resolved, n_samples)


def _generate_noise(
    n_samples: int,
    n_features: int,
    feature_types: FeatureTypes,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate independent type-specific synthetic noise.

    Continuous features are sampled from Uniform(0, 1), and binary features
    are sampled from Bernoulli(0.5).
    """
    X_noise = np.empty(
        (n_samples, n_features),
        dtype=float,
    )

    for feature in range(n_features):
        if feature_types[feature] == _CONTINUOUS:
            X_noise[:, feature] = rng.uniform(
                0.0,
                1.0,
                n_samples,
            )
        else:
            X_noise[:, feature] = rng.binomial(
                1,
                0.5,
                n_samples,
            )

    return X_noise


def _validate_noise_baselines(
    noise_baselines: NoiseBaselines,
) -> None:
    """Validate a type-specific baseline dictionary."""
    required_types = {
        _CONTINUOUS,
        _BINARY,
    }

    missing_types = required_types - set(
        noise_baselines
    )

    if missing_types:
        raise ValueError(
            "Missing noise baselines for feature types: "
            f"{sorted(missing_types)}"
        )

    for feature_type in required_types:
        baseline = noise_baselines[feature_type]

        if not isinstance(baseline, dict):
            raise ValueError(
                f"Baseline for '{feature_type}' must be a dictionary."
            )

        if "mean" not in baseline or "sd" not in baseline:
            raise ValueError(
                f"Baseline for '{feature_type}' must contain "
                "'mean' and 'sd'."
            )

        mean = float(baseline["mean"])
        sd = float(baseline["sd"])

        if not np.isfinite(mean):
            raise ValueError(
                f"Baseline mean for '{feature_type}' must be finite."
            )

        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(
                f"Baseline SD for '{feature_type}' must be finite "
                "and greater than zero."
            )


def make_noise_baselines(
    cont_mean: float,
    cont_sd: float,
    bin_mean: float,
    bin_sd: float,
) -> NoiseBaselines:
    """Create a validated continuous/binary baseline dictionary."""
    baselines: NoiseBaselines = {
        _CONTINUOUS: {
            "mean": float(cont_mean),
            "sd": float(cont_sd),
        },
        _BINARY: {
            "mean": float(bin_mean),
            "sd": float(bin_sd),
        },
    }

    _validate_noise_baselines(baselines)

    return baselines


def _summarize_scores(
    scores: list[float],
) -> Tuple[float, float]:
    """Return mean and sample SD with a positive-SD fallback."""
    if not scores:
        return 0.0, 1.0

    values = np.asarray(
        scores,
        dtype=float,
    )

    mean = float(values.mean())
    sd = float(values.std(ddof=1))

    return mean, sd if sd > 0 else 1.0


def get_enrichment_noise_baselines(
    X_dim: int,
    feature_types: FeatureTypes,
    if_params: dict,
    n_iter: int = 20,
    random_state: Optional[int] = 12345,
    n_noise: int = 256,
) -> Tuple[float, float, float, float]:
    """Estimate null baselines for enrichment-based AD-DIFFI.

    This function is intended for secondary type-specific calibration of raw
    enrichment scores. It repeatedly generates independent synthetic noise
    datasets, fits an Isolation Forest using the supplied settings, and
    computes raw enrichment AD-DIFFI scores.

    Continuous null features follow Uniform(0, 1). Binary null features follow
    Bernoulli(0.5).

    Returns
    -------
    tuple[float, float, float, float]
        ``(cont_mean, cont_sd, bin_mean, bin_sd)``.

    Notes
    -----
    The resulting baseline is a type-specific pure-noise reference. A
    standardized score derived from this baseline is a secondary null-departure
    measure and is not guaranteed to provide a universally comparable global
    ranking scale across continuous and binary features.
    """
    if n_iter < 2:
        raise ValueError(
            "n_iter must be at least 2."
        )

    if n_noise < 2:
        raise ValueError(
            "n_noise must be at least 2."
        )

    _validate_feature_types(
        feature_types,
        X_dim,
    )

    rng = np.random.default_rng(random_state)

    continuous_indices = [
        feature
        for feature in range(X_dim)
        if feature_types[feature] == _CONTINUOUS
    ]

    binary_indices = [
        feature
        for feature in range(X_dim)
        if feature_types[feature] == _BINARY
    ]

    continuous_scores: list[float] = []
    binary_scores: list[float] = []

    for iteration in range(n_iter):
        X_noise = _generate_noise(
            n_samples=n_noise,
            n_features=X_dim,
            feature_types=feature_types,
            rng=rng,
        )

        params = dict(if_params)
        params.pop("random_state", None)

        params["max_samples"] = _resolve_max_samples(
            params.get("max_samples", "auto"),
            n_noise,
        )

        noise_model = IsolationForest(
            **params,
            random_state=(
                None
                if random_state is None
                else random_state + iteration
            ),
        ).fit(X_noise)

        anomaly_mask = noise_model.predict(X_noise) == -1

        if anomaly_mask.all() or (~anomaly_mask).all():
            order = np.argsort(
                noise_model.score_samples(X_noise)
            )

            anomaly_mask = np.zeros(
                n_noise,
                dtype=bool,
            )

            contamination = params.get(
                "contamination",
                0.1,
            )

            n_outliers = max(
                1,
                int(
                    np.ceil(
                        float(contamination)
                        * n_noise
                    )
                ),
            )

            anomaly_mask[
                order[:n_outliers]
            ] = True

        noise_scores = compute_enrichment_ad_diffi(
            iforest=noise_model,
            X_data=X_noise,
            feature_types=feature_types,
            anomaly_mask=anomaly_mask,
        )

        if continuous_indices:
            continuous_scores.extend(
                noise_scores[
                    continuous_indices
                ].tolist()
            )

        if binary_indices:
            binary_scores.extend(
                noise_scores[
                    binary_indices
                ].tolist()
            )

    cont_mean, cont_sd = _summarize_scores(
        continuous_scores
    )

    bin_mean, bin_sd = _summarize_scores(
        binary_scores
    )

    return (
        cont_mean,
        cont_sd,
        bin_mean,
        bin_sd,
    )


def calculate_ad_diffi_zscore(
    raw_scores: np.ndarray,
    feature_types: FeatureTypes,
    noise_baselines: NoiseBaselines,
) -> np.ndarray:
    """Calculate secondary type-specific null-standardized scores.

    Parameters
    ----------
    raw_scores:
        One raw score per feature. This may be enrichment-based AD-DIFFI
        output or another score with a matching type-specific null baseline.

    feature_types:
        Mapping from feature index to "cont" or "bin".

    noise_baselines:
        Dictionary returned by ``make_noise_baselines``.

    Returns
    -------
    np.ndarray
        Type-specific Z-standardized scores.

    Notes
    -----
    This transformation is monotone within a feature type, so it preserves
    within-type ordering. It can alter the ordering between continuous and
    binary features. It should therefore be used as a secondary null-reference
    analysis rather than automatically as the primary global ranking score.
    """
    raw_scores = np.asarray(
        raw_scores,
        dtype=float,
    )

    if raw_scores.ndim != 1:
        raise ValueError(
            "raw_scores must be a one-dimensional array."
        )

    n_features = len(raw_scores)

    _validate_feature_types(
        feature_types,
        n_features,
    )

    if not np.all(np.isfinite(raw_scores)):
        raise ValueError(
            "raw_scores must contain only finite values."
        )

    _validate_noise_baselines(noise_baselines)

    z_scores = np.empty_like(
        raw_scores,
        dtype=float,
    )

    for feature in range(n_features):
        feature_type = feature_types[feature]
        mean = float(
            noise_baselines[feature_type]["mean"]
        )
        sd = float(
            noise_baselines[feature_type]["sd"]
        )

        z_scores[feature] = (
            raw_scores[feature] - mean
        ) / sd

    return z_scores


def calculate_enrichment_ad_diffi(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: np.ndarray,
    if_params: Optional[dict] = None,
    n_noise_iter: int = 20,
    noise_random_state: Optional[int] = 12345,
    n_noise: int = 256,
    return_zscore: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Compute primary raw enrichment AD-DIFFI and optional Z-scores.

    Parameters
    ----------
    iforest:
        A fitted IsolationForest.

    X_data:
        Attribution data matrix.

    feature_types:
        Mapping from feature index to "cont" or "bin".

    anomaly_mask:
        Boolean model-defined outlier mask.

    if_params:
        Isolation Forest parameters for secondary null calibration. Required
        only when ``return_zscore=True``.

    n_noise_iter, noise_random_state, n_noise:
        Settings for optional secondary pure-noise calibration.

    return_zscore:
        If False, return only primary raw enrichment scores. If True, return
        ``(raw_scores, z_scores)``.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Raw enrichment scores, optionally paired with secondary type-specific
        Z-scores.
    """
    raw_scores = compute_enrichment_ad_diffi(
        iforest=iforest,
        X_data=X_data,
        feature_types=feature_types,
        anomaly_mask=anomaly_mask,
    )

    if not return_zscore:
        return raw_scores

    if if_params is None:
        raise ValueError(
            "if_params is required when return_zscore=True."
        )

    cont_mean, cont_sd, bin_mean, bin_sd = (
        get_enrichment_noise_baselines(
            X_dim=X_data.shape[1],
            feature_types=feature_types,
            if_params=if_params,
            n_iter=n_noise_iter,
            random_state=noise_random_state,
            n_noise=n_noise,
        )
    )

    baselines = make_noise_baselines(
        cont_mean=cont_mean,
        cont_sd=cont_sd,
        bin_mean=bin_mean,
        bin_sd=bin_sd,
    )

    z_scores = calculate_ad_diffi_zscore(
        raw_scores=raw_scores,
        feature_types=feature_types,
        noise_baselines=baselines,
    )

    return raw_scores, z_scores


# ----------------------------------------------------------------
# Legacy ratio-based functions
# ----------------------------------------------------------------
# These functions are retained for backward compatibility with
# prior AD-DIFFI experiments. They are not the primary method
# described in the enrichment-based AD-DIFFI workflow.
# ----------------------------------------------------------------
def _legacy_feature_path_terms(
    estimator,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return legacy RSO ratio-based numerator and denominator terms."""
    n_samples, n_features = X_data.shape

    tree = estimator.tree_
    lambdas = _node_lambdas(tree)
    paths = _paths_for_rows(
        estimator,
        X_data,
    )

    numerator = np.zeros(
        (n_samples, n_features),
        dtype=float,
    )

    denominator = np.zeros(
        (n_samples, n_features),
        dtype=float,
    )

    for row, path in enumerate(paths):
        internal_path = [
            node
            for node in path
            if tree.feature[node] >= 0
        ]

        path_length = max(
            len(internal_path),
            1,
        )

        for node in internal_path:
            feature = int(tree.feature[node])

            if not _eligible_node(
                node,
                feature,
                feature_types,
            ):
                continue

            numerator[row, feature] += (
                lambdas[node] / path_length
            )

            denominator[row, feature] += 1.0

    return numerator, denominator


def compute_group_cfi(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute legacy group-specific conditional feature importances.

    This function supports the previous ratio-based AD-DIFFI formulation.
    It is retained for backward compatibility and diagnostic analyses.
    """
    X_data, anomaly_mask = _validate_inputs(
        iforest,
        X_data,
        feature_types,
        anomaly_mask,
    )

    if anomaly_mask is None:
        raise ValueError(
            "anomaly_mask is required."
        )

    n_features = X_data.shape[1]

    cfi_outliers = np.zeros(
        n_features,
        dtype=float,
    )

    cfi_inliers = np.zeros(
        n_features,
        dtype=float,
    )

    for estimator in iforest.estimators_:
        numerator, denominator = _legacy_feature_path_terms(
            estimator,
            X_data,
            feature_types,
        )

        numerator_outliers = numerator[
            anomaly_mask
        ].sum(axis=0)

        denominator_outliers = denominator[
            anomaly_mask
        ].sum(axis=0)

        numerator_inliers = numerator[
            ~anomaly_mask
        ].sum(axis=0)

        denominator_inliers = denominator[
            ~anomaly_mask
        ].sum(axis=0)

        cfi_outliers += np.divide(
            numerator_outliers,
            denominator_outliers,
            out=np.zeros_like(numerator_outliers),
            where=denominator_outliers > 0,
        )

        cfi_inliers += np.divide(
            numerator_inliers,
            denominator_inliers,
            out=np.zeros_like(numerator_inliers),
            where=denominator_inliers > 0,
        )

    n_trees = len(iforest.estimators_)

    return (
        cfi_outliers / n_trees,
        cfi_inliers / n_trees,
    )


def compute_raw_ad_diffi(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the legacy outlier-to-inlier CFI ratio score.

    This function implements the prior ratio-based AD-DIFFI formulation and
    is retained for backward compatibility. Use
    ``compute_enrichment_ad_diffi`` for the primary enrichment-based method.
    """
    cfi_outliers, cfi_inliers = compute_group_cfi(
        iforest,
        X_data,
        feature_types,
        anomaly_mask,
    )

    raw_scores = np.divide(
        cfi_outliers,
        cfi_inliers,
        out=cfi_outliers.copy(),
        where=cfi_inliers > eps,
    )

    return (
        cfi_outliers,
        cfi_inliers,
        raw_scores,
    )


def get_noise_baselines(
    X_dim: int,
    feature_types: FeatureTypes,
    if_params: dict,
    n_iter: int = 20,
    random_state: Optional[int] = 12345,
) -> Tuple[float, float, float, float]:
    """Estimate legacy ratio-score pure-noise baselines.

    Retained for backward compatibility. Use
    ``get_enrichment_noise_baselines`` for the primary enrichment method.
    """
    if n_iter < 2:
        raise ValueError(
            "n_iter must be at least 2."
        )

    _validate_feature_types(
        feature_types,
        X_dim,
    )

    n_noise = 256
    rng = np.random.default_rng(random_state)

    continuous_indices = [
        feature
        for feature in range(X_dim)
        if feature_types[feature] == _CONTINUOUS
    ]

    binary_indices = [
        feature
        for feature in range(X_dim)
        if feature_types[feature] == _BINARY
    ]

    continuous_scores: list[float] = []
    binary_scores: list[float] = []

    for iteration in range(n_iter):
        X_noise = _generate_noise(
            n_samples=n_noise,
            n_features=X_dim,
            feature_types=feature_types,
            rng=rng,
        )

        params = dict(if_params)
        params.pop("random_state", None)

        params["max_samples"] = _resolve_max_samples(
            params.get("max_samples", "auto"),
            n_noise,
        )

        noise_model = IsolationForest(
            **params,
            random_state=(
                None
                if random_state is None
                else random_state + iteration
            ),
        ).fit(X_noise)

        anomaly_mask = noise_model.predict(X_noise) == -1

        if anomaly_mask.all() or (~anomaly_mask).all():
            order = np.argsort(
                noise_model.score_samples(X_noise)
            )

            anomaly_mask = np.zeros(
                n_noise,
                dtype=bool,
            )

            anomaly_mask[
                order[:max(1, n_noise // 10)]
            ] = True

        _, _, raw_scores = compute_raw_ad_diffi(
            noise_model,
            X_noise,
            feature_types,
            anomaly_mask,
        )

        if continuous_indices:
            continuous_scores.extend(
                raw_scores[
                    continuous_indices
                ].tolist()
            )

        if binary_indices:
            binary_scores.extend(
                raw_scores[
                    binary_indices
                ].tolist()
            )

    cont_mean, cont_sd = _summarize_scores(
        continuous_scores
    )

    bin_mean, bin_sd = _summarize_scores(
        binary_scores
    )

    return (
        cont_mean,
        cont_sd,
        bin_mean,
        bin_sd,
    )


def calculate_ad_diffi(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    if_params: dict,
    anomaly_mask: np.ndarray,
    n_noise_iter: int = 20,
    noise_random_state: Optional[int] = 12345,
    eps: float = 1e-12,
) -> np.ndarray:
    """Calculate legacy ratio-based AD-DIFFI Z-scores.

    This convenience function retains the previous API. For new analyses,
    prefer ``calculate_enrichment_ad_diffi`` or
    ``compute_enrichment_ad_diffi``.
    """
    _, _, raw_scores = compute_raw_ad_diffi(
        iforest=iforest,
        X_data=X_data,
        feature_types=feature_types,
        anomaly_mask=anomaly_mask,
        eps=eps,
    )

    cont_mean, cont_sd, bin_mean, bin_sd = (
        get_noise_baselines(
            X_dim=X_data.shape[1],
            feature_types=feature_types,
            if_params=if_params,
            n_iter=n_noise_iter,
            random_state=noise_random_state,
        )
    )

    baselines = make_noise_baselines(
        cont_mean=cont_mean,
        cont_sd=cont_sd,
        bin_mean=bin_mean,
        bin_sd=bin_sd,
    )

    return calculate_ad_diffi_zscore(
        raw_scores=raw_scores,
        feature_types=feature_types,
        noise_baselines=baselines,
    )
