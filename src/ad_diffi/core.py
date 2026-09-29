from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest


FeatureTypes = Dict[int, str]
NoiseBaselines = Dict[str, Dict[str, float]]


def _validate_inputs(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: Optional[np.ndarray] = None,
) -> None:
    X_data = np.asarray(X_data)

    if X_data.ndim != 2:
        raise ValueError("X_data must be a two-dimensional array.")

    if not hasattr(iforest, "estimators_") or not iforest.estimators_:
        raise ValueError("iforest must be a fitted IsolationForest.")

    n_samples, n_features = X_data.shape

    if (
        hasattr(iforest, "n_features_in_")
        and iforest.n_features_in_ != n_features
    ):
        raise ValueError(
            "X_data and iforest have different feature counts."
        )

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
        if feature_types[feature] not in {"cont", "bin"}
    }

    if invalid_types:
        raise ValueError(
            "Feature types must be 'cont' or 'bin': "
            f"{invalid_types}"
        )

    if anomaly_mask is not None:
        anomaly_mask = np.asarray(anomaly_mask, dtype=bool)

        if anomaly_mask.shape != (n_samples,):
            raise ValueError(
                "anomaly_mask must have shape (n_samples,)."
            )

        if anomaly_mask.all() or (~anomaly_mask).all():
            raise ValueError(
                "Both outliers and inliers are required."
            )


def _node_lambdas(tree) -> np.ndarray:
    """Calculate raw node-level split-imbalance coefficients."""
    lambdas = np.zeros(tree.node_count, dtype=float)

    for node in range(tree.node_count):
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left == right:
            continue

        n_parent = float(tree.n_node_samples[node])

        if n_parent > 0:
            lambdas[node] = 1.0 - (
                min(
                    tree.n_node_samples[left],
                    tree.n_node_samples[right],
                )
                / n_parent
            )

    return lambdas


def _parent_array(tree) -> np.ndarray:
    """Return the parent-node index for every node in a tree."""
    parent = np.full(tree.node_count, -1, dtype=int)

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
    """Return root-to-terminal-node paths for all observations."""
    tree = estimator.tree_
    parent = _parent_array(tree)

    terminal_nodes = estimator.apply(X_data).astype(int)

    path_cache: dict[int, list[int]] = {}
    paths: list[list[int]] = []

    for terminal_node in terminal_nodes:
        terminal_node = int(terminal_node)

        if terminal_node not in path_cache:
            nodes = []
            node = terminal_node

            while node >= 0:
                nodes.append(node)
                node = parent[node]

            path_cache[terminal_node] = nodes[::-1]

        paths.append(path_cache[terminal_node])

    return paths


def _root_outlier_enrichment(
    estimator,
    X_data: np.ndarray,
    anomaly_mask: np.ndarray,
) -> float:
    """Calculate absolute outlier-prevalence difference across root children.

    The enrichment is

        | P(outlier | root-left) - P(outlier | root-right) |.

    It is zero if the root is a leaf or either root child receives no
    observations from X_data.
    """
    tree = estimator.tree_

    root = 0
    left = tree.children_left[root]
    right = tree.children_right[root]

    if left == right:
        return 0.0

    node_indicator = estimator.decision_path(X_data)

    in_left = node_indicator[:, left].toarray().ravel().astype(bool)
    in_right = node_indicator[:, right].toarray().ravel().astype(bool)

    n_left = int(in_left.sum())
    n_right = int(in_right.sum())

    if n_left == 0 or n_right == 0:
        return 0.0

    outlier_rate_left = float(anomaly_mask[in_left].mean())
    outlier_rate_right = float(anomaly_mask[in_right].mean())

    return abs(outlier_rate_left - outlier_rate_right)


def _feature_path_terms(
    estimator,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return feature-specific numerator and denominator terms.

    For an observation x and tree t, a continuous feature selected at internal
    node v contributes

        lambda(v) / h_t(x),

    where h_t(x) is the number of internal split nodes on the root-to-leaf
    path.

    Binary features are evaluated only at the root split under the
    Root-Split-Only constraint. If the root split feature is binary, its
    contribution is further multiplied by the root outlier-enrichment factor

        E_t = |P(outlier | root-left) - P(outlier | root-right)|.

    Thus, a binary root split receives large attribution only when it both
    creates an imbalanced partition and separates model-defined outliers from
    inliers across its two root children.
    """
    X_data = np.asarray(X_data, dtype=float)
    anomaly_mask = np.asarray(anomaly_mask, dtype=bool)

    n_samples, n_features = X_data.shape

    tree = estimator.tree_
    lambdas = _node_lambdas(tree)
    paths = _paths_for_rows(estimator, X_data)

    root_enrichment = _root_outlier_enrichment(
        estimator,
        X_data,
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

        path_length = max(len(internal_path), 1)

        for node in internal_path:
            feature = int(tree.feature[node])
            feature_type = feature_types[feature]

            if feature_type == "cont":
                contribution = (
                    lambdas[node] / path_length
                )

            elif feature_type == "bin" and node == 0:
                contribution = (
                    lambdas[node]
                    * root_enrichment
                    / path_length
                )

            else:
                continue

            numerator[row, feature] += contribution
            denominator[row, feature] += 1.0

    return numerator, denominator


def compute_group_cfi(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute forest-averaged outlier and inlier CFIs.

    For each tree and group, the CFI is the sum of feature-specific,
    inverse-path-length-weighted contributions divided by the corresponding
    feature-specific path-occurrence count. Continuous nodes use lambda(v),
    whereas binary root nodes use lambda(root) times root outlier enrichment.
    Tree-specific CFIs are averaged over all trees.
    """
    X_data = np.asarray(X_data, dtype=float)
    anomaly_mask = np.asarray(anomaly_mask, dtype=bool)

    _validate_inputs(
        iforest,
        X_data,
        feature_types,
        anomaly_mask,
    )

    n_features = X_data.shape[1]

    cfi_outliers = np.zeros(n_features, dtype=float)
    cfi_inliers = np.zeros(n_features, dtype=float)

    for estimator in iforest.estimators_:
        numerator, denominator = _feature_path_terms(
            estimator,
            X_data,
            feature_types,
            anomaly_mask,
        )

        numerator_outliers = numerator[anomaly_mask].sum(axis=0)
        denominator_outliers = denominator[anomaly_mask].sum(axis=0)

        numerator_inliers = numerator[~anomaly_mask].sum(axis=0)
        denominator_inliers = denominator[~anomaly_mask].sum(axis=0)

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
    """Compute outlier CFI, inlier CFI, and raw AD-DIFFI ratio."""
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

    return cfi_outliers, cfi_inliers, raw_scores

def _node_outlier_enrichment(
    estimator,
    X_data: np.ndarray,
    anomaly_mask: np.ndarray,
) -> np.ndarray:
    """Return outlier-prevalence contrasts for all internal tree nodes.

    For an internal node v, the enrichment is

        |P(outlier | left child of v) - P(outlier | right child of v)|.

    Leaf-node enrichment is zero. Node membership is evaluated using the
    full X_data and the supplied model-defined anomaly mask.
    """
    X_data = np.asarray(X_data, dtype=float)
    anomaly_mask = np.asarray(anomaly_mask, dtype=bool)

    tree = estimator.tree_
    node_indicator = estimator.decision_path(X_data)

    enrichment = np.zeros(tree.node_count, dtype=float)

    for node in range(tree.node_count):
        left = tree.children_left[node]
        right = tree.children_right[node]

        if left == right:
            continue

        in_left = (
            node_indicator[:, left]
            .toarray()
            .ravel()
            .astype(bool)
        )

        in_right = (
            node_indicator[:, right]
            .toarray()
            .ravel()
            .astype(bool)
        )

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


def _feature_enrichment_terms(
    estimator,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-sample feature contributions and path-occurrence counts.

    Continuous features contribute at all eligible internal nodes. Binary
    features contribute only if selected at the root node.

    For an eligible node v on the path of observation x, the contribution is

        lambda(v) * E(v) / h_t(x),

    where E(v) is the node-level outlier-prevalence contrast and h_t(x) is
    the number of internal split nodes in the path.
    """
    X_data = np.asarray(X_data, dtype=float)
    anomaly_mask = np.asarray(anomaly_mask, dtype=bool)

    n_samples, n_features = X_data.shape

    tree = estimator.tree_
    lambdas = _node_lambdas(tree)
    enrichment = _node_outlier_enrichment(
        estimator,
        X_data,
        anomaly_mask,
    )

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
            feature_type = feature_types[feature]

            is_eligible = (
                feature_type == "cont"
                or (
                    feature_type == "bin"
                    and node == 0
                )
            )

            if not is_eligible:
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
    """Compute enrichment-based raw AD-DIFFI feature importance.

    The score is the forest mean of tree-specific feature contributions. Each
    tree-specific contribution is the mean lambda(v) * E(v) / h_t(x) over
    feature-specific path occurrences.

    Continuous features are eligible at all internal split nodes. Binary
    features are eligible only when selected at the root split.
    """
    X_data = np.asarray(X_data, dtype=float)
    anomaly_mask = np.asarray(anomaly_mask, dtype=bool)

    _validate_inputs(
        iforest,
        X_data,
        feature_types,
        anomaly_mask,
    )

    n_features = X_data.shape[1]

    importance = np.zeros(
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

        numerator_total = numerator.sum(axis=0)
        denominator_total = denominator.sum(axis=0)

        importance += np.divide(
            numerator_total,
            denominator_total,
            out=np.zeros_like(numerator_total),
            where=denominator_total > 0,
        )

    return importance / len(iforest.estimators_)

def get_enrichment_noise_baselines(
    X_dim: int,
    feature_types: FeatureTypes,
    if_params: dict,
    n_iter: int = 20,
    random_state: Optional[int] = 12345,
    n_noise: int = 256,
) -> Tuple[float, float, float, float]:
    """Estimate type-specific null means and SDs for enrichment AD-DIFFI.

    Synthetic noise datasets are generated with independent feature-type-
    specific distributions:

    - continuous features: Uniform(0, 1)
    - binary features: Bernoulli(0.5)

    For each replication, an Isolation Forest is fitted using ``if_params``.
    Model-defined outliers are obtained using ``predict(X_noise) == -1``.
    Enrichment-based raw feature scores are then computed using
    ``compute_enrichment_ad_diffi``.

    Returns
    -------
    tuple[float, float, float, float]
        ``(cont_mean, cont_sd, bin_mean, bin_sd)``.
    """
    if n_iter < 2:
        raise ValueError("n_iter must be at least 2.")

    if n_noise < 2:
        raise ValueError("n_noise must be at least 2.")

    expected_indices = set(range(X_dim))

    if set(feature_types) != expected_indices:
        raise ValueError(
            "feature_types must contain exactly indices "
            f"0 through {X_dim - 1}."
        )

    invalid_types = {
        feature: feature_types[feature]
        for feature in range(X_dim)
        if feature_types[feature] not in {"cont", "bin"}
    }

    if invalid_types:
        raise ValueError(
            "Feature types must be 'cont' or 'bin': "
            f"{invalid_types}"
        )

    rng = np.random.default_rng(random_state)

    continuous_indices = [
        feature
        for feature in range(X_dim)
        if feature_types[feature] == "cont"
    ]

    binary_indices = [
        feature
        for feature in range(X_dim)
        if feature_types[feature] == "bin"
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

        noise_mask = noise_model.predict(X_noise) == -1

        if noise_mask.all() or (~noise_mask).all():
            order = np.argsort(
                noise_model.score_samples(X_noise)
            )

            noise_mask = np.zeros(
                n_noise,
                dtype=bool,
            )

            n_outliers = max(
                1,
                int(
                    np.ceil(
                        params.get("contamination", 0.1)
                        * n_noise
                    )
                ),
            )

            noise_mask[order[:n_outliers]] = True

        enrichment_scores = compute_enrichment_ad_diffi(
            iforest=noise_model,
            X_data=X_noise,
            feature_types=feature_types,
            anomaly_mask=noise_mask,
        )

        if continuous_indices:
            continuous_scores.extend(
                enrichment_scores[
                    continuous_indices
                ].tolist()
            )

        if binary_indices:
            binary_scores.extend(
                enrichment_scores[
                    binary_indices
                ].tolist()
            )

    def summarize(
        values: list[float],
    ) -> Tuple[float, float]:
        if not values:
            return 0.0, 1.0

        values_array = np.asarray(
            values,
            dtype=float,
        )

        mean = float(values_array.mean())
        sd = float(values_array.std(ddof=1))

        return mean, sd if sd > 0 else 1.0

    cont_mean, cont_sd = summarize(
        continuous_scores
    )

    bin_mean, bin_sd = summarize(
        binary_scores
    )

    return cont_mean, cont_sd, bin_mean, bin_sd
    
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

    return min(int(max_samples), n_samples)


def _generate_noise(
    n_samples: int,
    n_features: int,
    feature_types: FeatureTypes,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate type-specific maximum-entropy synthetic noise."""
    X_noise = np.empty(
        (n_samples, n_features),
        dtype=float,
    )

    for feature in range(n_features):
        if feature_types[feature] == "cont":
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


def get_noise_baselines(
    X_dim: int,
    feature_types: FeatureTypes,
    if_params: dict,
    n_iter: int = 20,
    random_state: Optional[int] = 12345,
) -> Tuple[float, float, float, float]:
    """Estimate type-specific null means and SDs of raw AD-DIFFI scores.

    The returned tuple is:

        (cont_mean, cont_sd, bin_mean, bin_sd)
    """
    if n_iter < 2:
        raise ValueError("n_iter must be at least 2.")

    expected_indices = set(range(X_dim))

    if set(feature_types) != expected_indices:
        raise ValueError(
            "feature_types must contain exactly indices "
            f"0 through {X_dim - 1}."
        )

    invalid_types = {
        feature: feature_types[feature]
        for feature in range(X_dim)
        if feature_types[feature] not in {"cont", "bin"}
    }

    if invalid_types:
        raise ValueError(
            "Feature types must be 'cont' or 'bin': "
            f"{invalid_types}"
        )

    n_noise = 256
    rng = np.random.default_rng(random_state)

    continuous_indices = [
        feature
        for feature in range(X_dim)
        if feature_types[feature] == "cont"
    ]

    binary_indices = [
        feature
        for feature in range(X_dim)
        if feature_types[feature] == "bin"
    ]

    continuous_scores: list[float] = []
    binary_scores: list[float] = []

    for iteration in range(n_iter):
        X_noise = _generate_noise(
            n_noise,
            X_dim,
            feature_types,
            rng,
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

        noise_mask = noise_model.predict(X_noise) == -1

        if noise_mask.all() or (~noise_mask).all():
            order = np.argsort(
                noise_model.score_samples(X_noise)
            )

            noise_mask = np.zeros(n_noise, dtype=bool)

            n_outliers = max(
                1,
                n_noise // 10,
            )

            noise_mask[order[:n_outliers]] = True

        _, _, raw_noise_scores = compute_raw_ad_diffi(
            noise_model,
            X_noise,
            feature_types,
            noise_mask,
        )

        if continuous_indices:
            continuous_scores.extend(
                raw_noise_scores[continuous_indices].tolist()
            )

        if binary_indices:
            binary_scores.extend(
                raw_noise_scores[binary_indices].tolist()
            )

    def summarize(
        values: list[float],
    ) -> Tuple[float, float]:
        if not values:
            return 0.0, 1.0

        values_array = np.asarray(values, dtype=float)

        mean = float(values_array.mean())
        sd = float(values_array.std(ddof=1))

        return mean, sd if sd > 0 else 1.0

    cont_mean, cont_sd = summarize(continuous_scores)
    bin_mean, bin_sd = summarize(binary_scores)

    return cont_mean, cont_sd, bin_mean, bin_sd


def make_noise_baselines(
    cont_mean: float,
    cont_sd: float,
    bin_mean: float,
    bin_sd: float,
) -> NoiseBaselines:
    """Create validated type-specific noise-baseline dictionary."""
    baselines: NoiseBaselines = {
        "cont": {
            "mean": float(cont_mean),
            "sd": float(cont_sd),
        },
        "bin": {
            "mean": float(bin_mean),
            "sd": float(bin_sd),
        },
    }

    _validate_noise_baselines(baselines)

    return baselines


def _validate_noise_baselines(
    noise_baselines: NoiseBaselines,
) -> None:
    """Validate a type-specific noise-baseline dictionary."""
    required_types = {"cont", "bin"}

    missing_types = required_types - set(noise_baselines)

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

        mean = baseline["mean"]
        sd = baseline["sd"]

        if not np.isfinite(mean):
            raise ValueError(
                f"Baseline mean for '{feature_type}' must be finite."
            )

        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(
                f"Baseline SD for '{feature_type}' must be finite "
                "and greater than zero."
            )


def calculate_ad_diffi_zscore(
    raw_scores: np.ndarray,
    feature_types: FeatureTypes,
    noise_baselines: NoiseBaselines,
) -> np.ndarray:
    """Standardize raw AD-DIFFI scores using precomputed noise baselines."""
    raw_scores = np.asarray(raw_scores, dtype=float)

    if raw_scores.ndim != 1:
        raise ValueError(
            "raw_scores must be a one-dimensional array."
        )

    n_features = len(raw_scores)

    expected_indices = set(range(n_features))

    if set(feature_types) != expected_indices:
        raise ValueError(
            "feature_types must contain exactly one valid type "
            "for every raw_scores index."
        )

    invalid_types = {
        feature: feature_types[feature]
        for feature in range(n_features)
        if feature_types[feature] not in {"cont", "bin"}
    }

    if invalid_types:
        raise ValueError(
            "Feature types must be 'cont' or 'bin': "
            f"{invalid_types}"
        )

    if not np.all(np.isfinite(raw_scores)):
        raise ValueError(
            "raw_scores must contain only finite values."
        )

    _validate_noise_baselines(noise_baselines)

    z_scores = np.empty_like(raw_scores, dtype=float)

    for feature in range(n_features):
        feature_type = feature_types[feature]

        mean = noise_baselines[feature_type]["mean"]
        sd = noise_baselines[feature_type]["sd"]

        z_scores[feature] = (
            raw_scores[feature] - mean
        ) / sd

    return z_scores


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
    """Calculate AD-DIFFI with an internally estimated noise baseline.

    For repeated analyses on the same scenario, use
    ``compute_raw_ad_diffi()``, ``get_noise_baselines()``,
    ``make_noise_baselines()``, and ``calculate_ad_diffi_zscore()`` instead.
    """
    X_data = np.asarray(X_data, dtype=float)
    anomaly_mask = np.asarray(anomaly_mask, dtype=bool)

    _validate_inputs(
        iforest,
        X_data,
        feature_types,
        anomaly_mask,
    )

    _, _, raw_scores = compute_raw_ad_diffi(
        iforest,
        X_data,
        feature_types,
        anomaly_mask,
        eps=eps,
    )

    cont_mean, cont_sd, bin_mean, bin_sd = get_noise_baselines(
        X_dim=X_data.shape[1],
        feature_types=feature_types,
        if_params=if_params,
        n_iter=n_noise_iter,
        random_state=noise_random_state,
    )

    noise_baselines = make_noise_baselines(
        cont_mean=cont_mean,
        cont_sd=cont_sd,
        bin_mean=bin_mean,
        bin_sd=bin_sd,
    )

    return calculate_ad_diffi_zscore(
        raw_scores=raw_scores,
        feature_types=feature_types,
        noise_baselines=noise_baselines,
    )
