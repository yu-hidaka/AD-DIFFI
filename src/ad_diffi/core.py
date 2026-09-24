from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest

FeatureTypes = Dict[int, str]


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
    if hasattr(iforest, "n_features_in_") and iforest.n_features_in_ != n_features:
        raise ValueError("X_data and iforest have different feature counts.")

    expected = set(range(n_features))
    missing = expected - set(feature_types)
    extra = set(feature_types) - expected
    if missing:
        raise ValueError(f"Missing feature types: {sorted(missing)}")
    if extra:
        raise ValueError(f"Invalid feature indices: {sorted(extra)}")
    invalid = {
        f: feature_types[f]
        for f in range(n_features)
        if feature_types[f] not in {"cont", "bin"}
    }
    if invalid:
        raise ValueError(f"Feature types must be 'cont' or 'bin': {invalid}")

    if anomaly_mask is not None:
        anomaly_mask = np.asarray(anomaly_mask, dtype=bool)
        if anomaly_mask.shape != (n_samples,):
            raise ValueError("anomaly_mask must have shape (n_samples,).")
        if anomaly_mask.all() or (~anomaly_mask).all():
            raise ValueError("Both outliers and inliers are required.")


def _node_lambdas(tree) -> np.ndarray:
    lambdas = np.zeros(tree.node_count, dtype=float)
    for node in range(tree.node_count):
        left, right = tree.children_left[node], tree.children_right[node]
        if left == right:
            continue
        n_parent = float(tree.n_node_samples[node])
        if n_parent > 0:
            lambdas[node] = 1.0 - min(
                tree.n_node_samples[left], tree.n_node_samples[right]
            ) / n_parent
    return lambdas


def _parent_array(tree) -> np.ndarray:
    parent = np.full(tree.node_count, -1, dtype=int)
    for node in range(tree.node_count):
        left, right = tree.children_left[node], tree.children_right[node]
        if left != right:
            parent[left] = node
            parent[right] = node
    return parent


def _paths_for_rows(estimator, X_data: np.ndarray) -> list[list[int]]:
    tree = estimator.tree_
    parent = _parent_array(tree)
    terminal_nodes = estimator.apply(X_data).astype(int)
    cache: dict[int, list[int]] = {}
    paths = []
    for terminal in terminal_nodes:
        terminal = int(terminal)
        if terminal not in cache:
            nodes = []
            node = terminal
            while node >= 0:
                nodes.append(node)
                node = parent[node]
            cache[terminal] = nodes[::-1]
        paths.append(cache[terminal])
    return paths


def _feature_path_terms(
    estimator,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    inverse_path_weight: str = "path_length",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return numerator and denominator terms for every sample and feature.

    For an observation x and tree t, a selected node v contributes

        lambda(v) / h_t(x)

    to the numerator, where h_t(x) is the number of internal split nodes on
    the observation's path. The denominator C counts feature-specific path
    appearances (data count x path occurrence count).

    Binary RSO features are evaluated at the root split. A binary feature not
    selected at the root contributes zero in that tree, as required by the
    literal RSO rule.
    """
    X_data = np.asarray(X_data, dtype=float)
    n_samples, n_features = X_data.shape
    tree = estimator.tree_
    lambdas = _node_lambdas(tree)
    paths = _paths_for_rows(estimator, X_data)

    numerator = np.zeros((n_samples, n_features), dtype=float)
    denominator = np.zeros((n_samples, n_features), dtype=float)

    for row, path in enumerate(paths):
        internal_path = [node for node in path if tree.feature[node] >= 0]
        if inverse_path_weight == "path_length":
            h = len(internal_path)
        elif inverse_path_weight == "depth":
            h = len(internal_path)
        else:
            raise ValueError("inverse_path_weight must be 'path_length' or 'depth'.")
        h = max(h, 1)

        for node in internal_path:
            feature = int(tree.feature[node])
            ftype = feature_types[feature]
            selected = ftype == "cont" or (ftype == "bin" and node == 0)
            if not selected:
                continue

            numerator[row, feature] += lambdas[node] / h
            denominator[row, feature] += 1.0

    return numerator, denominator


def compute_group_cfi(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute group-wise CFI = sum(FI) / sum(C).

    For each tree and group, the numerator is the sum of lambda(v)/h_t(x)
    terms. The denominator is the corresponding number of feature-specific
    path occurrences. CFI is then averaged over trees.
    """
    X_data = np.asarray(X_data, dtype=float)
    anomaly_mask = np.asarray(anomaly_mask, dtype=bool)
    _validate_inputs(iforest, X_data, feature_types, anomaly_mask)

    n_features = X_data.shape[1]
    cfi_out = np.zeros(n_features, dtype=float)
    cfi_in = np.zeros(n_features, dtype=float)

    for estimator in iforest.estimators_:
        numerator, denominator = _feature_path_terms(
            estimator,
            X_data,
            feature_types,
        )

        num_out = numerator[anomaly_mask].sum(axis=0)
        den_out = denominator[anomaly_mask].sum(axis=0)
        num_in = numerator[~anomaly_mask].sum(axis=0)
        den_in = denominator[~anomaly_mask].sum(axis=0)

        cfi_out += np.divide(
            num_out,
            den_out,
            out=np.zeros_like(num_out),
            where=den_out > 0,
        )
        cfi_in += np.divide(
            num_in,
            den_in,
            out=np.zeros_like(num_in),
            where=den_in > 0,
        )

    n_trees = len(iforest.estimators_)
    return cfi_out / n_trees, cfi_in / n_trees


def compute_raw_ad_diffi(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: FeatureTypes,
    anomaly_mask: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute outlier CFI, inlier CFI, and their raw ratio."""
    cfi_out, cfi_in = compute_group_cfi(
        iforest,
        X_data,
        feature_types,
        anomaly_mask,
    )
    raw = np.where(cfi_in > eps, cfi_out / cfi_in, cfi_out)
    return cfi_out, cfi_in, raw


def _resolve_max_samples(max_samples, n_samples: int) -> int:
    if max_samples == "auto":
        return min(256, n_samples)
    if isinstance(max_samples, (float, np.floating)):
        if not 0 < max_samples <= 1:
            raise ValueError("Float max_samples must be in (0, 1].")
        return max(2, int(np.ceil(max_samples * n_samples)))
    return min(int(max_samples), n_samples)


def _generate_noise(
    n_samples: int,
    n_features: int,
    feature_types: FeatureTypes,
    rng: np.random.Generator,
) -> np.ndarray:
    X_noise = np.empty((n_samples, n_features), dtype=float)
    for feature in range(n_features):
        if feature_types[feature] == "cont":
            X_noise[:, feature] = rng.uniform(0.0, 1.0, n_samples)
        else:
            X_noise[:, feature] = rng.binomial(1, 0.5, n_samples)
    return X_noise


def get_noise_baselines(
    X_dim: int,
    feature_types: FeatureTypes,
    if_params: dict,
    n_iter: int = 20,
    random_state: Optional[int] = 12345,
) -> Tuple[float, float, float, float]:
    """Estimate type-specific mean and SD of the same raw CFI ratio."""
    if n_iter < 2:
        raise ValueError("n_iter must be at least 2.")

    n_noise = 256
    rng = np.random.default_rng(random_state)
    cont_idx = [f for f in range(X_dim) if feature_types[f] == "cont"]
    bin_idx = [f for f in range(X_dim) if feature_types[f] == "bin"]
    cont_scores, bin_scores = [], []

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
            random_state=None if random_state is None else random_state + iteration,
        ).fit(X_noise)

        noise_mask = noise_model.predict(X_noise) == -1
        if noise_mask.all() or (~noise_mask).all():
            order = np.argsort(noise_model.score_samples(X_noise))
            noise_mask = np.zeros(n_noise, dtype=bool)
            noise_mask[order[:max(1, n_noise // 10)]] = True

        _, _, raw_noise = compute_raw_ad_diffi(
            noise_model,
            X_noise,
            feature_types,
            noise_mask,
        )
        if cont_idx:
            cont_scores.extend(raw_noise[cont_idx].tolist())
        if bin_idx:
            bin_scores.extend(raw_noise[bin_idx].tolist())

    def summarize(values: list[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 1.0
        values = np.asarray(values, dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        return mean, sd if sd > 0 else 1.0

    cont_mean, cont_sd = summarize(cont_scores)
    bin_mean, bin_sd = summarize(bin_scores)
    return cont_mean, cont_sd, bin_mean, bin_sd


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
    """Calculate theory-aligned AD-DIFFI."""
    X_data = np.asarray(X_data, dtype=float)
    anomaly_mask = np.asarray(anomaly_mask, dtype=bool)
    _validate_inputs(iforest, X_data, feature_types, anomaly_mask)

    _, _, raw_scores = compute_raw_ad_diffi(
        iforest,
        X_data,
        feature_types,
        anomaly_mask,
        eps=eps,
    )
    cont_mean, cont_sd, bin_mean, bin_sd = get_noise_baselines(
        X_data.shape[1],
        feature_types,
        if_params,
        n_iter=n_noise_iter,
        random_state=noise_random_state,
    )

    is_cont = np.array(
        [feature_types[f] == "cont" for f in range(X_data.shape[1])],
        dtype=bool,
    )
    return np.where(
        is_cont,
        (raw_scores - cont_mean) / cont_sd,
        (raw_scores - bin_mean) / bin_sd,
    )
