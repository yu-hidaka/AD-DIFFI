"""
AD-DIFFI: Adjusted Depth-based Isolation Forest Feature Importance
Root-Split-Only (RSO) constraint + Noise-based Z-Normalization for Mixed-Type Data

Core functions for Chapter 5 simulations and real-data applications.
Usage:
from ad_diffi import diffi_ib_binary_rso, run_noise_only_rso_simulation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Tuple, Dict, List

# --- Core Settings (Shared across simulations) ---
MIN_DEPTH = 1       # Minimum depth for leaf nodes (denominator safeguard)
N_ITER = 100        # Iterations for signal datasets
N_ITER_NOISE = 100  # Iterations for noise baseline
N_TREES = 100
MAX_SAMPLES = 256
CONTAMINATION = 0.05
RANDOM_SEED_BASE = 42

# --- Feature Definitions (Chapter 5: 8 mixed features) ---
FEATURE_DEFINITIONS = [
    {'name': 'X1_Strong_Cont', 'type': 'cont', 'is_signal': True, 'strength': 'strong'},
    {'name': 'X2_Weak_Cont_A', 'type': 'cont', 'is_signal': True, 'strength': 'weak'},
    {'name': 'X3_Weak_Cont_B', 'type': 'cont', 'is_signal': True, 'strength': 'weak'},
    {'name': 'X4_Strong_Bin', 'type': 'bin', 'is_signal': True, 'strength': 'strong'},
    {'name': 'X5_Weak_Bin', 'type': 'bin', 'is_signal': True, 'strength': 'weak'},
    {'name': 'X6_Cont_Noise', 'type': 'cont', 'is_signal': False, 'strength': 'noise'},
    {'name': 'X7_Bin_Noise_A', 'type': 'bin', 'is_signal': False, 'strength': 'noise'},
    {'name': 'X8_Bin_Noise_B', 'type': 'bin', 'is_signal': False, 'strength': 'noise'},
]
# Feature indices and types
FEATURE_TYPES = {i: f['type'] for i, f in enumerate(FEATURE_DEFINITIONS)}
FEATURE_NAMES = [f['name'] for f in FEATURE_DEFINITIONS]
CONT_INDICES = [i for i, f in enumerate(FEATURE_DEFINITIONS) if f['type'] == 'cont']
BIN_INDICES = [i for i, f in enumerate(FEATURE_DEFINITIONS) if f['type'] == 'bin']

# --- Data Generation ---
def generate_mixed_signal_dataset_general(n_total: int = 1000, contamination: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """Generate mixed signal + noise dataset for anomaly detection testing."""
    n_normal = int(n_total * (1 - contamination))
    n_anomaly = n_total - n_normal
    rng = np.random.default_rng(seed=seed)
    signal_features = [f for f in FEATURE_DEFINITIONS if f['is_signal']]
    n_signals = len(signal_features)
    n_a_per_signal = n_anomaly // n_signals if n_signals > 0 else 0
    X_dict = {}
    for i, feat in enumerate(FEATURE_DEFINITIONS):
        name, ftype, is_sig, strength = feat['name'], feat['type'], feat['is_signal'], feat['strength']
        if is_sig:
            n_a_current = n_a_per_signal + (1 if i < (n_anomaly % n_signals) else 0)
            if ftype == 'cont':
                loc_sig = 50 if strength == 'strong' else 15
                scale_sig = 0.5 if strength == 'strong' else 1
                X_normal = rng.normal(loc=10, scale=1, size=n_normal)
                X_sig = rng.normal(loc=loc_sig, scale=scale_sig, size=n_a_current)
                X_all = np.concatenate([X_normal, X_sig])
            elif ftype == 'bin':
                p_sig_one = 0.9 if strength == 'strong' else 0.5
                X_normal = np.zeros(n_normal)
                X_sig_ones = np.ones(int(n_a_current * p_sig_one))
                X_sig_zeros = np.zeros(n_a_current - len(X_sig_ones))
                X_all = np.concatenate([X_normal, X_sig_ones, X_sig_zeros])
            if len(X_all) < n_total:
                X_filler = rng.choice([0, 1], size=n_total - len(X_all), p=[0.5, 0.5]) if ftype == 'bin' else rng.uniform(0, 20, size=n_total - len(X_all))
                X_all = np.concatenate([X_all, X_filler])
            X_dict[name] = X_all[:n_total]
        else:
            if ftype == 'cont':
                X_dict[name] = rng.uniform(low=0, high=20, size=n_total)
            elif ftype == 'bin':
                X_dict[name] = rng.choice([0, 1], size=n_total, p=[0.5, 0.5])
    X = pd.DataFrame(X_dict)
    df_all = X.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df_all

def generate_noise_only_dataset_general(n_total: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate pure noise dataset for baseline estimation."""
    rng = np.random.default_rng(seed=seed)
    X_dict = {}
    for feat in FEATURE_DEFINITIONS:
        name, ftype = feat['name'], feat['type']
        if ftype == 'cont':
            X_dict[name] = rng.uniform(low=0, high=20, size=n_total)
        elif ftype == 'bin':
            X_dict[name] = rng.choice([0, 1], size=n_total, p=[0.5, 0.5])
    return pd.DataFrame(X_dict)

# --- RSO-Constrained DIFFI Computation ---
def diffi_ib_binary_rso(iforest: IsolationForest, X_data: np.ndarray, feature_types: Dict[int, str]) -> np.ndarray:
    """Compute DIFFI scores with Root-Split-Only (RSO) constraint for binary features."""
    num_feat = X_data.shape[1]
    estimators = iforest.estimators_
    cfi_outliers_ib = np.zeros(num_feat, dtype=float)
    cfi_inliers_ib = np.zeros(num_feat, dtype=float)
    counter_outliers_ib = np.zeros(num_feat, dtype=int)
    counter_inliers_ib = np.zeros(num_feat, dtype=int)
    in_bag_samples = iforest.estimators_samples_

    global_scores = iforest.decision_function(X_data)
    global_threshold = np.percentile(global_scores, 100 * iforest.contamination)

    for k, estimator in enumerate(estimators):
        in_bag_sample = list(in_bag_samples[k])
        X_ib = X_data[in_bag_sample, :]
        scores_ib = global_scores[in_bag_sample]

        X_outliers_ib = X_ib[scores_ib < global_threshold]
        X_inliers_ib = X_ib[scores_ib >= global_threshold]
        if X_inliers_ib.shape[0] == 0 or X_outliers_ib.shape[0] == 0:
            continue

        tree = estimator.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        n_samples_node = tree.n_node_samples

        # Compute node depths
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        stack = [(0, -1)]
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
            if children_left[node_id] != children_right[node_id]:
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))

        # Lambda adjustment
        current_lambda_adj = np.zeros(shape=n_nodes, dtype=float)
        for node in range(n_nodes):
            if children_left[node] != children_right[node]:
                n_parent = n_samples_node[node]
                n_left = n_samples_node[children_left[node]]
                n_right = n_samples_node[children_right[node]]
                ratio_small = np.min([n_left, n_right]) / n_parent
                current_lambda_adj[node] = 1.0 - ratio_small

        def calculate_contribution(X_path, cfi, counter):
            node_indicator_all_points = estimator.decision_path(X_path)
            node_indicator_all_points_array = node_indicator_all_points.toarray()
            for i in range(len(X_path)):
                path = list(np.where(node_indicator_all_points_array[i] == 1)[0])
                if len(path) == 0:
                    continue

                depth_leaf = node_depth[path[-1]]
                h_leaf = max(depth_leaf, MIN_DEPTH)
                depth_scaling = 1.0 / h_leaf

                for node in path:
                    current_feature_index = feature[node]

                    if current_feature_index >= 0:
                        feature_type = feature_types[current_feature_index]

                        # --- RSO Constraint: Skip binary splits except root (depth=0) ---
                        if feature_type == 'bin' and node_depth[node] > 0:
                            continue
                        # -------------------------------------------------------------

                        lambda_val = current_lambda_adj[node]
                        contribution = depth_scaling * lambda_val
                        cfi[current_feature_index] += contribution
                        counter[current_feature_index] += 1

        calculate_contribution(X_outliers_ib, cfi_outliers_ib, counter_outliers_ib)
        calculate_contribution(X_inliers_ib, cfi_inliers_ib, counter_inliers_ib)

    fi_outliers_ib = np.where(counter_outliers_ib > 0, cfi_outliers_ib / counter_outliers_ib, 0)
    fi_inliers_ib = np.where(counter_inliers_ib > 0, cfi_inliers_ib / counter_inliers_ib, 0)
    fi_ib_rso = np.divide(fi_outliers_ib, fi_inliers_ib, out=np.zeros_like(fi_outliers_ib), where=fi_inliers_ib != 0)
    return fi_ib_rso

# --- Noise Baseline ---
def run_noise_only_rso_simulation(n_iter_noise: int, feature_types: Dict[int, str]) -> Tuple[float, float, float, float]:
    """Establish noise baselines (mean/SD) for continuous and binary features using RSO DIFFI."""
    print(f"\n--- Establishing Noise Baseline (DIFFI RSO) over {n_iter_noise} iterations ---")
    raw_scores_list = []

    for k in range(n_iter_noise):
        X_data = generate_noise_only_dataset_general(seed=RANDOM_SEED_BASE + k).values
        iforest = IsolationForest(
            n_estimators=N_TREES, max_samples=MAX_SAMPLES,
            contamination=CONTAMINATION, random_state=k, bootstrap=False,
        )
        iforest.fit(X_data)
        fi_diffi_raw = diffi_ib_binary_rso(iforest, X_data, feature_types)
        raw_scores_list.append(fi_diffi_raw)

    fi_diffi_matrix = np.vstack(raw_scores_list)

    # Continuous features baseline
    cont_scores = fi_diffi_matrix[:, CONT_INDICES]
    mean_cont_noise_fi = cont_scores.mean() if CONT_INDICES else 0.0
    sd_cont_noise_fi = cont_scores.std() if CONT_INDICES else 0.0

    # Binary features baseline
    bin_scores = fi_diffi_matrix[:, BIN_INDICES]
    mean_binary_noise_fi = bin_scores.mean() if BIN_INDICES else 0.0
    sd_binary_noise_fi = bin_scores.std() if BIN_INDICES else 0.0

    print(f"    Cont. FI Noise Baseline (Mean): {mean_cont_noise_fi:.6f}, (SD): {sd_cont_noise_fi:.6f}")
    print(f"    Binary FI Noise Baseline (Mean): {mean_binary_noise_fi:.6f}, (SD): {sd_binary_noise_fi:.6f}")

    return mean_cont_noise_fi, sd_cont_noise_fi, mean_binary_noise_fi, sd_binary_noise_fi

# --- Score Collection and Standardization ---
def diffi_score_collection_and_standardize_rso(
    n_iter: int, feature_types: Dict[int, str],
    cont_mean: float, cont_sd: float, bin_mean: float, bin_sd: float
) -> pd.DataFrame:
    """Collect raw DIFFI RSO scores over iterations and apply type-specific Z-standardization (AD-DIFFI)."""
    raw_scores_list = []
    print(f"\n--- Running {n_iter} iterations for DIFFI RSO scores ---")

    for k in range(n_iter):
        X_data = generate_mixed_signal_dataset_general(seed=RANDOM_SEED_BASE + k).values
        iforest = IsolationForest(
            n_estimators=N_TREES, max_samples=MAX_SAMPLES,
            contamination=CONTAMINATION, random_state=k, bootstrap=False,
        )
        iforest.fit(X_data)
        fi_diffi_raw = diffi_ib_binary_rso(iforest, X_data, feature_types)
        raw_scores_list.append(fi_diffi_raw)

    fi_diffi_matrix = np.vstack(raw_scores_list)
    df_raw = pd.DataFrame(fi_diffi_matrix, columns=FEATURE_NAMES)
    results = {}

    # 1. Raw DIFFI RSO scores
    df_raw_scenario = df_raw.copy()
    df_raw_scenario['Scenario'] = '1_Raw_Score (DIFFI RSO)'
    results['1_Raw_Score (DIFFI RSO)'] = df_raw_scenario

    # 2. Z-Score AD-DIFFI RSO (type-specific normalization)
    df_z_scenario = df_raw.copy()
    for i, feat_name in enumerate(FEATURE_NAMES):
        feat_type = FEATURE_TYPES[i]
        if feat_type == 'cont':
            if cont_sd > 0:
                df_z_scenario.iloc[:, i] = (df_raw.iloc[:, i] - cont_mean) / cont_sd
            else:
                df_z_scenario.iloc[:, i] = 0.0
        elif feat_type == 'bin':
            if bin_sd > 0:
                df_z_scenario.iloc[:, i] = (df_raw.iloc[:, i] - bin_mean) / bin_sd
            else:
                df_z_scenario.iloc[:, i] = 0.0

    df_z_scenario['Scenario'] = '2_Z_Score (AD-DIFFI RSO)'
    results['2_Z_Score (AD-DIFFI RSO)'] = df_z_scenario

    return pd.concat([df for df in results.values()]).reset_index(drop=True)
