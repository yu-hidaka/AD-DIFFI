# ============================================================================
# Real Data Analysis: Breast Cancer (Table S2)
# Comparison of Original DIFFI vs AD-DIFFI (RSO + Z-normalization)
# For Google Colab Execution
# ============================================================================

# =============================================================================
# Cell 1: Setup and Install Dependencies
# =============================================================================

!pip install -q kaggle scikit-learn pandas numpy matplotlib seaborn scipy lifelines
print("All dependencies installed.")

# =============================================================================
# Cell 2: Data Setup Function for Colab
# =============================================================================

import os
from pathlib import Path
from google.colab import files
import pandas as pd

def setup_colab_data():
    """
    Automatic data download for Colab environment.

    Supports two methods:
    1. Kaggle API: Download from Kaggle directly
    2. Google Drive: Manual upload to Drive

    Returns:
        str: Path to Breast_Cancer.csv file
    """
    data_dir = Path('/tmp/ad_diffi_data')
    data_dir.mkdir(exist_ok=True)
    csv_file = data_dir / 'Breast_Cancer.csv'

    # Check if dataset already exists
    if csv_file.exists():
        print("[INFO] Dataset already downloaded at {}".format(csv_file))
        return str(csv_file)

    print("Downloading Breast Cancer dataset from Kaggle...")
    print("\n[SETUP INSTRUCTIONS]")
    print("1. Visit: https://www.kaggle.com/settings/account")
    print("2. Click 'Create New API Token'")
    print("3. Download kaggle.json")
    print("4. Upload the file when prompted below\n")

    # File upload dialog
    uploaded = files.upload()

    if 'kaggle.json' not in uploaded:
        print("[ERROR] kaggle.json not found")
        return None

    # Configure Kaggle
    import subprocess
    subprocess.run(['pip', 'install', '-q', 'kaggle'], check=True)
    subprocess.run(['mkdir', '-p', str(Path.home() / '.kaggle')], check=True)
    subprocess.run(['mv', 'kaggle.json', str(Path.home() / '.kaggle' / 'kaggle.json')], check=True)
    subprocess.run(['chmod', '600', str(Path.home() / '.kaggle' / 'kaggle.json')], check=True)

    # Download dataset (修正後)
    print("\nDownloading...")
    !kaggle datasets download -d reihanenamdari/breast-cancer -p {str(data_dir)} --unzip --force

    if csv_file.exists():
        print("[SUCCESS] Dataset saved to: {}".format(csv_file))
        return str(csv_file)
    else:
        print("[ERROR] Download failed. Check dataset slug or internet.")
        print("Alternative: Manual download from https://www.kaggle.com/datasets/reihanenamdari/breast-cancer")
        return None


# =============================================================================
# Cell 3: Core Functions - Original DIFFI Implementation
# =============================================================================

import numpy as np
import time
from math import ceil
from sklearn.ensemble import IsolationForest
from typing import Tuple

def _get_iic(estimator, predictions, is_leaves, adjust_iic: bool = True) -> np.ndarray:
    """
    Compute Induced Imbalance Coefficient (IIC).

    Args:
        estimator: Fitted isolation tree estimator
        predictions: Input samples
        is_leaves: Boolean array marking leaf nodes
        adjust_iic: Whether to normalize IIC to [0.5, 1.0]

    Returns:
        lambda_: IIC values for each node
    """
    desired_min = 0.5
    desired_max = 1.0
    epsilon = 0.0

    n_nodes = estimator.tree_.node_count
    lambda_ = np.zeros(n_nodes)
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right

    if predictions.shape[0] == 0:
        return lambda_

    # Compute sample counts in each node
    node_indicator = estimator.decision_path(predictions).toarray()
    num_samples_in_node = np.sum(node_indicator, axis=0)

    for node in range(n_nodes):
        num_samples_current = num_samples_in_node[node]
        num_samples_left = num_samples_in_node[children_left[node]]
        num_samples_right = num_samples_in_node[children_right[node]]

        if num_samples_current <= 1 or is_leaves[node]:
            lambda_[node] = -1
        elif num_samples_left == 0 or num_samples_right == 0:
            lambda_[node] = epsilon
        else:
            # Compute imbalance ratio
            if num_samples_current % 2 == 0:
                current_min = 0.5
            else:
                current_min = ceil(num_samples_current / 2) / num_samples_current
            current_max = (num_samples_current - 1) / num_samples_current
            max_child_ratio = max(num_samples_left, num_samples_right) / num_samples_current

            if adjust_iic and current_min != current_max:
                lambda_[node] = ((max_child_ratio - current_min) / (current_max - current_min)) \
                               * (desired_max - desired_min) + desired_min
            else:
                lambda_[node] = max_child_ratio

    return lambda_


def diffi_ib(iforest: IsolationForest, X: np.ndarray, adjust_iic: bool = True) -> Tuple[np.ndarray, float]:
    """
    Original DIFFI: Depth-based Isolation Forest Feature Importance.

    Args:
        iforest: Fitted IsolationForest model
        X: Input data (n_samples, n_features)
        adjust_iic: Whether to adjust IIC scaling

    Returns:
        fi_ib: Feature importance scores (n_features,)
        exec_time: Computation time in seconds
    """
    start = time.time()
    num_features = X.shape[1]
    estimators = iforest.estimators_
    in_bag_samples = iforest.estimators_samples_

    # Initialize accumulators
    cfi_outliers = np.zeros(num_features, dtype=float)
    cfi_inliers = np.zeros(num_features, dtype=float)
    cnt_outliers = np.zeros(num_features, dtype=int)
    cnt_inliers = np.zeros(num_features, dtype=int)

    # Compute anomaly threshold
    global_scores = iforest.decision_function(X)
    threshold = np.percentile(global_scores, 100 * iforest.contamination)

    for k, estimator in enumerate(estimators):
        in_bag_index = list(in_bag_samples[k])
        X_ib = X[in_bag_index, :]
        scores_ib = global_scores[in_bag_index]

        X_outliers = X_ib[scores_ib < threshold]
        X_inliers = X_ib[scores_ib >= threshold]

        if X_inliers.shape[0] == 0 or X_outliers.shape[0] == 0:
            continue

        # Extract tree structure
        tree = estimator.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature

        # Compute node depths
        node_depth = np.zeros(n_nodes, dtype=np.int64)
        is_leaves = np.zeros(n_nodes, dtype=bool)

        stack = [(0, 0)]
        while stack:
            node_id, depth = stack.pop()
            node_depth[node_id] = depth
            if children_left[node_id] != children_right[node_id]:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        # Process outliers
        lambda_outliers = _get_iic(estimator, X_outliers, is_leaves, adjust_iic)
        node_indicator_outliers = estimator.decision_path(X_outliers).toarray()

        for i in range(len(X_outliers)):
            path = np.where(node_indicator_outliers[i] == 1)[0]
            if len(path) == 0:
                continue
            depth = node_depth[path[-1]]
            for node in path:
                current_feature = feature[node]
                if current_feature >= 0 and lambda_outliers[node] != -1:
                    cfi_outliers[current_feature] += (1.0 / depth) * lambda_outliers[node]
                    cnt_outliers[current_feature] += 1

        # Process inliers
        lambda_inliers = _get_iic(estimator, X_inliers, is_leaves, adjust_iic)
        node_indicator_inliers = estimator.decision_path(X_inliers).toarray()

        for i in range(len(X_inliers)):
            path = np.where(node_indicator_inliers[i] == 1)[0]
            if len(path) == 0:
                continue
            depth = node_depth[path[-1]]
            for node in path:
                current_feature = feature[node]
                if current_feature >= 0 and lambda_inliers[node] != -1:
                    cfi_inliers[current_feature] += (1.0 / depth) * lambda_inliers[node]
                    cnt_inliers[current_feature] += 1

    # Normalize and compute ratio
    fi_outliers = np.where(cnt_outliers > 0, cfi_outliers / cnt_outliers, 0)
    fi_inliers = np.where(cnt_inliers > 0, cfi_inliers / cnt_inliers, 0)

    with np.errstate(divide='ignore', invalid='ignore'):
        fi_ib = np.divide(fi_outliers, fi_inliers, out=np.zeros_like(fi_outliers),
                         where=fi_inliers != 0)

    exec_time = time.time() - start
    return fi_ib, exec_time


# =============================================================================
# Cell 4: AD-DIFFI Implementation (RSO + Z-normalization)
# =============================================================================

from typing import Dict, Tuple

MIN_DEPTH = 1

def diffi_ib_binary_rso(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: Dict[int, str]
) -> np.ndarray:
    """
    AD-DIFFI with Root-Split-Only (RSO) constraint for binary features.

    Args:
        iforest: Fitted IsolationForest model
        X_data: Input data (n_samples, n_features)
        feature_types: Dict mapping feature index to 'cont' or 'bin'

    Returns:
        fi_rso: Feature importance scores with RSO constraint
    """
    num_features = X_data.shape[1]
    estimators = iforest.estimators_

    cfi_outliers = np.zeros(num_features, dtype=float)
    cfi_inliers = np.zeros(num_features, dtype=float)
    cnt_outliers = np.zeros(num_features, dtype=int)
    cnt_inliers = np.zeros(num_features, dtype=int)

    # Compute anomaly threshold
    global_scores = iforest.decision_function(X_data)
    threshold = np.percentile(global_scores, 100 * iforest.contamination)

    for k, estimator in enumerate(estimators):
        in_bag_index = iforest.estimators_samples_[k]
        X_ib = X_data[in_bag_index]
        scores_ib = global_scores[in_bag_index]

        X_outliers = X_ib[scores_ib < threshold]
        X_inliers = X_ib[scores_ib >= threshold]

        if X_outliers.shape[0] == 0 or X_inliers.shape[0] == 0:
            continue

        # Extract tree structure
        tree = estimator.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature_index = tree.feature
        n_node_samples = tree.n_node_samples

        # Compute node depths
        node_depth = np.zeros(n_nodes, dtype=int)
        stack = [(0, 0)]
        while stack:
            node_id, depth = stack.pop()
            node_depth[node_id] = depth
            if children_left[node_id] != children_right[node_id]:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))

        # Compute adjusted lambda (imbalance coefficient)
        lambda_adjusted = np.zeros(n_nodes, dtype=float)
        for node in range(n_nodes):
            if children_left[node] != children_right[node]:
                n_parent = n_node_samples[node]
                n_left = n_node_samples[children_left[node]]
                n_right = n_node_samples[children_right[node]]
                ratio_small = min(n_left, n_right) / n_parent
                lambda_adjusted[node] = 1.0 - ratio_small

        def accumulate_contributions(X_subset, cfi, cnt):
            """Accumulate feature importance contributions from samples."""
            node_indicator = estimator.decision_path(X_subset).toarray()
            for sample_idx in range(X_subset.shape[0]):
                path = np.where(node_indicator[sample_idx] == 1)[0]
                if len(path) == 0:
                    continue

                depth_leaf = node_depth[path[-1]]
                h_leaf = max(depth_leaf, MIN_DEPTH)
                depth_scale = 1.0 / h_leaf

                for node in path:
                    feat_idx = feature_index[node]
                    if feat_idx < 0:
                        continue

                    feature_type = feature_types[feat_idx]
                    # RSO constraint: binary features only split at root (depth=0)
                    if feature_type == 'bin' and node_depth[node] > 0:
                        continue

                    contribution = depth_scale * lambda_adjusted[node]
                    cfi[feat_idx] += contribution
                    cnt[feat_idx] += 1

        accumulate_contributions(X_outliers, cfi_outliers, cnt_outliers)
        accumulate_contributions(X_inliers, cfi_inliers, cnt_inliers)

    # Normalize and compute ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        fi_outliers = np.where(cnt_outliers > 0, cfi_outliers / cnt_outliers, 0)
        fi_inliers = np.where(cnt_inliers > 0, cfi_inliers / cnt_inliers, 0)
        fi_rso = np.divide(fi_outliers, fi_inliers, out=np.zeros_like(fi_outliers),
                          where=fi_inliers != 0)

    return fi_rso


def noise_baseline_rso(
    X_dim: int,
    feature_types: Dict[int, str],
    if_params: Dict,
    n_iter: int = 50
) -> Tuple[float, float, float, float]:
    """
    Establish noise baseline for RSO-based DIFFI scores.

    Uses random uniform noise to determine null distribution.

    Args:
        X_dim: Number of features
        feature_types: Dict mapping feature index to 'cont' or 'bin'
        if_params: Isolation Forest hyperparameters
        n_iter: Number of iterations

    Returns:
        Tuple of (cont_mean, cont_sd, bin_mean, bin_sd)
    """
    print("Establishing noise baseline (RSO DIFFI) over {} iterations...".format(n_iter))
    scores_list = []

    cont_indices = [i for i, t in feature_types.items() if t == 'cont']
    bin_indices = [i for i, t in feature_types.items() if t == 'bin']

    for k in range(n_iter):
        X_noise = np.random.uniform(0, 1, size=(2000, X_dim))
        iforest = IsolationForest(random_state=k, **if_params)
        iforest.fit(X_noise)
        fi = diffi_ib_binary_rso(iforest, X_noise, feature_types)
        scores_list.append(fi)

    M = np.vstack(scores_list)

    cont_scores = M[:, cont_indices] if cont_indices else np.array([])
    bin_scores = M[:, bin_indices] if bin_indices else np.array([])

    cont_mean = float(cont_scores.mean()) if len(cont_indices) > 0 else 1.0
    cont_sd = float(cont_scores.std()) if len(cont_indices) > 0 else 1.0
    bin_mean = float(bin_scores.mean()) if len(bin_indices) > 0 else 1.0
    bin_sd = float(bin_scores.std()) if len(bin_indices) > 0 else 1.0

    print("  Continuous (mean={:.4f}, std={:.4f})".format(cont_mean, cont_sd))
    print("  Binary      (mean={:.4f}, std={:.4f})".format(bin_mean, bin_sd))

    return cont_mean, cont_sd, bin_mean, bin_sd


# =============================================================================
# Cell 5: Comparison Function
# =============================================================================

from sklearn.preprocessing import StandardScaler
from typing import List

def compare_diffi_vs_ad_diffi(
    X: np.ndarray,
    feature_names: List[str],
    feature_types: Dict[int, str],
    if_params: Dict,
    n_iter_noise: int = 50,
    n_iter_main: int = 50,
) -> pd.DataFrame:
    """
    Compare Original DIFFI vs AD-DIFFI (RSO + Z-normalization).

    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        feature_types: Dict mapping feature index to 'cont' or 'bin'
        if_params: Isolation Forest parameters
        n_iter_noise: Iterations for noise baseline
        n_iter_main: Iterations for main comparison

    Returns:
        DataFrame with importance scores, ranks and rank changes
    """
    # Standardize data
    X_scaled = StandardScaler().fit_transform(X)

    # Establish noise baseline for AD-DIFFI
    cont_mean, cont_sd, bin_mean, bin_sd = noise_baseline_rso(
        X_dim=X_scaled.shape[1],
        feature_types=feature_types,
        if_params=if_params,
        n_iter=n_iter_noise
    )

    # Run main comparison iterations
    orig_scores_list = []
    rso_raw_list = []

    print("Running {} iterations for feature importance comparison...".format(n_iter_main))
    for k in range(n_iter_main):
        iforest = IsolationForest(random_state=k, **if_params)
        iforest.fit(X_scaled)

        # Original DIFFI
        fi_orig, _ = diffi_ib(iforest, X_scaled, adjust_iic=True)
        orig_scores_list.append(fi_orig)

        # AD-DIFFI RSO (raw)
        fi_rso = diffi_ib_binary_rso(iforest, X_scaled, feature_types)
        rso_raw_list.append(fi_rso)

    mean_orig = np.mean(orig_scores_list, axis=0)
    M_rso_raw = np.vstack(rso_raw_list)

    # Z-normalize RSO scores
    rso_z_scores = M_rso_raw.copy()
    for i in range(len(feature_names)):
        if feature_types[i] == 'cont':
            if cont_sd > 0:
                rso_z_scores[:, i] = (M_rso_raw[:, i] - cont_mean) / cont_sd
            else:
                rso_z_scores[:, i] = 0.0
        else:  # binary
            if bin_sd > 0:
                rso_z_scores[:, i] = (M_rso_raw[:, i] - bin_mean) / bin_sd
            else:
                rso_z_scores[:, i] = 0.0

    mean_rso_z = np.mean(rso_z_scores, axis=0)

    # Build comparison table
    compare_df = pd.DataFrame({
        "Feature": feature_names,
        "Type": [feature_types[i] for i in range(len(feature_names))],
        "Original_DIFFI": np.round(mean_orig, 4),
        "AD_DIFFI_RSO_Z": np.round(mean_rso_z, 4),
    })

    compare_df["Rank_Original"] = compare_df["Original_DIFFI"].rank(ascending=False)
    compare_df["Rank_AD_DIFFI"] = compare_df["AD_DIFFI_RSO_Z"].rank(ascending=False)
    compare_df["Rank_Change"] = compare_df["Rank_AD_DIFFI"] - compare_df["Rank_Original"]

    return compare_df.sort_values("AD_DIFFI_RSO_Z", ascending=False)

# =============================================================================
# Cell 6: Breast Cancer Dataset Processing
# =============================================================================

def create_binary_features_breast_cancer(df: pd.DataFrame) -> pd.DataFrame:

    df_processed = df[[
        'Age', 'Tumor Size', 'Regional Node Examined',
        'Reginol Node Positive', 'Survival Months', 'Status'
    ]].copy()

    df_processed['Race_White'] = (df['Race'] == 'White').astype(int)
    df_processed['Marital_Married'] = df['Marital Status'].isin(
        ['Married', 'Married-spouse-absent']
    ).astype(int)
    df_processed['T_Stage_T1'] = (df['T Stage '] == 'T1').astype(int)
    df_processed['N_Stage_N0'] = (df['N Stage'] == 'N0').astype(int)
    df_processed['Stage_II'] = (df['6th Stage'] == 'II').astype(int)

    diff_map = {
        'Poorly differentiated': 0,
        'Moderately differentiated': 1,
        'Well differentiated': 1,
        'Poorly differentiated; Moderately differentiated': 0
    }
    df_processed['Differentiate_High'] = df['differentiate'].map(diff_map).fillna(1).astype(int)
    df_processed['Grade_Low'] = df['Grade'].isin(['Grade I', 'Grade II']).astype(int)
    df_processed['A_Stage_Regional'] = (df['A Stage'] == 'Regional').astype(int)
    df_processed['Estrogen_Positive'] = (df['Estrogen Status'] == 'Positive').astype(int)
    df_processed['Progesterone_Positive'] = (df['Progesterone Status'] == 'Positive').astype(int)

    return df_processed


def evaluate_cox_cindex(
    df: pd.DataFrame,
    feature_list: List[str],
    time_col: str = 'Survival Months',
    event_col: str = 'Status_num',
    random_seed: int = 0,
    max_features: int = 10
) -> float:
    """
    Robust Cox C-index with multicollinearity handling.
    """
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold

    np.random.seed(random_seed)

    X_raw = df[feature_list].fillna(df[feature_list].median()).copy()
    T = df[time_col].values
    E = df[event_col].values

    valid_mask = ~(pd.isna(T) | pd.isna(E))
    if valid_mask.sum() < 20:
        print(f"[WARNING] Insufficient data: {valid_mask.sum()} samples")
        return 0.0

    X = X_raw[valid_mask].reset_index(drop=True)
    T_valid = T[valid_mask]
    E_valid = E[valid_mask]

    n_samples, n_features = X.shape
    print(f"[DEBUG] Cox: {n_samples} samples, {n_features} features")

    if n_features == 0:
        return 0.0

    selector = VarianceThreshold(threshold=0.01)
    X_var = selector.fit_transform(X)
    selected_features = [feature_list[i] for i in selector.get_support(indices=True)]

    if len(selected_features) == 0:
        print("[WARNING] No features after variance filtering")
        return 0.0

    X_filtered = pd.DataFrame(X_var, columns=selected_features)
    print(f"[DEBUG] After variance filter: {len(selected_features)} features")

    if len(selected_features) > max_features:
        corr_matrix = X_filtered.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [column for column in upper.columns if any(upper[column] > 0.95)]

        remaining_features = [f for f in selected_features if f not in high_corr]
        if len(remaining_features) > max_features:
            remaining_features = remaining_features[:max_features]

        X_filtered = X_filtered[remaining_features]
        print(f"[DEBUG] After correlation filter: {len(remaining_features)} features")
    else:
        remaining_features = selected_features

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_filtered),
        columns=remaining_features
    )

    survival_df = pd.DataFrame({
        time_col: T_valid,
        'event': E_valid.astype(bool)
    })
    cox_df = pd.concat([survival_df, X_scaled], axis=1)

    try:
        cph = CoxPHFitter()
        cph.fit(
            cox_df,
            duration_col=time_col,
            event_col='event',
            show_progress=False,
            robust=True
        )

        risk_scores = -cph.predict_partial_hazard(X_scaled)
        cindex = concordance_index(T_valid, risk_scores, E_valid)

        print(f"[SUCCESS] Cox C-index: {cindex:.4f} ({len(remaining_features)} features)")
        return float(cindex)

    except Exception as e:
        print(f"[ERROR] Cox failed: {str(e)[:100]}")
        print("  Using null model C-index...")

        if len(remaining_features) > 0:
            single_feature = remaining_features[0]
            single_df = pd.concat([
                survival_df,
                X_scaled[[single_feature]]
            ], axis=1)

            try:
                cph_single = CoxPHFitter()
                cph_single.fit(single_df, duration_col=time_col, event_col='event', show_progress=False)
                risk_single = -cph_single.predict_partial_hazard(X_scaled[[single_feature]])
                cindex_single = concordance_index(T_valid, risk_single, E_valid)
                print(f"[FALLBACK] Single feature C-index: {cindex_single:.4f}")
                return float(cindex_single)
            except:
                pass

        return 0.5


def evaluate_if_auc(
    df: pd.DataFrame,
    feature_list: List[str],
    y: np.ndarray,
    if_params: Dict,
    random_state: int = 0
) -> float:

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    X = df[feature_list].fillna(df[feature_list].median()).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    from sklearn.ensemble import IsolationForest
    iforest = IsolationForest(**if_params, random_state=random_state)
    iforest.fit(X_train_sc)

    scores = -iforest.decision_function(X_test_sc)
    auc = roc_auc_score(y_test, scores)
    return auc

# =============================================================================
# Cell 7: Load Data and Run Analysis (Updated with seeds and Cox C-index)
# =============================================================================

def run_breast_cancer_analysis(random_seed: int = 42):
    """Main analysis execution with reproducible seeds."""

    # Set global random seeds for reproducibility
    np.random.seed(random_seed)
    import random
    random.seed(random_seed)

    print("="*70)
    print("AD-DIFFI Real Data Analysis: Breast Cancer (Table S2)")
    print("Random Seed: {}".format(random_seed))
    print("="*70)

    # Data setup
    data_path = setup_colab_data()

    if not data_path:
        print("[ERROR] Failed to load dataset")
        return

    # Load and process data
    df_BC = pd.read_csv(data_path)
    df_BC_processed = create_binary_features_breast_cancer(df_BC)

    # Feature definitions
    feature_cols = [c for c in df_BC_processed.columns
                    if c not in ['Status', 'Survival Months']]

    print("[INFO] Available features:", feature_cols)

    FEATURE_DEFINITIONS = [
        {'name': col, 'type': 'bin' if df_BC_processed[col].nunique() <= 10 else 'cont'}
        for col in feature_cols
    ]
    FEATURE_TYPES = {i: f['type'] for i, f in enumerate(FEATURE_DEFINITIONS)}
    FEATURE_NAMES = [f['name'] for f in FEATURE_DEFINITIONS]

    print("[INFO] Dataset: {} samples, {} features".format(
        len(df_BC_processed), len(feature_cols)
    ))
    n_cont = sum(1 for f in FEATURE_DEFINITIONS if f['type'] == 'cont')
    n_bin = sum(1 for f in FEATURE_DEFINITIONS if f['type'] == 'bin')
    print("[INFO] Feature types: {} continuous, {} binary".format(n_cont, n_bin))

    # Prepare data
    X_data = df_BC_processed[feature_cols].fillna(
        df_BC_processed[feature_cols].median()
    ).values

    # Isolation Forest parameters
    IF_PARAMS = {
        'n_estimators': 200,
        'max_samples': 512,
        'contamination': 0.05,
        'max_features': 1.0,
        'bootstrap': False,
    }

    # Run comparison
    print("="*70)
    print("Running feature importance comparison...")
    print("="*70)

    compare_results = compare_diffi_vs_ad_diffi(
        X=X_data,
        feature_names=FEATURE_NAMES,
        feature_types=FEATURE_TYPES,
        if_params=IF_PARAMS,
        n_iter_noise=20,
        n_iter_main=20,
    )

    print("\nFeature Importance Comparison (Table S2)")
    print(compare_results.to_markdown(index=False))

    # Prepare target variables
    df_BC_processed["Status_num"] = (df_BC_processed["Status"] != "Alive").astype(int)
    y_status = df_BC_processed["Status_num"].values

    # Evaluate performance metrics
    print("="*70)
    print("Performance Evaluation (IF AUC + Cox C-index)")
    print("="*70)

    # All features
    auc_all = evaluate_if_auc(df_BC_processed, feature_cols, y_status, IF_PARAMS, random_seed)
    cindex_all = evaluate_cox_cindex(df_BC_processed, feature_cols, random_seed=random_seed)
    print("All {:2d} features:  IF AUC={:.4f}, Cox C-index={:.4f}".format(
      len(feature_cols), auc_all, cindex_all))

    # Original DIFFI top-6
    top_orig = compare_results.sort_values("Original_DIFFI", ascending=False).head(6)
    top_orig_feats = top_orig["Feature"].tolist()
    auc_orig6 = evaluate_if_auc(df_BC_processed, top_orig_feats, y_status, IF_PARAMS, random_seed)
    cindex_orig6 = evaluate_cox_cindex(df_BC_processed, top_orig_feats, random_seed=random_seed)
    print("Original DIFFI top-6: IF AUC={:.4f}, Cox C-index={:.4f}".format(auc_orig6, cindex_orig6))

    # AD-DIFFI top-6
    top_ad = compare_results.sort_values("AD_DIFFI_RSO_Z", ascending=False).head(6)
    top_ad_feats = top_ad["Feature"].tolist()
    auc_ad6 = evaluate_if_auc(df_BC_processed, top_ad_feats, y_status, IF_PARAMS, random_seed)
    cindex_ad6 = evaluate_cox_cindex(df_BC_processed, top_ad_feats, random_seed=random_seed)
    print("AD-DIFFI top-6:     IF AUC={:.4f}, Cox C-index={:.4f}".format(auc_ad6, cindex_ad6))

    # Summary table
    print("\nSummary Table:")
    summary_data = {
        "Method": ["All Features", "Original DIFFI Top-6", "AD-DIFFI Top-6"],
        "IF AUC": [round(auc_all, 4), round(auc_orig6, 4), round(auc_ad6, 4)],
        "Cox C-index": [round(cindex_all, 4), round(cindex_orig6, 4), round(cindex_ad6, 4)]
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_markdown(index=False))

    print("="*70)
    print("Analysis complete.")
    print("="*70)

    # Return results for download
    results_dict = {
        'feature_importance': compare_results,
        'performance_summary': summary_df,
        'top_orig_features': top_orig_feats,
        'top_ad_features': top_ad_feats,
        'metrics': {
            'all': {'if_auc': auc_all, 'cindex': cindex_all},
            'orig_top6': {'if_auc': auc_orig6, 'cindex': cindex_orig6},
            'ad_top6': {'if_auc': auc_ad6, 'cindex': cindex_ad6}
        }
    }
    return results_dict


# =============================================================================
# Cell 8: Download Results (Updated)
# =============================================================================

def save_and_download_results(results_dict: dict, random_seed: int = 42):
    """
    Save all results and prepare for download.

    Args:
        results_dict: Dictionary containing all analysis results
        random_seed: Random seed used
    """
    import json

    # Save feature importance
    compare_results = results_dict['feature_importance']
    output_filename = f'breast_cancer_analysis_seed{random_seed}.csv'
    compare_results.to_csv(output_filename, index=False)

    # Save performance summary
    summary_df = results_dict['performance_summary']
    summary_filename = f'breast_cancer_performance_seed{random_seed}.csv'
    summary_df.to_csv(summary_filename, index=False)

    # Save detailed metrics
    metrics_filename = f'breast_cancer_metrics_seed{random_seed}.json'
    with open(metrics_filename, 'w') as f:
        json.dump(results_dict['metrics'], f, indent=2)

    print("[INFO] Files saved:")
    print(f"  - {output_filename}")
    print(f"  - {summary_filename}")
    print(f"  - {metrics_filename}")

    # Download in Colab
    from google.colab import files
    files.download(output_filename)
    files.download(summary_filename)
    files.download(metrics_filename)
    print("[INFO] Downloads started...")


# =============================================================================
# Main Execution (Run in Colab) - Updated with seed control
# =============================================================================

if __name__ == "__main__":
    # Run analysis with reproducible seed
    RANDOM_SEED = 42  # Change this value to get different reproducible results
    results = run_breast_cancer_analysis(random_seed=RANDOM_SEED)

    # Save and download results
    if results is not None:
        save_and_download_results(results, random_seed=RANDOM_SEED)
