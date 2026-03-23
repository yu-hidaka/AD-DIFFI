# ============================================================================
# Real Data Analysis: Stroke Prediction (Table S3)
# Comparison of Original DIFFI vs AD-DIFFI (RSO + Z-normalization)
# For Google Colab Execution
# ============================================================================

# =============================================================================
# Cell 1: Setup and Install Dependencies
# =============================================================================

!pip install -q kaggle scikit-learn pandas numpy lifelines
print("All dependencies installed.")

# =============================================================================
# Cell 2: Data Setup Functions for Colab
# =============================================================================

import os
import subprocess
from pathlib import Path
from google.colab import files
import pandas as pd
import numpy as np

def setup_colab_stroke_data():
    """
    Automatic Stroke dataset download for Colab environment.
    Kaggle API: Download from fedesoriano/stroke-prediction-dataset
    """
    data_dir = Path('/tmp/ad_diffi_data')
    data_dir.mkdir(exist_ok=True)
    csv_file = data_dir / 'stroke_data.csv'

    if csv_file.exists():
        print(f"[INFO] Dataset already downloaded at {csv_file}")
        return str(csv_file)

    print("Downloading Stroke Prediction dataset from Kaggle...")
    print("\n[SETUP INSTRUCTIONS]")
    print("1. Visit: https://www.kaggle.com/settings/account")
    print("2. Click 'Create New API Token'")
    print("3. Download kaggle.json")
    print("4. Upload the file when prompted below\n")

    uploaded = files.upload()

    if 'kaggle.json' not in uploaded:
        print("[ERROR] kaggle.json not found")
        return None

    # Configure Kaggle
    subprocess.run(['pip', 'install', '-q', 'kaggle'], check=True)
    subprocess.run(['mkdir', '-p', str(Path.home() / '.kaggle')], check=True)
    subprocess.run(['mv', 'kaggle.json', str(Path.home() / '.kaggle' / 'kaggle.json')], check=True)
    subprocess.run(['chmod', '600', str(Path.home() / '.kaggle' / 'kaggle.json')], check=True)

    # Download dataset
    print("\nDownloading...")
    result = subprocess.run([
        'kaggle', 'datasets', 'download', '-d', 'fedesoriano/stroke-prediction-dataset',
        '-p', str(data_dir), '--unzip', '--force', '-q'
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] Kaggle download failed: {result.stderr}")
        return None

    raw_csv = data_dir / 'healthcare-dataset-stroke-data.csv'
    if raw_csv.exists():
        csv_file.write_bytes(raw_csv.read_bytes())
        raw_csv.unlink()
        print(f"[SUCCESS] Dataset saved to: {csv_file}")
        return str(csv_file)
    else:
        print("[ERROR] Expected file not found after download")
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
# Cell 5: Comparison Function (Stroke Prediction)
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


def create_features_stroke(df_raw: pd.DataFrame, random_seed: int = 42):
    """
    Stroke dataset preprocessing matching Annthyroid/Breast cancer pipeline.
    """
    print("Original shape:", df_raw.shape)

    # Clean data
    df = df_raw.dropna(subset=["bmi"])
    df = df[df["gender"] != "Other"].copy()

    # Stratified subsampling if too large
    if len(df) > 2000:
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(
            df, train_size=2000, random_state=random_seed,
            stratify=df["stroke"]
        )

    print(f"After cleaning: {df.shape}, stroke ratio={df['stroke'].mean():.3f}")

    # Continuous features
    cont_features = ["age", "avg_glucose_level", "bmi"]

    # Native binary features
    bin_features_native = ["hypertension", "heart_disease"]

    # Categorical one-hot encoding
    cat_features = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    df_cat = pd.get_dummies(df[cat_features], prefix_sep='_', drop_first=True)

    # Combine features
    feature_cols = cont_features + bin_features_native + df_cat.columns.tolist()
    df_processed = pd.concat([df[cont_features + bin_features_native], df_cat], axis=1)

    # Outlier labels (stroke=1 → outlier)
    df_processed["Outlier_label"] = np.where(df["stroke"] == 1, "o", "normal")

    # Numeric conversion + imputation
    for c in cont_features:
        df_processed[c] = pd.to_numeric(df_processed[c], errors="coerce")
    df_processed[cont_features] = df_processed[cont_features].fillna(df_processed[cont_features].median())

    print(f"Processed shape: {df_processed.shape}")
    print(f"Continuous ({len(cont_features)}): {cont_features}")
    print(f"Binary ({len(feature_cols)-len(cont_features)}): total binary/dummy features")

    return df_processed, feature_cols, "Outlier_label"

# =============================================================================
# Cell 6: Evaluation Functions (IF AUC + Logistic Regression AUC)
# =============================================================================

def evaluate_if_auc(
    df: pd.DataFrame,
    feature_list: list,
    y: np.ndarray,
    if_params: dict,
    random_state: int = 0,
) -> float:
    """Isolation Forest AUC (standardized across datasets)."""
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

    iforest = IsolationForest(**if_params, random_state=random_state)
    iforest.fit(X_train_sc)
    scores = -iforest.decision_function(X_test_sc)
    auc = roc_auc_score(y_test, scores)
    return auc

def evaluate_lr_auc(
    df: pd.DataFrame,
    feature_list: list,
    y: np.ndarray,
    random_state: int = 0,
) -> float:
    """Logistic Regression AUC (standardized across datasets)."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = df[feature_list].fillna(df[feature_list].median()).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train_sc, y_train)
    probs = clf.predict_proba(X_test_sc)[:, 1]
    auc = roc_auc_score(y_test, probs)
    return auc

# =============================================================================
# Cell 7: Load Data and Run Analysis
# =============================================================================

def run_stroke_analysis(random_seed: int = 42):
    """Main analysis execution with reproducible seeds."""
    import random
    np.random.seed(random_seed)
    random.seed(random_seed)

    print("=" * 70)
    print("AD-DIFFI Real Data Analysis: Stroke Prediction (Table S3)")
    print(f"Random Seed: {random_seed}")
    print("=" * 70)

    # 1. Data setup
    data_path = setup_colab_stroke_data()
    if not data_path:
        print("[ERROR] Failed to load dataset")
        return None

    df_raw = pd.read_csv(data_path)

    # 2. Preprocess features
    df_processed, feature_cols, label_col = create_features_stroke(df_raw, random_seed)
    X = df_processed[feature_cols].values
    y = (df_processed[label_col] == "o").astype(int).values

    FEATURE_NAMES = feature_cols
    FEATURE_TYPES = {
        i: "cont" if name in ["age", "avg_glucose_level", "bmi"] else "bin"
        for i, name in enumerate(FEATURE_NAMES)
    }

    print("[INFO] Dataset: {} samples, {} features".format(
        len(df_processed), len(feature_cols)
    ))
    n_cont = sum(1 for t in FEATURE_TYPES.values() if t == "cont")
    n_bin = sum(1 for t in FEATURE_TYPES.values() if t == "bin")
    print("[INFO] Feature types: {} continuous, {} binary".format(n_cont, n_bin))

    # 3. Isolation Forest parameters (real-data setting - unified)
    IF_PARAMS = {
        "n_estimators": 100,
        "max_samples": 512,
        "contamination": 0.05,
        "max_features": 1.0,
        "bootstrap": False,
    }

    # 4. DIFFI vs AD-DIFFI comparison
    print("=" * 70)
    print("Running feature importance comparison...")
    print("=" * 70)

    compare_results = compare_diffi_vs_ad_diffi(
        X=X,
        feature_names=FEATURE_NAMES,
        feature_types=FEATURE_TYPES,
        if_params=IF_PARAMS,
        n_iter_noise=20,  # Unified with Breast/Annthyroid
        n_iter_main=20,   # Unified with Breast/Annthyroid
    )

    print("\nFeature Importance Comparison (Table S3)")
    print(compare_results.to_markdown(index=False))

    # 5. Performance evaluation
    print("=" * 70)
    print("Performance Evaluation (IF AUC + Logistic Regression AUC)")
    print("=" * 70)

    k = 6  # Unified top-k

    # All features
    auc_if_all = evaluate_if_auc(df_processed, FEATURE_NAMES, y, IF_PARAMS, random_seed)
    auc_lr_all = evaluate_lr_auc(df_processed, FEATURE_NAMES, y, random_seed)

    # Original DIFFI top-k
    top_orig = compare_results.sort_values("Original_DIFFI", ascending=False).head(k)
    top_orig_feats = top_orig["Feature"].tolist()
    auc_if_orig_k = evaluate_if_auc(df_processed, top_orig_feats, y, IF_PARAMS, random_seed)
    auc_lr_orig_k = evaluate_lr_auc(df_processed, top_orig_feats, y, random_seed)

    # AD-DIFFI top-k
    top_ad = compare_results.sort_values("AD_DIFFI_RSO_Z", ascending=False).head(k)
    top_ad_feats = top_ad["Feature"].tolist()
    auc_if_ad_k = evaluate_if_auc(df_processed, top_ad_feats, y, IF_PARAMS, random_seed)
    auc_lr_ad_k = evaluate_lr_auc(df_processed, top_ad_feats, y, random_seed)

    # Summary table (unified format)
    print("\nSummary Table (Stroke):")
    summary_df = pd.DataFrame(
        {
            "Method": [
                f"All Features ({len(FEATURE_NAMES)})",
                f"Original DIFFI Top-{k}",
                f"AD-DIFFI Top-{k}",
            ],
            "IF AUC": [
                round(auc_if_all, 4),
                round(auc_if_orig_k, 4),
                round(auc_if_ad_k, 4),
            ],
            "LR AUC": [
                round(auc_lr_all, 4),
                round(auc_lr_orig_k, 4),
                round(auc_lr_ad_k, 4),
            ],
        }
    )
    print(summary_df.to_markdown(index=False))

    # Rank stability
    rank_corr = compare_results["Rank_Original"].corr(
        compare_results["Rank_AD_DIFFI"], method="spearman"
    )
    mean_abs_rank_change = compare_results["Rank_Change"].abs().mean()
    print(f"\nRank Stability (Spearman ρ): {rank_corr:.4f}")
    print(f"Mean |RankΔ|: {mean_abs_rank_change:.3f}")

    print("=" * 70)
    print("Analysis complete.")
    print("=" * 70)

    results_dict = {
        "feature_importance": compare_results,
        "performance_summary": summary_df,
        "top_orig_features": top_orig_feats,
        "top_ad_features": top_ad_feats,
        "metrics": {
            "all": {"if_auc": auc_if_all, "lr_auc": auc_lr_all},
            "orig_topk": {"if_auc": auc_if_orig_k, "lr_auc": auc_lr_orig_k},
            "ad_topk": {"if_auc": auc_if_ad_k, "lr_auc": auc_lr_ad_k},
        },
    }
    return results_dict

# =============================================================================
# Cell 8: Save and Download Results
# =============================================================================

def save_and_download_results_stroke(results_dict: dict, random_seed: int = 42):
    """Save all results and prepare for download (Stroke)."""
    import json
    from pathlib import Path

    output_dir = Path("ad_diffi_results")
    output_dir.mkdir(exist_ok=True)

    fi_df = results_dict["feature_importance"]
    summary_df = results_dict["performance_summary"]

    fi_file = output_dir / f"stroke_feature_importance_seed{random_seed}.csv"
    summary_file = output_dir / f"stroke_performance_seed{random_seed}.csv"
    metrics_file = output_dir / f"stroke_metrics_seed{random_seed}.json"

    fi_df.to_csv(fi_file, index=False)
    summary_df.to_csv(summary_file, index=False)
    with open(metrics_file, "w") as f:
        json.dump(results_dict["metrics"], f, indent=2)

    print("[INFO] Files saved:")
    print(f"  - {fi_file}")
    print(f"  - {summary_file}")
    print(f"  - {metrics_file}")

    try:
        from google.colab import files
        files.download(str(fi_file))
        files.download(str(summary_file))
        files.download(str(metrics_file))
        print("[INFO] Downloads started...")
    except Exception:
        print("[INFO] Not in Colab; please download files manually.")

# =============================================================================
# Main Execution (Run in Colab) - Stroke
# =============================================================================

if __name__ == "__main__":
    RANDOM_SEED = 42
    results = run_stroke_analysis(random_seed=RANDOM_SEED)
    if results is not None:
        save_and_download_results_stroke(results, random_seed=RANDOM_SEED)
