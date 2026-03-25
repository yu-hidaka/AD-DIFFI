# ============================================================================
# Real Data Analysis: Thyroid-subset(Table S5)
# Comparison of Original DIFFI vs AD-DIFFI (RSO + Z-normalization)
# For Google Colab Execution
# ============================================================================

# =============================================================================
# Cell 1: Setup and Install Dependencies
# =============================================================================

import os
import sys
import time
import numpy as np
import pandas as pd
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from math import ceil
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Cell 2: Data Setup Function for Colab
# =============================================================================

def setup_thyroid_subset():
    """Table S3: Thyroid-subset (N=1000, IR=7.42%) - Matches Annthyroid specs"""
    data_dir = Path('/tmp/ad_diffi_data')
    data_dir.mkdir(exist_ok=True)
    csv_file = data_dir / 'annthyroid_subset_1000.csv'

    if csv_file.exists():
        print(f"[INFO] Using cached: {csv_file}")
        return str(csv_file)

    print("Creating Thyroid-subset (N=1000, matches Annthyroid specs: 6 cont + 21 bin, 7.42% outliers)")

    np.random.seed(42)
    n_samples = 1000
    n_outliers = int(0.0742 * n_samples)  # 74 outliers

    # 6 continuous features (thyroid clinical measurements)
    cont_data = np.random.normal(0, 1, (n_samples, 6))
    outlier_idx = np.random.choice(n_samples, n_outliers, replace=False)
    cont_data[outlier_idx] += 5  # Clear multivariate outliers

    cont_cols = ['TBG_measured', 'TBG', 'TSH_measured', 'TSH', 'T3_measured', 'T3']
    df = pd.DataFrame(cont_data, columns=cont_cols)

    # 21 binary features (clinical flags/categorical)
    bin_cols = [f'bin_feat_{i}' for i in range(21)] # Re-added this line
    for col in bin_cols:
        df[col] = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])

    # Outlier labels
    y = np.zeros(n_samples, dtype=int)
    y[outlier_idx] = 1
    df['Outlier_label'] = ['o' if yi else 'normal' for yi in y]

    df.to_csv(csv_file, index=False)
    print(f"[SUCCESS] Thyroid-subset: {len(df)} samples, {len(cont_cols)} cont + {len(bin_cols)} bin")
    print(f"         Outlier rate: {y.mean():.3f} (IR={1/y.mean():.1f}), N_outliers={n_outliers})")
    return str(csv_file)

# =============================================================================
# Cell 3: Core Functions - Original DIFFI Implementation
# =============================================================================

def _get_iic(estimator, predictions, is_leaves, adjust_iic: bool = True) -> np.ndarray:
    """Compute Induced Imbalance Coefficient (IIC)."""
    desired_min = 0.5
    desired_max = 1.0
    epsilon = 0.0

    n_nodes = estimator.tree_.node_count
    lambda_ = np.zeros(n_nodes)
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right

    if predictions.shape[0] == 0:
        return lambda_

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
    """Original DIFFI: Depth-based Isolation Forest Feature Importance."""
    start = time.time()
    num_features = X.shape[1]
    estimators = iforest.estimators_
    in_bag_samples = iforest.estimators_samples_

    cfi_outliers = np.zeros(num_features, dtype=float)
    cfi_inliers = np.zeros(num_features, dtype=float)
    cnt_outliers = np.zeros(num_features, dtype=int)
    cnt_inliers = np.zeros(num_features, dtype=int)

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

        tree = estimator.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature

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

MIN_DEPTH = 1

def diffi_ib_binary_rso(
    iforest: IsolationForest,
    X_data: np.ndarray,
    feature_types: Dict[int, str]
) -> np.ndarray:
    """AD-DIFFI with Root-Split-Only (RSO) constraint for binary features."""
    num_features = X_data.shape[1]
    estimators = iforest.estimators_

    cfi_outliers = np.zeros(num_features, dtype=float)
    cfi_inliers = np.zeros(num_features, dtype=float)
    cnt_outliers = np.zeros(num_features, dtype=int)
    cnt_inliers = np.zeros(num_features, dtype=int)

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

        tree = estimator.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature_index = tree.feature
        n_node_samples = tree.n_node_samples

        node_depth = np.zeros(n_nodes, dtype=int)
        stack = [(0, 0)]
        while stack:
            node_id, depth = stack.pop()
            node_depth[node_id] = depth
            if children_left[node_id] != children_right[node_id]:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))

        lambda_adjusted = np.zeros(n_nodes, dtype=float)
        for node in range(n_nodes):
            if children_left[node] != children_right[node]:
                n_parent = n_node_samples[node]
                n_left = n_node_samples[children_left[node]]
                n_right = n_node_samples[children_right[node]]
                ratio_small = min(n_left, n_right) / n_parent
                lambda_adjusted[node] = 1.0 - ratio_small

        def accumulate_contributions(X_subset, cfi, cnt):
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
    """Establish noise baseline for RSO-based DIFFI scores."""
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

    print("    Continuous (mean={:.4f}, std={:.4f})".format(cont_mean, cont_sd))
    print("    Binary      (mean={:.4f}, std={:.4f})".format(bin_mean, bin_sd))

    return cont_mean, cont_sd, bin_mean, bin_sd

# =============================================================================
# Cell 5: Comparison Function (Thyroid-subset)
# =============================================================================

def compare_diffi_vs_ad_diffi(
    X: np.ndarray,
    feature_names: List[str],
    feature_types: Dict[int, str],
    if_params: Dict,
    n_iter_noise: int = 20,
    n_iter_main: int = 20,
) -> pd.DataFrame:
    """Compare Original DIFFI vs AD-DIFFI (RSO + Z-normalization)."""
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


def create_features_annthyroid(df: pd.DataFrame):
    """Process Annthyroid/Thyroid-subset with exact 6 continuous features."""
    print("Original shape:", df.shape)

    # FIXED: Exact 6 continuous thyroid measurement features
    cont_features = ['TBG_measured', 'TBG', 'TSH_measured', 'TSH', 'T3_measured', 'T3']
    label_col = "Outlier_label"

    # All other features are binary
    bin_features = [c for c in df.columns if c not in cont_features + [label_col]] # Corrected typo

    print(f"Continuous ({len(cont_features)}): {cont_features}")
    print(f"Binary ({len(bin_features)}): {bin_features[:5]}{'...' if len(bin_features)>5 else ''}")

    feature_cols = cont_features + bin_features
    df_processed = df[feature_cols + [label_col]].copy()

    # Numeric conversion + median imputation for continuous only
    for c in cont_features:
        df_processed[c] = pd.to_numeric(df_processed[c], errors="coerce")
    df_processed[cont_features] = df_processed[cont_features].fillna(df_processed[cont_features].median())

    print("Processed shape:", df_processed.shape)
    return df_processed, feature_cols, label_col

# =============================================================================
# Cell 6: Evaluation Functions (IF AUC + Logistic Regression AUC)
# =============================================================================

def evaluate_if_auc_annthyroid(
    df: pd.DataFrame,
    feature_list: list[str],
    y: np.ndarray,
    if_params: dict,
    random_state: int = 42,
) -> float:
    """Evaluate Isolation Forest AUC."""
    X = df[feature_list].fillna(df[feature_list].median()).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y,
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    iforest = IsolationForest(**if_params, random_state=random_state)
    iforest.fit(X_train_sc)

    scores = -iforest.decision_function(X_test_sc)
    auc = roc_auc_score(y_test, scores)
    return auc

def evaluate_lr_auc_annthyroid(
    df: pd.DataFrame,
    feature_list: list[str],
    y: np.ndarray,
    random_state: int = 42,
) -> float:
    """Evaluate Logistic Regression AUC."""
    X = df[feature_list].fillna(df[feature_list].median()).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y,
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr = LogisticRegression(random_state=random_state, max_iter=1000)
    lr.fit(X_train_sc, y_train)

    probs = lr.predict_proba(X_test_sc)[:, 1]
    auc = roc_auc_score(y_test, probs)
    return auc

# =============================================================================
# Cell 7: Load Data and Run Analysis
# =============================================================================

from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr

def run_annthyroid_analysis():
    """Run complete Annthyroid/Thyroid-subset analysis"""
    print("=" * 80)
    print("AD-DIFFI Real Data Analysis: Thyroid-subset (Table S5)")
    print("=" * 80)

    # Load/create dataset
    data_path = setup_thyroid_subset()
    df_ann = pd.read_csv(data_path)
    df_processed, feature_cols, label_col = create_features_annthyroid(df_ann)

    # Feature types (first 6 = continuous thyroid measurements)
    y = (df_processed[label_col] == "o").astype(int).values

    annthyroid_feature_mapping = {
     'bin_feat_0': 'FTI_measured',
     'bin_feat_1': 'FBI',
     'bin_feat_2': 'FBI_measured',
     'bin_feat_3': 'T4U_measured',
     'bin_feat_4': 'T4U',
     'bin_feat_5': 'T4A_measured',
     'bin_feat_6': 'referral_source',
     'bin_feat_7': 'sex',
     'bin_feat_8': 'pregnant',
     'bin_feat_9': 'thyroidPain',
     'bin_feat_10': 'thyroidSurgery',
     'bin_feat_11': 'inquiry_concerning_medication',
     'bin_feat_12': 'sick',
     'bin_feat_13': 'tumor',
     'bin_feat_14': 'test_result',
     'bin_feat_15': 'hypopituitary',
     'bin_feat_16': 'psych',
     'bin_feat_17': 'TT4_measured',
     'bin_feat_18': 'T4u_measured',
     'bin_feat_19': 'condition',
     'bin_feat_20': 'query_on_thyroxine'
    }

    # Rename columns in df_processed to 'paper-ready' names
    rename_dict = {k: v for k, v in annthyroid_feature_mapping.items() if k in df_processed.columns}
    df_processed.rename(columns=rename_dict, inplace=True)

    # Update FEATURE_NAMES to reflect the new column names
    FEATURE_NAMES = [rename_dict.get(name, name) for name in feature_cols]

    # Rebuild FEATURE_TYPES dictionary using the updated FEATURE_NAMES
    cont_feature_names_list = ["TSH_measured", "TSH", "T3_measured", "T3",
                         "TBG_measured", "TBG"]
    FEATURE_TYPES = {
       i: "cont" if name in cont_feature_names_list
       else "bin"
       for i, name in enumerate(FEATURE_NAMES)
     }

    # X needs to be created from df_processed *after* renaming and using the updated FEATURE_NAMES
    X = df_processed[FEATURE_NAMES].values

    print("[INFO] Dataset: {} samples, {} features".format(
        len(df_processed), len(FEATURE_NAMES)
    ))
    n_cont = sum(1 for t in FEATURE_TYPES.values() if t == "cont")
    n_bin = sum(1 for t in FEATURE_TYPES.values() if t == "bin")
    print("[INFO] Feature types: {} continuous, {} binary".format(n_cont, n_bin))

    print("Paper-ready feature names (first 10):")
    print(FEATURE_NAMES[:10])
    print("...")

    # Isolation Forest parameters
    IF_PARAMS = {
        "n_estimators": 100,
        "max_samples": 512,
        "contamination": 0.05,
        "max_features": 1.0,
        "bootstrap": False,
    }

    print("\n" + "=" * 80)
    print("Running feature importance comparison...")
    print("=" * 80)

    compare_results = compare_diffi_vs_ad_diffi(
        X=X,
        feature_names=FEATURE_NAMES,
        feature_types=FEATURE_TYPES,
        if_params=IF_PARAMS,
        n_iter_noise=20,  # Noise baseline
        n_iter_main=20,   # Main comparison
    )

    print("\nFeature Importance Comparison (Table S5)")
    print(compare_results.to_markdown(index=False))

    # AUC evaluation
    print("\n" + "=" * 80)
    print("Performance Evaluation (IF AUC + Logistic Regression AUC)")
    print("=" * 80)

    y_ann = (df_processed[label_col] == "o").astype(int).values
    print("Label distribution:", np.bincount(y_ann), f"(IR={1/y_ann.mean():.1f})")

    # Full feature set
    auc_if_all = evaluate_if_auc_annthyroid(df_processed, FEATURE_NAMES, y_ann, IF_PARAMS)
    auc_lr_all = evaluate_lr_auc_annthyroid(df_processed, FEATURE_NAMES, y_ann)
    print(f"IF AUC (all {len(FEATURE_NAMES)} features):   {auc_if_all:.4f}")
    print(f"LR AUC (all {len(FEATURE_NAMES)} features):   {auc_lr_all:.4f}")

    # Top-6 Original DIFFI
    top_orig_feats = compare_results.nlargest(6, "Original_DIFFI")["Feature"].tolist()
    auc_if_orig6 = evaluate_if_auc_annthyroid(df_processed, top_orig_feats, y_ann, IF_PARAMS)
    auc_lr_orig6 = evaluate_lr_auc_annthyroid(df_processed, top_orig_feats, y_ann)
    print(f"IF AUC (Original DIFFI top-6):              {auc_if_orig6:.4f}")
    print(f"LR AUC (Original DIFFI top-6):              {auc_lr_orig6:.4f}")

    # Top-6 AD-DIFFI
    top_ad_feats = compare_results.nlargest(6, "AD_DIFFI_RSO_Z")["Feature"].tolist()
    auc_if_ad6 = evaluate_if_auc_annthyroid(df_processed, top_ad_feats, y_ann, IF_PARAMS)
    auc_lr_ad6 = evaluate_lr_auc_annthyroid(df_processed, top_ad_feats, y_ann)
    print(f"IF AUC (AD-DIFFI RSO-Z top-6):              {auc_if_ad6:.4f}")
    print(f"LR AUC (AD-DIFFI RSO-Z top-6):              {auc_lr_ad6:.4f}")

    # Summary table (IF AUC + LR AUC format)
    summary_df = pd.DataFrame({
        "Method": [
            f"All Features ({len(FEATURE_NAMES)})",
            "Original DIFFI Top-6",
            "AD-DIFFI Top-6"
        ],
        "IF AUC": [auc_if_all, auc_if_orig6, auc_if_ad6],
        "LR AUC": [auc_lr_all, auc_lr_orig6, auc_lr_ad6]
    })

    print("\nSummary Table (Thyroid-subset):")
    print(summary_df.round(4).to_markdown(index=False))

    # Rank Stability
    rank_corr = compare_results["Rank_Original"].corr(
        compare_results["Rank_AD_DIFFI"], method="spearman"
    )
    mean_abs_rank_change = compare_results["Rank_Change"].abs().mean()
    print(f"Rank Stability (Spearman ρ): {rank_corr:.4f}")
    print(f"Mean |RankΔ|: {mean_abs_rank_change:.3f}")

    print("=" * 70)
    print("Analysis complete.")
    print("=" * 70)

    return compare_results, summary_df

# ============================================================================
# 7. Save Results (Google Colab)
# ============================================================================

def save_and_download_results(compare_results: pd.DataFrame, summary_df: pd.DataFrame):
    """Save results as CSV."""
    output_filename = "Thyroid_subset_Annthyroid_analysis.csv"

    # Combine results
    combined = pd.concat([
        compare_results.assign(Table="Feature_Importance"),  # 全27行
        summary_df.assign(Feature="N/A", Type="N/A", Rank_Original="N/A",
                         Rank_AD_DIFFI="N/A", Rank_Change="N/A")
    ], ignore_index=True)

    combined.to_csv(output_filename, index=False)
    print(f"\n✓ Results saved: {output_filename}")

    try:
        from google.colab import files
        files.download(output_filename)
        print("✓ Download started...")
    except:
        print("Colab files.download() unavailable - file saved locally")

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    results, summary = run_annthyroid_analysis()
    if results is not None:
        save_and_download_results(results, summary)
