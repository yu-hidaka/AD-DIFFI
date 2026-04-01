%%writefile /content/AD-DIFFI/src/ad_diffi/benchmarks.py
"""
Performance and Stability Benchmarks for AD-DIFFI.
This module provides logic to compare Feature Importance (FI) rankings 
and evaluate anomaly detection performance using AUC.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from scipy.stats import spearmanr

# =============================================================================
# 1. FEATURE IMPORTANCE COMPARISON LOGIC
# =============================================================================

def run_diffi_comparison_benchmark(
    X: np.ndarray,
    feature_names: List[str],
    feature_types: Dict[int, str],
    if_params: Dict[str, Any],
    diffi_func_orig: Any,      
    diffi_func_ad: Any,        
    noise_baseline_func: Any,  
    n_iter_noise: int = 20,
    n_iter_main: int = 20,
) -> pd.DataFrame:
    """
    Executes a benchmark comparing Original DIFFI and AD-DIFFI (RSO + Z-Normalization).
    """
    X_scaled = StandardScaler().fit_transform(X)

    # 1. Establish Noise Baseline
    baselines = noise_baseline_func(
        X_dim=X_scaled.shape[1],
        feature_types=feature_types,
        if_params=if_params,
        n_iter=n_iter_noise
    )
    cont_mean, cont_sd, bin_mean, bin_sd = baselines

    orig_scores_list = []
    ad_raw_list = []

    print(f"Running {n_iter_main} iterations for feature importance comparison...")
    
    for k in range(n_iter_main):
        current_params = if_params.copy()
        current_params.pop('random_state', None)
        
        iforest = IsolationForest(random_state=k, **current_params)
        iforest.fit(X_scaled)

        # Pass the noise_baselines to the original function
        res_orig = diffi_func_orig(iforest, X_scaled, noise_baselines=baselines)
        fi_orig = res_orig[0] if isinstance(res_orig, (tuple, list)) else res_orig
        orig_scores_list.append(fi_orig)

        # Calculate AD-DIFFI (RSO)
        fi_ad = diffi_func_ad(iforest, X_scaled, feature_types)
        ad_raw_list.append(fi_ad)

    # Calculate means
    mean_orig = np.mean(orig_scores_list, axis=0)
    matrix_ad_raw = np.vstack(ad_raw_list)

    # 2. Apply Z-normalization for AD-DIFFI
    ad_z_scores = matrix_ad_raw.copy()
    for i in range(len(feature_names)):
        f_type = feature_types[i]
        mu, sigma = (cont_mean, cont_sd) if f_type == 'cont' else (bin_mean, bin_sd)
        
        if sigma > 0:
            ad_z_scores[:, i] = (matrix_ad_raw[:, i] - mu) / sigma
        else:
            ad_z_scores[:, i] = 0.0

    mean_ad_z = np.mean(ad_z_scores, axis=0)

    # 3. Create Comparison Report
    compare_df = pd.DataFrame({
        "Feature": feature_names,
        "Type": [feature_types[i] for i in range(len(feature_names))],
        "Original_DIFFI": np.round(mean_orig, 4),
        "AD_DIFFI_RSO_Z": np.round(mean_ad_z, 4),
    })

    compare_df["Rank_Original"] = compare_df["Original_DIFFI"].rank(ascending=False)
    compare_df["Rank_AD_DIFFI"] = compare_df["AD_DIFFI_RSO_Z"].rank(ascending=False)
    compare_df["Rank_Change"] = compare_df["Rank_AD_DIFFI"] - compare_df["Rank_Original"]

    return compare_df.sort_values("AD_DIFFI_RSO_Z", ascending=False)

# =============================================================================
# 2. PERFORMANCE EVALUATION METRICS
# =============================================================================

def evaluate_feature_set_auc(
    df: pd.DataFrame,
    features: List[str],
    target: np.ndarray,
    model_type: str = 'IF',
    params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.3,
    random_state: int = 42
) -> float:
    if not features:
        return 0.5

    X = df[features].fillna(df[features].median()).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=test_size, random_state=random_state, stratify=target
    )
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    if model_type.upper() == 'IF':
        if_params = (params or {}).copy()
        if_params.pop('random_state', None)
        model = IsolationForest(**if_params, random_state=random_state)
        model.fit(X_train_sc)
        scores = -model.decision_function(X_test_sc)
    else:
        model = LogisticRegression(max_iter=1000, random_state=random_state)
        model.fit(X_train_sc, y_train)
        scores = model.predict_proba(X_test_sc)[:, 1]

    return float(roc_auc_score(y_test, scores))

def calculate_rank_stability(importance_df, col_a='Original_DIFFI', col_b='AD_DIFFI_RSO_Z'):
    rank_a = importance_df[col_a].rank(ascending=False)
    rank_b = importance_df[col_b].rank(ascending=False)
    rho, _ = spearmanr(rank_a, rank_b)
    mard = np.abs(rank_a - rank_b).mean()
    return {'spearman_rho': float(rho), 'mean_abs_rank_delta': float(mard)}

def get_top_k_features(importance_df, score_col, k=6):
    return importance_df.sort_values(by=score_col, ascending=False).head(k)['Feature'].tolist()
