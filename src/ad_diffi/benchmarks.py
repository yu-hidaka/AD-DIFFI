"""
Performance and Stability Benchmarks for AD-DIFFI.
Includes Survival Analysis (Cox C-index) for datasets like Breast Cancer.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import spearmanr

# =============================================================================
# 1. FEATURE IMPORTANCE COMPARISON LOGIC (Existing)
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

    for k in range(n_iter_main):
        if_params_copy = if_params.copy()
        if_params_copy.pop('random_state', None)
        iforest = IsolationForest(random_state=k, **if_params_copy)
        iforest.fit(X_scaled)

        res_orig = diffi_func_orig(iforest, X_scaled, noise_baselines=baselines)
        fi_orig = res_orig[0] if isinstance(res_orig, (tuple, list)) else res_orig
        orig_scores_list.append(fi_orig)

        fi_ad = diffi_func_ad(iforest, X_scaled, feature_types)
        ad_raw_list.append(fi_ad)

    mean_orig = np.mean(orig_scores_list, axis=0)
    matrix_ad_raw = np.vstack(ad_raw_list)

    # 2. Apply Z-normalization
    ad_z_scores = matrix_ad_raw.copy()
    for i in range(len(feature_names)):
        f_type = feature_types[i]
        mu, sigma = (cont_mean, cont_sd) if f_type == 'cont' else (bin_mean, bin_sd)
        ad_z_scores[:, i] = (matrix_ad_raw[:, i] - mu) / sigma if sigma > 0 else 0.0

    mean_ad_z = np.mean(ad_z_scores, axis=0)

    compare_df = pd.DataFrame({
        "Feature": feature_names,
        "Type": [feature_types[i] for i in range(len(feature_names))],
        "Original_DIFFI": np.round(mean_orig, 4),
        "AD_DIFFI_RSO_Z": np.round(mean_ad_z, 4),
    })

    compare_df["Rank_Original"] = compare_df["Original_DIFFI"].rank(ascending=False).astype(int)
    compare_df["Rank_AD_DIFFI"] = compare_df["AD_DIFFI_RSO_Z"].rank(ascending=False).astype(int)
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
    if not features: return 0.5
    X = df[features].fillna(df[features].median()).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=test_size, random_state=random_state, stratify=target
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    if model_type.upper() == 'IF':
        p = (params or {}).copy(); p.pop('random_state', None)
        model = IsolationForest(**p, random_state=random_state).fit(X_train_sc)
        scores = -model.decision_function(X_test_sc)
    else:
        model = LogisticRegression(max_iter=1000, random_state=random_state).fit(X_train_sc, y_train)
        scores = model.predict_proba(X_test_sc)[:, 1]
    return float(roc_auc_score(y_test, scores))

def evaluate_cox_cindex(
    df: pd.DataFrame,
    feature_list: List[str],
    time_col: str = 'Survival Months',
    event_col: str = 'Status_num',
    max_features: int = 10,
    random_seed: int = 42
) -> float:
    """
    Robust Cox C-index for Survival data (e.g. Breast Cancer).
    """
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    # 1. Data Cleaning
    cols = feature_list + [time_col, event_col]
    X_raw = df[cols].dropna().reset_index(drop=True)
    if len(X_raw) < 20: return 0.5

    T = X_raw[time_col]
    E = X_raw[event_col].astype(bool)
    X = X_raw[feature_list]

    # 2. Multicollinearity & Variance Filter
    selector = VarianceThreshold(threshold=0.01)
    X_var = selector.fit_transform(X)
    selected_feats = [feature_list[i] for i in selector.get_support(indices=True)]
    
    X_filtered = pd.DataFrame(X_var, columns=selected_feats)
    if len(selected_feats) > max_features:
        # Simple correlation filter to keep Top-K stable
        corr = X_filtered.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr = [c for c in upper.columns if any(upper[c] > 0.95)]
        final_feats = [f for f in selected_feats if f not in high_corr][:max_features]
        X_filtered = X_filtered[final_feats]
    else:
        final_feats = selected_feats

    # 3. Model Fitting
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_filtered), columns=final_feats)
    cox_df = pd.concat([X_scaled, T, E.rename('event')], axis=1)

    try:
        cph = CoxPHFitter(penalizer=0.1) # Add regularization for stability
        cph.fit(cox_df, duration_col=time_col, event_col='event')
        # Negative risk scores correlate with longer survival
        risk = -cph.predict_partial_hazard(X_scaled)
        return float(concordance_index(T, risk, E))
    except:
        return 0.5

def calculate_rank_stability(importance_df, col_a='Original_DIFFI', col_b='AD_DIFFI_RSO_Z'):
    rho, _ = spearmanr(importance_df[col_a].rank(ascending=False), 
                       importance_df[col_b].rank(ascending=False))
    mard = (importance_df[col_a].rank(ascending=False) - importance_df[col_b].rank(ascending=False)).abs().mean()
    return {'spearman_rho': float(rho), 'mean_abs_rank_delta': float(mard)}

def get_top_k_features(importance_df, score_col, k=6):
    return importance_df.sort_values(by=score_col, ascending=False).head(k)['Feature'].tolist()
