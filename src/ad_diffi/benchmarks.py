"""
Performance and Stability Benchmarks for AD-DIFFI.
Includes AUC for classification, C-index for survival analysis, and DIFFI comparison logic.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from scipy.stats import spearmanr

# External dependency check for Survival Analysis
try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

# =============================================================================
# 1. Feature Importance Comparison Logic
# =============================================================================

def run_diffi_comparison_benchmark(
    X: np.ndarray,
    feature_names: List[str],
    feature_types: Dict[int, str],
    if_params: Dict[str, Any],
    diffi_func_orig,           # Inject original DIFFI function
    diffi_func_ad,             # Inject AD-DIFFI (RSO) function
    noise_baseline_func,       # Inject noise baseline calculation function
    n_iter_noise: int = 20,
    n_iter_main: int = 20,
) -> pd.DataFrame:
    """
    Compare Original DIFFI vs AD-DIFFI (RSO + Z-normalization).
    """
    X_scaled = StandardScaler().fit_transform(X)

    # 1. Get Noise Baseline (RSO-based)
    cont_mean, cont_sd, bin_mean, bin_sd = noise_baseline_func(
        X_dim=X_scaled.shape[1],
        feature_types=feature_types,
        if_params=if_params,
        n_iter=n_iter_noise
    )

    orig_scores_list = []
    rso_raw_list = []

    print(f"Running {n_iter_main} iterations for feature importance comparison...")
    for k in range(n_iter_main):
        iforest = IsolationForest(random_state=k, **if_params)
        iforest.fit(X_scaled)

        # Original DIFFI (adjust_iic=True)
        fi_orig, _ = diffi_func_orig(iforest, X_scaled, adjust_iic=True)
        orig_scores_list.append(fi_orig)

        # AD-DIFFI Raw (RSO-enabled)
        fi_rso = diffi_func_ad(iforest, X_scaled, feature_types)
        rso_raw_list.append(fi_rso)

    # Calculate means
    mean_orig = np.mean(orig_scores_list, axis=0)
    M_rso_raw = np.vstack(rso_raw_list)

    # 2. Z-normalization (Baseline Correction)
    rso_z_scores = M_rso_raw.copy()
    for i in range(len(feature_names)):
        f_type = feature_types[i]
        mu, sigma = (cont_mean, cont_sd) if f_type == 'cont' else (bin_mean, bin_sd)
        
        if sigma > 0:
            rso_z_scores[:, i] = (M_rso_raw[:, i] - mu) / sigma
        else:
            rso_z_scores[:, i] = 0.0

    mean_rso_z = np.mean(rso_z_scores, axis=0)

    # Aggregate Results
    compare_df = pd.DataFrame({
        "Feature": feature_names,
        "Type": [feature_types[i] for i in range(len(feature_names))],
        "Original_DIFFI": np.round(mean_orig, 4),
        "AD_DIFFI_RSO_Z": np.round(mean_rso_z, 4),
    })

    # Ranking and Delta Calculation
    compare_df["Rank_Original"] = compare_df["Original_DIFFI"].rank(ascending=False)
    compare_df["Rank_AD_DIFFI"] = compare_df["AD_DIFFI_RSO_Z"].rank(ascending=False)
    compare_df["Rank_Change"] = compare_df["Rank_AD_DIFFI"] - compare_df["Rank_Original"]

    return compare_df.sort_values("AD_DIFFI_RSO_Z", ascending=False)


# =============================================================================
# 2. Evaluation Metrics (AUC & C-Index)
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
    """
    Evaluate AUC for classification tasks (e.g., Annthyroid).
    """
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
        model = IsolationForest(**(params or {}), random_state=random_state)
        model.fit(X_train_sc)
        scores = -model.decision_function(X_test_sc)
    elif model_type.upper() == 'LR':
        model = LogisticRegression(max_iter=1000, random_state=random_state)
        model.fit(X_train_sc, y_train)
        scores = model.predict_proba(X_test_sc)[:, 1]
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return float(roc_auc_score(y_test, scores))

def evaluate_feature_set_cindex(
    df: pd.DataFrame,
    features: List[str],
    duration_col: str,
    event_col: str,
    test_size: float = 0.3,
    random_state: int = 42
) -> float:
    """
    Evaluate C-Index for survival analysis tasks (Cox PH Model).
    """
    if not HAS_LIFELINES:
        print("[ERROR] lifelines library is not installed.")
        return 0.0
    
    if not features:
        return 0.5

    data = df[features + [duration_col, event_col]].copy()
    data = data.fillna(data.median())
    
    train_df, test_df = train_test_split(
        data, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])

    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(train_df, duration_col=duration_col, event_col=event_col)
        scores = cph.predict_partial_hazard(test_df)
        c_index = concordance_index(test_df[duration_col], -scores, test_df[event_col])
        return float(c_index)
    except Exception as e:
        print(f"[WARNING] Cox model failed: {e}")
        return 0.5

# =============================================================================
# 3. Stability Helpers
# =============================================================================

def calculate_rank_stability(
    importance_df: pd.DataFrame, 
    col_a: str = 'Original_DIFFI', 
    col_b: str = 'AD_DIFFI_RSO_Z'
) -> Dict[str, float]:
    """
    Calculate Spearman's Rho and Mean Absolute Rank Delta (MARD).
    """
    rank_a = importance_df[col_a].rank(ascending=False)
    rank_b = importance_df[col_b].rank(ascending=False)
    
    rho, _ = spearmanr(rank_a, rank_b)
    mard = np.abs(rank_a - rank_b).mean()
    
    return {'spearman_rho': float(rho), 'mean_abs_rank_delta': float(mard)}

def get_top_k_features(importance_df: pd.DataFrame, score_col: str, k: int = 6) -> List[str]:
    """
    Get names of the top-k features based on score.
    """
    return importance_df.sort_values(by=score_col, ascending=False).head(k)['Feature'].tolist()
