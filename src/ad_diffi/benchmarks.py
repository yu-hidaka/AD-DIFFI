"""
Performance and Stability Benchmarks for AD-DIFFI.
Includes AUC for classification and C-index for survival analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from scipy.stats import spearmanr

# For Survival Analysis
try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

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
    Evaluate AUC for classification tasks (Annthyroid, etc.)
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
    Evaluate Concordance Index using Cox Proportional Hazards Model (Breast Cancer, etc.)
    """
    if not HAS_LIFELINES:
        print("[ERROR] lifelines library is not installed.")
        return 0.0
    
    if not features:
        return 0.5

    # Prepare data for lifelines
    data = df[features + [duration_col, event_col]].copy()
    data = data.fillna(data.median())
    
    # Simple split
    train_df, test_df = train_test_split(
        data, test_size=test_size, random_state=random_state
    )

    # Standardize features (recommended for Cox regression convergence)
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])

    try:
        cph = CoxPHFitter(penalizer=0.1) # Add slight regularization for stability
        cph.fit(train_df, duration_col=duration_col, event_col=event_col)
        
        # Calculate C-index on test set
        scores = cph.predict_partial_hazard(test_df)
        c_index = concordance_index(test_df[duration_col], -scores, test_df[event_col])
        return float(c_index)
    except Exception as e:
        print(f"[WARNING] Cox model failed: {e}")
        return 0.5

def calculate_rank_stability(
    importance_df: pd.DataFrame, 
    col_a: str = 'Original_DIFFI', 
    col_back: str = 'AD_DIFFI'
) -> Dict[str, float]:
    """
    Calculate Spearman's Rho and MARD between two methods.
    """
    rank_a = importance_df[col_a].rank(ascending=False)
    rank_b = importance_df[col_back].rank(ascending=False)
    
    rho, _ = spearmanr(rank_a, rank_b)
    mard = np.abs(rank_a - rank_b).mean()
    
    return {'spearman_rho': float(rho), 'mean_abs_rank_delta': float(mard)}

def get_top_k_features(importance_df: pd.DataFrame, score_col: str, k: int = 6) -> List[str]:
    """
    Extract the top-k feature names.
    """
    return importance_df.sort_values(by=score_col, ascending=False).head(k)['Feature'].tolist()
