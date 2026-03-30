"""
Performance and Stability Benchmarks for AD-DIFFI.
This module provides utilities to evaluate the predictive performance of 
feature sets and the rank stability between different importance methods.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from scipy.stats import spearmanr


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
    Evaluate the Area Under the ROC Curve (AUC) for a specific feature set.
    
    Args:
        df: Input DataFrame containing the features.
        features: List of feature names to evaluate.
        target: Binary ground truth labels (0 for normal, 1 for anomaly).
        model_type: Type of model to evaluate ('IF' for Isolation Forest, 
                    'LR' for Logistic Regression).
        params: Dictionary of hyperparameters for the model.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Seed for reproducibility.

    Returns:
        float: Calculated ROC AUC score.
    """
    if not features:
        return 0.5  # Random baseline for empty feature set

    # Handle missing values using median imputation
    X = df[features].fillna(df[features].median()).values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=target
    )
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    if model_type.upper() == 'IF':
        # Evaluation as an Unsupervised Anomaly Detection task
        model = IsolationForest(**(params or {}), random_state=random_state)
        model.fit(X_train_sc)
        # Higher score = more anomalous. decision_function returns lower for anomalies.
        scores = -model.decision_function(X_test_sc)
    elif model_type.upper() == 'LR':
        # Evaluation as a Supervised Classification task
        model = LogisticRegression(max_iter=1000, random_state=random_state)
        model.fit(X_train_sc, y_train)
        scores = model.predict_proba(X_test_sc)[:, 1]
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'IF' or 'LR'.")

    return float(roc_auc_score(y_test, scores))


def calculate_rank_stability(
    importance_df: pd.DataFrame, 
    col_method_a: str, 
    col_method_b: str
) -> Dict[str, float]:
    """
    Calculate stability metrics between two feature importance methods.
    
    Args:
        importance_df: DataFrame containing the importance scores.
        col_method_a: Column name for the baseline method (e.g., 'Original_DIFFI').
        col_method_b: Column name for the proposed method (e.g., 'AD_DIFFI_RSO_Z').

    Returns:
        Dict: Dictionary containing Spearman's Rho and Mean Absolute Rank Change.
    """
    # Assign ranks (descending: highest importance = rank 1)
    rank_a = importance_df[col_method_a].rank(ascending=False)
    rank_b = importance_df[col_method_b].rank(ascending=False)
    
    # Spearman's rank correlation coefficient (rho)
    rho, _ = spearmanr(rank_a, rank_b)
    
    # Mean Absolute Rank Delta (MARD)
    mean_abs_delta = np.abs(rank_a - rank_b).mean()
    
    return {
        'spearman_rho': float(rho),
        'mean_abs_rank_delta': float(mean_abs_delta)
    }


def get_top_k_features(
    importance_df: pd.DataFrame, 
    method_col: str, 
    k: int = 5,
    feature_col: str = 'Feature'
) -> List[str]:
    """
    Extract the top-k feature names based on a specific importance score.
    """
    top_k = importance_df.sort_values(by=method_col, ascending=False).head(k)
    return top_k[feature_col].tolist()
