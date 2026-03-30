import numpy as np
import pandas as pd
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# Import core functions from core.py
# from ad_diffi.core import diffi_ib_binary_rso, calculate_ad_diffi_zscore

# =============================================================================
# 1. DATA LOADING & PREPROCESSING
# =============================================================================

def download_adbench_dataset(dataset_name: str) -> str:
    """
    Download a dataset from the ADBench public repository.
    """
    data_dir = Path('/tmp/ad_diffi_data')
    data_dir.mkdir(exist_ok=True)
    csv_file = data_dir / f"{dataset_name}.csv"

    if csv_file.exists():
        return str(csv_file)

    url = f"https://raw.githubusercontent.com/Minqi824/ADBench/main/data/{dataset_name}.csv"
    try:
        urllib.request.urlretrieve(url, str(csv_file))
        return str(csv_file)
    except Exception as e:
        print(f"[ERROR] Failed to download {dataset_name}: {e}")
        return ""

def get_feature_metadata(df: pd.DataFrame, dataset_name: str, label_col: str = "Outlier_label") -> Tuple[List[str], Dict[int, str]]:
    """
    Identify continuous and binary features based on dataset-specific rules.
    """
    all_cols = [c for c in df.columns if c != label_col]
    
    # Dataset-specific continuous feature definitions
    cont_map = {
        "annthyroid": ['TBG', 'TSH', 'T3', 'TT4', 'T4U', 'FTI'],
        "breast_cancer": [col for col in all_cols if "mean" in col or "worst" in col or "error" in col],
        "hepatitis": ['age', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime']
    }
    
    target_cont = cont_map.get(dataset_name, [])
    
    # Filter columns that actually exist in the dataframe
    actual_cont = [c for c in target_cont if c in df.columns]
    
    # If no specific rule, assume non-binary (more than 2 unique values) are continuous
    if not actual_cont:
        actual_cont = [c for c in all_cols if df[c].nunique() > 2]

    feature_names = all_cols
    feature_types = {i: "cont" if col in actual_cont else "bin" for i, col in enumerate(feature_names)}
    
    return feature_names, feature_types

# =============================================================================
# 2. NOISE BASELINE GENERATION
# =============================================================================

def get_noise_baseline(
    X_dim: int, 
    feature_types: Dict[int, str], 
    if_params: Dict[str, Any], 
    n_iter: int = 50
) -> Dict[str, Dict[str, float]]:
    """
    Establish a noise baseline for Z-normalization using uniform noise data.
    """
    # Note: Requires diffi_ib_binary_rso from core.py
    from ad_diffi.core import diffi_ib_binary_rso
    
    scores_list = []
    cont_indices = [i for i, t in feature_types.items() if t == 'cont']
    bin_indices = [i for i, t in feature_types.items() if t == 'bin']

    for k in range(n_iter):
        X_noise = np.random.uniform(0, 1, size=(2000, X_dim))
        if_model = IsolationForest(random_state=k, **if_params)
        if_model.fit(X_noise)
        
        fi = diffi_ib_binary_rso(if_model, X_noise, feature_types)
        scores_list.append(fi)

    M = np.vstack(scores_list)
    
    baselines = {}
    for key, indices in [('cont', cont_indices), ('bin', bin_indices)]:
        if indices:
            sub_scores = M[:, indices]
            baselines[key] = {'mean': float(sub_scores.mean()), 'sd': float(sub_scores.std())}
        else:
            baselines[key] = {'mean': 0.0, 'sd': 1.0}
            
    return baselines

# =============================================================================
# 3. PERFORMANCE EVALUATION
# =============================================================================

def evaluate_auc(
    df: pd.DataFrame, 
    features: List[str], 
    y: np.ndarray, 
    model_type: str = "IF", 
    if_params: Optional[Dict] = None, 
    seed: int = 42
) -> float:
    """
    Evaluate Anomaly Detection performance using AUC (Isolation Forest or Logistic Regression).
    """
    X = df[features].fillna(df[features].median()).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "IF":
        params = if_params if if_params else {}
        model = IsolationForest(**params, random_state=seed)
        model.fit(X_train_scaled)
        # Use negative decision function for anomaly score
        scores = -model.decision_function(X_test_scaled)
    else:
        model = LogisticRegression(max_iter=1000, random_state=seed)
        model.fit(X_train_scaled, y_train)
        scores = model.predict_proba(X_test_scaled)[:, 1]

    return float(roc_auc_score(y_test, scores))

# =============================================================================
# 4. SUMMARY & REPORTING
# =============================================================================

def create_importance_report(
    feature_names: List[str],
    feature_types: Dict[int, str],
    orig_scores: np.ndarray,
    ad_scores: np.ndarray
) -> pd.DataFrame:
    """
    Create a comparative DataFrame of feature importance scores and ranks.
    """
    report = pd.DataFrame({
        "Feature": feature_names,
        "Type": [feature_types[i] for i in range(len(feature_names))],
        "Original_DIFFI": np.round(orig_scores, 4),
        "AD_DIFFI": np.round(ad_scores, 4)
    })
    
    report["Rank_Orig"] = report["Original_DIFFI"].rank(ascending=False)
    report["Rank_AD"] = report["AD_DIFFI"].rank(ascending=False)
    report["Rank_Delta"] = report["Rank_AD"] - report["Rank_Orig"]
    
    return report.sort_values("AD_DIFFI", ascending=False)
