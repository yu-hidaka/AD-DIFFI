import numpy as np
import pandas as pd
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Any

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
    
    # Dataset-specific continuous feature definitions (case-insensitive keys)
    cont_map = {
        "annthyroid": ['tbg', 'tsh', 't3', 'tt4', 't4u', 'fti'],
        "breast_cancer": [col for col in all_cols if any(x in col.lower() for x in ["mean", "worst", "error"])],
        "hepatitis": ['age', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime']
    }
    
    target_cont = cont_map.get(dataset_name.lower(), [])
    
    # Identify actual continuous columns present in the dataframe (case-insensitive check)
    actual_cont = []
    for col in all_cols:
        if col.lower() in target_cont or col in target_cont:
            actual_cont.append(col)
    
    # Fallback: if no specific rule matched, assume features with >2 unique values are continuous
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
    from sklearn.ensemble import IsolationForest
    from ad_diffi.core import diffi_ib_binary_rso
    
    scores_list = []
    cont_indices = [i for i, t in feature_types.items() if t == 'cont']
    bin_indices = [i for i, t in feature_types.items() if t == 'bin']

    for k in range(n_iter):
        # Generate 2000 samples of pure noise
        X_noise = np.random.uniform(0, 1, size=(2000, X_dim))
        
        if_model = IsolationForest(random_state=k, **if_params)
        if_model.fit(X_noise)
        
        fi = diffi_ib_binary_rso(if_model, X_noise, feature_types)
        scores_list.append(fi)

    M = np.vstack(scores_list)
    
    baselines = {}
    for key, indices in [('cont', cont_indices), ('bin', bin_indices)]:
        if indices:
            # Calculate mean/sd across all noise iterations for this feature type
            sub_scores = M[:, indices]
            baselines[key] = {
                'mean': float(np.mean(sub_scores)), 
                'sd': float(np.std(sub_scores))
            }
        else:
            baselines[key] = {'mean': 0.0, 'sd': 1.0}
            
    return baselines

# =============================================================================
# 3. SUMMARY & REPORTING
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
    
    # Higher score means higher importance (Rank 1 = Most important)
    report["Rank_Orig"] = report["Original_DIFFI"].rank(ascending=False, method='min')
    report["Rank_AD"] = report["AD_DIFFI"].rank(ascending=False, method='min')
    report["Rank_Delta"] = report["Rank_Orig"] - report["Rank_AD"] # Positive means rank improved in AD-DIFFI
    
    return report.sort_values("AD_DIFFI", ascending=False)
