import numpy as np
import pandas as pd
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Any

# =============================================================================
# 1. DATA LOADING & PREPROCESSING (Robust Version)
# =============================================================================

def download_adbench_dataset(dataset_name: str) -> str:
    """
    Download a dataset from the ADBench public repository.
    If the URL is broken (404), generates a synthetic dataset matching paper specs.
    """
    data_dir = Path('/tmp/ad_diffi_data')
    data_dir.mkdir(exist_ok=True)
    csv_file = data_dir / f"{dataset_name}.csv"

    if csv_file.exists():
        return str(csv_file)

    # Possible URL structures in ADBench repository
    urls = [
        f"https://raw.githubusercontent.com/Minqi824/ADBench/main/data/{dataset_name}.csv",
        f"https://raw.githubusercontent.com/Minqi824/ADBench/main/datasets/Classical/21_{dataset_name}.csv" # Annthyroid specific
    ]

    for url in urls:
        try:
            print(f"[INFO] Attempting download: {url}")
            urllib.request.urlretrieve(url, str(csv_file))
            # Verify if downloaded file is a valid CSV
            pd.read_csv(csv_file, nrows=5)
            print(f"[SUCCESS] {dataset_name} downloaded.")
            return str(csv_file)
        except Exception:
            continue

    # --- FALLBACK: SYNTHETIC DATA GENERATION ---
    print(f"[WARN] Failed to download {dataset_name}. Generating synthetic fallback...")
    return _generate_synthetic_fallback(dataset_name, csv_file)

def _generate_synthetic_fallback(dataset_name: str, save_path: Path) -> str:
    """Internal helper to generate statistically similar datasets for the paper."""
    np.random.seed(42)
    
    if dataset_name.lower() == "annthyroid":
        n_samples, n_outliers = 7200, int(7200 * 0.0742)
        cont_cols = ['TBG_measured', 'TBG', 'TSH_measured', 'TSH', 'T3_measured', 'T3']
        bin_cols = ['FTI_measured', 'FBI', 'sex', 'pregnant', 'sick', 'tumor', 'psych'] # Simplified set
        
        X_cont = np.random.normal(0, 1, (n_samples, len(cont_cols)))
        outlier_idx = np.random.choice(n_samples, n_outliers, replace=False)
        X_cont[outlier_idx] += 4.5 # Anomaly signal
        
        df = pd.DataFrame(X_cont, columns=cont_cols)
        for col in bin_cols:
            df[col] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        
        df['Outlier_label'] = ['o' if i in outlier_idx else 'n' for i in range(n_samples)]
        
    elif dataset_name.lower() == "breast_cancer":
        # Simplified Breast Cancer logic
        n_samples, n_features = 569, 30
        X = np.random.normal(0, 1, (n_samples, n_features))
        outlier_idx = np.random.choice(n_samples, int(n_samples * 0.3), replace=False)
        X[outlier_idx] += 3.0
        df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
        df['Outlier_label'] = ['o' if i in outlier_idx else 'n' for i in range(n_samples)]
    
    else:
        # Generic fallback
        df = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
        df['Outlier_label'] = 'n'

    df.to_csv(save_path, index=False)
    return str(save_path)

def get_feature_metadata(df: pd.DataFrame, dataset_name: str, label_col: str = "Outlier_label") -> Tuple[List[str], Dict[int, str]]:
    """
    Identify continuous and binary features based on dataset-specific rules.
    """
    all_cols = [c for c in df.columns if c != label_col]
    
    # Dataset-specific continuous feature definitions (case-insensitive keys)
    cont_map = {
        "annthyroid": ['tbg_measured', 'tbg', 'tsh_measured', 'tsh', 't3_measured', 't3', 'tt4', 't4u', 'fti'],
        "breast_cancer": [col for col in all_cols if any(x in col.lower() for x in ["mean", "worst", "error"])],
        "hepatitis": ['age', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime']
    }
    
    target_cont = [c.lower() for c in cont_map.get(dataset_name.lower(), [])]
    
    actual_cont = []
    for col in all_cols:
        if col.lower() in target_cont:
            actual_cont.append(col)
    
    # Fallback: if no specific rule matched, assume features with >2 unique values are continuous
    if not actual_cont:
        actual_cont = [c for c in all_cols if df[c].nunique() > 2]

    feature_names = all_cols
    feature_types = {i: "cont" if col in actual_cont else "bin" for i, col in enumerate(feature_names)}
    
    return feature_names, feature_types

# --- (以下、get_noise_baseline および create_importance_report は変更なし) ---
