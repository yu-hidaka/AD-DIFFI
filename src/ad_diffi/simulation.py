import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from .core import diffi_ib_binary_rso, calculate_ad_diffi_zscore
from .baseline_methods import original_diffi_importance

# --- Simulation 2 & 6.1 Settings ---
N_TREES = 100
MAX_SAMPLES = 256
CONTAMINATION = 0.05
N_ITER = 100

def generate_mixed_signal_dataset(n_total=1000, contamination=0.05, seed=42):
    """
    Simulation 2 / 6.1: 
    Generates a dataset with 8 features (Continuous/Binary x Strong/Weak/Noise).
    """
    rng = np.random.default_rng(seed=seed)
    n_normal = int(n_total * (1 - contamination))
    n_anomaly = n_total - n_normal
    
    # Feature definitions (Matches Table in Section 3.5.3/6.1)
    data = {}
    
    # 1. Continuous Signal (Strong)
    data['X1_Strong_Cont'] = np.concatenate([rng.normal(10, 1, n_normal), rng.normal(50, 0.5, n_anomaly)])
    # 2. Continuous Signal (Weak)
    data['X2_Weak_Cont'] = np.concatenate([rng.normal(10, 1, n_normal), rng.normal(15, 1, n_anomaly)])
    # 3. Binary Signal (Strong)
    data['X3_Strong_Bin'] = np.concatenate([np.zeros(n_normal), rng.choice([0, 1], n_anomaly, p=[0.1, 0.9])])
    # 4. Binary Signal (Weak)
    data['X4_Weak_Bin'] = np.concatenate([np.zeros(n_normal), rng.choice([0, 1], n_anomaly, p=[0.5, 0.5])])
    # 5. Continuous Noise
    data['X5_Noise_Cont'] = rng.uniform(0, 20, n_total)
    # 6. Binary Noise
    data['X6_Noise_Bin'] = rng.choice([0, 1], n_total, p=[0.5, 0.5])
    
    df = pd.DataFrame(data)
    # Add labels for evaluation if needed
    df['y'] = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

def run_comparative_simulation(n_iter=N_ITER):
    """
    Comprehensive Simulation (Section 6.1):
    Compare Original DIFFI vs. AD-DIFFI (RSO + Z-score).
    """
    # 1. Pre-calculate Noise Baseline for AD-DIFFI
    # (Simplified baseline for example, in real use run on pure noise data)
    noise_baselines = {
        'cont': {'mean': 0.15, 'sd': 0.05}, 
        'bin': {'mean': 0.08, 'sd': 0.02}
    }
    
    feature_types = {0: 'cont', 1: 'cont', 2: 'bin', 3: 'bin', 4: 'cont', 5: 'bin'}
    results_list = []

    for i in range(n_iter):
        df = generate_mixed_signal_dataset(seed=42+i)
        X = df.drop(columns=['y']).values
        
        model = IsolationForest(n_estimators=N_TREES, contamination=CONTAMINATION, random_state=i)
        model.fit(X)
        
        # --- Original DIFFI ---
        fi_orig, _, _ = original_diffi_importance(model, X, adjust_iic=True)
        
        # --- AD-DIFFI (RSO + Z-score) ---
        raw_rso = diffi_ib_binary_rso(model, X, feature_types)
        ad_diffi = calculate_ad_diffi_zscore(raw_rso, feature_types, noise_baselines)
        
        # Store results
        res_iter = []
        for feat_idx in range(X.shape[1]):
            res_iter.append({
                'iteration': i,
                'feature': df.columns[feat_idx],
                'original_diffi': fi_orig[feat_idx],
                'ad_diffi': ad_diffi[feat_idx]
            })
        results_list.extend(res_iter)
        
    return pd.DataFrame(results_list)
