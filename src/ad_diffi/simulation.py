import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Tuple, Dict, List
from .core import diffi_ib_binary_rso, calculate_ad_diffi_zscore

# --- Chapter 5: Simulation Settings ---
N_TREES = 100
MAX_SAMPLES = 256
CONTAMINATION = 0.05
RANDOM_SEED_BASE = 42

# Feature Definitions for Chapter 5 (8 mixed features)
FEATURE_DEFINITIONS = [
    {'name': 'X1_Strong_Cont', 'type': 'cont', 'is_signal': True, 'strength': 'strong'},
    {'name': 'X2_Weak_Cont_A', 'type': 'cont', 'is_signal': True, 'strength': 'weak'},
    {'name': 'X3_Weak_Cont_B', 'type': 'cont', 'is_signal': True, 'strength': 'weak'},
    {'name': 'X4_Strong_Bin', 'type': 'bin', 'is_signal': True, 'strength': 'strong'},
    {'name': 'X5_Weak_Bin', 'type': 'bin', 'is_signal': True, 'strength': 'weak'},
    {'name': 'X6_Cont_Noise', 'type': 'cont', 'is_signal': False, 'strength': 'noise'},
    {'name': 'X7_Bin_Noise_A', 'type': 'bin', 'is_signal': False, 'strength': 'noise'},
    {'name': 'X8_Bin_Noise_B', 'type': 'bin', 'is_signal': False, 'strength': 'noise'},
]

FEATURE_NAMES = [f['name'] for f in FEATURE_DEFINITIONS]
FEATURE_TYPES = {i: f['type'] for i, f in enumerate(FEATURE_DEFINITIONS)}
CONT_INDICES = [i for i, f in enumerate(FEATURE_DEFINITIONS) if f['type'] == 'cont']
BIN_INDICES = [i for i, f in enumerate(FEATURE_DEFINITIONS) if f['type'] == 'bin']

def generate_mixed_signal_dataset(n_total: int = 1000, contamination: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """Generate mixed signal + noise dataset for Chapter 5 simulation."""
    n_normal = int(n_total * (1 - contamination))
    n_anomaly = n_total - n_normal
    rng = np.random.default_rng(seed=seed)
    
    signal_features = [f for f in FEATURE_DEFINITIONS if f['is_signal']]
    n_signals = len(signal_features)
    n_a_per_signal = n_anomaly // n_signals if n_signals > 0 else 0
    
    X_dict = {}
    for i, feat in enumerate(FEATURE_DEFINITIONS):
        name, ftype, is_sig, strength = feat['name'], feat['type'], feat['is_signal'], feat['strength']
        if is_sig:
            n_a_current = n_a_per_signal + (1 if i < (n_anomaly % n_signals) else 0)
            if ftype == 'cont':
                loc_sig = 50 if strength == 'strong' else 15
                scale_sig = 0.5 if strength == 'strong' else 1
                X_normal = rng.normal(loc=10, scale=1, size=n_normal)
                X_sig = rng.normal(loc=loc_sig, scale=scale_sig, size=n_a_current)
                X_all = np.concatenate([X_normal, X_sig])
            elif ftype == 'bin':
                p_sig_one = 0.9 if strength == 'strong' else 0.5
                X_normal = np.zeros(n_normal)
                X_sig_ones = np.ones(int(n_a_current * p_sig_one))
                X_sig_zeros = np.zeros(n_a_current - len(X_sig_ones))
                X_all = np.concatenate([X_normal, X_sig_ones, X_sig_zeros])
            X_dict[name] = X_all[:n_total]
        else:
            if ftype == 'cont':
                X_dict[name] = rng.uniform(low=0, high=20, size=n_total)
            elif ftype == 'bin':
                X_dict[name] = rng.choice([0, 1], size=n_total, p=[0.5, 0.5])
                
    return pd.DataFrame(X_dict).sample(frac=1, random_state=seed).reset_index(drop=True)

def get_noise_baselines(n_iter: int = 100) -> Dict[str, Dict[str, float]]:
    """Establish noise baselines for standardization."""
    raw_scores_list = []
    rng = np.random.default_rng(seed=RANDOM_SEED_BASE)
    
    for k in range(n_iter):
        # Generate pure noise data
        X_noise = {}
        for feat in FEATURE_DEFINITIONS:
            if feat['type'] == 'cont':
                X_noise[feat['name']] = rng.uniform(0, 20, 1000)
            else:
                X_noise[feat['name']] = rng.choice([0, 1], 1000)
        X_data = pd.DataFrame(X_noise).values
        
        model = IsolationForest(n_estimators=N_TREES, random_state=k).fit(X_data)
        raw_scores = diffi_ib_binary_rso(model, X_data, FEATURE_TYPES)
        raw_scores_list.append(raw_scores)
        
    matrix = np.vstack(raw_scores_list)
    return {
        'cont': {'mean': matrix[:, CONT_INDICES].mean(), 'sd': matrix[:, CONT_INDICES].std()},
        'bin': {'mean': matrix[:, BIN_INDICES].mean(), 'sd': matrix[:, BIN_INDICES].std()}
    }

def run_chapter_5_simulation(n_iter: int = 100):
    """Run the full Chapter 5 simulation and return aggregated results."""
    baselines = get_noise_baselines(n_iter=100)
    all_results = []

    for k in range(n_iter):
        df = generate_mixed_signal_dataset(seed=RANDOM_SEED_BASE + k)
        X_data = df.values
        model = IsolationForest(n_estimators=N_TREES, random_state=k).fit(X_data)
        
        # 1. Raw DIFFI-RSO
        raw = diffi_ib_binary_rso(model, X_data, FEATURE_TYPES)
        
        # 2. AD-DIFFI (Z-score)
        ad_diffi = calculate_ad_diffi_zscore(raw, FEATURE_TYPES, baselines)
        
        all_results.append(ad_diffi)
        
    return pd.DataFrame(np.vstack(all_results), columns=FEATURE_NAMES)
