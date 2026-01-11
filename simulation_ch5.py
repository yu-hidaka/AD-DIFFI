# =============================================================================
# AD-DIFFI Chapter 5: Colab-Ready Complete Simulation (RSO + Z-Normalization)
# Copy-paste single cell → Run → Ch5 Figure + Table instant reproduction
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# --- 1. Core Settings & Features (Ch5) ---
MIN_DEPTH = 1; N_ITER = 100; N_ITER_NOISE = 100; N_TREES = 100
MAX_SAMPLES = 256; CONTAMINATION = 0.05; RANDOM_SEED_BASE = 42

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

FEATURE_TYPES = {i: f['type'] for i, f in enumerate(FEATURE_DEFINITIONS)}
FEATURE_NAMES = [f['name'] for f in FEATURE_DEFINITIONS]
CONT_INDICES = [i for i, f in enumerate(FEATURE_DEFINITIONS) if f['type'] == 'cont']
BIN_INDICES = [i for i, f in enumerate(FEATURE_DEFINITIONS) if f['type'] == 'bin']

FEATURE_LABELS = {
    'X1_Strong_Cont': 'X1 (S Cont.)', 'X2_Weak_Cont_A': 'X2 (W Cont. A)',
    'X3_Weak_Cont_B': 'X3 (W Cont. B)', 'X4_Strong_Bin': 'X4 (S Bin.)',
    'X5_Weak_Bin': 'X5 (W Bin.)', 'X6_Cont_Noise': 'X6 (Cont. Noise)',
    'X7_Bin_Noise_A': 'X7 (Bin. Noise A)', 'X8_Bin_Noise_B': 'X8 (Bin. Noise B)',
}

# --- 2. Data Generation (Signal + Noise) ---
def generate_mixed_signal_dataset_general(n_total=1000, contamination=0.05, seed=42):
    n_normal = int(n_total * (1 - contamination)); n_anomaly = n_total - n_normal
    rng = np.random.default_rng(seed); signal_features = [f for f in FEATURE_DEFINITIONS if f['is_signal']]
    n_signals = len(signal_features); n_a_per_signal = n_anomaly // n_signals if n_signals > 0 else 0
    X_dict = {}
    for i, feat in enumerate(FEATURE_DEFINITIONS):
        name, ftype, is_sig, strength = feat['name'], feat['type'], feat['is_signal'], feat['strength']
        if is_sig:
            n_a_current = n_a_per_signal + (1 if i < (n_anomaly % n_signals) else 0)
            if ftype == 'cont':
                loc_sig = 50 if strength == 'strong' else 15; scale_sig = 0.5 if strength == 'strong' else 1
                X_normal = rng.normal(10, 1, n_normal); X_sig = rng.normal(loc_sig, scale_sig, n_a_current)
                X_all = np.concatenate([X_normal, X_sig])
            else:  # bin
                p_sig_one = 0.9 if strength == 'strong' else 0.5
                X_normal = np.zeros(n_normal); X_sig_ones = np.ones(int(n_a_current * p_sig_one))
                X_sig_zeros = np.zeros(n_a_current - len(X_sig_ones)); X_all = np.concatenate([X_normal, X_sig_ones, X_sig_zeros])
            if len(X_all) < n_total:
                X_filler = rng.choice([0,1], n_total-len(X_all), [0.5,0.5]) if ftype=='bin' else rng.uniform(0,20,n_total-len(X_all))
                X_all = np.concatenate([X_all, X_filler]); X_dict[name] = X_all[:n_total]
        else:
            X_dict[name] = rng.uniform(0,20,n_total) if ftype=='cont' else rng.choice([0,1], n_total, [0.5,0.5])
    X = pd.DataFrame(X_dict).sample(frac=1, random_state=seed).reset_index(drop=True)
    return X

def generate_noise_only_dataset_general(n_total=1000, seed=42):
    rng = np.random.default_rng(seed); X_dict = {}
    for feat in FEATURE_DEFINITIONS:
        name, ftype = feat['name'], feat['type']
        X_dict[name] = rng.uniform(0,20,n_total) if ftype=='cont' else rng.choice([0,1], n_total, [0.5,0.5])
    return pd.DataFrame(X_dict)

# --- 3. RSO DIFFI Core ---
def diffi_ib_binary_rso(iforest, X_data, feature_types):
    num_feat = X_data.shape[1]; estimators = iforest.estimators_
    cfi_outliers_ib = np.zeros(num_feat); cfi_inliers_ib = np.zeros(num_feat)
    counter_outliers_ib = np.zeros(num_feat, int); counter_inliers_ib = np.zeros(num_feat, int)
    in_bag_samples = iforest.estimators_samples_
    global_scores = iforest.decision_function(X_data)
    global_threshold = np.percentile(global_scores, 100 * iforest.contamination)

    for k, estimator in enumerate(estimators):
        in_bag_sample = list(in_bag_samples[k]); X_ib = X_data[in_bag_sample,:]
        scores_ib = global_scores[in_bag_sample]; X_outliers_ib = X_ib[scores_ib < global_threshold]
        X_inliers_ib = X_ib[scores_ib >= global_threshold]
        if len(X_inliers_ib) == 0 or len(X_outliers_ib) == 0: continue

        tree = estimator.tree_; n_nodes = tree.node_count
        children_left, children_right, feature, n_samples_node = tree.children_left, tree.children_right, tree.feature, tree.n_node_samples

        node_depth = np.zeros(n_nodes, int); stack = [(0, -1)]
        while stack:
            node_id, parent_depth = stack.pop(); node_depth[node_id] = parent_depth + 1
            if children_left[node_id] != children_right[node_id]:
                stack += [(children_left[node_id], parent_depth+1), (children_right[node_id], parent_depth+1)]

        current_lambda_adj = np.zeros(n_nodes)
        for node in range(n_nodes):
            if children_left[node] != children_right[node]:
                n_parent, n_left, n_right = n_samples_node[node], n_samples_node[children_left[node]], n_samples_node[children_right[node]]
                current_lambda_adj[node] = 1.0 - min(n_left, n_right) / n_parent

        def calculate_contribution(X_path, cfi, counter):
            node_indicator = estimator.decision_path(X_path).toarray()
            for i in range(len(X_path)):
                path = np.where(node_indicator[i] == 1)[0]
                if len(path) == 0: continue
                depth_leaf = node_depth[path[-1]]; h_leaf = max(depth_leaf, MIN_DEPTH)
                depth_scaling = 1.0 / h_leaf
                for node in path:
                    feat_idx = feature[node]
                    if feat_idx >= 0 and (feature_types[feat_idx] != 'bin' or node_depth[node] == 0):
                        lambda_val = current_lambda_adj[node]
                        cfi[feat_idx] += depth_scaling * lambda_val; counter[feat_idx] += 1

        calculate_contribution(X_outliers_ib, cfi_outliers_ib, counter_outliers_ib)
        calculate_contribution(X_inliers_ib, cfi_inliers_ib, counter_inliers_ib)

    fi_outliers = np.divide(cfi_outliers_ib, counter_outliers_ib, out=np.zeros_like(cfi_outliers_ib), where=counter_outliers_ib>0)
    fi_inliers = np.divide(cfi_inliers_ib, counter_inliers_ib, out=np.zeros_like(cfi_inliers_ib), where=counter_inliers_ib>0)
    return np.divide(fi_outliers, fi_inliers, out=np.zeros_like(fi_outliers), where=fi_inliers!=0)

# --- 4. Noise Baseline ---
def run_noise_only_rso_simulation(n_iter_noise, feature_types):
    print(f"\n--- Noise Baseline (RSO DIFFI, {n_iter_noise} iters) ---")
    raw_scores_list = []
    for k in range(n_iter_noise):
        X = generate_noise_only_dataset_general(seed=RANDOM_SEED_BASE+k).values
        iforest = IsolationForest(n_estimators=N_TREES, max_samples=MAX_SAMPLES, contamination=CONTAMINATION, random_state=k, bootstrap=False).fit(X)
        raw_scores_list.append(diffi_ib_binary_rso(iforest, X, feature_types))
    
    fi_matrix = np.vstack(raw_scores_list)
    cont_mean = fi_matrix[:,CONT_INDICES].mean() if CONT_INDICES else 0
    cont_sd = fi_matrix[:,CONT_INDICES].std() if CONT_INDICES else 0
    bin_mean = fi_matrix[:,BIN_INDICES].mean() if BIN_INDICES else 0
    bin_sd = fi_matrix[:,BIN_INDICES].std() if BIN_INDICES else 0
    
    print(f"  Cont baseline: mean={cont_mean:.6f}, sd={cont_sd:.6f}")
    print(f"  Bin baseline:  mean={bin_mean:.6f}, sd={bin_sd:.6f}")
    return cont_mean, cont_sd, bin_mean, bin_sd

# --- 5. Signal Scores + AD-DIFFI ---
def diffi_score_collection_and_standardize_rso(n_iter, feature_types, cont_mean, cont_sd, bin_mean, bin_sd):
    print(f"\n--- Signal Scores (RSO DIFFI, {n_iter} iters) ---")
    raw_scores_list = []
    for k in range(n_iter):
        X = generate_mixed_signal_dataset_general(seed=RANDOM_SEED_BASE+k).values
        iforest = IsolationForest(n_estimators=N_TREES, max_samples=MAX_SAMPLES, contamination=CONTAMINATION, random_state=k, bootstrap=False).fit(X)
        raw_scores_list.append(diffi_ib_binary_rso(iforest, X, feature_types))
    
    fi_matrix = np.vstack(raw_scores_list); df_raw = pd.DataFrame(fi_matrix, columns=FEATURE_NAMES)
    
    # Raw scores
    df_raw['Scenario'] = '1_Raw (DIFFI RSO)'
    
    # Z-standardized AD-DIFFI
    df_z = df_raw.copy()
    for i, name in enumerate(FEATURE_NAMES):
        ftype = FEATURE_TYPES[i]
        if ftype == 'cont': df_z.iloc[:,i] = (df_raw.iloc[:,i] - cont_mean) / cont_sd if cont_sd > 0 else 0
        else: df_z.iloc[:,i] = (df_raw.iloc[:,i] - bin_mean) / bin_sd if bin_sd > 0 else 0
    df_z['Scenario'] = '2_Z (AD-DIFFI RSO)'
    
    return pd.concat([df_raw, df_z]).reset_index(drop=True)

# --- 6. Plot & Summary ---
def plot_and_summarize():
    # Baselines
    cont_mean, cont_sd, bin_mean, bin_sd = run_noise_only_rso_simulation(N_ITER_NOISE, FEATURE_TYPES)
    
    # Scores
    df_results = diffi_score_collection_and_standardize_rso(N_ITER, FEATURE_TYPES, cont_mean, cont_sd, bin_mean, bin_sd)
    
    # Melt for plotting
    df_plot = df_results.melt(id_vars='Scenario', value_name='Score', var_name='Feature')
    df_plot['Label'] = df_plot['Feature'].map(FEATURE_LABELS)
    
    # Boxplot
    fig, ax = plt.subplots(1,1,figsize=(16,8))
    sns.boxplot(data=df_plot, x='Label', y='Score', hue='Scenario', ax=ax, 
                palette=['#1f77b4','#ff7f0e'], width=0.7)
    ax.axhline(1.0, color='r', ls='--', lw=1.5, label='Raw=1.0')
    ax.axhline(0.0, color='k', ls=':', lw=1.5, label='Z=0.0')
    ax.set_title(f'Ch5: RSO DIFFI vs AD-DIFFI (N={N_ITER})'); ax.tick_params(axis='x', rotation=20)
    plt.legend(); plt.tight_layout(); plt.show()
    
    # Save
    plt.savefig('ch5_ad_diffi.png', dpi=300, bbox_inches='tight')
    
    # Summary table
    summary = df_plot.groupby(['Label','Scenario'])['Score'].agg(['mean','std']).round(4).reset_index()
    print("\n### Ch5 Results Table ###")
    print(summary.to_markdown(index=False))
    
    print("\n✓ Ch5 simulation complete! PNG saved.")

# === RUN ===
plot_and_summarize()
