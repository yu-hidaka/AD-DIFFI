# =============================================================================
# AD-DIFFI Chapter 5: Complete Simulation (RSO + Z-Normalization)
# Colab-ready: Copy-paste single cell → Run → Ch5 Figure + Table reproduction
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# --- 1. Core Settings & Features (Chapter 5) ---
MIN_DEPTH = 1
N_ITER = 100          # Signal iterations
N_ITER_NOISE = 100    # Noise baseline iterations
N_TREES = 100
MAX_SAMPLES = 256
CONTAMINATION = 0.05
RANDOM_SEED_BASE = 42

FEATURE_DEFINITIONS = [
    {'name': 'X1_Strong_Cont', 'type': 'cont', 'is_signal': True,  'strength': 'strong'},
    {'name': 'X2_Weak_Cont_A', 'type': 'cont', 'is_signal': True,  'strength': 'weak'},
    {'name': 'X3_Weak_Cont_B', 'type': 'cont', 'is_signal': True,  'strength': 'weak'},
    {'name': 'X4_Strong_Bin',  'type': 'bin',  'is_signal': True,  'strength': 'strong'},
    {'name': 'X5_Weak_Bin',    'type': 'bin',  'is_signal': True,  'strength': 'weak'},
    {'name': 'X6_Cont_Noise',  'type': 'cont', 'is_signal': False, 'strength': 'noise'},
    {'name': 'X7_Bin_Noise_A', 'type': 'bin',  'is_signal': False, 'strength': 'noise'},
    {'name': 'X8_Bin_Noise_B', 'type': 'bin',  'is_signal': False, 'strength': 'noise'},
]

FEATURE_TYPES = {i: f['type'] for i, f in enumerate(FEATURE_DEFINITIONS)}
FEATURE_NAMES = [f['name'] for f in FEATURE_DEFINITIONS]
CONT_INDICES = [i for i, f in enumerate(FEATURE_DEFINITIONS) if f['type'] == 'cont']
BIN_INDICES = [i for i, f in enumerate(FEATURE_DEFINITIONS) if f['type'] == 'bin']

# --- 2. Data Generation Functions ---
def generate_mixed_signal_dataset(n_total=1000, contamination=0.05, seed=42):
    """Generate mixed dataset with signal anomalies in specified features"""
    n_normal = int(n_total * (1 - contamination))
    n_anomaly = n_total - n_normal
    rng = np.random.default_rng(seed)
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
                X_normal = rng.normal(10, 1, n_normal)
                X_sig = rng.normal(loc_sig, scale_sig, n_a_current)
                X_all = np.concatenate([X_normal, X_sig])
            else:  # binary
                p_sig_one = 0.9 if strength == 'strong' else 0.5
                X_normal = np.zeros(n_normal)
                X_sig_ones = np.ones(int(n_a_current * p_sig_one))
                X_sig_zeros = np.zeros(n_a_current - len(X_sig_ones))
                X_all = np.concatenate([X_normal, X_sig_ones, X_sig_zeros])
            
            if len(X_all) < n_total:
                X_filler = (rng.choice([0,1], n_total-len(X_all), [0.5,0.5]) if ftype=='bin' 
                           else rng.uniform(0,20, n_total-len(X_all)))
                X_all = np.concatenate([X_all, X_filler])
                X_dict[name] = X_all[:n_total]
        else:
            X_dict[name] = (rng.uniform(0,20,n_total) if ftype=='cont' 
                           else rng.choice([0,1], n_total, [0.5,0.5]))
    
    X = pd.DataFrame(X_dict).sample(frac=1, random_state=seed).reset_index(drop=True)
    return X

def generate_noise_only_dataset(n_total=1000, seed=42):
    """Generate pure noise dataset (no signal features)"""
    rng = np.random.default_rng(seed)
    X_dict = {}
    for feat in FEATURE_DEFINITIONS:
        name, ftype = feat['name'], feat['type']
        X_dict[name] = (rng.uniform(0,20,n_total) if ftype=='cont' 
                       else rng.choice([0,1], n_total, [0.5,0.5]))
    return pd.DataFrame(X_dict)

# --- 3. Original DIFFI (Raw scores, no RSO adjustments) ---
def diffi_ib_binary_original(iforest, X_data):
    """Original DIFFI implementation without RSO adjustments"""
    num_feat = X_data.shape[1]
    estimators = iforest.estimators_
    cfi_outliers_ib = np.zeros(num_feat)
    cfi_inliers_ib = np.zeros(num_feat)
    counter_outliers_ib = np.zeros(num_feat, int)
    counter_inliers_ib = np.zeros(num_feat, int)
    in_bag_samples = iforest.estimators_samples_
    global_scores = iforest.decision_function(X_data)
    global_threshold = np.percentile(global_scores, 100 * iforest.contamination)

    for k, estimator in enumerate(estimators):
        in_bag_sample = list(in_bag_samples[k])
        X_ib = X_data[in_bag_sample,:]
        scores_ib = global_scores[in_bag_sample]
        X_outliers_ib = X_ib[scores_ib < global_threshold]
        X_inliers_ib = X_ib[scores_ib >= global_threshold]
        if len(X_inliers_ib) == 0 or len(X_outliers_ib) == 0: 
            continue

        tree = estimator.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        n_samples_node = tree.n_node_samples

        # Compute node depths
        node_depth = np.zeros(n_nodes, int)
        stack = [(0, -1)]
        while stack:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
            if children_left[node_id] != children_right[node_id]:
                stack += [(children_left[node_id], parent_depth+1), 
                         (children_right[node_id], parent_depth+1)]

        def calculate_contribution(X_path, cfi, counter):
            node_indicator = estimator.decision_path(X_path).toarray()
            for i in range(len(X_path)):
                path = np.where(node_indicator[i] == 1)[0]
                if len(path) == 0: continue
                depth_leaf = node_depth[path[-1]]
                h_leaf = max(depth_leaf, MIN_DEPTH)
                depth_scaling = 1.0 / h_leaf
                for node in path:
                    feat_idx = feature[node]
                    if feat_idx >= 0:
                        cfi[feat_idx] += depth_scaling
                        counter[feat_idx] += 1

        calculate_contribution(X_outliers_ib, cfi_outliers_ib, counter_outliers_ib)
        calculate_contribution(X_inliers_ib, cfi_inliers_ib, counter_inliers_ib)

    fi_outliers = np.divide(cfi_outliers_ib, counter_outliers_ib, 
                           out=np.zeros_like(cfi_outliers_ib), where=counter_outliers_ib>0)
    fi_inliers = np.divide(cfi_inliers_ib, counter_inliers_ib, 
                          out=np.zeros_like(cfi_inliers_ib), where=counter_inliers_ib>0)
    return np.divide(fi_outliers, fi_inliers, out=np.zeros_like(fi_outliers), where=fi_inliers!=0)

# --- 4. RSO DIFFI (Root-Split-Only with lambda adjustment) ---
def diffi_ib_binary_rso(iforest, X_data, feature_types):
    """RSO DIFFI with lambda adjustment for balanced splits"""
    num_feat = X_data.shape[1]
    estimators = iforest.estimators_
    cfi_outliers_ib = np.zeros(num_feat)
    cfi_inliers_ib = np.zeros(num_feat)
    counter_outliers_ib = np.zeros(num_feat, int)
    counter_inliers_ib = np.zeros(num_feat, int)
    in_bag_samples = iforest.estimators_samples_
    global_scores = iforest.decision_function(X_data)
    global_threshold = np.percentile(global_scores, 100 * iforest.contamination)

    for k, estimator in enumerate(estimators):
        in_bag_sample = list(in_bag_samples[k])
        X_ib = X_data[in_bag_sample,:]
        scores_ib = global_scores[in_bag_sample]
        X_outliers_ib = X_ib[scores_ib < global_threshold]
        X_inliers_ib = X_ib[scores_ib >= global_threshold]
        if len(X_inliers_ib) == 0 or len(X_outliers_ib) == 0: 
            continue

        tree = estimator.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        n_samples_node = tree.n_node_samples

        # Compute node depths
        node_depth = np.zeros(n_nodes, int)
        stack = [(0, -1)]
        while stack:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
            if children_left[node_id] != children_right[node_id]:
                stack += [(children_left[node_id], parent_depth+1), 
                         (children_right[node_id], parent_depth+1)]

        # Lambda adjustment for balanced splits
        current_lambda_adj = np.zeros(n_nodes)
        for node in range(n_nodes):
            if children_left[node] != children_right[node]:
                n_parent = n_samples_node[node]
                n_left = n_samples_node[children_left[node]]
                n_right = n_samples_node[children_right[node]]
                current_lambda_adj[node] = 1.0 - min(n_left, n_right) / n_parent

        def calculate_contribution(X_path, cfi, counter):
            node_indicator = estimator.decision_path(X_path).toarray()
            for i in range(len(X_path)):
                path = np.where(node_indicator[i] == 1)[0]
                if len(path) == 0: continue
                depth_leaf = node_depth[path[-1]]
                h_leaf = max(depth_leaf, MIN_DEPTH)
                depth_scaling = 1.0 / h_leaf
                for node in path:
                    feat_idx = feature[node]
                    if feat_idx >= 0 and (feature_types[feat_idx] != 'bin' or node_depth[node] == 0):
                        lambda_val = current_lambda_adj[node]
                        cfi[feat_idx] += depth_scaling * lambda_val
                        counter[feat_idx] += 1

        calculate_contribution(X_outliers_ib, cfi_outliers_ib, counter_outliers_ib)
        calculate_contribution(X_inliers_ib, cfi_inliers_ib, counter_inliers_ib)

    fi_outliers = np.divide(cfi_outliers_ib, counter_outliers_ib, 
                           out=np.zeros_like(cfi_outliers_ib), where=counter_outliers_ib>0)
    fi_inliers = np.divide(cfi_inliers_ib, counter_inliers_ib, 
                          out=np.zeros_like(cfi_inliers_ib), where=counter_inliers_ib>0)
    return np.divide(fi_outliers, fi_inliers, out=np.zeros_like(fi_outliers), where=fi_inliers!=0)

# --- 5. Noise Baseline Computation ---
def compute_noise_baseline(n_iter_noise, feature_types):
    """Compute noise baseline for Z-normalization (separate for cont/bin)"""
    print(f"\n--- Computing Noise Baseline (RSO DIFFI, {n_iter_noise} iterations) ---")
    raw_scores_list = []
    
    for k in range(n_iter_noise):
        X = generate_noise_only_dataset(seed=RANDOM_SEED_BASE+k).values
        iforest = IsolationForest(
            n_estimators=N_TREES, max_samples=MAX_SAMPLES, 
            contamination=CONTAMINATION, random_state=k, bootstrap=False
        ).fit(X)
        raw_scores_list.append(diffi_ib_binary_rso(iforest, X, feature_types))
    
    fi_matrix = np.vstack(raw_scores_list)
    cont_mean = fi_matrix[:,CONT_INDICES].mean() if CONT_INDICES else 0
    cont_sd = fi_matrix[:,CONT_INDICES].std() if CONT_INDICES else 0
    bin_mean = fi_matrix[:,BIN_INDICES].mean() if BIN_INDICES else 0
    bin_sd = fi_matrix[:,BIN_INDICES].std() if BIN_INDICES else 0
    
    print(f"  Continuous features baseline: mean={cont_mean:.6f}, sd={cont_sd:.6f}")
    print(f"  Binary features baseline:    mean={bin_mean:.6f}, sd={bin_sd:.6f}")
    return cont_mean, cont_sd, bin_mean, bin_sd

# --- 6. Collect Both Methods ---
def collect_orig_raw_and_rso_z(n_iter, feature_types, cont_mean, cont_sd, bin_mean, bin_sd):
    """Collect Original DIFFI (Raw) and AD-DIFFI RSO (Z-normalized) scores"""
    print(f"\n--- Collecting Signal Scores ({n_iter} iterations) ---")
    
    orig_raw_scores = []
    rso_raw_scores = []
    
    for k in range(n_iter):
        # Generate signal dataset
        X = generate_mixed_signal_dataset(seed=RANDOM_SEED_BASE+k).values
        
        # Original DIFFI (Raw scores)
        iforest_orig = IsolationForest(
            n_estimators=N_TREES, max_samples=MAX_SAMPLES,
            contamination=CONTAMINATION, random_state=k, bootstrap=False
        ).fit(X)
        orig_raw_scores.append(diffi_ib_binary_original(iforest_orig, X))
        
        # RSO DIFFI (Raw scores, before Z-normalization)
        iforest_rso = IsolationForest(
            n_estimators=N_TREES, max_samples=MAX_SAMPLES,
            contamination=CONTAMINATION, random_state=k, bootstrap=False
        ).fit(X)
        rso_raw_scores.append(diffi_ib_binary_rso(iforest_rso, X, feature_types))
    
    # Original DIFFI DataFrame
    df_orig_raw = pd.DataFrame(np.vstack(orig_raw_scores), columns=FEATURE_NAMES)
    df_orig_raw['Scenario'] = 'Orig_Raw'
    
    # RSO DIFFI (Z-normalized to AD-DIFFI)
    df_rso_raw = pd.DataFrame(np.vstack(rso_raw_scores), columns=FEATURE_NAMES)
    df_rso_z = df_rso_raw.copy()
    
    for i, name in enumerate(FEATURE_NAMES):
        ftype = FEATURE_TYPES[i]
        if ftype == 'cont':
            df_rso_z.iloc[:,i] = (df_rso_raw.iloc[:,i] - cont_mean) / cont_sd if cont_sd > 0 else 0
        else:
            df_rso_z.iloc[:,i] = (df_rso_raw.iloc[:,i] - bin_mean) / bin_sd if bin_sd > 0 else 0
    
    df_rso_z['Scenario'] = 'RSO_Z'
    
    return pd.concat([df_orig_raw, df_rso_z]).reset_index(drop=True)

# --- 7. Visualization & Summary (Original Chapter 5 Style) ---
def plot_orig_vs_rso_z(combined_df: pd.DataFrame, feature_defs: List[Dict]):
    """Plot Original DIFFI vs AD-DIFFI RSO (Z-normalized) as side-by-side boxplots"""
    
    all_features = [f['name'] for f in feature_defs]
    feature_labels = {
        'X1_Strong_Cont': 'X1 (S Cont.)', 'X2_Weak_Cont_A': 'X2 (W Cont. A)',
        'X3_Weak_Cont_B': 'X3 (W Cont. B)', 'X4_Strong_Bin': 'X4 (S Bin.)',
        'X5_Weak_Bin': 'X5 (W Bin.)', 'X6_Cont_Noise': 'X6 (Cont. Noise)',
        'X7_Bin_Noise_A': 'X7 (Bin. Noise A)', 'X8_Bin_Noise_B': 'X8 (Bin. Noise B)',
    }

    # Melt to long format
    df_plot = combined_df[['Scenario'] + all_features].melt(
        id_vars='Scenario', var_name='Feature', value_name='Score'
    )
    df_plot['Feature_Label'] = df_plot['Feature'].map(feature_labels)

    feature_order = [feature_labels[f['name']] for f in feature_defs]

    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # Left: Original DIFFI (Raw)
    sns.boxplot(
        x='Feature_Label', y='Score',
        data=df_plot[df_plot['Scenario'] == 'Orig_Raw'],
        order=feature_order, color='#1f77b4', width=0.7, 
        showfliers=False, ax=ax1
    )
    ax1.set_ylabel('DIFFI score', fontsize=12)
    ax1.set_xlabel('Feature', fontsize=12)
    ax1.set_title('A: Original DIFFI', fontsize=14, fontweight='normal')
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.grid(axis='y', linestyle=':', alpha=0.4)

    # Right: AD-DIFFI RSO (Z-normalized)
    sns.boxplot(
        x='Feature_Label', y='Score',
        data=df_plot[df_plot['Scenario'] == 'RSO_Z'],
        order=feature_order, color='#ff7f0e', width=0.7, 
        showfliers=False, ax=ax2
    )
    ax2.set_ylabel('AD-DIFFI score', fontsize=12)
    ax2.set_xlabel('Feature', fontsize=12)
    ax2.set_title('B: AD-DIFFI', fontsize=14, fontweight='normal')
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.4)

    # Final adjustments
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, wspace=0.3)
    
    # Save high-resolution figure
    plt.savefig('Original_DIFFI_vs_AD-DIFFI.png', bbox_inches='tight', 
                dpi=300, facecolor='white')
    print("✅ Figure saved: Original_DIFFI_vs_AD-DIFFI.png (300 DPI)")

    plt.show()

    # Summary table (pivot format)
    avg_scores = (
        df_plot.groupby(['Scenario', 'Feature_Label'])['Score']
        .agg(['mean', 'std']).reset_index()
    )
    avg_scores = avg_scores.pivot(
        index='Feature_Label', columns='Scenario', values=['mean', 'std']
    )
    feature_sort_order = [feature_labels[f['name']] for f in FEATURE_DEFINITIONS]
    avg_scores = avg_scores.reindex(feature_sort_order)

    print("\n### Chapter 5 Results Table ###")
    print(avg_scores.round(4).to_markdown())

# =============================================================================
# EXECUTION: Reproduce Chapter 5 Figure & Table exactly
# =============================================================================
if __name__ == '__main__':
    # Step 1: Compute RSO noise baseline
    cont_mean, cont_sd, bin_mean, bin_sd = compute_noise_baseline(
        N_ITER_NOISE, FEATURE_TYPES
    )
    
    # Step 2: Collect Original DIFFI (Raw) & AD-DIFFI RSO (Z-normalized)
    combined_results_df = collect_orig_raw_and_rso_z(
        N_ITER, FEATURE_TYPES, cont_mean, cont_sd, bin_mean, bin_sd
    )
    
    # Step 3: Generate Chapter 5 figure (side-by-side) + summary table
    plot_orig_vs_rso_z(combined_results_df, FEATURE_DEFINITIONS)
    
    print("\n✓ Chapter 5 simulation complete!")
    print("  - High-resolution PNG saved")
    print("  - Summary table printed")
