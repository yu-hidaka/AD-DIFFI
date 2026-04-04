import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict, Optional, Tuple

# Minimum depth for leaf nodes to prevent division by zero
MIN_DEPTH = 1

def diffi_ib_binary_rso(iforest: IsolationForest, X_data: np.ndarray, feature_types: Dict[int, str]) -> np.ndarray:
    """
    Compute Raw DIFFI scores with Root-Split-Only (RSO) constraint.
    
    Args:
        iforest: Trained sklearn IsolationForest model.
        X_data: Input data as a numpy array.
        feature_types: Dictionary mapping feature index to type ('cont' or 'bin').
                      Example: {0: 'cont', 1: 'bin', 2: 'bin'}
                      
    Returns:
        np.ndarray: Raw DIFFI-RSO importance scores for each feature.
    """
    num_feat = X_data.shape[1]
    estimators = iforest.estimators_
    in_bag_samples = iforest.estimators_samples_
    
    cfi_outliers_ib = np.zeros(num_feat, dtype=float)
    cfi_inliers_ib = np.zeros(num_feat, dtype=float)
    counter_outliers_ib = np.zeros(num_feat, dtype=int)
    counter_inliers_ib = np.zeros(num_feat, dtype=int)

    # Calculate anomaly scores and threshold
    global_scores = iforest.decision_function(X_data)
    global_threshold = np.percentile(global_scores, 100 * iforest.contamination)

    for k, estimator in enumerate(estimators):
        in_bag_sample = list(in_bag_samples[k])
        X_ib = X_data[in_bag_sample, :]
        scores_ib = global_scores[in_bag_sample]

        X_outliers_ib = X_ib[scores_ib < global_threshold]
        X_inliers_ib = X_ib[scores_ib >= global_threshold]
        
        if X_inliers_ib.shape[0] == 0 or X_outliers_ib.shape[0] == 0:
            continue

        tree = estimator.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        n_samples_node = tree.n_node_samples

        # 1. Compute node depths
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        stack = [(0, -1)]
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
            if children_left[node_id] != children_right[node_id]:
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))

        # 2. Compute Lambda adjustment (imbalance factor)
        current_lambda_adj = np.zeros(shape=n_nodes, dtype=float)
        for node in range(n_nodes):
            if children_left[node] != children_right[node]:
                n_parent = n_samples_node[node]
                n_left = n_samples_node[children_left[node]]
                n_right = n_samples_node[children_right[node]]
                ratio_small = np.min([n_left, n_right]) / n_parent
                current_lambda_adj[node] = 1.0 - ratio_small

        # 3. Path-based contribution calculation
        def calculate_contribution(X_path, cfi, counter):
            if X_path.shape[0] == 0:
                return
            node_indicator = estimator.decision_path(X_path).toarray()
            
            for i in range(len(X_path)):
                path = np.where(node_indicator[i] == 1)[0]
                if len(path) == 0:
                    continue

                h_leaf = max(node_depth[path[-1]], MIN_DEPTH)
                depth_scaling = 1.0 / h_leaf

                for node in path:
                    idx = feature[node]
                    if idx >= 0:
                        # Apply RSO Constraint: Binary features only contribute at root (depth 0)
                        f_type = feature_types.get(idx, 'cont')
                        if f_type == 'bin' and node_depth[node] > 0:
                            continue
                        
                        cfi[idx] += depth_scaling * current_lambda_adj[node]
                        counter[idx] += 1

        calculate_contribution(X_outliers_ib, cfi_outliers_ib, counter_outliers_ib)
        calculate_contribution(X_inliers_ib, cfi_inliers_ib, counter_inliers_ib)

    # Final raw score calculation
    fi_outliers = np.where(counter_outliers_ib > 0, cfi_outliers_ib / counter_outliers_ib, 0)
    fi_inliers = np.where(counter_inliers_ib > 0, cfi_inliers_ib / counter_inliers_ib, 0)
    
    return np.divide(fi_outliers, fi_inliers, out=np.zeros_like(fi_outliers), where=fi_inliers != 0)


def calculate_ad_diffi_zscore(
    raw_scores: np.ndarray, 
    feature_types: Dict[int, str], 
    noise_baselines: Dict[str, Dict[str, float]]
) -> np.ndarray:
    """
    Apply type-specific Z-score normalization (AD-DIFFI).
    
    Args:
        raw_scores: Raw RSO-DIFFI scores.
        feature_types: Dictionary mapping feature index to type.
        noise_baselines: Dictionary with mean/sd for 'cont' and 'bin'.
                        Example: {'cont': {'mean': 0.5, 'sd': 0.1}, 'bin': {...}}
    
    Returns:
        np.ndarray: Normalized AD-DIFFI scores.
    """
    z_scores = np.zeros_like(raw_scores)
    for i in range(len(raw_scores)):
        ftype = feature_types.get(i, 'cont')
        stats = noise_baselines.get(ftype, {'mean': 0.0, 'sd': 1.0})
        
        if stats['sd'] > 0:
            z_scores[i] = (raw_scores[i] - stats['mean']) / stats['sd']
        else:
            z_scores[i] = 0.0
            
    return z_scores

def get_noise_baselines(X_dim: int, feature_types: Dict[int, str], if_params: Dict, n_iter: int = 20) -> Tuple[float, float, float, float]:
    """
    Establish noise baselines for Z-score normalization.
    Returns: (cont_mean, cont_sd, bin_mean, bin_sd)
    """
    noise_scores = []
    cont_idx = [i for i, t in feature_types.items() if t == 'cont']
    bin_idx = [i for i, t in feature_types.items() if t == 'bin']

    for i in range(n_iter):
        X_noise = np.random.uniform(0, 1, (if_params.get('max_samples', 512), X_dim))
        p = if_params.copy()
        p.pop('random_state', None)
        m_noise = IsolationForest(**p, random_state=i).fit(X_noise)
        
        # Calculate raw RSO scores for this noise instance
        scores = diffi_ib_binary_rso(m_noise, X_noise, feature_types)
        noise_scores.append(scores)

    M_noise = np.vstack(noise_scores)
    
    c_mean, c_std = (M_noise[:, cont_idx].mean(), M_noise[:, cont_idx].std()) if cont_idx else (0.0, 1.0)
    b_mean, b_std = (M_noise[:, bin_idx].mean(), M_noise[:, bin_idx].std()) if bin_idx else (0.0, 1.0)
    
    return c_mean, (c_std if c_std > 0 else 1.0), b_mean, (b_std if b_std > 0 else 1.0)
