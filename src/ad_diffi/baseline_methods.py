import numpy as np
from math import ceil
from sklearn.ensemble import IsolationForest

def _get_iic(estimator, predictions, is_leaves, adjust_iic=True):
    """
    Internal function to compute the Imbalance Isolation Contribution (IIC).
    This is a core component of the original DIFFI algorithm.
    """
    desired_min = 0.5
    desired_max = 1.0
    epsilon = 0.0
    n_nodes = estimator.tree_.node_count
    lambda_ = np.zeros(n_nodes)
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right

    if predictions.shape[0] == 0:
        return lambda_

    node_indicator_all_samples = estimator.decision_path(predictions).toarray()
    num_samples_in_node = np.sum(node_indicator_all_samples, axis=0)

    for node in range(n_nodes):
        num_samples_in_current_node = num_samples_in_node[node]
        num_samples_in_left_children = num_samples_in_node[children_left[node]]
        num_samples_in_right_children = num_samples_in_node[children_right[node]]

        if num_samples_in_current_node <= 1 or is_leaves[node]:
            lambda_[node] = -1
        elif num_samples_in_left_children == 0 or num_samples_in_right_children == 0:
            lambda_[node] = epsilon
        else:
            current_min = 0.5 if num_samples_in_current_node % 2 == 0 else ceil(num_samples_in_current_node / 2) / num_samples_in_current_node
            current_max = (num_samples_in_current_node - 1) / num_samples_in_current_node
            tmp = np.max([num_samples_in_left_children, num_samples_in_right_children]) / num_samples_in_current_node

            if adjust_iic and current_min != current_max:
                lambda_[node] = ((tmp - current_min) / (current_max - current_min)) * (desired_max - desired_min) + desired_min
            else:
                lambda_[node] = tmp
    return lambda_

def original_diffi_importance(iforest: IsolationForest, X: np.ndarray, adjust_iic: True) -> tuple:
    """
    Compute the original DIFFI feature importance.
    
    Returns:
        fi_ib: Global feature importance scores.
        cfi_out: Cumulative importance for outliers (Numerator).
        cfi_in: Cumulative importance for inliers (Denominator).
    """
    num_feat = X.shape[1]
    estimators = iforest.estimators_
    in_bag_samples = iforest.estimators_samples_
    
    cfi_outliers_ib = np.zeros(num_feat, dtype=float)
    cfi_inliers_ib = np.zeros(num_feat, dtype=float)
    counter_outliers_ib = np.zeros(num_feat, dtype=int)
    counter_inliers_ib = np.zeros(num_feat, dtype=int)

    # Calculate anomaly scores to separate outliers/inliers
    as_full = iforest.decision_function(X)

    for k, estimator in enumerate(estimators):
        in_bag_sample = list(in_bag_samples[k])
        X_ib = X[in_bag_sample, :]
        as_ib = as_full[in_bag_sample]

        # In original DIFFI, scores < 0 are outliers
        X_outliers_ib = X_ib[as_ib < 0]
        X_inliers_ib = X_ib[as_ib > 0]

        if X_inliers_ib.shape[0] == 0 or X_outliers_ib.shape[0] == 0:
            continue

        tree = estimator.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        
        # Compute node depths and identify leaves
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
            if children_left[node_id] != children_right[node_id]:
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True

        def accumulate_cfi(X_sub, cfi_array, counter_array):
            lambdas = _get_iic(estimator, X_sub, is_leaves, adjust_iic)
            node_indicator = estimator.decision_path(X_sub).toarray()
            
            for i in range(len(X_sub)):
                path = np.where(node_indicator[i] == 1)[0]
                if len(path) == 0: continue
                depth_leaf = node_depth[path[-1]]
                
                for node in path:
                    feat_idx = feature[node]
                    l_val = lambdas[node]
                    if feat_idx >= 0 and l_val != -1:
                        cfi_array[feat_idx] += (1.0 / depth_leaf) * l_val
                        counter_array[feat_idx] += 1

        accumulate_cfi(X_outliers_ib, cfi_outliers_ib, counter_outliers_ib)
        accumulate_cfi(X_inliers_ib, cfi_inliers_ib, counter_inliers_ib)

    # Final Importance Calculation (Ratio of Outlier/Inlier CFI)
    fi_out = np.where(counter_outliers_ib > 0, cfi_outliers_ib / counter_outliers_ib, 0)
    fi_in = np.where(counter_inliers_ib > 0, cfi_inliers_ib / counter_inliers_ib, 0)
    fi_ib = np.divide(fi_out, fi_in, out=np.zeros_like(fi_out), where=fi_in != 0)

    return fi_ib, cfi_outliers_ib, cfi_inliers_ib
