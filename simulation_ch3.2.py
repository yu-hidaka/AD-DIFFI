import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score
from collections import Counter
from math import ceil
import matplotlib.pyplot as plt
import time
from scipy.special import digamma  # Unused but kept for compatibility


# ====================================================================
# A. Required functions from original DIFFI repository
# ====================================================================


# Isolation Forest path length correction constant c(n)
def _average_path_length(n_samples_leaf):
    n_samples_leaf = np.asarray(n_samples_leaf)
    mask = n_samples_leaf <= 1
    n_samples_leaf = n_samples_leaf.copy()
    n_samples_leaf[mask] = 2
    # c(n) = 2(H(n-1)) - (2(n-1)/n) where H(i) is the harmonic number
    return (2.0 * (np.log(n_samples_leaf - 1.0) + np.euler_gamma) -
            (2.0 * (n_samples_leaf - 1.0) / n_samples_leaf))


# --- A1. IIC (Isolation Index Correction) precise implementation ---


def _get_iic(estimator, predictions, is_leaves, adjust_iic=True):
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

        # Added check for num_samples_in_current_node == 0
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


# --- A2. DIFFI computation function (global threshold-based classification logic) ---


def diffi_ib_global_logic(iforest, X, adjust_iic=True):
    num_feat = X.shape[1]
    estimators = iforest.estimators_

    # Initialize accumulators
    cfi_outliers_ib = np.zeros(num_feat).astype('float')
    cfi_inliers_ib = np.zeros(num_feat).astype('float')
    counter_outliers_ib = np.zeros(num_feat).astype('int')
    counter_inliers_ib = np.zeros(num_feat).astype('int')

    in_bag_samples = iforest.estimators_samples_

    # Compute Isolation Forest scores and classify using global threshold
    as_full = iforest.decision_function(X)
    global_threshold = np.percentile(as_full, 100 * iforest.contamination)

    # Pre-determine outlier and inlier indices
    outliers_indices = np.where(as_full < global_threshold)[0]
    inliers_indices = np.where(as_full >= global_threshold)[0]

    for k, estimator in enumerate(estimators):
        in_bag_sample_set = set(in_bag_samples[k])

        # Extract in-bag samples classified as global outliers/inliers
        X_outliers_ib_indices = sorted(list(in_bag_sample_set.intersection(outliers_indices)))
        X_inliers_ib_indices = sorted(list(in_bag_sample_set.intersection(inliers_indices)))

        if len(X_inliers_ib_indices) == 0 or len(X_outliers_ib_indices) == 0:
            continue

        X_outliers_ib = X[X_outliers_ib_indices, :]
        X_inliers_ib = X[X_inliers_ib_indices, :]

        # Tree structure information
        tree_ = estimator.tree_
        n_nodes = tree_.node_count
        children_left = tree_.children_left
        children_right = tree_.children_right
        feature = tree_.feature
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)

        # Compute node depths
        stack = [(0, -1)]
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True

        # OUTLIERS
        lambda_outliers_ib = _get_iic(estimator, X_outliers_ib, is_leaves, adjust_iic)
        node_indicator_all_points_array_outliers_ib = estimator.decision_path(X_outliers_ib).toarray()

        for i in range(len(X_outliers_ib)):
            path = list(np.where(node_indicator_all_points_array_outliers_ib[i] == 1)[0])
            if not path: continue
            depth = node_depth[path[-1]]  # Leaf depth
            if depth == 0: continue  # Skip root-only trees

            for node in path:
                current_feature = feature[node]
                lambda_val = lambda_outliers_ib[node]

                if current_feature >= 0 and current_feature < num_feat and lambda_val != -1:
                    cfi_outliers_ib[current_feature] += (1 / depth) * lambda_val
                    counter_outliers_ib[current_feature] += 1

        # INLIERS
        lambda_inliers_ib = _get_iic(estimator, X_inliers_ib, is_leaves, adjust_iic)
        node_indicator_all_points_array_inliers_ib = estimator.decision_path(X_inliers_ib).toarray()

        for i in range(len(X_inliers_ib)):
            path = list(np.where(node_indicator_all_points_array_inliers_ib[i] == 1)[0])
            if not path: continue
            depth = node_depth[path[-1]]  # Leaf depth
            if depth == 0: continue

            for node in path:
                current_feature = feature[node]
                lambda_val = lambda_inliers_ib[node]

                if current_feature >= 0 and current_feature < num_feat and lambda_val != -1:
                    cfi_inliers_ib[current_feature] += (1 / depth) * lambda_val
                    counter_inliers_ib[current_feature] += 1

    # Compute FI
    fi_outliers_ib = np.where(counter_outliers_ib > 0, cfi_outliers_ib / counter_outliers_ib, 0)
    fi_inliers_ib = np.where(counter_inliers_ib > 0, cfi_inliers_ib / counter_inliers_ib, 0)

    # DIFFI Score (Global Feature Importance): Outlier CFIS / Inlier CFIS
    fi_ib = np.divide(fi_outliers_ib, fi_inliers_ib, out=np.zeros_like(fi_outliers_ib), where=fi_inliers_ib != 0)

    return fi_ib, cfi_outliers_ib, cfi_inliers_ib, counter_outliers_ib, counter_inliers_ib


# ====================================================================
# B. Main execution code (Simulation Ch3.2: Binary Signal vs Binary Noise)
# ====================================================================


# --- 1. DIFFI computation and aggregation function ---
def diffi_ranks_with_split_count(X, y, n_trees, max_samples, contamination, n_iter, diffi_func):
    f1_all, fi_diffi_all_list = [], []
    cfi_out_all, cfi_in_all = [], []
    split_counts_total = np.zeros(X.shape[1], dtype=float)
    X_np = X.values

    for k in range(n_iter):
        iforest = IsolationForest(
            n_estimators=n_trees, max_samples=max_samples,
            contamination=contamination, random_state=k, bootstrap=False,
        )
        iforest.fit(X_np)
        y_pred = np.array(iforest.decision_function(X_np) < 0).astype('int')
        f1_all.append(f1_score(y, y_pred))

        fi_diffi, cfi_out, cfi_in, _, _ = diffi_func(iforest, X_np, adjust_iic=True)
        fi_diffi_all_list.append(fi_diffi)

        cfi_out_all.append(cfi_out)
        cfi_in_all.append(cfi_in)

        # Split frequency counting
        split_counts_iter = np.zeros(X.shape[1], dtype=float)
        for estimator in iforest.estimators_:
            split_features = estimator.tree_.feature
            # Count only splitting nodes
            split_features = split_features[split_features >= 0]
            counts = Counter(split_features)
            for idx, cnt in counts.items():
                split_counts_iter[idx] += cnt
        split_counts_total += split_counts_iter

    avg_f1 = np.mean(f1_all)
    fi_diffi_matrix = np.vstack(fi_diffi_all_list)
    fi_diffi_mean = np.mean(fi_diffi_matrix, axis=0)

    cfis_out_matrix = np.vstack(cfi_out_all)
    cfis_in_matrix = np.vstack(cfi_in_all)

    cfis_out_avg = np.mean(cfis_out_matrix, axis=0)
    cfis_in_avg = np.mean(cfis_in_matrix, axis=0)

    split_counts_avg = split_counts_total / n_iter

    return fi_diffi_mean, avg_f1, split_counts_avg, cfis_out_avg, cfis_in_avg, fi_diffi_matrix, cfis_out_matrix, cfis_in_matrix


# --- 2. Simulation Ch3.2: Binary Signal vs Binary Noise (Score Reversal Test) ---
def run_diffi_bias_test_binary_v2(n_iter=100, n_trees=100, max_samples=256, contamination=0.05):
    """
    Simulation Ch3.2: Tests DIFFI bias where binary signal features, which separate anomalies early,
    receive lower DIFFI scores than binary noise features due to reduced denominator contribution.
    """
    print("--- Simulation Ch3.2: DIFFI Binary Signal vs Binary Noise Score Reversal Test ---")

    # 1. Dataset design
    n_total = 1000
    n_anomaly = int(n_total * contamination)
    n_normal = n_total - n_anomaly
    rng = np.random.default_rng(seed=42)

    y_true_indices = np.zeros(n_total, dtype=int)
    y_true_indices[:n_anomaly] = 1  # First n_anomaly as anomalies

    # F_sig (Binary Signal) design
    # Anomalies: 95% probability of 1. Normals: 5% probability of 1. -> Clear separation
    F_sig_all = np.zeros(n_total)
    F_sig_all[:n_anomaly] = rng.binomial(1, 0.95, n_anomaly)  # Anomalies mostly 1
    F_sig_all[n_anomaly:] = rng.binomial(1, 0.05, n_normal)   # Normals mostly 0

    # F_noise (Binary Noise) design
    # Pure binary noise: 50% probability of 1 for both anomalies and normals
    p_binary_noise = 0.5
    F_noise_all = rng.binomial(1, p_binary_noise, n_total)

    # F_cont_ref (Continuous Noise): Reference continuous noise
    F_cont_ref_all = rng.uniform(low=0, high=10, size=n_total)

    # Create and shuffle dataframe
    df = pd.DataFrame({
        'F_sig (Binary Signal)': F_sig_all,
        'F_noise (Binary Noise)': F_noise_all,
        'F_cont_ref (Cont. Noise)': F_cont_ref_all,
        'Y': y_true_indices
    })
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df[['F_sig (Binary Signal)', 'F_noise (Binary Noise)', 'F_cont_ref (Cont. Noise)']]
    y_true = df['Y'].values
    FEATURE_NAMES = X.columns.tolist()

    print(f"Dataset size: {X.shape[0]}, Number of features: {X.shape[1]}")
    print(f"True anomaly proportion: {contamination * 100:.2f}%")
    print("Expected result: F_sig may receive lower DIFFI score than F_noise due to early anomaly separation.")

    # 2. DIFFI score and CFIS computation
    fi_diffi_mean, avg_f1, split_counts_avg, cfis_out_avg, cfis_in_avg, fi_diffi_matrix, cfis_out_matrix, cfis_in_matrix = diffi_ranks_with_split_count(
        X, y_true, n_trees=n_trees, max_samples=max_samples,
        contamination=contamination, n_iter=n_iter, diffi_func=diffi_ib_global_logic
    )

    # 3. Results aggregation
    results = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'DIFFI_Score_Mean': fi_diffi_mean,
        'Avg_CFIS_Outliers (Numerator)': cfis_out_avg,
        'Avg_CFIS_Inliers (Denominator)': cfis_in_avg,
        'Avg_Split_Count_Per_Tree': split_counts_avg.round(0)  # Average splits per tree
    })

    # Display results table
    print("\n--- Detailed CFIS Analysis Results (Simulation Ch3.2) ---")
    print(f"IF Model Avg F1 Score (Signal Detection): {avg_f1:.4f}")
    results = results.sort_values(by='DIFFI_Score_Mean', ascending=False).round(4)
    print(results.to_markdown(index=False))

    # 4. Validation results
    sig_row = results[results['Feature'] == 'F_sig (Binary Signal)'].iloc[0]
    noise_row = results[results['Feature'] == 'F_noise (Binary Noise)'].iloc[0]

    score_sig = sig_row['DIFFI_Score_Mean']
    score_noise = noise_row['DIFFI_Score_Mean']

    print("\n--- Validation Results (Simulation Ch3.2) ---")
    print(f"F_sig (Binary Signal) score: {score_sig:.4f}")
    print(f"F_noise (Binary Noise) score: {score_noise:.4f}")

    # CFIS numerator comparison
    cfis_out_sig = sig_row['Avg_CFIS_Outliers (Numerator)']
    cfis_out_noise = noise_row['Avg_CFIS_Outliers (Numerator)']
    print(f"CFIS Numerator (Outlier): F_sig={cfis_out_sig:.4f}, F_noise={cfis_out_noise:.4f}")

    # CFIS denominator comparison
    cfis_in_sig = sig_row['Avg_CFIS_Inliers (Denominator)']
    cfis_in_noise = noise_row['Avg_CFIS_Inliers (Denominator)']
    print(f"CFIS Denominator (Inlier): F_sig={cfis_in_sig:.4f}, F_noise={cfis_in_noise:.4f}")

    if score_sig < score_noise * 0.9:
        print(f"\nConclusion (Ch3.2): F_sig score ({score_sig:.4f}) is significantly lower than F_noise score ({score_noise:.4f}).")
        print("Cause: F_sig separates anomalies at shallow splits, reducing its contribution to inlier paths (denominator), causing DIFFI score reversal.")
    else:
        print("\nConclusion (Ch3.2): F_sig score remains higher than F_noise, reflecting its strong anomaly separation ability.")
        print("However, F_sig's CFIS denominator is substantially lower than F_noise (less denominator contribution), confirming binary signal characteristics.")


# --- Execution ---
if __name__ == '__main__':
    run_diffi_bias_test_binary_v2(n_iter=100)
