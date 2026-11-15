"""Depth-sensitivity and split-purity analysis under the current S1 design.

For each of 100 S1 replications (same data-generating mechanism and
Isolation Forest settings as the main simulation), this script

(A) recomputes Original DIFFI and the raw AD-DIFFI enrichment rule while
    restricting binary-feature contributions to splits at depth <= d_max
    (d_max = 0 corresponds to the root-split-only rule; d_max = "all"
    reproduces Original DIFFI and an all-depth variant of the raw rule);
(B) records, for every split node, the proportion of model-defined
    outliers in the node and the child-node outlier-rate contrast E(v).

Continuous features are always evaluated at all depths.
"""
from __future__ import annotations

import sys
import json
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

# Repository-relative paths (this file lives in <repo>/scripts/)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
for _p in (REPO / "src", SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from validate_permutation_budget import (  # noqa: E402
    FEATURE_NAMES, FEATURE_TYPES, SCENARIOS, generate_dataset, make_seed,
)

MASTER_SEED = 42
N_REP = 100
if len(sys.argv) > 1 and sys.argv[1].isdigit():  # notebook-safe
    N_REP = int(sys.argv[1])
N_TREES = 100
MAX_SAMPLES = 256
MAX_D = 12  # depth bins (trees with max_samples=256 have depth <= 8)
DEPTH_LIMITS = [0, 1, 2, 3, 4, 5, 6, MAX_D]  # MAX_D == all depths
OUT = REPO / "results" / "supp_fig_s2_depth_purity"
OUT.mkdir(parents=True, exist_ok=True)

F = len(FEATURE_NAMES)
IS_BIN = np.array([FEATURE_TYPES[j] == "bin" for j in range(F)])
SIGNAL = np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
BIN_IDX = np.where(IS_BIN)[0]
CONT_IDX = np.where(~IS_BIN)[0]


def node_depths(tree):
    depth = np.zeros(tree.node_count, dtype=int)
    stack = [(0, 0)]
    while stack:
        n, d = stack.pop()
        depth[n] = d
        if tree.children_left[n] != tree.children_right[n]:
            stack.append((tree.children_left[n], d + 1))
            stack.append((tree.children_right[n], d + 1))
    return depth


def original_iic(tree, counts, adjust=True):
    """IIC exactly as in ad_diffi.benchmark._get_iic."""
    vals = np.full(tree.node_count, -1.0)
    for node in range(tree.node_count):
        l, r = tree.children_left[node], tree.children_right[node]
        if l == r:
            continue
        n = counts[node]
        if n <= 1:
            continue
        nl, nr = counts[l], counts[r]
        if nl == 0 or nr == 0:
            vals[node] = 0.0
            continue
        cmin = ceil(n / 2) / n
        cmax = (n - 1) / n
        imb = max(nl, nr) / n
        vals[node] = ((imb - cmin) / (cmax - cmin) * 0.5 + 0.5
                      if adjust and cmin != cmax else imb)
    return vals


def per_tree_terms(model, X, mask):
    """Return per-tree (feature x depth) accumulators and split records."""
    T = len(model.estimators_)
    o_num = np.zeros((F, MAX_D + 1)); o_cnt = np.zeros((F, MAX_D + 1))
    i_num = np.zeros((F, MAX_D + 1)); i_cnt = np.zeros((F, MAX_D + 1))
    r_num = np.zeros((T, F, MAX_D + 1)); r_den = np.zeros((T, F, MAX_D + 1))
    splits = []
    scores = model.decision_function(X)

    for t, (est, inbag) in enumerate(zip(model.estimators_,
                                         model.estimators_samples_)):
        tree = est.tree_
        depth = node_depths(tree)
        feat = tree.feature
        internal = tree.children_left != tree.children_right
        dbin = np.minimum(depth, MAX_D)

        # ---------------- Original DIFFI terms (in-bag) ----------------
        Xb = X[inbag]; sb = scores[inbag]
        for sub, num, cnt in ((Xb[sb < 0], o_num, o_cnt),
                              (Xb[sb >= 0], i_num, i_cnt)):
            if len(sub) == 0:
                continue
            M = est.decision_path(sub).tocsr()
            counts = np.asarray(M.sum(axis=0)).ravel()
            coef = original_iic(tree, counts)
            leaf_depth = depth[est.apply(sub)]
            w = np.where(leaf_depth > 0, 1.0 / np.maximum(leaf_depth, 1), 0.0)
            valid_row = (leaf_depth > 0).astype(float)
            node_w = M.T @ w
            node_c = M.T @ valid_row
            use = internal & (coef != -1)
            for node in np.where(use)[0]:
                num[feat[node], dbin[node]] += coef[node] * node_w[node]
                cnt[feat[node], dbin[node]] += node_c[node]
        # Original DIFFI skips trees lacking either outliers or inliers.
        # (Handled by accumulating only when both exist; see check below.)
        if (sb < 0).sum() == 0 or (sb >= 0).sum() == 0:
            raise RuntimeError("tree without outliers or inliers")

        # ---------------- Raw enrichment terms (all rows) ---------------
        M = est.decision_path(X).tocsr().astype(bool)
        n_node = np.asarray(M.sum(axis=0)).ravel()
        n_out = np.asarray(M[mask].sum(axis=0)).ravel()
        n_samp = tree.n_node_samples
        h = np.maximum(depth[est.apply(X)], 1)
        inv_h = M.T.astype(float) @ (1.0 / h)
        for node in np.where(internal)[0]:
            l, r = tree.children_left[node], tree.children_right[node]
            lam = 1.0 - min(n_samp[l], n_samp[r]) / n_samp[node]
            if n_node[l] == 0 or n_node[r] == 0:
                E = 0.0
            else:
                E = abs(n_out[l] / n_node[l] - n_out[r] / n_node[r])
            j = feat[node]
            r_num[t, j, dbin[node]] += lam * E * inv_h[node]
            r_den[t, j, dbin[node]] += n_node[node]
            if n_node[node] > 0:
                splits.append((t, j, int(depth[node]), int(n_node[node]),
                               n_out[node] / n_node[node], E))
    return (o_num, o_cnt, i_num, i_cnt, r_num, r_den), splits


def limited_scores(terms, d):
    o_num, o_cnt, i_num, i_cnt, r_num, r_den = terms
    keep = np.ones((F, MAX_D + 1), dtype=bool)
    keep[IS_BIN, d + 1:] = False

    def ratio(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b > 0)

    fo = ratio((o_num * keep).sum(1), (o_cnt * keep).sum(1))
    fi = ratio((i_num * keep).sum(1), (i_cnt * keep).sum(1))
    orig = ratio(fo, fi)

    tn = (r_num * keep).sum(2); td = (r_den * keep).sum(2)
    ts = ratio(tn, td)
    valid = td > 0
    raw = ratio((ts * valid).sum(0), valid.sum(0).astype(float))
    return orig, raw


def auc(scores, idx):
    return roc_auc_score(SIGNAL[idx].astype(int), scores[idx])


def main():
    scenario = SCENARIOS["S1"]
    rows, split_rows = [], []
    for rep in range(N_REP):
        data_seed = make_seed(MASTER_SEED, 1, rep, 0)
        forest_seed = make_seed(MASTER_SEED, 1, rep, 1)
        X, y = generate_dataset(data_seed, scenario)
        model = IsolationForest(
            n_estimators=N_TREES, max_samples=min(MAX_SAMPLES, len(X)),
            contamination=scenario["contamination"], max_features=1.0,
            bootstrap=False, random_state=forest_seed).fit(X)
        mask = model.predict(X) == -1
        terms, splits = per_tree_terms(model, X, mask)
        for d in DEPTH_LIMITS:
            orig, raw = limited_scores(terms, d)
            for meth, s in (("Original DIFFI", orig), ("Raw rule", raw)):
                rec = {"rep": rep, "d_max": d, "method": meth,
                       "auc_overall": auc(s, np.arange(F)),
                       "auc_binary": auc(s, BIN_IDX),
                       "auc_cont": auc(s, CONT_IDX)}
                rec.update({FEATURE_NAMES[j]: s[j] for j in range(F)})
                rows.append(rec)
        for (t, j, dep, n, pur, E) in splits:
            split_rows.append((rep, t, FEATURE_NAMES[j], dep, n, pur, E))
        if rep == 0:
            check(model, X, mask, terms)
        print(f"rep {rep + 1}/{N_REP}", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "depth_sensitivity_s1.csv", index=False)
    pd.DataFrame(split_rows, columns=["rep", "tree", "feature", "depth",
                                      "n_node", "outlier_proportion",
                                      "enrichment"]).to_csv(
        OUT / "split_purity_s1.csv", index=False)
    json.dump({"master_seed": MASTER_SEED, "n_rep": N_REP,
               "n_trees": N_TREES, "max_samples": MAX_SAMPLES,
               "depth_limits": DEPTH_LIMITS, "all_depth_code": MAX_D},
              open(OUT / "settings.json", "w"), indent=2)


def check(model, X, mask, terms):
    """Verify against the repository implementations."""
    from ad_diffi.benchmark import original_diffi_importance
    from ad_diffi.core import compute_raw_ad_diffi
    ref_orig = original_diffi_importance(model, X)[0]
    ref_raw = compute_raw_ad_diffi(iforest=model, X_data=X,
                                   feature_types=FEATURE_TYPES,
                                   anomaly_mask=mask)
    orig_all, _ = limited_scores(terms, MAX_D)
    _, raw_rso = limited_scores(terms, 0)
    print("check Original (d=all):", np.max(np.abs(orig_all - ref_orig)))
    print("check Raw (d=0):", np.max(np.abs(raw_rso - ref_raw)))
    from ad_diffi.permutation import get_model_anomaly_mask
    print("mask identical:", np.array_equal(mask,
                                            get_model_anomaly_mask(model, X)))


if __name__ == "__main__":
    main()
