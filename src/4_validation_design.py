
"""
4_validation_design.py

Builds and audits the cross-validation folds used for development-stage
model evaluation, feature selection, hyperparameter tuning, threshold
selection, and hybrid-model cross-fitting.

Only development data is used. The test datasets are not loaded by this
script.

RT-IoT2022:
    - Creates stratified 5-fold cross-validation partitions at row level.
    - Uses attack_type for stratification where sufficient records are
      available, falling back to the binary 'malicious' label for rare
      attack types.
    - Uses the binary target 'malicious', where 0 represents benign
      traffic and 1 represents malicious traffic.
    - Produces fold summaries and attack-type distributions for auditing.

IoT-23:
    - Creates two candidate 3-fold validation designs for comparison.
    - Design A uses stratified row-level folds.
    - Design B uses partial scenario grouping to assess generalisation
      to scenarios not included in the corresponding training fold.
    - Distributes records from CTU-IoT-Malware-Capture-35-1 across folds
      using stratification because this capture accounts for most of the
      development data and cannot be assigned wholly to one fold.
    - Assigns the remaining scenarios as whole groups to folds.
    - Uses exhaustive search to select a scenario allocation that balances
      benign and malicious class counts across folds.
    - Requires each validation fold to contain at least 30 benign and
      30 malicious records from scenarios excluded from its training data.
    - Audits class distributions, scenario assignments, and fold overlap.

The script saves row-to-fold assignments and summary reports for
examining class balance and the degree of scenario separation.

The row-level IoT-23 design does not provide scenario-level separation.
The partially grouped design provides this separation for whole scenarios,
but records from Capture-35-1 remain divided across folds.

Usage:
    python 4_validation_design.py
    python 4_validation_design.py rt_iot2022
    python 4_validation_design.py iot23
"""


import os
import sys
import json
import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "04_validation")
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET = "malicious"                  # 0 = benign, 1 = malicious
ROW_ID = "row_id"
CLASSES = [0, 1]
RANDOM_STATE = 42
RT_K = 5
IOT_K = 3
CARVE_OUT = "CTU-IoT-Malware-Capture-35-1"
MIN_ROWS_PER_CLASS = 30


def strat_key(fine, coarse, k):
    """Fine label; values with fewer than k rows fall back to the coarse label."""
    fine = fine.astype(str)
    coarse = coarse.astype(str)
    key = fine.where(fine.map(fine.value_counts()) >= k, "fallback__" + coarse)
    key = key.where(key.map(key.value_counts()) >= k, coarse)
    return key


def stratified_folds(key, k):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    fold = np.empty(len(key), dtype=np.int8)
    for f, (_, val_idx) in enumerate(skf.split(np.zeros(len(key)), key)):
        fold[val_idx] = f
    return fold


def fold_class_table(frame, fold_col, k):
    rows = []
    for f in range(k):
        val = frame[frame[fold_col] == f]
        trn = frame[frame[fold_col] != f]
        r = {"fold": f, "validation_rows": len(val), "training_rows": len(trn)}
        for c in CLASSES:
            r[f"val_{c}"] = int((val[TARGET] == c).sum())
            r[f"train_{c}"] = int((trn[TARGET] == c).sum())
        rows.append(r)
    return pd.DataFrame(rows)


# ======================================================================
# RT-IoT2022
# ======================================================================

def design_rt():
    print("=" * 70 + f"\nRT-IoT2022: stratified {RT_K}-fold (row-level)\n" + "=" * 70)
    dev = pd.read_csv(os.path.join(PROCESSED_DIR, "rt_iot2022_train.csv"),
                      usecols=[ROW_ID, "attack_type", TARGET])
    key = strat_key(dev["attack_type"], dev[TARGET], RT_K)
    dev["fold"] = stratified_folds(key, RT_K)

    fallback = sorted(dev.loc[key.str.startswith("fallback__") | key.isin([str(c) for c in CLASSES]),
                              "attack_type"].unique())
    table = fold_class_table(dev, "fold", RT_K)
    by_attack = pd.crosstab(dev["attack_type"], dev["fold"])

    dev[[ROW_ID, "fold"]].to_csv(os.path.join(PROCESSED_DIR, "rt_iot2022_cv_folds.csv"), index=False)
    table.to_csv(os.path.join(RESULTS_DIR, "rt_iot2022_fold_summary.csv"), index=False)
    by_attack.to_csv(os.path.join(RESULTS_DIR, "rt_iot2022_fold_by_attack_type.csv"))

    print(f"Development rows: {len(dev):,}")
    print(f"Attack types stratified via the binary-label fallback (< {RT_K} rows): {fallback or 'none'}")
    print("\nPer fold (validation / training class counts):\n" + table.to_string(index=False))
    print("\nValidation rows per attack type per fold:\n" + by_attack.to_string() + "\n")
    return {"rt_k": RT_K, "rt_fallback_attack_types": fallback}


# ======================================================================
# IoT-23
# ======================================================================

def best_group_assignment(ct, k):
    """Exhaustive search over all k^n whole-scenario fold assignments."""
    names = ct.index.tolist()
    n = len(names)
    counts = ct[CLASSES].to_numpy(dtype=np.float64)          # (n, classes)
    totals = counts.sum(axis=0)
    n_comb = k ** n
    print(f"  Exhaustively evaluating all {n_comb:,} assignments of {n} whole scenarios to {k} folds...")
    combos = np.arange(n_comb, dtype=np.int64)
    assign = (combos[:, None] // (k ** np.arange(n, dtype=np.int64))[None, :]) % k   # (n_comb, n)

    val = np.zeros((n_comb, k, len(CLASSES)))
    for f in range(k):
        val[:, f, :] = (assign == f).astype(np.float64) @ counts
    score = (np.abs(val - totals / k) / totals).sum(axis=(1, 2))
    feasible = (val >= MIN_ROWS_PER_CLASS).all(axis=(1, 2))
    if not feasible.any():
        raise RuntimeError(f"No assignment gives every fold >= {MIN_ROWS_PER_CLASS} unseen-scenario "
                           f"rows of every class. Report this before choosing a design.")
    best = int(np.argmin(np.where(feasible, score, np.inf)))
    print(f"  Feasible assignments: {int(feasible.sum()):,}; selected spread score {score[best]:.4f}")
    return dict(zip(names, assign[best].tolist())), int(feasible.sum()), float(score[best])


def design_iot23():
    print("=" * 70 + f"\nIoT-23: {IOT_K}-fold, design A (row-level) vs B (partial grouping)\n" + "=" * 70)
    dev = pd.read_csv(os.path.join(PROCESSED_DIR, "iot23_train.csv"),
                      usecols=[ROW_ID, "scenario", "detailed_label", TARGET])
    is_c = (dev["scenario"] == CARVE_OUT).to_numpy()
    print(f"Development rows: {len(dev):,} | {CARVE_OUT}: {is_c.sum():,} ({is_c.mean():.2%})")

    # ---- A: row-level
    dev["fold_A_rowlevel"] = stratified_folds(strat_key(dev["detailed_label"], dev[TARGET], IOT_K), IOT_K)

    # ---- B: partial grouping
    fold_b = np.full(len(dev), -1, dtype=np.int8)
    sub = dev.loc[is_c]
    fold_b[is_c] = stratified_folds(strat_key(sub["detailed_label"], sub[TARGET], IOT_K), IOT_K)
    others = dev.loc[~is_c]
    ct = others.groupby(["scenario", TARGET]).size().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
    print("\nNon-carve-out development scenarios (whole groups):\n" + ct.to_string())
    assignment, n_feasible, spread = best_group_assignment(ct, IOT_K)
    fold_b[~is_c] = others["scenario"].map(assignment).to_numpy(dtype=np.int8)
    dev["fold_B_partial_grouped"] = fold_b
    dev["scenario_group"] = np.where(is_c, "capture_35_1_rows", "whole_scenario")

    # ---- audit tables
    tab_a = fold_class_table(dev, "fold_A_rowlevel", IOT_K)
    tab_b_all = fold_class_table(dev, "fold_B_partial_grouped", IOT_K)
    tab_b_c35 = fold_class_table(dev.loc[is_c], "fold_B_partial_grouped", IOT_K)
    tab_b_whole = fold_class_table(dev.loc[~is_c], "fold_B_partial_grouped", IOT_K)
    scen_fold = pd.DataFrame({"scenario": list(assignment), "fold_B": list(assignment.values())}) \
        .merge(ct.reset_index(), on="scenario").sort_values(["fold_B", "scenario"])

    # Scenario overlap check: under B, a whole scenario must never be in both
    # training and validation of the same fold.
    leaks = [s for s, g in dev.loc[~is_c].groupby("scenario")["fold_B_partial_grouped"] if g.nunique() != 1]
    if leaks:
        raise RuntimeError(f"Scenarios split across folds under design B: {leaks}")

    dev[[ROW_ID, "scenario_group", "fold_A_rowlevel", "fold_B_partial_grouped"]].to_csv(
        os.path.join(PROCESSED_DIR, "iot23_cv_folds.csv"), index=False)
    tab_a.to_csv(os.path.join(RESULTS_DIR, "iot23_foldA_rowlevel_summary.csv"), index=False)
    tab_b_all.to_csv(os.path.join(RESULTS_DIR, "iot23_foldB_all_rows_summary.csv"), index=False)
    tab_b_c35.to_csv(os.path.join(RESULTS_DIR, "iot23_foldB_capture35_rows_summary.csv"), index=False)
    tab_b_whole.to_csv(os.path.join(RESULTS_DIR, "iot23_foldB_whole_scenario_rows_summary.csv"), index=False)
    scen_fold.to_csv(os.path.join(RESULTS_DIR, "iot23_foldB_scenario_assignment.csv"), index=False)

    print("\nDesign B -- whole-scenario fold assignment:\n" + scen_fold.to_string(index=False))
    print("\nDesign A (row-level) -- per fold:\n" + tab_a.to_string(index=False))
    print("\nDesign B -- all rows per fold:\n" + tab_b_all.to_string(index=False))
    print("\nDesign B -- Capture-35-1 rows per fold:\n" + tab_b_c35.to_string(index=False))
    print("\nDesign B -- whole-scenario rows per fold (validation = scenarios unseen in training):\n"
          + tab_b_whole.to_string(index=False))
    return {"iot23_k": IOT_K, "min_rows_per_class_unseen_validation": MIN_ROWS_PER_CLASS,
            "design_B_feasible_assignments": n_feasible, "design_B_spread_score": spread,
            "design_B_assignment": assignment}


if __name__ == "__main__":
    targets = sys.argv[1:] or ["rt_iot2022", "iot23"]
    manifest = {"script": "4_validation_design.py", "random_state": RANDOM_STATE,
                "started": datetime.datetime.now().isoformat(timespec="seconds")}
    for name in targets:
        if name == "rt_iot2022":
            manifest.update(design_rt())
        elif name == "iot23":
            manifest.update(design_iot23())
        else:
            raise SystemExit(f"Unknown dataset '{name}'.")
    manifest["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(RESULTS_DIR, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("Done. Outputs: data/processed/*_cv_folds.csv, results/04_validation/")