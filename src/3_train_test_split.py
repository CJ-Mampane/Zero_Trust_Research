
"""
3_train_test_split.py

Splits the cleaned RT-IoT2022 and IoT-23 datasets into development
and test sets using a target test proportion of 30%.

The target is the binary dataset label 'malicious', where 0 represents
benign traffic and 1 represents malicious traffic.

RT-IoT2022:
    - Creates a stratified random 70/30 split.
    - Stratifies using attack_type where possible to preserve the
      distribution of fine-grained attack labels across both partitions.
    - Uses row-level splitting because the dataset does not provide
      a suitable scenario, session, device, or time identifier.
    - Assigns each row a stable row_id for tracking records across
      subsequent processing stages.

IoT-23:
    - Uses a scenario-aware split to evaluate generalisation to
      previously unseen network captures.
    - Assigns whole scenarios to either development or test wherever
      feasible, keeping their records together.
    - Identifies large scenarios that require an internal split to
      achieve a test proportion close to 30%.
    - Selects scenario allocations by minimising the difference between
      the test set's binary class proportions and those of the full
      dataset, while accounting for the target test-set size.
    - Stratifies internally split scenarios using detailed_label where
      possible.
    - Requires benign and malicious records to appear in both the
      development and test partitions, including the wholly held-out
      test scenarios.

For both datasets, the script checks class presence, preserves relevant
metadata columns, writes the development and test CSV files, and
generates reports describing class distributions and split composition.

The row_id, scenario, label, detailed_label, attack_type, and
row_multiplicity columns are metadata and must not be used as model
features unless explicitly justified by the modelling methodology.

Usage:
    python 3_train_test_split.py
    python 3_train_test_split.py rt_iot2022
    python 3_train_test_split.py iot23
"""


import os
import sys
import json
import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "03_split")
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET = "malicious"                  # 0 = benign, 1 = malicious
ROW_ID = "row_id"
CLASSES = [0, 1]
TEST_SIZE = 0.30
RANDOM_STATE = 42                     # same seed as the historical pipeline
EXACT_FEASIBILITY_TOLERANCE = 0.02    # fixed convention (historical value)
ROW_COUNT_DEVIATION_WEIGHT = 1000.0
EXACT_ENUMERATION_MAX_ITEMS = 24


def stratification_key(frame, fine_col):
    """Fine-grained label; values with <2 rows fall back to the binary label."""
    fine = frame[fine_col].astype(str)
    counts = fine.value_counts()
    key = fine.where(fine.map(counts) >= 2, "fallback__" + frame[TARGET].astype(str))
    counts2 = key.value_counts()
    key = key.where(key.map(counts2) >= 2, frame[TARGET].astype(str))
    return key, sorted(counts[counts < 2].index.tolist())


def class_presence_check(train, test, extra=""):
    problems = []
    for name, part in (("development", train), ("test", test)):
        present = set(part[TARGET].unique())
        for c in CLASSES:
            if c not in present:
                problems.append(f"'{c}' has ZERO rows in {name}{extra}")
    if problems:
        raise RuntimeError("Class missing from a partition: " + "; ".join(problems))


def breakdown(train, test, col):
    t = pd.DataFrame({"development_rows": train[col].value_counts(),
                      "test_rows": test[col].value_counts()}).fillna(0).astype("int64")
    t["total"] = t.sum(axis=1)
    t["test_share"] = (t["test_rows"] / t["total"]).round(4)
    return t.sort_values("total", ascending=False)


# ======================================================================
# RT-IoT2022
# ======================================================================

def split_rt_iot2022():
    print("=" * 70 + "\nRT-IoT2022 (stratified random 70/30)\n" + "=" * 70)
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "rt_iot2022_clean.csv"), low_memory=False)
    if df[TARGET].isna().any():
        raise ValueError("Missing 'malicious' values in rt_iot2022_clean.csv")
    df.insert(0, ROW_ID, np.arange(len(df), dtype=np.int64))

    key, fallback = stratification_key(df, "attack_type")
    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=key,
                                   random_state=RANDOM_STATE)
    train, test = train.sort_values(ROW_ID), test.sort_values(ROW_ID)
    class_presence_check(train, test)

    train.to_csv(os.path.join(PROCESSED_DIR, "rt_iot2022_train.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, "rt_iot2022_test.csv"), index=False)

    by_attack = breakdown(train, test, "attack_type")
    by_state = breakdown(train, test, TARGET)
    by_attack.to_csv(os.path.join(RESULTS_DIR, "rt_iot2022_split_by_attack_type.csv"))
    by_state.to_csv(os.path.join(RESULTS_DIR, "rt_iot2022_split_by_malicious.csv"))

    print(f"Development: {len(train):,} rows | Test: {len(test):,} rows "
          f"({len(test) / len(df):.2%} test)")
    print(f"Fine labels with <2 rows (fell back to the binary label): {fallback or 'none'}")
    print("\nBy attack type:\n" + by_attack.to_string())
    print("\nBy benign (0) / malicious (1):\n" + by_state.to_string() + "\n")
    return {"rt_development_rows": len(train), "rt_test_rows": len(test)}


# ======================================================================
# IoT-23 -- scenario-aware
# ======================================================================

def closest_subset_sum(values, target):
    """Exact: best achievable subset sum to `target` (boolean DP over sums)."""
    total = int(sum(values))
    reach = np.zeros(total + 1, dtype=bool)
    reach[0] = True
    for v in values:
        v = int(v)
        if v == 0:
            continue
        reach[v:] |= reach[:total + 1 - v].copy()
    sums = np.flatnonzero(reach)
    return int(sums[np.argmin(np.abs(sums - target))])


def find_carveouts(scenario_totals):
    remaining = dict(scenario_totals)
    carveout = []
    while remaining:
        pool_total = sum(remaining.values())
        target = round(TEST_SIZE * pool_total)
        achieved = closest_subset_sum(list(remaining.values()), target)
        deviation = abs(achieved - target) / pool_total if pool_total else 0
        print(f"  feasibility with {len(remaining)} scenarios: best achievable test share "
              f"{achieved / pool_total:.2%} (deviation {deviation:.2%})")
        if deviation <= EXACT_FEASIBILITY_TOLERANCE or len(remaining) <= 1:
            return list(remaining.keys()), carveout
        largest = max(remaining, key=remaining.get)
        print(f"    -> not within {EXACT_FEASIBILITY_TOLERANCE:.0%}; carving out the largest "
              f"scenario: {largest} ({remaining[largest]:,} rows)")
        carveout.append(largest)
        del remaining[largest]
    return [], carveout


def exact_joint_optimum(pool_names, crosstab, carveout_names, overall_target_test,
                        overall_props):
    n = len(pool_names)
    if n > EXACT_ENUMERATION_MAX_ITEMS:
        raise RuntimeError(f"{n} pool scenarios exceeds the exact-enumeration cap "
                           f"({EXACT_ENUMERATION_MAX_ITEMS}); not expected for IoT-23.")
    totals = crosstab.loc[pool_names, "scenario_total"].to_numpy(dtype=np.int64)
    class_vals = crosstab.loc[pool_names, CLASSES].to_numpy(dtype=np.float64)
    pool_class_total = class_vals.sum(axis=0)
    carve_total = int(crosstab.loc[carveout_names, "scenario_total"].sum()) if carveout_names else 0
    carve_class = (crosstab.loc[carveout_names, CLASSES].sum().to_numpy(dtype=np.float64)
                   if carveout_names else np.zeros(len(CLASSES)))

    n_combos = 1 << n
    print(f"  Exhaustively evaluating all {n_combos:,} whole-scenario allocations...")
    combos = np.arange(n_combos, dtype=np.int64)
    pool_sum = np.zeros(n_combos, dtype=np.int64)
    pool_cls = np.zeros((n_combos, len(CLASSES)), dtype=np.float64)
    for i in range(n):
        b = ((combos >> i) & 1).astype(bool)
        pool_sum[b] += totals[i]
        pool_cls[b] += class_vals[i]
    del combos

    frac = ((overall_target_test - pool_sum) / carve_total if carve_total
            else np.zeros(n_combos))
    frac_c = np.clip(frac, 0, 1)
    row_pen = np.where((frac >= 0) & (frac <= 1), 0.0,
                       ROW_COUNT_DEVIATION_WEIGHT * np.abs(np.clip(frac, -10, 10) - frac_c))
    test_cls = pool_cls + frac_c[:, None] * carve_class[None, :]
    test_tot = test_cls.sum(axis=1)
    test_props = test_cls / np.where(test_tot == 0, 1, test_tot)[:, None]
    class_dev = np.abs(test_props - overall_props[None, :]).sum(axis=1)
    score = class_dev + row_pen

    # Change c: class-presence constraint
    train_cls = (pool_class_total[None, :] - pool_cls) + (1 - frac_c)[:, None] * carve_class[None, :]
    feasible = (test_cls.min(axis=1) > 0) & (train_cls.min(axis=1) > 0) & (pool_cls.min(axis=1) > 0)

    best_unconstrained = int(np.argmin(score))
    if not feasible.any():
        raise RuntimeError("No allocation satisfies the class-presence constraint.")
    best = int(np.argmin(np.where(feasible, score, np.inf)))
    names_for = lambda idx: [pool_names[i] for i in range(n) if (idx >> i) & 1]
    info = {
        "class_proportion_L1_deviation": float(class_dev[best]),
        "constraint_changed_allocation": best != best_unconstrained,
        "unconstrained_optimum_test_scenarios": names_for(best_unconstrained),
        "unconstrained_optimum_satisfied_constraint": bool(feasible[best_unconstrained]),
        "n_feasible_allocations": int(feasible.sum()),
    }
    print(f"  Selected allocation: class-proportion L1 deviation = {class_dev[best]:.4f}; "
          f"constraint changed the allocation: {info['constraint_changed_allocation']}")
    return names_for(best), float(frac_c[best]), info


def split_iot23():
    print("=" * 70 + "\nIoT-23 (scenario-aware 70/30)\n" + "=" * 70)
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "iot23_clean.csv"), low_memory=False)
    if df[TARGET].isna().any():
        raise ValueError("Missing 'malicious' values in iot23_clean.csv")
    df.insert(0, ROW_ID, np.arange(len(df), dtype=np.int64))
    n_total = len(df)

    crosstab = df.groupby(["scenario", TARGET]).size().unstack(fill_value=0)
    for c in CLASSES:
        if c not in crosstab.columns:
            crosstab[c] = 0
    crosstab = crosstab[CLASSES]
    crosstab["scenario_total"] = crosstab.sum(axis=1)

    overall_target_test = round(TEST_SIZE * n_total)
    overall_props = crosstab[CLASSES].sum().to_numpy(dtype=np.float64) / n_total

    pool, carveouts = find_carveouts(crosstab["scenario_total"].to_dict())
    test_scenarios, carve_frac, info = exact_joint_optimum(
        pool, crosstab, carveouts, overall_target_test, overall_props)

    split = pd.Series("development", index=df.index)
    split[df["scenario"].isin(test_scenarios)] = "test"
    carve_details = {}
    for scen in carveouts:
        idx = df.index[df["scenario"] == scen]
        key, fallback = stratification_key(df.loc[idx], "detailed_label")
        _, test_idx = train_test_split(idx, test_size=carve_frac, stratify=key,
                                       random_state=RANDOM_STATE)
        split[test_idx] = "test"
        carve_details[scen] = {"rows": len(idx), "test_rows": len(test_idx),
                               "test_fraction": carve_frac,
                               "fine_labels_fallback_to_binary_label": fallback}

    train, test = df.loc[split == "development"], df.loc[split == "test"]
    class_presence_check(train, test)
    held_out = test[~test["scenario"].isin(carveouts)]
    missing_held_out = [c for c in CLASSES if c not in set(held_out[TARGET])]
    if missing_held_out:
        raise RuntimeError(f"Wholly held-out test scenarios lack: {missing_held_out}")

    train.to_csv(os.path.join(PROCESSED_DIR, "iot23_train.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, "iot23_test.csv"), index=False)

    # ---------------- audit outputs ----------------
    audit = pd.crosstab([df["scenario"], df[TARGET]], split).reindex(
        columns=["development", "test"], fill_value=0)
    audit = audit.unstack(TARGET, fill_value=0)
    audit.columns = [f"{part}_{cls}" for part, cls in audit.columns]
    audit["allocation"] = ["carve-out (internal stratified split)" if s in carveouts
                           else ("whole -> test" if s in test_scenarios else "whole -> development")
                           for s in audit.index]
    audit["rows"] = df.groupby("scenario").size().reindex(audit.index)
    audit = audit.sort_values("rows", ascending=False)
    audit.to_csv(os.path.join(RESULTS_DIR, "iot23_scenario_split_audit.csv"))

    by_state = breakdown(train, test, TARGET)
    by_detail = breakdown(train, test, "detailed_label")
    by_state.to_csv(os.path.join(RESULTS_DIR, "iot23_split_by_malicious.csv"))
    by_detail.to_csv(os.path.join(RESULTS_DIR, "iot23_split_by_detailed_label.csv"))

    in_carve_train = train["scenario"].isin(carveouts)
    in_carve_test = test["scenario"].isin(carveouts)
    groups = pd.DataFrame({
        "development_rows": [int(in_carve_train.sum()), int((~in_carve_train).sum())],
        "test_rows": [int(in_carve_test.sum()), int((~in_carve_test).sum())],
    }, index=["carve-out scenario(s)", "whole scenarios"])
    groups["development_share"] = (groups["development_rows"] / len(train)).round(4)
    groups["test_share"] = (groups["test_rows"] / len(test)).round(4)
    groups.to_csv(os.path.join(RESULTS_DIR, "iot23_scenario_group_shares.csv"))
    held_out_by_state = held_out[TARGET].value_counts().reindex(CLASSES, fill_value=0)

    summary = {
        "rows_total": n_total,
        "development_rows": len(train), "test_rows": len(test),
        "test_fraction": round(len(test) / n_total, 6),
        "carveout_scenarios": carve_details,
        "whole_scenarios_to_test": sorted(test_scenarios),
        "whole_scenarios_to_development": sorted(s for s in pool if s not in test_scenarios),
        "held_out_test_rows_by_malicious": {str(k): int(v) for k, v in held_out_by_state.items()},
        **info,
    }
    with open(os.path.join(RESULTS_DIR, "iot23_split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDevelopment: {len(train):,} | Test: {len(test):,} "
          f"({len(test) / n_total:.2%} test)")
    print(f"Carve-out: {carve_details}")
    print(f"Whole scenarios -> TEST ({len(test_scenarios)}): {sorted(test_scenarios)}")
    print(f"Whole scenarios -> DEVELOPMENT ({len(pool) - len(test_scenarios)}): "
          f"{sorted(s for s in pool if s not in test_scenarios)}")
    print("\nScenario-group shares:\n" + groups.to_string())
    print("\nWholly held-out test rows, benign (0) / malicious (1):\n" + held_out_by_state.to_string())
    print("\nBy benign (0) / malicious (1):\n" + by_state.to_string())
    print("\nBy detailed label:\n" + by_detail.to_string())
    print("\nScenario audit:\n" + audit.to_string())
    return {"iot23_development_rows": len(train), "iot23_test_rows": len(test)}


if __name__ == "__main__":
    targets = sys.argv[1:] or ["rt_iot2022", "iot23"]
    manifest = {"script": "3_train_test_split.py", "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "feasibility_tolerance": EXACT_FEASIBILITY_TOLERANCE,
                "started": datetime.datetime.now().isoformat(timespec="seconds")}
    for name in targets:
        if name == "rt_iot2022":
            manifest.update(split_rt_iot2022())
        elif name == "iot23":
            manifest.update(split_iot23())
        else:
            raise SystemExit(f"Unknown dataset '{name}'. Use rt_iot2022 and/or iot23.")
    manifest["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(RESULTS_DIR, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nDone. Outputs: data/processed/*_train.csv, *_test.csv, results/03_split/")