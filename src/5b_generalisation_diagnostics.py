"""
5b_generalisation_diagnostics.py

Performs diagnostic analysis of IoT-23 generalisation using development
data only. It does not make feature-selection decisions, modify previous
outputs, or change the frozen feature-selection protocol.

The analysis examines whether individual flow records can distinguish
benign from malicious traffic in scenarios held out during development.
The target is the binary label 'malicious' (0 = benign, 1 = malicious).

Diagnostics:

    1. Feature-vector ambiguity
       - Compares each held-out whole-scenario validation row with
         training rows using its exact encoded feature vector.
       - Classifies vectors as unseen, seen only with the same class,
         seen only with the opposite class, or seen with both classes.
       - Summarises ambiguity by true class, detailed label, and scenario.
       - Measures how often identical feature vectors occur with both
         benign and malicious labels in the development data.

    2. Classification error analysis
       - Evaluates Random Forest and XGBoost using the existing
         partially grouped cross-validation folds.
       - Produces confusion matrices for held-out whole-scenario rows.
       - Reports binary macro-F1 for these rows.
       - Reports macro-F1 for Capture-35-1 rows, using the classes
         present in those rows.
       - Summarises predictions by detailed label and scenario.

Two feature sets are compared:
    - all_filtered: all variables surviving the feature-selection filters.
    - selected: variables selected by the feature-selection procedure.

Usage on terminal:
    python src/5b_generalisation_diagnostics.py
"""

import os
import gc
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FS_DIR = os.path.join(PROJECT_ROOT, "results", "05_feature_selection")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "05b_generalisation_diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)

DS = "iot23"
TARGET, ROW_ID = "malicious", "row_id"
CLASS_CODE = {"benign": 0, "malicious": 1}
CLASS_NAME = {v: k for k, v in CLASS_CODE.items()}
RANDOM_STATE = 42
MODELS = ["random_forest", "xgboost"]


def load():
    dev = pd.read_csv(os.path.join(PROCESSED_DIR, f"{DS}_train.csv"), low_memory=False)
    folds = pd.read_csv(os.path.join(PROCESSED_DIR, f"{DS}_cv_folds.csv"))
    dev = dev.merge(folds, on=ROW_ID, how="left", validate="one_to_one")
    with open(os.path.join(FS_DIR, f"{DS}_filter_log.json")) as f:
        flog = json.load(f)
    with open(os.path.join(FS_DIR, f"{DS}_selection.json")) as f:
        sel = json.load(f)
    return dev, flog, sel


def encode(dev, variables, numeric, levels):
    arrays, names = [], []
    for v in variables:
        if v in numeric:
            arrays.append(dev[v].to_numpy(dtype=np.float32))
            names.append(v)
        else:
            s = dev[v].astype(object).where(dev[v].notna(), "<missing>").astype(str).to_numpy()
            for lvl in levels[v]:
                arrays.append((s == lvl).astype(np.float32))
                names.append(f"{v}={lvl}")
    return np.column_stack(arrays), names


def row_hash(X):
    return pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()


def ambiguity(h, y, fold, sel_rows, f):
    train = fold != f
    pairs = pd.DataFrame({"h": h[train], "bit": (1 << y[train])}).drop_duplicates()
    mask_by_h = pairs.groupby("h")["bit"].sum()
    val = (fold == f) & sel_rows
    m = pd.Series(h[val]).map(mask_by_h).fillna(0).astype(np.int64).to_numpy()
    own = 1 << y[val]
    cat = np.where(m == 0, "unseen_vector",
          np.where(m == own, "seen_same_class_only",
          np.where((m & own) == 0, "seen_other_class_only", "seen_both")))
    return val, cat


def fit(name, X, y):
    if name == "random_forest":
        m = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(X, y)
    else:
        m = XGBClassifier(n_estimators=200, tree_method="hist", eval_metric="logloss",
                          random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return m


def main():
    t0 = time.perf_counter()
    dev, flog, sel = load()
    fold = dev["fold_B_partial_grouped"].to_numpy(dtype=np.int64)
    sel_rows = (dev["scenario_group"] == "whole_scenario").to_numpy()
    carve = ~sel_rows
    y = dev[TARGET].to_numpy(dtype=np.int64)
    levels = {v: d["kept"] for v, d in flog["categorical_levels"].items()}
    numeric = flog["surviving_numeric"]
    sets = {"all_filtered": flog["surviving_variables"],
            "selected": sel["selected_variables"]}
    K = len(np.unique(fold))

    amb_rows, conf_rows, label_rows, scen_rows, score_rows = [], [], [], [], []
    for set_name, variables in sets.items():
        X, names = encode(dev, variables, numeric, levels)
        h = row_hash(X)
        # within-development conflicts in this feature space (all rows)
        pairs = pd.DataFrame({"h": h, "y": y}).drop_duplicates()
        n_cls = pairs.groupby("h")["y"].nunique()
        conflicted = pd.Series(h).map(n_cls).to_numpy() > 1
        print(f"\n=== {set_name}: {len(variables)} variables, {X.shape[1]} columns ===")
        print(f"  distinct feature vectors in development: {len(n_cls):,} "
              f"(of {len(h):,} rows)")
        print(f"  development rows whose vector carries both benign and malicious: "
              f"{conflicted.sum():,} ({conflicted.mean():.2%})")
        score_rows.append({"feature_set": set_name, "fold": "all", "model": "none",
                           "metric": "share_dev_rows_vector_with_both_classes",
                           "value": float(conflicted.mean())})

        # ---- A. ambiguity
        for f in range(K):
            val, cat = ambiguity(h, y, fold, sel_rows, f)
            vdf = pd.DataFrame({"category": cat, "true": dev.loc[val, TARGET].map(CLASS_NAME).to_numpy(),
                                "detailed_label": dev.loc[val, "detailed_label"].to_numpy(),
                                "scenario": dev.loc[val, "scenario"].to_numpy()})
            for by in ("true", "detailed_label", "scenario"):
                t = pd.crosstab(vdf[by], vdf["category"], normalize="index").round(4)
                t["n_rows"] = vdf[by].value_counts()
                t = t.reset_index().rename(columns={by: "group"})
                t.insert(0, "grouped_by", by)
                t.insert(0, "fold", f)
                t.insert(0, "feature_set", set_name)
                amb_rows.append(t)
            overall = vdf["category"].value_counts(normalize=True).round(4).to_dict()
            print(f"  fold {f} unseen-capture rows ({len(vdf):,}): {overall}")

        # ---- B. models
        for f in range(K):
            tr, va = fold != f, fold == f
            for name in MODELS:
                model = fit(name, X[tr], y[tr])
                pred = model.predict(X[va])
                del model
                gc.collect()
                yv, sv, cv = y[va], sel_rows[va], carve[va]
                cm = confusion_matrix(yv[sv], pred[sv], labels=[0, 1])
                for i in range(2):
                    for j in range(2):
                        conf_rows.append({"feature_set": set_name, "fold": f, "model": name,
                                          "true": CLASS_NAME[i], "predicted": CLASS_NAME[j],
                                          "rows": int(cm[i, j])})
                present_c = sorted(set(yv[cv]))
                scores = {
                    "macro_f1_unseen_capture_rows": f1_score(yv[sv], pred[sv], labels=[0, 1],
                                                             average="macro", zero_division=0),
                    "macro_f1_capture35_rows_present_classes": f1_score(
                        yv[cv], pred[cv], labels=present_c, average="macro", zero_division=0),
                }
                for k, v in scores.items():
                    score_rows.append({"feature_set": set_name, "fold": f, "model": name,
                                       "metric": k, "value": float(v)})
                pdf = pd.DataFrame({"pred": [CLASS_NAME[p] for p in pred[sv]],
                                    "detailed_label": dev.loc[va, "detailed_label"].to_numpy()[sv],
                                    "scenario": dev.loc[va, "scenario"].to_numpy()[sv],
                                    "true": [CLASS_NAME[t] for t in yv[sv]]})
                for by, store in (("detailed_label", label_rows), ("scenario", scen_rows)):
                    t = pd.crosstab([pdf[by], pdf["true"]], pdf["pred"], normalize="index").round(4)
                    t = t.reindex(columns=list(CLASS_CODE), fill_value=0)
                    t["n_rows"] = pdf.groupby([by, "true"]).size()
                    t = t.reset_index()
                    t.insert(0, "model", name)
                    t.insert(0, "fold", f)
                    t.insert(0, "feature_set", set_name)
                    store.append(t)
                print(f"  fold {f} {name:13s} unseen macro-F1 {scores['macro_f1_unseen_capture_rows']:.4f} | "
                      f"Capture-35-1 macro-F1 (present classes) "
                      f"{scores['macro_f1_capture35_rows_present_classes']:.4f}")
                print("    confusion (rows=true benign/malicious, cols=pred benign/malicious):\n    " +
                      str(cm).replace("\n", "\n    "))
        del X
        gc.collect()

    pd.concat(amb_rows).to_csv(os.path.join(OUT_DIR, "ambiguity_unseen_capture_rows.csv"), index=False)
    pd.DataFrame(conf_rows).to_csv(os.path.join(OUT_DIR, "confusion_unseen_capture_rows.csv"), index=False)
    pd.concat(label_rows).to_csv(os.path.join(OUT_DIR, "predictions_by_detailed_label.csv"), index=False)
    pd.concat(scen_rows).to_csv(os.path.join(OUT_DIR, "predictions_by_scenario.csv"), index=False)
    pd.DataFrame(score_rows).to_csv(os.path.join(OUT_DIR, "scores.csv"), index=False)

    amb = pd.concat(amb_rows)
    print("\n=== Ambiguity by TRUE CLASS (share of unseen-capture rows) ===")
    print(amb[amb["grouped_by"] == "true"].to_string(index=False))
    print("\n=== Ambiguity by DETAILED LABEL ===")
    print(amb[amb["grouped_by"] == "detailed_label"].to_string(index=False))
    print(f"\nDone in {time.perf_counter() - t0:.0f}s. Outputs: results/05b_generalisation_diagnostics/")


if __name__ == "__main__":
    main()