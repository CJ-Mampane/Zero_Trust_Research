"""
7_tuning.py

Tunes the Phase-1 binary risk models, XGBoost and Random Forest, using
development data only. The test partitions are never opened.

The target is 'malicious' (0 = benign, 1 = malicious). Each model estimates
the probability of malicious traffic, P(malicious), from the feature-selected
FS-2 dataset.

Hyperparameter tuning uses 60 randomly sampled candidates per model, evaluated
using the same cross-validation folds and selected features as FS-2:
    - RT-IoT2022: stratified 5-fold cross-validation.
    - IoT-23: 3-fold design B with partial scenario grouping.

Candidate selection is based on the highest mean binary macro-F1 score across
the decision rows in each validation fold. For IoT-23, decision rows are
restricted to rows from whole scenarios unseen in that fold's training data.
For RT-IoT2022, all validation rows are decision rows. Ties are resolved by
candidate order. Class balancing is applied during model fitting.

Additional diagnostics are recorded but are not used for candidate selection:
    - Standard deviation of decision-row macro-F1 across folds.
    - Log-loss on decision rows.
    - Macro-F1 across all validation rows.
    - IoT-23 Capture-35-1 macro-F1 over classes present.
    - Model fitting time and per-fold results.

Each candidate's parameters are saved before fitting. Individual
(model, candidate, fold) fits are checkpointed so completed work can be
reused after interruption. MemoryError failures are logged, and candidates
with any failed fold are excluded from selection.

The search is non-nested, so the selected configuration's development score
may be optimistically biased. Final performance must be assessed separately
using the held-out test partitions, with this limitation disclosed.

Usage:
    python src/7_tuning.py rt_iot2022
    python src/7_tuning.py iot23
    python src/7_tuning.py
"""

import os
import sys
import gc
import json
import time
import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FS_DIR = os.path.join(PROJECT_ROOT, "results", "05_feature_selection")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "07_tuning")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_CANDIDATES = 60
TARGET, ROW_ID = "malicious", "row_id"          # 0 = benign, 1 = malicious
LABELS = [0, 1]
MODELS = ["xgboost", "random_forest"]
FOLD_COL = {"iot23": "fold_B_partial_grouped", "rt_iot2022": "fold"}
NON_FEATURES = {ROW_ID, TARGET, "fold", "scenario_group", "fold_A_rowlevel", "fold_B_partial_grouped"}


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


# ----------------------------------------------------------------------
# search spaces and candidates
# ----------------------------------------------------------------------

def sample_candidates(model, n, seed):
    rng = np.random.default_rng(seed)
    cands = []
    for _ in range(n):
        if model == "random_forest":
            p = {"n_estimators": int(rng.integers(100, 401)),
                 "max_depth": [None, 10, 20, 30][int(rng.integers(0, 4))],
                 "min_samples_leaf": int(rng.integers(1, 11)),
                 "max_features": ["sqrt", "log2", 0.5][int(rng.integers(0, 3))]}
        else:
            p = {"n_estimators": int(rng.integers(100, 401)),
                 "max_depth": int(rng.integers(3, 10)),
                 "learning_rate": float(rng.uniform(0.01, 0.30)),
                 "subsample": float(rng.uniform(0.6, 1.0)),
                 "colsample_bytree": float(rng.uniform(0.6, 1.0)),
                 "min_child_weight": int(rng.integers(1, 8)),
                 "gamma": float(rng.uniform(0.0, 0.5)),
                 "reg_alpha": float(rng.uniform(0.0, 1.0)),
                 "reg_lambda": float(rng.uniform(0.5, 2.0))}
        cands.append(p)
    return cands


def make_model(model, params):
    if model == "random_forest":
        return RandomForestClassifier(**params, class_weight="balanced",
                                      random_state=RANDOM_STATE, n_jobs=-1)
    return XGBClassifier(**params, tree_method="hist", eval_metric="logloss",
                         random_state=RANDOM_STATE, n_jobs=-1)


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------

def load(ds):
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ds}_train_selected.csv"), low_memory=False)
    folds = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ds}_cv_folds.csv"))
    df = df.merge(folds, on=ROW_ID, how="left", validate="one_to_one")
    with open(os.path.join(FS_DIR, f"{ds}_selection.json")) as f:
        sel_spec = json.load(f)
    feat_cols = [c for c in df.columns if c not in NON_FEATURES]
    if feat_cols != sel_spec["selected_columns"]:
        raise ValueError(f"Feature columns {feat_cols} do not match FS-2 selection "
                         f"{sel_spec['selected_columns']}")
    fold = df[FOLD_COL[ds]].to_numpy(dtype=np.int64)
    if ds == "iot23":
        decision_rows = (df["scenario_group"] == "whole_scenario").to_numpy()
        carve = ~decision_rows
    else:
        decision_rows = np.ones(len(df), dtype=bool)
        carve = None
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.int64)
    if not set(np.unique(y)) <= set(LABELS):
        raise ValueError("Target values other than 0/1 found.")
    print(f"  rows {len(df):,} | features {len(feat_cols)} | folds {sorted(set(fold))} | "
          f"decision rows {decision_rows.sum():,}")
    return X, y, fold, decision_rows, carve, feat_cols


# ----------------------------------------------------------------------
# checkpoint
# ----------------------------------------------------------------------

class Checkpoint:
    def __init__(self, path):
        self.path, self.done = path, {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self.done[r["key"]] = r
        print(f"  checkpoint: {len(self.done)} completed fits")

    def get(self, key):
        return self.done.get(key)

    def put(self, key, rec):
        rec = {"key": key, **rec}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.done[key] = rec


def macro_f1(y, p, labels=(0, 1)):
    return float(f1_score(y, p, labels=list(labels), average="macro", zero_division=0))


# ----------------------------------------------------------------------
# search
# ----------------------------------------------------------------------

def evaluate_fold(model, params, X, y, fold, f, decision_rows, carve):
    tr, va = fold != f, fold == f
    m = make_model(model, params)
    t0 = time.perf_counter()
    if model == "xgboost":
        m.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
    else:
        m.fit(X[tr], y[tr])
    fit_s = time.perf_counter() - t0
    if list(m.classes_) != LABELS:
        raise RuntimeError(f"Unexpected class order {m.classes_}")
    proba = m.predict_proba(X[va])
    del m
    gc.collect()
    pred = proba.argmax(axis=1)
    yv, dv = y[va], decision_rows[va]
    rec = {"fit_seconds": fit_s,
           "macro_f1_decision_rows": macro_f1(yv[dv], pred[dv]),
           "logloss_decision_rows": float(log_loss(yv[dv], np.clip(proba[dv], 1e-15, 1), labels=LABELS)),
           "macro_f1_all_validation_rows": macro_f1(yv, pred)}
    if carve is not None:
        cv = carve[va]
        rec["macro_f1_capture35_present_classes"] = macro_f1(yv[cv], pred[cv], sorted(set(yv[cv])))
    return rec


def run(ds):
    print("=" * 70 + f"\nTUNING: {ds} (development data only)\n" + "=" * 70)
    X, y, fold, decision_rows, carve, feat_cols = load(ds)
    K = len(np.unique(fold))
    ck = Checkpoint(os.path.join(OUT_DIR, f"{ds}_checkpoint.jsonl"))
    summary = {}
    n_total = len(MODELS) * N_CANDIDATES * K
    done_now, t_start = 0, time.perf_counter()

    for mi, model in enumerate(MODELS):
        cands = sample_candidates(model, N_CANDIDATES, seed=RANDOM_STATE + mi)
        cand_path = os.path.join(OUT_DIR, f"{ds}_{model}_candidates.json")
        if os.path.exists(cand_path):
            with open(cand_path) as f:
                if json.load(f) != json.loads(json.dumps(cands)):
                    raise SystemExit(f"Saved candidate list {cand_path} differs from the generated "
                                     f"one; refusing to mix searches.")
        else:
            with open(cand_path, "w") as f:
                json.dump(cands, f, indent=1)

        rows = []
        for ci, params in enumerate(cands):
            for f in range(K):
                key = f"{model}|cand={ci}|fold={f}"
                rec = ck.get(key)
                if rec is None:
                    try:
                        rec = evaluate_fold(model, params, X, y, fold, f, decision_rows, carve)
                        rec["status"] = "ok"
                    except MemoryError:
                        gc.collect()
                        rec = {"status": "failed_memory"}
                    rec.update({"model": model, "candidate": ci, "fold": f})
                    ck.put(key, rec)
                    done_now += 1
                    el = time.perf_counter() - t_start
                    remaining = n_total - len(ck.done)
                    eta_h = el / done_now * remaining / 3600
                    print(f"  {model:13s} cand {ci:2d} fold {f} "
                          f"{rec.get('status')} F1(decision)={rec.get('macro_f1_decision_rows', float('nan')):.4f} "
                          f"({rec.get('fit_seconds', 0):.0f}s fit) | ~{eta_h:.1f} h left")
                rows.append({**{k: v for k, v in rec.items() if k != "key"}, **{f"param_{k}": v for k, v in params.items()}})

        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(OUT_DIR, f"{ds}_{model}_search_by_fold.csv"), index=False)
        ok = res.groupby("candidate")["status"].apply(lambda s: (s == "ok").all())
        agg = res[res["candidate"].isin(ok[ok].index)].groupby("candidate").agg(
            mean_macro_f1_decision_rows=("macro_f1_decision_rows", "mean"),
            sd_macro_f1_decision_rows=("macro_f1_decision_rows", "std"),
            mean_logloss_decision_rows=("logloss_decision_rows", "mean"),
            mean_macro_f1_all_rows=("macro_f1_all_validation_rows", "mean"),
            mean_fit_seconds=("fit_seconds", "mean"))
        if "macro_f1_capture35_present_classes" in res.columns:
            agg["mean_macro_f1_capture35"] = res[res["candidate"].isin(ok[ok].index)] \
                .groupby("candidate")["macro_f1_capture35_present_classes"].mean()
        agg = agg.sort_values(["mean_macro_f1_decision_rows"], ascending=False, kind="mergesort")
        agg.to_csv(os.path.join(OUT_DIR, f"{ds}_{model}_search_summary.csv"))
        failed = sorted(set(ok[~ok].index))
        best_c = int(agg.index[0])
        best = {"dataset": ds, "model": model, "candidate": best_c, "params": cands[best_c],
                "mean_macro_f1_decision_rows": float(agg.iloc[0]["mean_macro_f1_decision_rows"]),
                "sd_macro_f1_decision_rows": float(agg.iloc[0]["sd_macro_f1_decision_rows"]),
                "failed_candidates": failed, "n_candidates": N_CANDIDATES, "folds": K,
                "features": feat_cols, "selected_on": now(),
                "note": "non-nested random search; development score optimistically biased"}
        with open(os.path.join(OUT_DIR, f"{ds}_{model}_best_params.json"), "w") as f:
            json.dump(best, f, indent=2)
        summary[model] = best
        print(f"\n  {model}: top 5 candidates by mean macro-F1 on decision rows")
        print(agg.head(5).round(4).to_string())
        print(f"  SELECTED candidate {best_c}: {cands[best_c]}")
        if failed:
            print(f"  FAILED candidates (excluded, logged): {failed}")
        print()

    with open(os.path.join(OUT_DIR, f"{ds}_run_manifest.json"), "w") as f:
        json.dump({"dataset": ds, "finished": now(), "n_candidates": N_CANDIDATES,
                   "random_state": RANDOM_STATE, "selected": summary}, f, indent=2)


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["rt_iot2022", "iot23"]:
        if ds not in FOLD_COL:
            raise SystemExit(f"Unknown dataset '{ds}'.")
        run(ds)
    print("Done. Outputs: results/07_tuning/")