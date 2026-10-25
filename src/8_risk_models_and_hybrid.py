"""
8_risk_models_and_hybrid.py

Trains and evaluates four Phase-1 binary risk-estimation arrangements using
development data only. The test partitions are never opened.

The target is 'malicious' (0 = benign, 1 = malicious). Each arrangement
produces a risk score r = P(malicious) for each flow. Trusted, Suspicious,
and Untrusted levels are not assigned in this script; they are derived from
the risk scores using the cost-based thresholds in 9_decision_layer.py.

Arrangements:
    - xgboost_alone: XGBoost comparison model.
    - random_forest_alone: Random Forest comparison model.
    - hybrid_xgb_rf: primary hybrid, with XGBoost as the first stage and
      Random Forest as the second stage.
    - hybrid_rf_xgb: secondary hybrid, with the model roles reversed.

Both models use the tuned hyperparameters from 7_tuning.py. The same
FS-2-selected features and cross-validation folds are used. Hybrid models
use stacked generalisation with out-of-fold first-stage risk scores to
train the second stage, avoiding the use of in-sample first-stage scores
as training inputs.

For each outer fold, the first stage generates out-of-fold scores for the
other folds, which are combined with the selected features to train the
second stage. The fitted second stage then predicts the outer validation
fold using risk scores from the first stage trained on the other folds.
Pooled out-of-fold predictions provide a development-set risk score for
every row.

Final single models are fitted on all development rows. Final hybrid
second stages are trained using the selected features and the corresponding
single-model out-of-fold risk scores; their first stages are the single
models refitted on all development rows. Class balancing is applied during
training.

Evaluation includes threshold-free risk-score metrics:
    - ROC-AUC and average precision.
    - Log-loss and Brier score.
    - Mean predicted risk.

Reference classification metrics are also calculated at r >= 0.5, including
accuracy, per-class precision, recall and F1, macro-F1, Matthews correlation
coefficient, and confusion matrices. These reference metrics do not assign
the final trust levels.

For IoT-23, metrics are reported for unseen-capture decision rows,
Capture-35-1 rows, and all rows. For RT-IoT2022, all rows form the evaluation
group. Results also include per-fold metrics and risk-score summaries by
attack type or detailed label on the decision rows.

The script measures single-flow prediction latency (median and 95th
percentile), batch prediction throughput, and model size on disk for each
arrangement. Latency measurements use up to 1,000 randomly selected
development rows and single-threaded prediction.

Fold predictions are cached as NumPy files and reused when available,
allowing completed fold computations to be reused after interruption.

Usage:
    python src/8_risk_models_and_hybrid.py rt_iot2022
    python src/8_risk_models_and_hybrid.py iot23
"""

import os
import sys
import gc
import json
import time
import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, precision_recall_fscore_support, accuracy_score,
                             matthews_corrcoef, log_loss, confusion_matrix,
                             roc_auc_score, average_precision_score)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FS_DIR = os.path.join(PROJECT_ROOT, "results", "05_feature_selection")
TUNE_DIR = os.path.join(PROJECT_ROOT, "results", "07_tuning")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "08_risk_models")
NPY_DIR = os.path.join(OUT_DIR, "fold_predictions")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
for d in (OUT_DIR, NPY_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
TARGET, ROW_ID = "malicious", "row_id"          # 0 = benign, 1 = malicious
CLASS_CODE = {"benign": 0, "malicious": 1}
CLASS_NAME = {v: k for k, v in CLASS_CODE.items()}
LABELS = [0, 1]
FOLD_COL = {"iot23": "fold_B_partial_grouped", "rt_iot2022": "fold"}
FINE_LABEL = {"iot23": "detailed_label", "rt_iot2022": "attack_type"}
NON_FEATURES = {ROW_ID, TARGET, "fold", "scenario_group", "fold_A_rowlevel", "fold_B_partial_grouped"}
SINGLE = {"xgboost_alone": "xgboost", "random_forest_alone": "random_forest"}
HYBRIDS = {"hybrid_xgb_rf": ("xgboost", "random_forest"),
           "hybrid_rf_xgb": ("random_forest", "xgboost")}
ARRANGEMENTS = list(SINGLE) + list(HYBRIDS)


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")

# data and models

def load(ds):
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ds}_train_selected.csv"), low_memory=False)
    folds = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ds}_cv_folds.csv"))
    df = df.merge(folds, on=ROW_ID, how="left", validate="one_to_one")
    fine = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ds}_train.csv"),
                       usecols=[ROW_ID, FINE_LABEL[ds]], low_memory=False)
    df = df.merge(fine, on=ROW_ID, how="left", validate="one_to_one")
    with open(os.path.join(FS_DIR, f"{ds}_selection.json")) as f:
        cols = json.load(f)["selected_columns"]
    feat = [c for c in df.columns if c not in NON_FEATURES | {FINE_LABEL[ds]}]
    if feat != cols:
        raise ValueError(f"Feature mismatch: {feat} vs FS-2 {cols}")
    X = df[cols].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.int64)
    fold = df[FOLD_COL[ds]].to_numpy(dtype=np.int64)
    groups = {"all": np.ones(len(df), dtype=bool)}
    if ds == "iot23":
        groups["unseen_capture_rows"] = (df["scenario_group"] == "whole_scenario").to_numpy()
        groups["capture35_rows"] = ~groups["unseen_capture_rows"]
    return df, X, y, fold, groups, cols


def tuned_params(ds, model):
    with open(os.path.join(TUNE_DIR, f"{ds}_{model}_best_params.json")) as f:
        return json.load(f)["params"]


def make(model, params, n_jobs=-1):
    if model == "random_forest":
        return RandomForestClassifier(**params, class_weight="balanced",
                                      random_state=RANDOM_STATE, n_jobs=n_jobs)
    return XGBClassifier(**params, tree_method="hist", eval_metric="logloss",
                         random_state=RANDOM_STATE, n_jobs=n_jobs)


def fit(model, params, X, y):
    if set(np.unique(y)) != set(LABELS):
        raise RuntimeError("Training set lacks a class.")
    m = make(model, params)
    if model == "xgboost":
        m.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        m.fit(X, y)
    if list(m.classes_) != LABELS:
        raise RuntimeError(f"Unexpected class order {m.classes_}")
    return m


def risk(model, X):
    """Risk score r = P(malicious) as a column vector (n, 1)."""
    return model.predict_proba(X)[:, [1]].astype(np.float64)


def cached(path, compute):
    if os.path.exists(path):
        return np.load(path)
    arr = compute()
    np.save(path, arr)
    return arr


# out-of-fold predictions
def single_oof(ds, model, params, X, y, fold):
    oof = np.zeros((len(y), 1), dtype=np.float64)
    for f in np.unique(fold):
        def compute():
            t0 = time.perf_counter()
            m = fit(model, params, X[fold != f], y[fold != f])
            p = risk(m, X[fold == f])
            print(f"    {model:13s} fold {f}: fit+predict {time.perf_counter() - t0:.0f}s")
            del m
            gc.collect()
            return p
        oof[fold == f] = cached(os.path.join(NPY_DIR, f"{ds}_{model}_alone_fold{f}.npy"), compute)
    return oof


def hybrid_oof(ds, name, lookout, bouncer, params, X, y, fold, single_oofs):
    """lookout = first stage, bouncer = second stage."""
    oof = np.zeros((len(y), 1), dtype=np.float64)
    for f in np.unique(fold):
        def compute():
            T = fold != f
            inner = np.zeros((len(y), 1), dtype=np.float64)
            for g in np.unique(fold[T]):
                def inner_compute():
                    t0 = time.perf_counter()
                    m = fit(lookout, params[lookout], X[T & (fold != g)], y[T & (fold != g)])
                    p = risk(m, X[fold == g])
                    print(f"    {name} outer {f} inner {g}: first stage {time.perf_counter() - t0:.0f}s")
                    del m
                    gc.collect()
                    return p
                inner[fold == g] = cached(
                    os.path.join(NPY_DIR, f"{ds}_{name}_outer{f}_inner{g}.npy"), inner_compute)
            t0 = time.perf_counter()
            b = fit(bouncer, params[bouncer], np.hstack([X[T], inner[T]]).astype(np.float32), y[T])
            # lookout trained on all of T == single-model fold-f model -> reuse its predictions
            Zf = np.hstack([X[fold == f], single_oofs[lookout][fold == f]]).astype(np.float32)
            p = risk(b, Zf)
            print(f"    {name} outer {f}: second stage {time.perf_counter() - t0:.0f}s")
            del b
            gc.collect()
            return p
        oof[fold == f] = cached(os.path.join(NPY_DIR, f"{ds}_{name}_fold{f}.npy"), compute)
    return oof

# metrics
def normalise(r):
    """Risk score as a 1-D float64 array clipped to [0, 1]."""
    return np.clip(np.asarray(r, dtype=np.float64).reshape(-1), 0.0, 1.0)


def metric_block(y, r):
    r = normalise(r)
    pred = (r >= 0.5).astype(np.int64)
    present = sorted(set(y))
    both = len(present) == 2
    p, rc, f1, s = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    rr = np.clip(r, 1e-15, 1 - 1e-15)
    out = {"n": int(len(y)),
           "roc_auc": float(roc_auc_score(y, r)) if both else float("nan"),
           "average_precision": float(average_precision_score(y, r)) if both else float("nan"),
           "log_loss": float(log_loss(y, np.column_stack([1 - rr, rr]), labels=LABELS)),
           "brier": float(((r - y) ** 2).mean()),
           "mean_risk": float(r.mean()),
           "accuracy_at_0.5": float(accuracy_score(y, pred)),
           "macro_f1_at_0.5_present_classes": float(f1_score(y, pred, labels=present, average="macro",
                                                             zero_division=0)),
           "mcc_at_0.5": float(matthews_corrcoef(y, pred))}
    for k in LABELS:
        if s[k] or (pred == k).any():
            out[f"precision_{CLASS_NAME[k]}"] = float(p[k])
            out[f"recall_{CLASS_NAME[k]}"] = float(rc[k])
            out[f"f1_{CLASS_NAME[k]}"] = float(f1[k])
    out["confusion_rows_true_BM_cols_pred_BM_at_0.5"] = confusion_matrix(y, pred, labels=LABELS).tolist()
    return out

# final models and latency
def final_models(ds, params, X, y, single_oofs):
    paths = {}
    for arr, model in SINGLE.items():
        p = os.path.join(MODEL_DIR, f"{ds}_{arr}.joblib")
        if not os.path.exists(p):
            t0 = time.perf_counter()
            joblib.dump(fit(model, params[model], X, y), p)
            print(f"    final {arr}: {time.perf_counter() - t0:.0f}s")
        paths[arr] = p
    for name, (lookout, bouncer) in HYBRIDS.items():
        p = os.path.join(MODEL_DIR, f"{ds}_{name}_bouncer.joblib")
        if not os.path.exists(p):
            t0 = time.perf_counter()
            Z = np.hstack([X, single_oofs[lookout]]).astype(np.float32)
            joblib.dump(fit(bouncer, params[bouncer], Z, y), p)
            print(f"    final {name} second stage: {time.perf_counter() - t0:.0f}s")
        paths[name] = p
    return paths


def latency(ds, paths, X):
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(1000, len(X)), replace=False)
    loaded = {}
    for arr in SINGLE:
        m = joblib.load(paths[arr])
        m.set_params(n_jobs=1)
        loaded[arr] = m
    rows = []
    for arr in ARRANGEMENTS:
        if arr in SINGLE:
            predict = loaded[arr].predict_proba
            size = os.path.getsize(paths[arr])
        else:
            lookout = loaded[f"{HYBRIDS[arr][0]}_alone"]
            b = joblib.load(paths[arr])
            b.set_params(n_jobs=1)
            predict = (lambda xx, lk=lookout, bb=b:
                       bb.predict_proba(np.hstack([xx, lk.predict_proba(xx)[:, [1]]]).astype(np.float32)))
            size = os.path.getsize(paths[f"{HYBRIDS[arr][0]}_alone"]) + os.path.getsize(paths[arr])
        times = []
        for i in idx:
            t0 = time.perf_counter()
            predict(X[i:i + 1])
            times.append((time.perf_counter() - t0) * 1000)
        nb = min(100_000, len(X))
        t0 = time.perf_counter()
        predict(X[:nb])
        batch_s = time.perf_counter() - t0
        rows.append({"arrangement": arr, "single_flow_ms_median": float(np.median(times)),
                     "single_flow_ms_p95": float(np.percentile(times, 95)),
                     "batch_rows": nb, "batch_flows_per_second": nb / batch_s,
                     "model_size_mb": size / 1e6, "n_jobs": 1})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------

def run(ds):
    print("=" * 70 + f"\nRISK MODELS + HYBRID: {ds} (development data only)\n" + "=" * 70)
    df, X, y, fold, groups, cols = load(ds)
    params = {m: tuned_params(ds, m) for m in ("xgboost", "random_forest")}
    print(f"  rows {len(y):,} | features {len(cols)} | folds {sorted(set(fold))}")
    print(f"  tuned params: {params}")

    print("\n  [1] single-model out-of-fold probabilities")
    oofs = {}
    single_by_model = {}
    for arr, model in SINGLE.items():
        oofs[arr] = single_oof(ds, model, params[model], X, y, fold)
        single_by_model[model] = oofs[arr]

    print("\n  [2] hybrid out-of-fold probabilities (nested cross-fitting)")
    for name, (lookout, bouncer) in HYBRIDS.items():
        oofs[name] = hybrid_oof(ds, name, lookout, bouncer, params, X, y, fold, single_by_model)
    oofs = {arr: normalise(p) for arr, p in oofs.items()}

    # save pooled out-of-fold risk scores
    out = pd.DataFrame({ROW_ID: df[ROW_ID].to_numpy(), "fold": fold,
                        "true_class": y, FINE_LABEL[ds]: df[FINE_LABEL[ds]].to_numpy()})
    if ds == "iot23":
        out["scenario_group"] = df["scenario_group"].to_numpy()
    for arr in ARRANGEMENTS:
        out[f"{arr}_risk"] = oofs[arr].astype(np.float32)
    oof_path = os.path.join(OUT_DIR, f"{ds}_oof_probabilities.csv")
    out.to_csv(oof_path, index=False, float_format="%.6f")

    print("\n  [3] metrics of the risk score (pooled out-of-fold and per fold)")
    rows, conf, fine_rows = [], {}, []
    for arr in ARRANGEMENTS:
        for g, mask in groups.items():
            rows.append({"arrangement": arr, "group": g, "fold": "pooled",
                         **{k: v for k, v in metric_block(y[mask], oofs[arr][mask]).items()
                            if not k.startswith("confusion")}})
            conf[f"{arr}|{g}"] = metric_block(y[mask], oofs[arr][mask])["confusion_rows_true_BM_cols_pred_BM_at_0.5"]
            for f in np.unique(fold):
                m2 = mask & (fold == f)
                if m2.sum():
                    rows.append({"arrangement": arr, "group": g, "fold": int(f),
                                 **{k: v for k, v in metric_block(y[m2], oofs[arr][m2]).items()
                                    if not k.startswith("confusion")}})
        dec = groups.get("unseen_capture_rows", groups["all"])
        fr = pd.DataFrame({FINE_LABEL[ds]: df.loc[dec, FINE_LABEL[ds]].to_numpy(),
                           "true_class": y[dec], "risk": oofs[arr][dec]})
        t = fr.groupby(FINE_LABEL[ds]).agg(true_class=("true_class", "first"), n=("risk", "size"),
                                           mean_risk=("risk", "mean"), median_risk=("risk", "median"),
                                           share_risk_ge_0_5=("risk", lambda v: float((v >= 0.5).mean())))
        t.insert(0, "arrangement", arr)
        fine_rows.append(t.round(4).reset_index())
    met = pd.DataFrame(rows)
    met.to_csv(os.path.join(OUT_DIR, f"{ds}_metrics.csv"), index=False)
    pd.concat(fine_rows).to_csv(os.path.join(OUT_DIR, f"{ds}_predictions_by_fine_label.csv"), index=False)
    with open(os.path.join(OUT_DIR, f"{ds}_confusion_matrices.json"), "w") as f:
        json.dump(conf, f, indent=1)

    print("\n  [4] final models (all development rows)")
    paths = final_models(ds, params, X, y, single_by_model)

    print("\n  [5] latency and size")
    lat = latency(ds, paths, X)
    lat.to_csv(os.path.join(OUT_DIR, f"{ds}_latency.csv"), index=False)

    show = ["arrangement", "group", "roc_auc", "average_precision", "log_loss", "brier",
            "macro_f1_at_0.5_present_classes", "mcc_at_0.5", "recall_benign", "recall_malicious"]
    pooled = met[met["fold"] == "pooled"]
    print("\n  Pooled out-of-fold metrics:")
    print(pooled[[c for c in show if c in pooled.columns]].round(4).to_string(index=False))
    fold_view = met[met["fold"] != "pooled"].pivot_table(
        index=["arrangement", "group"], columns="fold", values="roc_auc")
    print("\n  ROC-AUC per fold:\n" + fold_view.round(4).to_string())
    for key, cm in conf.items():
        if key.endswith("unseen_capture_rows") or ds == "rt_iot2022":
            print(f"\n  Confusion at r >= 0.5, {key} (rows true benign/malicious, cols predicted):\n  {np.array(cm)}")
    fine = pd.concat(fine_rows)
    print("\n  Mean risk score by attack type on decision rows:\n" +
          fine.pivot_table(index=FINE_LABEL[ds], columns="arrangement", values="mean_risk").round(3).to_string())
    print("\n  Latency / size:\n" + lat.round(4).to_string(index=False))
    with open(os.path.join(OUT_DIR, f"{ds}_run_manifest.json"), "w") as f:
        json.dump({"dataset": ds, "finished": now(), "params": params,
                   "final_models": paths, "oof_file": oof_path}, f, indent=2)
    print(f"\n  Saved: {oof_path}\n")


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["rt_iot2022", "iot23"]:
        if ds not in FOLD_COL:
            raise SystemExit(f"Unknown dataset '{ds}'.")
        run(ds)
    print("Done. Outputs: results/08_risk_models/, models/")