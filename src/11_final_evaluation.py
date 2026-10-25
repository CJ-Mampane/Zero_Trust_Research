"""
11_final_evaluation.py

FINAL EVALUATION on the 30% test partitions, using ONLY frozen artefacts:

    models/{ds}_xgboost_alone.joblib, {ds}_random_forest_alone.joblib,
    models/{ds}_hybrid_xgb_rf_bouncer.joblib, {ds}_hybrid_rf_xgb_bouncer.joblib   (script 8;
        "bouncer" = the hybrid's second stage)
    results/05_feature_selection/{ds}_selection.json                               (FS-2 encoding)
    results/09_decision_layer/{ds}_thresholds.csv     rows chosen on "all development folds"
    results/09_decision_layer/{ds}_sensitivity.csv    thresholds for every (b, rho, C)
    results/10_trust_lifecycle/iot23_lifecycle_parameters.csv   same

Target: the datasets' own binary label 'malicious'. Every arrangement gives a
risk score r = P(malicious); the trust levels Trusted / Suspicious / Untrusted
come only from the frozen thresholds (Allow / Restrict / Reject).

Nothing is fitted, tuned, recalibrated or selected here. Every decision rule,
metric and lifecycle function is IMPORTED from scripts 8, 9 and 10, so the test
evaluation uses the identical code that produced the development results.

DISCLOSURE (also written to the run folder): the test partitions were
examined before, including a completed final evaluation of an earlier version
of this method. This run is a SUBSEQUENT, DISCLOSED evaluation, not an
untouched confirmatory test.

What it does
------------
[0] Integrity: SHA-256 manifest of every frozen input. Checks: models load with
    class order [0, 1] and the expected input width; thresholds / parameters
    are complete; development re-encoding with the FS-2 spec reproduces
    {ds}_train_selected.csv exactly; partitions are disjoint and complete.
[1] Multiplicity-weighting verification (as before): weighted results are
    written ONLY if every check passes.
[2] Per-request evaluation, four arrangements, both datasets:
    Phase 1 (risk score): ROC-AUC, average precision, log-loss, Brier; at
    r >= 0.5 for reference: accuracy, macro-F1, MCC, per-class recall.
    Phase 2: reference without a trust layer (r >= 0.5 -> Reject, else Allow)
    + frozen cost-minimising thresholds at C in {1, 2, 5, 10, 20} (primary
    b = rho = 1/3): false trust, over-restriction, share of flows per trust
    level, share malicious within each level, mean cost.
    Sensitivity: the frozen thresholds of every (b, rho, C) from script 9
    applied to the test data (point estimates).
    Groups: RT all rows; IoT-23 Capture-35-1 test rows, wholly held-out
    captures (pooled), all rows, and every held-out capture separately.
    Trust level reached by each attack type / detailed label.
    95% percentile bootstrap intervals (B = 1000, seed 42): row resampling for
    RT and Capture-35-1; CLUSTER resampling of the held-out captures; both
    combined for IoT-23 all rows. ROC-AUC intervals use 1,000-bin score
    histograms (point estimates are exact).
[3] Trust lifecycle (IoT-23 held-out captures, internal devices only), as
    before, with the single risk score.
[4] Thesis tables/figures, METRIC_DEFINITIONS.md, DISCLOSURE.md, manifest,
    COMPLETED marker with output hashes.

Re-running: allowed and never destructive. Each run writes a new folder
results/11_final_evaluation/run_<timestamp>/ (older runs are never touched).
If a COMPLETED run already exists, the script refuses to start unless
--confirm-rerun "<reason>" is given; the reason, time and input hashes are
appended to results/11_final_evaluation/run_log.jsonl. This prevents an
accidental second look without making recovery impossible (e.g. after a crash
or power cut a run without COMPLETED does not block a restart).

Usage (from the project root):
    python src/11_final_evaluation.py --verify-only      # checks + manifest; opens NO test labels
    python src/11_final_evaluation.py                    # full evaluation
    python src/11_final_evaluation.py --skip-lifecycle   # per-request only (minutes)
    python src/11_final_evaluation.py --confirm-rerun "reason"   # only if a completed run exists
"""

import os
import sys
import gc
import json
import time
import hashlib
import platform
import argparse
import datetime
import importlib.util

import joblib
import numpy as np
import pandas as pd
from scipy.signal import lfilter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
INTERIM_SCEN = os.path.join(PROJECT_ROOT, "data", "interim", "iot23_scenarios")
CLEAN_SCEN = os.path.join(PROJECT_ROOT, "data", "interim", "iot23_clean_scenarios")
RESULTS = os.path.join(PROJECT_ROOT, "results")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUT_ROOT = os.path.join(RESULTS, "11_final_evaluation")
CACHE_DIR = os.path.join(OUT_ROOT, "replay_cache")

RANDOM_STATE = 42
CLASSES = ["benign", "malicious"]                 # dataset labels (truth)
LEVELS = ["Trusted", "Suspicious", "Untrusted"]   # derived from the risk score
ACTIONS = ["Allow", "Restrict", "Reject"]
AUC_BINS = 1000
COST_RATIOS = [1, 2, 5, 10, 20]
CARVE = "CTU-IoT-Malware-Capture-35-1"
FINAL_ROWS = "all development folds"
ROWSETS = {"rt_iot2022": ["all"], "iot23": ["unseen_capture_rows", "all"]}   # first = primary
# flows per vectorised block in the lifecycle replay, and the capture size up to which
# script 10's own code is also run as a cross-check (environment overrides exist only
# for testing the chunked code path on small data; they do not change any result)
LEAN_CHUNK = int(os.environ.get("ZT_LEAN_CHUNK", 4_000_000))
REFERENCE_MAX_FLOWS = int(os.environ.get("ZT_REFERENCE_MAX_FLOWS", 3_000_000))



# imports of the frozen pipeline code

def import_script(alias, filename):
    path = os.path.join(SCRIPT_DIR, filename)
    if not os.path.exists(path):
        raise SystemExit(f"Required pipeline script not found: {path}")
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S8 = import_script("s8_risk_models", "8_risk_models_and_hybrid.py")
S9 = import_script("s9_decision_layer", "9_decision_layer.py")
S10 = import_script("s10_trust_lifecycle", "10_trust_lifecycle.py")
def normalise(r):
    """Risk score as a 1-D float64 array clipped to [0, 1] (as script 8)."""
    return np.clip(np.asarray(r, dtype=np.float64).reshape(-1), 0.0, 1.0)


ARRANGEMENTS = list(S8.ARRANGEMENTS)          # xgboost_alone, random_forest_alone, hybrid_xgb_rf, hybrid_rf_xgb
FINE = S8.FINE_LABEL                          # {'iot23': 'detailed_label', 'rt_iot2022': 'attack_type'}
assert S9.COST_RATIOS == COST_RATIOS and S10.COST_RATIOS == COST_RATIOS
assert sorted(S9.ARRANGEMENTS) == sorted(ARRANGEMENTS) == sorted(S10.ARRANGEMENTS)


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def rel(p):
    return os.path.relpath(p, PROJECT_ROOT).replace("\\", "/")


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class Log:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *msg):
        line = " ".join(str(m) for m in msg)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


# ======================================================================
# [0] integrity
# ======================================================================

def frozen_paths(datasets):
    files = []
    for ds in datasets:
        files += [os.path.join(MODEL_DIR, f"{ds}_{a}.joblib") for a in S8.SINGLE]
        files += [os.path.join(MODEL_DIR, f"{ds}_{h}_bouncer.joblib") for h in S8.HYBRIDS]
        files += [os.path.join(RESULTS, "05_feature_selection", f"{ds}_selection.json"),
                  os.path.join(RESULTS, "09_decision_layer", f"{ds}_thresholds.csv"),
                  os.path.join(RESULTS, "09_decision_layer", f"{ds}_sensitivity.csv"),
                  os.path.join(RESULTS, "07_tuning", f"{ds}_xgboost_best_params.json"),
                  os.path.join(RESULTS, "07_tuning", f"{ds}_random_forest_best_params.json"),
                  os.path.join(PROCESSED_DIR, f"{ds}_train.csv"),
                  os.path.join(PROCESSED_DIR, f"{ds}_test.csv"),
                  os.path.join(PROCESSED_DIR, f"{ds}_cv_folds.csv"),
                  os.path.join(PROCESSED_DIR, f"{ds}_train_selected.csv")]
    if "iot23" in datasets:
        files += [os.path.join(RESULTS, "10_trust_lifecycle", "iot23_lifecycle_parameters.csv"),
                  os.path.join(RESULTS, "03_split", "iot23_split_summary.json")]
    for d in ("src", "config", "docs"):
        root = os.path.join(PROJECT_ROOT, d)
        if os.path.isdir(root):
            for name in sorted(os.listdir(root)):
                p = os.path.join(root, name)
                if os.path.isfile(p) and not name.endswith(".pyc"):
                    files.append(p)
    lock = os.path.join(RESULTS, "05_feature_selection", "protocol_lock.json")
    if os.path.exists(lock):
        files.append(lock)
    return files


def environment():
    import sklearn
    import scipy
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "processor": platform.processor(), "cpu_count": os.cpu_count(),
           "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
           "scipy": scipy.__version__, "joblib": joblib.__version__}
    try:
        import xgboost
        env["xgboost"] = xgboost.__version__
    except Exception as e:                       # noqa: BLE001
        env["xgboost"] = f"not importable: {e}"
    try:
        import psutil
        env["total_ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:                            # noqa: BLE001
        env["total_ram_gb"] = "unknown"
    return env


def build_manifest(datasets, log):
    log("\n[0] fingerprinting frozen inputs (SHA-256)")
    missing, rows = [], []
    for p in frozen_paths(datasets):
        if not os.path.exists(p):
            missing.append(rel(p))
            continue
        t0 = time.perf_counter()
        rows.append({"file": rel(p), "bytes": os.path.getsize(p), "sha256": sha256(p),
                     "modified": datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")})
        if time.perf_counter() - t0 > 5:
            log(f"  hashed {rel(p)} ({rows[-1]['bytes'] / 1e6:.0f} MB)")
    must = [m for m in missing if not m.startswith(("docs/", "config/"))]
    if must:
        raise SystemExit(f"Frozen inputs missing: {must}")
    interim = []
    if "iot23" in datasets and os.path.isdir(INTERIM_SCEN):
        for name in sorted(os.listdir(INTERIM_SCEN)):
            p = os.path.join(INTERIM_SCEN, name)
            if name.endswith(".csv"):
                interim.append({"file": rel(p), "bytes": os.path.getsize(p),
                                "modified": datetime.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds"),
                                "note": "size/mtime only (tens of GB); content verified by the replay checks"})
    log(f"  {len(rows)} files hashed")
    return {"files": rows, "interim_raw_flow_files": interim, "environment": environment()}


def load_models(ds, n_cols):
    m = {}
    for arr in S8.SINGLE:
        mod = joblib.load(os.path.join(MODEL_DIR, f"{ds}_{arr}.joblib"))
        if list(mod.classes_) != [0, 1]:
            raise SystemExit(f"{ds} {arr}: class order {list(mod.classes_)} != [0, 1]")
        if int(mod.n_features_in_) != n_cols:
            raise SystemExit(f"{ds} {arr}: expects {mod.n_features_in_} features, FS-2 gives {n_cols}")
        m[arr] = mod
    for h in S8.HYBRIDS:
        b = joblib.load(os.path.join(MODEL_DIR, f"{ds}_{h}_bouncer.joblib"))
        if list(b.classes_) != [0, 1]:
            raise SystemExit(f"{ds} {h} second stage: class order {list(b.classes_)}")
        if int(b.n_features_in_) != n_cols + 1:
            raise SystemExit(f"{ds} {h} second stage: expects {b.n_features_in_} inputs, needs {n_cols + 1}")
        m[h] = b
    return m



def predict_all(models, X):
    """Exactly as in script 8: first stage = the refit single model; second-stage
    input = [features, first-stage risk score] as float32. Returns r = P(malicious)."""
    out = {}
    for arr in S8.SINGLE:
        out[arr] = normalise(models[arr].predict_proba(X)[:, 1])
    for h, (lookout, _) in S8.HYBRIDS.items():
        look = models[f"{lookout}_alone"].predict_proba(X)[:, [1]]
        out[h] = normalise(models[h].predict_proba(np.hstack([X, look]).astype(np.float32))[:, 1])
    return out



def encode(df, spec):
    """FS-2 encoding (identical to 5_feature_selection.encode)."""
    arrays, names = [], []
    for v in spec["selected_variables"]:
        if v in spec["numeric_variables"]:
            arrays.append(df[v].to_numpy(dtype=np.float32))
            names.append(v)
        else:
            s = df[v].astype(object).where(df[v].notna(), "<missing>").astype(str).to_numpy()
            for lvl in spec["categorical_levels"][v]:
                arrays.append((s == lvl).astype(np.float32))
                names.append(f"{v}={lvl}")
    if names != spec["selected_columns"]:
        raise SystemExit(f"Encoding produced {names}, spec says {spec['selected_columns']}")
    return np.column_stack(arrays).astype(np.float32, copy=False)


def load_spec(ds):
    with open(os.path.join(RESULTS, "05_feature_selection", f"{ds}_selection.json")) as f:
        return json.load(f)


def check_encoder(ds, spec, log):
    """Re-encode the development partition and compare with the saved
    {ds}_train_selected.csv: proves the test encoding uses the same rule."""
    dev = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ds}_train.csv"), low_memory=False)
    saved = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ds}_train_selected.csv"), low_memory=False)
    if not np.array_equal(dev["row_id"].to_numpy(), saved["row_id"].to_numpy()):
        raise SystemExit(f"{ds}: row order of train.csv and train_selected.csv differs")
    X = encode(dev, spec)
    S = saved[spec["selected_columns"]].to_numpy(dtype=np.float32)
    if not np.array_equal(X, S):
        bad = int((X != S).sum())
        raise SystemExit(f"{ds}: re-encoding development data differs from train_selected in {bad} cells")
    log(f"  {ds}: FS-2 encoder reproduces train_selected exactly ({X.shape[0]:,} x {X.shape[1]})")
    del dev, saved, X, S
    gc.collect()


def load_thresholds(ds):
    t = pd.read_csv(os.path.join(RESULTS, "09_decision_layer", f"{ds}_thresholds.csv"))
    t = t[t["chosen_on"] == FINAL_ROWS]
    out = {}
    for arr in ARRANGEMENTS:
        for rs in ROWSETS[ds]:
            for r in COST_RATIOS:
                m = t[(t["arrangement"] == arr) & (t["threshold_rows"] == rs) & (t["cost_ratio"] == r)]
                if len(m) != 1:
                    raise SystemExit(f"{ds}: {len(m)} frozen threshold rows for {arr}/{rs}/ratio {r}")
                out[(arr, rs, r)] = (float(m["tau1_restrict"].iloc[0]), float(m["tau2_reject"].iloc[0]))
    return out


def load_sensitivity(ds):
    """Frozen thresholds of script 9's sensitivity analysis (primary selection rows)."""
    t = pd.read_csv(os.path.join(RESULTS, "09_decision_layer", f"{ds}_sensitivity.csv"))
    out = {}
    for _, x in t.iterrows():
        out[(x["arrangement"], float(x["b"]), float(x["rho"]), int(x["cost_ratio"]))] = (
            float(x["tau1_restrict"]), float(x["tau2_reject"]))
    return out



def load_lifecycle_params():
    p = pd.read_csv(os.path.join(RESULTS, "10_trust_lifecycle", "iot23_lifecycle_parameters.csv"))
    p = p[p["chosen_on"] == FINAL_ROWS]
    out = {}
    for arr in ARRANGEMENTS:
        for r in COST_RATIOS:
            for rule in ("contextual", "singular"):
                m = p[(p["arrangement"] == arr) & (p["cost_ratio"] == r) & (p["rule"] == rule)]
                if len(m) != 1:
                    raise SystemExit(f"lifecycle: {len(m)} frozen rows for {arr}/{r}/{rule}")
                row = m.iloc[0]
                out[(arr, r, rule)] = {"lambda": float(row["lambda"]), "probation_m": int(row["probation_m"]),
                                       "tau1": float(row["tau1"]), "tau2": float(row["tau2"])}
                if rule == "singular" and (out[(arr, r, rule)]["lambda"] != 0 or out[(arr, r, rule)]["probation_m"] != 0):
                    raise SystemExit(f"lifecycle: singular rule with lambda/probation != 0 for {arr}/{r}")
    return out



# [1] multiplicity verification

def verify_multiplicity(ds, train_mult, test_df, log):
    """Returns (ok, details). train_mult: DataFrame with scenario (IoT) + row_multiplicity."""
    det = {}
    ok = True
    if ds == "rt_iot2022":
        with open(os.path.join(RESULTS, "02_cleaning", "rt_iot2022_cleaning_summary.json")) as f:
            raw = int(json.load(f)["rows_raw"])
        tot = int(train_mult["row_multiplicity"].sum()) + int(test_df["row_multiplicity"].sum())
        ok = tot == raw
        det = {"rows_raw_from_cleaning": raw, "sum_multiplicity_dev_plus_test": tot}
    else:
        both = pd.concat([train_mult[["scenario", "row_multiplicity"]],
                          test_df[["scenario", "row_multiplicity"]]])
        got = both.groupby("scenario")["row_multiplicity"].sum()
        per = []
        for scen, s in got.items():
            sp = os.path.join(CLEAN_SCEN, f"{scen}.stats.json")
            if not os.path.exists(sp):
                per.append({"scenario": scen, "ok": False, "reason": f"missing {rel(sp)}"})
                ok = False
                continue
            with open(sp) as f:
                st = json.load(f)
            exp = int(st["rows_raw"]) - int(st["rows_label_violations"])
            per.append({"scenario": scen, "expected": exp, "sum_multiplicity": int(s), "ok": int(s) == exp})
            ok &= int(s) == exp
        det = {"per_scenario": per}
    log(f"  {ds}: multiplicities reconcile with raw counts from cleaning: {ok}")
    return ok, det


# CM-based metrics (used for bootstrap and weighted results; checked
# against scripts 8/9 on the unweighted point estimates)

COST = {r: S9.cost_matrix(r, S9.B_PRIMARY, S9.RHO_PRIMARY) for r in COST_RATIOS}


def cm_classification(cm):
    """Phase-1 metrics at r >= 0.5 from a 2x2 matrix (rows true benign/malicious,
    cols predicted); names as in script 8's metric_block."""
    cm = np.asarray(cm, dtype=np.float64)
    n = cm.sum()
    tt, tp = cm.sum(axis=1), cm.sum(axis=0)
    out = {"n": n}
    if n == 0:
        return out
    diag = np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp > 0, diag / tp, 0.0)
        rec = np.where(tt > 0, diag / tt, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    present = tt > 0
    out["accuracy_at_0.5"] = diag.sum() / n
    out["macro_f1_at_0.5_present_classes"] = f1[present].mean()
    cov = diag.sum() * n - (tp * tt).sum()
    den = np.sqrt((n ** 2 - (tp ** 2).sum()) * (n ** 2 - (tt ** 2).sum()))
    out["mcc_at_0.5"] = cov / den if den > 0 else 0.0
    for k, c in enumerate(CLASSES):
        if tt[k] > 0 or tp[k] > 0:
            out[f"precision_{c}"] = prec[k]
            out[f"recall_{c}"] = rec[k]
            out[f"f1_{c}"] = f1[k]
    return out


def cm_decision(cm, cost=None):
    """Phase-2 metrics from a 2x3 matrix (rows true benign/malicious, cols
    Allow/Restrict/Reject); same definitions as 9_decision_layer.decision_metrics."""
    cm = np.asarray(cm, dtype=np.float64)
    n = cm.sum()
    tt = cm.sum(axis=1)
    nan = float("nan")
    out = {"n": n}
    if n == 0:
        return out
    out["false_trust_rate_allow_given_malicious"] = cm[1, 0] / tt[1] if tt[1] else nan
    out["malicious_restricted_rate"] = cm[1, 1] / tt[1] if tt[1] else nan
    out["malicious_rejected_rate"] = cm[1, 2] / tt[1] if tt[1] else nan
    out["over_restriction_rate_given_benign"] = (cm[0, 1] + cm[0, 2]) / tt[0] if tt[0] else nan
    out["benign_restricted_rate"] = cm[0, 1] / tt[0] if tt[0] else nan
    out["benign_rejected_rate"] = cm[0, 2] / tt[0] if tt[0] else nan
    col = cm.sum(axis=0)
    for k, lvl in enumerate(LEVELS):
        out[f"share_{lvl}"] = col[k] / n
        out[f"malicious_share_within_{lvl}"] = cm[1, k] / col[k] if col[k] else nan
    for r in COST_RATIOS:
        out[f"mean_cost_C{r}"] = (COST[r] * cm).sum() / n
    if cost is not None:
        out["mean_cost"] = (cost * cm).sum() / n
    return out


def row_losses(y, r):
    """Per-row log-loss and Brier score of the risk score, as in script 8."""
    rr = np.clip(r, 1e-15, 1 - 1e-15)
    ll = -(y * np.log(rr) + (1 - y) * np.log(1 - rr))
    br = (r - y) ** 2
    return np.stack([ll, br], axis=1)              # (n, 2)


LOSS_NAMES = ["log_loss", "brier"]


def score_bins(r):
    return np.minimum((np.asarray(r) * AUC_BINS).astype(np.int64), AUC_BINS - 1)


def auc_from_hist(h):
    """ROC-AUC from per-class score histograms h (2, bins); ties within a bin count 1/2."""
    neg, pos = h[0], h[1]
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return np.nan
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (P * N))


def close(a, b, tol=1e-9):
    if (a is None or (isinstance(a, float) and np.isnan(a))) and (b is None or (isinstance(b, float) and np.isnan(b))):
        return True
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


# [2] per-request evaluation

def variants(ds, thr):
    """Decision variants: reference without a trust layer + frozen thresholds."""
    v = [("no_trust_layer_r_ge_0.5_reject", "-", None, None, None)]
    for rs in ROWSETS[ds]:
        for r in COST_RATIOS:
            v.append(("cost_minimising_thresholds", rs, r) + (None, None))
    return v



def per_request(ds, models, spec, thr, sens_thr, log, run_dir, B, weights_ok_hint):
    log("\n" + "=" * 70 + f"\n[2] PER-REQUEST EVALUATION: {ds}\n" + "=" * 70)
    test = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ds}_test.csv"), low_memory=False)
    X = encode(test, spec)
    if test["malicious"].isna().any():
        raise SystemExit(f"{ds}: missing 'malicious' labels in test")
    y = test["malicious"].to_numpy(dtype=np.int64)
    if not set(np.unique(y)) <= {0, 1}:
        raise SystemExit(f"{ds}: test labels other than 0/1")
    mult = test["row_multiplicity"].to_numpy(dtype=np.float64)
    fine = test[FINE[ds]].astype(str).to_numpy()
    t0 = time.perf_counter()
    P = predict_all(models, X)
    log(f"  {len(y):,} test rows, {X.shape[1]} encoded columns; predictions in {time.perf_counter() - t0:.0f}s")

    # save risk scores (row-level, for audit)
    prob = pd.DataFrame({"row_id": test["row_id"].to_numpy(), "true_class": y, FINE[ds]: fine,
                         "row_multiplicity": test["row_multiplicity"].to_numpy()})
    if ds == "iot23":
        prob.insert(1, "scenario", test["scenario"].to_numpy())
    for arr in ARRANGEMENTS:
        prob[f"{arr}_risk"] = P[arr].astype(np.float32)
    prob_path = os.path.join(run_dir, f"{ds}_test_probabilities.csv")
    prob.to_csv(prob_path, index=False, float_format="%.6f")
    # Development decisions (scripts 9, 10) were made on risk scores stored by
    # script 8 as float32 with 6 decimals, then clipped. The test decisions go
    # through the identical storage step; Phase-1 metrics use the in-memory scores.
    stored = pd.read_csv(prob_path, low_memory=False)
    if not np.array_equal(stored["row_id"].to_numpy(), test["row_id"].to_numpy()):
        raise SystemExit(f"{ds}: stored risk file row order differs")
    Rdec = {arr: np.clip(stored[f"{arr}_risk"].to_numpy(dtype=np.float64), 0, 1) for arr in ARRANGEMENTS}
    del stored

    # groups
    groups = {"all": np.ones(len(y), dtype=bool)}
    caps = []
    if ds == "iot23":
        scen = test["scenario"].astype(str).to_numpy()
        groups["capture35_test_rows"] = scen == CARVE
        groups["heldout_captures"] = scen != CARVE
        caps = sorted(set(scen[scen != CARVE]))
        for c in caps:
            groups[f"capture:{c}"] = scen == c

    # actions per variant
    V = variants(ds, thr)
    actions = {}
    phase1_pred = {arr: (P[arr] >= 0.5).astype(np.int64) for arr in ARRANGEMENTS}
    for arr in ARRANGEMENTS:
        r = Rdec[arr]
        actions[(arr, "no_trust_layer_r_ge_0.5_reject", "-", None)] = (r >= 0.5).astype(np.int64) * 2
        for (_, rs, cr, *_rest) in V[1:]:
            t1, t2 = thr[(arr, rs, cr)]
            actions[(arr, "cost_minimising_thresholds", rs, cr)] = S9.decide(r, t1, t2)
    keys = list(actions)
    losses = {arr: row_losses(y, P[arr]) for arr in ARRANGEMENTS}
    hist_bin = {arr: score_bins(P[arr]) for arr in ARRANGEMENTS}

    # ---------------- point estimates ----------------
    p1_rows, dec_rows, cms = [], [], {}
    for g, gm in groups.items():
        for weighting in ("unweighted", "multiplicity_weighted"):
            w = np.ones(gm.sum()) if weighting == "unweighted" else mult[gm]
            yy = y[gm]
            both = len(set(yy)) == 2
            for arr in ARRANGEMENTS:
                pred = phase1_pred[arr][gm]
                cm = np.bincount(yy * 2 + pred, weights=w, minlength=4).reshape(2, 2)
                row = {"dataset": ds, "arrangement": arr, "group": g, "weighting": weighting,
                       **cm_classification(cm)}
                L = (losses[arr][gm] * w[:, None]).sum(axis=0) / w.sum()
                row.update(dict(zip(LOSS_NAMES, L)))
                row["roc_auc"] = float(S8.roc_auc_score(yy, P[arr][gm], sample_weight=w)) if both else np.nan
                row["average_precision"] = (float(S8.average_precision_score(yy, P[arr][gm], sample_weight=w))
                                            if both else np.nan)
                row["mean_risk"] = float((P[arr][gm] * w).sum() / w.sum())
                if weighting == "unweighted":           # cross-check with script 8's own function
                    ref = S8.metric_block(yy, P[arr][gm])
                    for k in ("accuracy_at_0.5", "macro_f1_at_0.5_present_classes", "mcc_at_0.5", "brier",
                              "roc_auc", "average_precision"):
                        if not close(row[k], ref[k], 1e-7):
                            raise SystemExit(f"metric mismatch vs script 8: {ds}/{arr}/{g}/{k}: {row[k]} vs {ref[k]}")
                    if not close(row["log_loss"], ref["log_loss"], 1e-6):
                        raise SystemExit(f"log-loss mismatch vs script 8: {row['log_loss']} vs {ref['log_loss']}")
                p1_rows.append(row)
                cms[f"phase1|{arr}|{g}|{weighting}"] = cm.tolist()
            for key in keys:
                arr, rule, rs, cr = key
                act = actions[key][gm]
                cm = np.bincount(yy * 3 + act, weights=w, minlength=6).reshape(2, 3)
                t1, t2 = (thr[(arr, rs, cr)] if rule == "cost_minimising_thresholds" else (None, None))
                row = {"dataset": ds, "arrangement": arr, "decision_rule": rule, "threshold_rows": rs,
                       "cost_ratio": cr, "tau1": t1, "tau2": t2, "group": g, "weighting": weighting,
                       **cm_decision(cm)}
                if weighting == "unweighted":           # cross-check with script 9's own function
                    ref = S9.decision_metrics(yy, act)
                    for k in ("false_trust_rate_allow_given_malicious", "over_restriction_rate_given_benign",
                              "benign_rejected_rate", "share_Suspicious", "mean_cost_C1", "mean_cost_C20"):
                        if not close(row[k], ref[k], 1e-9):
                            raise SystemExit(f"decision metric mismatch vs script 9: {key}/{g}/{k}: {row[k]} vs {ref[k]}")
                dec_rows.append(row)
                cms[f"decision|{arr}|{rule}|{rs}|{cr}|{g}|{weighting}"] = cm.tolist()
    log("  point estimates computed; CM-based metrics match scripts 8/9 exactly on unweighted data")

    # ---------------- sensitivity (frozen thresholds of every b, rho, C; point estimates) ----------------
    sens_rows = []
    sens_groups = ["all"] if ds == "rt_iot2022" else ["capture35_test_rows", "heldout_captures", "all"]
    for (arr, b, rho, cr), (t1, t2) in sens_thr.items():
        act_all = S9.decide(Rdec[arr], t1, t2)
        cost = S9.cost_matrix(cr, b, rho)
        for g in sens_groups:
            gm = groups[g]
            cm = np.bincount(y[gm] * 3 + act_all[gm], minlength=6).reshape(2, 3)
            sens_rows.append({"dataset": ds, "arrangement": arr, "b": b, "rho": rho, "b_plus_rho": round(b + rho, 4),
                              "cost_ratio": cr, "tau1": t1, "tau2": t2, "group": g, **cm_decision(cm, cost)})
    sens = pd.DataFrame(sens_rows)

    # ---------------- bootstrap ----------------
    ci_p1, ci_dec = {}, {}
    if B > 0:
        log(f"  bootstrap: B = {B} (row resampling for RT / Capture-35-1, cluster resampling of held-out captures)")
        t0 = time.perf_counter()
        ci_p1, ci_dec = bootstrap(ds, y, mult, actions, keys, phase1_pred, losses, hist_bin, groups, caps, B)
        log(f"  bootstrap done in {time.perf_counter() - t0:.0f}s")

    p1 = attach_ci(pd.DataFrame(p1_rows), ci_p1, ["arrangement", "group", "weighting"])
    dec = attach_ci(pd.DataFrame(dec_rows), ci_dec,
                    ["arrangement", "decision_rule", "threshold_rows", "cost_ratio", "group", "weighting"])

    # ---------------- trust level reached by each attack type ----------------
    dev_fine = set(pd.read_csv(os.path.join(PROCESSED_DIR, f"{ds}_train.csv"), usecols=[FINE[ds]],
                               low_memory=False)[FINE[ds]].astype(str))
    class_of = dict(zip(test[FINE[ds]].astype(str), y))
    sub_rows = []
    sub_groups = {k: v for k, v in groups.items() if not k.startswith("capture:")}
    primary_rs = ROWSETS[ds][0]
    for g, gm in sub_groups.items():
        for lab in sorted(set(fine[gm])):
            lm = gm & (fine == lab)
            for key in keys:
                arr, rule, rs, cr = key
                if rule == "cost_minimising_thresholds" and rs != primary_rs:
                    continue
                act = actions[key][lm]
                for weighting in ("unweighted", "multiplicity_weighted"):
                    w = np.ones(lm.sum()) if weighting == "unweighted" else mult[lm]
                    shares = np.bincount(act, weights=w, minlength=3) / w.sum()
                    sub_rows.append({"dataset": ds, "group": g, FINE[ds]: lab, "true_class": CLASSES[class_of[lab]],
                                     "in_development_data": lab in dev_fine, "n_rows": int(lm.sum()),
                                     "sum_multiplicity": float(mult[lm].sum()), "arrangement": arr,
                                     "decision_rule": rule, "threshold_rows": rs, "cost_ratio": cr,
                                     "weighting": weighting, "share_Trusted_allow": shares[0],
                                     "share_Suspicious_restrict": shares[1], "share_Untrusted_reject": shares[2],
                                     "mean_risk_hybrid_xgb_rf": float(P["hybrid_xgb_rf"][lm].mean())})
    sub = pd.DataFrame(sub_rows)
    return {"p1": p1, "dec": dec, "sub": sub, "sens": sens, "cms": cms, "n_test": len(y), "caps": caps}



def attach_ci(df, ci, keycols):
    if not ci:
        return df
    lo_cols, hi_cols = {}, {}
    for i, row in df.iterrows():
        key = tuple(row[k] if not (isinstance(row[k], float) and np.isnan(row[k])) else None for k in keycols)
        for metric, (lo, hi) in ci.get(key, {}).items():
            lo_cols.setdefault(f"{metric}_ci_low", {})[i] = lo
            hi_cols.setdefault(f"{metric}_ci_high", {})[i] = hi
    for c, vals in {**lo_cols, **hi_cols}.items():
        df[c] = pd.Series(vals)
    return df


CI_P1 = ["roc_auc", "accuracy_at_0.5", "macro_f1_at_0.5_present_classes", "mcc_at_0.5", "recall_benign",
         "recall_malicious"] + LOSS_NAMES
CI_DEC = ["false_trust_rate_allow_given_malicious", "over_restriction_rate_given_benign",
          "benign_rejected_rate", "malicious_restricted_rate", "share_Trusted", "share_Suspicious",
          "share_Untrusted", "malicious_share_within_Suspicious"] + [f"mean_cost_C{r}" for r in COST_RATIOS]


def bootstrap(ds, y, mult, actions, keys, phase1_pred, losses, hist_bin, groups, caps, B):
    """Joint resampling: in replicate b, ONE resample of rows/captures is used
    for every arrangement, variant and metric, so intervals are comparable.
    RT: rows. IoT-23: Capture-35-1 rows (rows) + held-out captures (clusters);
    'all' = both resamples of the same replicate combined. Each variant's
    statistics are a 2x3 cell count (decisions; Phase-1 predictions use cells
    0, 1, 3, 4) and, per arrangement, loss sums and a score histogram (ROC-AUC)."""
    rng = np.random.default_rng(RANDOM_STATE)
    nD = len(keys)
    nA = len(ARRANGEMENTS)
    codes = np.stack([y * 3 + actions[k] for k in keys] +
                     [y * 3 + phase1_pred[a] for a in ARRANGEMENTS]).astype(np.int64)   # (nV, n)
    nV = codes.shape[0]
    Lstack = np.stack([losses[a] for a in ARRANGEMENTS])                              # (nA, n, 2)
    Hcodes = np.stack([y * AUC_BINS + hist_bin[a] for a in ARRANGEMENTS])             # (nA, n)
    NL, NH = len(LOSS_NAMES), 2 * AUC_BINS

    if ds == "rt_iot2022":
        row_groups = {"all": groups["all"]}
        clus = None
    else:
        row_groups = {"capture35_test_rows": groups["capture35_test_rows"]}
        clus = {}
        for weighting in ("unweighted", "multiplicity_weighted"):
            cmc = np.zeros((len(caps), nV, 6))
            lsum = np.zeros((len(caps), nA, NL))
            hsum = np.zeros((len(caps), nA, NH))
            wsum = np.zeros(len(caps))
            for ci, c in enumerate(caps):
                m = groups[f"capture:{c}"]
                w = np.ones(m.sum()) if weighting == "unweighted" else mult[m]
                for v in range(nV):
                    cmc[ci, v] = np.bincount(codes[v, m], weights=w, minlength=6)
                for a in range(nA):
                    hsum[ci, a] = np.bincount(Hcodes[a, m], weights=w, minlength=NH)
                lsum[ci] = (Lstack[:, m, :] * w[None, :, None]).sum(axis=1)
                wsum[ci] = w.sum()
            clus[weighting] = (cmc, lsum, hsum, wsum)

    out_cm, out_l, out_h = {}, {}, {}
    grp_names = list(row_groups) + ([] if clus is None else ["heldout_captures", "all"])
    for g in grp_names:
        for weighting in ("unweighted", "multiplicity_weighted"):
            out_cm[(g, weighting)] = np.zeros((B, nV, 6))
            out_l[(g, weighting)] = np.zeros((B, nA, NL))
            out_h[(g, weighting)] = np.zeros((B, nA, NH))
    sub = {g: (np.flatnonzero(m), codes[:, m], Lstack[:, m, :], Hcodes[:, m], mult[m]) for g, m in row_groups.items()}
    for b in range(B):
        res_rows = {}
        for g, (idx, cg, lg, hg, mg) in sub.items():
            cnt = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx)).astype(np.float64)
            for weighting in ("unweighted", "multiplicity_weighted"):
                w = cnt if weighting == "unweighted" else cnt * mg
                cmb = np.stack([np.bincount(cg[v], weights=w, minlength=6) for v in range(nV)])
                hb = np.stack([np.bincount(hg[a], weights=w, minlength=NH) for a in range(nA)])
                lb = (lg * w[None, :, None]).sum(axis=1)
                out_cm[(g, weighting)][b] = cmb
                out_h[(g, weighting)][b] = hb
                out_l[(g, weighting)][b] = lb / w.sum()
                res_rows[(g, weighting)] = (cmb, lb, hb, w.sum())
        if clus is not None:
            cc = np.bincount(rng.integers(0, len(caps), len(caps)), minlength=len(caps)).astype(np.float64)
            for weighting in ("unweighted", "multiplicity_weighted"):
                cmc, lsum, hsum, wsum = clus[weighting]
                cmb = np.tensordot(cc, cmc, axes=1)
                lb = np.tensordot(cc, lsum, axes=1)
                hb = np.tensordot(cc, hsum, axes=1)
                ws = cc @ wsum
                out_cm[("heldout_captures", weighting)][b] = cmb
                out_h[("heldout_captures", weighting)][b] = hb
                out_l[("heldout_captures", weighting)][b] = lb / ws
                c35cm, c35l, c35h, c35w = res_rows[("capture35_test_rows", weighting)]
                out_cm[("all", weighting)][b] = cmb + c35cm
                out_h[("all", weighting)][b] = hb + c35h
                out_l[("all", weighting)][b] = (lb + c35l) / (ws + c35w)

    ci_p1, ci_dec = {}, {}
    for (g, weighting), arrs in out_cm.items():
        for vi, key in enumerate(keys):
            arr, rule, rs, r = key
            vals = [cm_decision(arrs[b, vi].reshape(2, 3)) for b in range(B)]
            ci_dec[(arr, rule, rs, r, g, weighting)] = {
                m: tuple(np.nanpercentile([v.get(m, np.nan) for v in vals], [2.5, 97.5])) for m in CI_DEC}
        for ai, arr in enumerate(ARRANGEMENTS):
            vi = nD + ai
            cl = [cm_classification(arrs[b, vi].reshape(2, 3)[:, :2]) for b in range(B)]
            d = {m: tuple(np.nanpercentile([v.get(m, np.nan) for v in cl], [2.5, 97.5]))
                 for m in CI_P1 if m not in LOSS_NAMES and m != "roc_auc"}
            for li, ln in enumerate(LOSS_NAMES):
                d[ln] = tuple(np.nanpercentile(out_l[(g, weighting)][:, ai, li], [2.5, 97.5]))
            aucs = [auc_from_hist(out_h[(g, weighting)][b, ai].reshape(2, AUC_BINS)) for b in range(B)]
            d["roc_auc"] = tuple(np.nanpercentile(aucs, [2.5, 97.5])) if np.isfinite(aucs).any() else (np.nan, np.nan)
            ci_p1[(arr, g, weighting)] = d
    return ci_p1, ci_dec



# ======================================================================
# [3] lifecycle replay (IoT-23 held-out captures)
# ======================================================================

def build_test_capture(cap, offset, n_unique, cap_rows, log):
    """Stream the raw flows of one held-out capture, map each flow to its
    deduplicated test row (same key and first-appearance rule as cleaning and
    script 10), verify, count raw multiplicities, keep internal-device flows
    sorted by device then time. Cached (restartable)."""
    path = os.path.join(CACHE_DIR, f"{cap}_test_v1.npz")
    if os.path.exists(path):
        return path
    t0 = time.perf_counter()
    stats_p = os.path.join(INTERIM_SCEN, f"{cap}.stats.json")
    with open(stats_p) as f:
        n_lines = int(json.load(f)["rows_written"])
    dur_ref = cap_rows["duration"].to_numpy(dtype=np.float64)
    pkt_ref = cap_rows["orig_pkts"].to_numpy(dtype=np.float64)
    lab_ref = cap_rows["detailed_label"].astype(str).to_numpy()
    rank_of, host_of, internal_of = {}, {}, []
    counts = np.zeros(n_unique, dtype=np.int64)
    rank_all = np.empty(n_lines, dtype=np.int32)       # internal flows only are kept; n_lines = upper bound
    host_all = np.empty(n_lines, dtype=np.int32)
    ts_all = np.empty(n_lines, dtype=np.float64)
    k = 0
    n_seen = 0
    bad = {"duration": 0, "orig_pkts": 0, "label": 0, "nan_ts": 0, "unknown_key": 0}
    for chunk in pd.read_csv(os.path.join(INTERIM_SCEN, f"{cap}.csv"), dtype=str,
                             keep_default_na=False, chunksize=S10.CHUNK):
        chunk = S10.standardize(chunk)
        for c in ("label", "detailed_label"):
            chunk[c] = chunk[c].str.strip().str.lower()
        for c in S10.DASH:
            is_dash = chunk[c].eq("-")
            chunk[c] = pd.to_numeric(chunk[c].where(~is_dash), errors="coerce").fillna(0.0).astype("float64")
        key_cols = [c for c in chunk.columns if c not in S10.IDENT]
        h = pd.util.hash_pandas_object(chunk[key_cols], index=False).to_numpy()
        uniq, first = np.unique(h, return_index=True)
        for u in uniq[np.argsort(first)].tolist():
            if u not in rank_of:
                rank_of[u] = len(rank_of)
        rk = pd.Series(h).map(rank_of).to_numpy(dtype=np.int64)
        if (rk >= n_unique).any():
            bad["unknown_key"] += int((rk >= n_unique).sum())
            rk = np.minimum(rk, n_unique - 1)
        counts += np.bincount(rk, minlength=n_unique)
        bad["duration"] += int((chunk["duration"].to_numpy(dtype=np.float64) != dur_ref[rk]).sum())
        bad["orig_pkts"] += int((pd.to_numeric(chunk["orig_pkts"], errors="coerce").to_numpy(dtype=np.float64)
                                 != pkt_ref[rk]).sum())
        bad["label"] += int((chunk["detailed_label"].to_numpy(dtype=object) != lab_ref[rk]).sum())
        ts = pd.to_numeric(chunk["ts"], errors="coerce").to_numpy(dtype=np.float64)
        bad["nan_ts"] += int(np.isnan(ts).sum())
        hosts = chunk["id_orig_h"].to_numpy()
        for hst in pd.unique(hosts):
            if hst not in host_of:
                host_of[hst] = len(host_of)
                internal_of.append(S10.is_internal(hst))
        hc = pd.Series(hosts).map(host_of).to_numpy(dtype=np.int32)
        keep = np.asarray(internal_of, dtype=bool)[hc]
        m = int(keep.sum())
        if k + m > n_lines:
            raise RuntimeError(f"{cap}: more flows than recorded by the loader ({n_lines:,})")
        rank_all[k:k + m] = rk[keep]
        host_all[k:k + m] = hc[keep]
        ts_all[k:k + m] = ts[keep]
        k += m
        n_seen += len(chunk)
        del chunk, h, rk, ts, hosts, hc, keep
    if n_seen != n_lines:
        raise RuntimeError(f"{cap}: read {n_seen:,} flows, loader wrote {n_lines:,}")
    if len(rank_of) != n_unique or any(bad.values()):
        raise RuntimeError(f"{cap}: mapping verification failed: unique keys {len(rank_of)} vs "
                           f"{n_unique} test rows; mismatches {bad}")
    mult_ok = bool(np.array_equal(counts, cap_rows["row_multiplicity"].to_numpy(dtype=np.int64)))
    rank_all, host_all, ts_all = rank_all[:k], host_all[:k], ts_all[:k]
    order = np.lexsort((ts_all, host_all))             # device primary, time secondary (as script 10)
    rank_all = rank_all[order]
    ts_all = ts_all[order]
    host_all = host_all[order]
    del order
    gc.collect()
    names = np.array([str(h) for h in host_of.keys()])
    np.savez(path, rank=rank_all, ts=ts_all, host=host_all, host_names=names,
             host_internal=np.asarray(internal_of, dtype=bool), n_raw=np.int64(n_seen),
             mult_ok=np.bool_(mult_ok), offset=np.int64(offset))
    log(f"    {cap}: {n_seen:,} raw flows -> {n_unique:,} test rows verified; "
        f"{k:,} flows from internal devices; multiplicities replay exactly: {mult_ok} "
        f"({time.perf_counter() - t0:.0f}s)")
    del rank_all, ts_all, host_all
    gc.collect()
    return path


def contextual_actions(r, starts, ends, p):
    """Lean, chunked version of script 10's decide(): same beta reputation
    (alpha_-1 = beta_-1 = 1, forgetting lambda), same rule and probation.
    r: float32 risk scores in device-time order."""
    lam, m, t1, t2 = p["lambda"], p["probation_m"], p["tau1"], p["tau2"]
    n = len(r)
    act = np.empty(n, dtype=np.int8)
    for s, e in zip(starts, ends):
        st = {"a": 1.0, "b": 1.0}
        for c0 in range(s, e, LEAN_CHUNK):
            c1 = min(e, c0 + LEAN_CHUNK)
            x = r[c0:c1].astype(np.float64)
            if lam == 0.0:
                cr = x
            else:
                a, _ = lfilter([1.0], [1.0, -lam], 1.0 - x, zi=[lam * st["a"]])
                b, _ = lfilter([1.0], [1.0, -lam], x, zi=[lam * st["b"]])
                st = {"a": a[-1], "b": b[-1]}
                cr = b / (a + b)
            a = np.where(cr >= t2, 2, np.where(cr >= t1, 1, 0)).astype(np.int8)
            if m > 0 and c0 - s < m:
                lim = min(c1, s + m) - c0
                head = a[:lim]
                head[head == 0] = 1
            act[c0:c1] = a
    return act



def lean_lifecycle(y, act, ts, starts, ends):
    """Per-device lifecycle quantities; columns identical to script 10's
    lifecycle() entity table, plus stability / false-revocation extras."""
    BIG = S10.BIG
    ent, extra = [], []
    cm = np.zeros((2, 3), dtype=np.int64)
    cm_state = np.zeros((2, 3), dtype=np.int64)
    cm_dev = np.zeros((2, 3), dtype=np.float64)
    comp = {"post_onset_flows": 0, "post_onset_allowed": 0}
    for s, e in zip(starts, ends):
        yd, ad, td = y[s:e], act[s:e], ts[s:e]
        n = e - s
        mal = yd > 0
        onset = int(np.argmax(mal)) if mal.any() else BIG
        is_allow = ad == 0
        first_allow = int(np.argmax(is_allow)) if is_allow.any() else BIG
        allowed = int(is_allow.sum())
        if onset < BIG:
            post_rej = ad[onset:] == 2
            first_rej = onset + int(np.argmax(post_rej)) if post_rej.any() else BIG
            allow_post = int(is_allow[onset:].sum())
            post_nonallow = ad[onset:] != 0
            first_nonallow = onset + int(np.argmax(post_nonallow)) if post_nonallow.any() else BIG
        else:
            first_rej = first_nonallow = BIG
            allow_post = 0
        if first_rej < BIG:
            retrust = int(is_allow[first_rej + 1:].sum())
            n_after_rej = n - first_rej - 1
        else:
            retrust = n_after_rej = 0
        rev_mask = (ad[1:] > 0) & (ad[:-1] == 0)
        revocations = int(rev_mask.sum())

        def tsec(p):
            return float(td[p] - td[0]) if p < BIG else np.nan
        ent.append({
            "flows": n, "malicious": onset < BIG,
            "onset_pos": float(onset) if onset < BIG else np.nan,
            "flows_to_first_allow": float(first_allow) if first_allow < BIG else np.nan,
            "seconds_to_first_allow": tsec(first_allow),
            "share_allowed": allowed / n,
            "revocations_after_trust": revocations,
            "allows_after_onset": float(allow_post) if onset < BIG else np.nan,
            "flows_after_onset": float(n - onset) if onset < BIG else np.nan,
            "delay_flows_onset_to_reject": float(first_rej - onset) if (onset < BIG and first_rej < BIG) else np.nan,
            "delay_seconds_onset_to_reject": (tsec(first_rej) - tsec(onset)) if (onset < BIG and first_rej < BIG) else np.nan,
            "ever_rejected_after_onset": first_rej < BIG,
            "allows_after_first_reject": float(retrust) if first_rej < BIG else np.nan,
            "flows_after_first_reject": float(n_after_rej) if first_rej < BIG else np.nan,
        })
        pre = onset if onset < BIG else n
        ystate = np.maximum.accumulate(yd)
        same_flow = yd[1:] == yd[:-1]
        same_state = ystate[1:] == ystate[:-1]
        change = ad[1:] != ad[:-1]
        extra.append({
            "pre_onset_flows": pre,
            "pre_onset_allowed_ever": bool(is_allow[:pre].any()) if pre > 0 else False,
            "pre_onset_restricted_or_rejected": int((ad[:pre] != 0).sum()),
            "pre_onset_rejected": int((ad[:pre] == 2).sum()),
            "pre_onset_revocations": int(rev_mask[:max(pre - 1, 0)].sum()),
            "delay_flows_onset_to_first_nonallow": float(first_nonallow - onset) if (onset < BIG and first_nonallow < BIG) else np.nan,
            "delay_seconds_onset_to_first_nonallow": (tsec(first_nonallow) - tsec(onset)) if (onset < BIG and first_nonallow < BIG) else np.nan,
            "pairs_same_flow_truth": int(same_flow.sum()),
            "flips_same_flow_truth": int((same_flow & change).sum()),
            "pairs_same_device_state": int(same_state.sum()),
            "flips_same_device_state": int((same_state & change).sum()),
        })
        dev_state_cm = np.zeros((2, 3), dtype=np.int64)
        for c0 in range(0, n, LEAN_CHUNK):
            c1 = min(n, c0 + LEAN_CHUNK)
            cm += np.bincount(yd[c0:c1].astype(np.int64) * 3 + ad[c0:c1], minlength=6).reshape(2, 3)
            dev_state_cm += np.bincount(ystate[c0:c1].astype(np.int64) * 3 + ad[c0:c1], minlength=6).reshape(2, 3)
        cm_state += dev_state_cm
        dev_state_cm = dev_state_cm.astype(np.float64)
        rs = dev_state_cm.sum(axis=1, keepdims=True)
        cm_dev += np.divide(dev_state_cm, rs, out=np.zeros_like(dev_state_cm), where=rs > 0)
        if onset < BIG:
            comp["post_onset_flows"] += n - onset
            comp["post_onset_allowed"] += allow_post
    return pd.DataFrame(ent), pd.DataFrame(extra), cm, cm_state, cm_dev, comp


def reference_check(cap, y, act_lean, r, ts, starts, ends, arr, p, lean_out):
    """Run script 10's own decide()/lifecycle() on this capture and compare."""
    n = len(y)
    pos = np.arange(n) - np.repeat(starts, ends - starts)
    d = {"pos": pos, "y": y.astype(np.int8), "ts": ts, f"{arr}_r": r}
    d["y_state"] = S10.device_state(d["y"], pos)
    act_ref = S10.decide(d, arr, p)
    diff = np.flatnonzero(act_ref != act_lean)
    if len(diff):
        # tolerate only flows whose contextual risk sits within 1e-9 of a threshold
        cr = S10.contextual(r, pos, p["lambda"])
        near = (np.abs(cr[diff] - p["tau1"]) < 1e-9) | (np.abs(cr[diff] - p["tau2"]) < 1e-9)
        if not near.all():
            raise RuntimeError(f"{cap}/{arr}/{p}: lean decisions differ from script 10 on "
                               f"{int((~near).sum())} flows not explained by rounding at a threshold")
        return {"capture": cap, "arrangement": arr, **p, "flows": n, "action_differences": int(len(diff)),
                "explained_by_threshold_rounding": int(near.sum()), "lifecycle_identical": "not compared"}
    ent_ref, cm_ref, cms_ref, cmd_ref, comp_ref = S10.lifecycle(d, act_ref)
    ent_lean, _, cm_l, cms_l, cmd_l, comp_l = lean_out
    same = (np.array_equal(cm_ref, cm_l) and np.array_equal(cms_ref, cms_l)
            and np.allclose(cmd_ref, cmd_l, rtol=1e-9, atol=1e-9) and comp_ref == comp_l)
    a = ent_ref.reset_index(drop=True)
    b = ent_lean[a.columns].reset_index(drop=True)
    for c in a.columns:
        x, z = a[c].to_numpy(dtype=np.float64), b[c].to_numpy(dtype=np.float64)
        if not np.allclose(x, z, rtol=1e-9, atol=1e-9, equal_nan=True):
            same = False
    if not same:
        raise RuntimeError(f"{cap}/{arr}/{p}: lean lifecycle differs from script 10's lifecycle()")
    return {"capture": cap, "arrangement": arr, **p, "flows": n, "action_differences": 0,
            "explained_by_threshold_rounding": 0, "lifecycle_identical": True}



def lifecycle_stage(test_probs, life_params, split_summary, log, run_dir, B):
    log("\n" + "=" * 70 + "\n[3] TRUST LIFECYCLE ON THE WHOLLY HELD-OUT IoT-23 CAPTURES\n" + "=" * 70)
    os.makedirs(CACHE_DIR, exist_ok=True)
    caps = sorted(split_summary["whole_scenarios_to_test"])
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "iot23_test.csv"), low_memory=False,
                       usecols=["row_id", "scenario", "duration", "orig_pkts", "detailed_label",
                                "row_multiplicity", "malicious"])
    probs = test_probs.set_index("row_id")
    sizes = []
    for cap in caps:
        with open(os.path.join(INTERIM_SCEN, f"{cap}.stats.json")) as f:
            sizes.append(int(json.load(f)["rows_written"]))
    order = [c for _, c in sorted(zip(sizes, caps))]          # small captures first: early failure
    log(f"  captures (smallest first): {order}")

    # unique parameter sets
    psets = {}
    for (arr, r, rule), p in life_params.items():
        key = (arr, p["lambda"], p["probation_m"], p["tau1"], p["tau2"])
        psets.setdefault(key, []).append((r, rule))
    log(f"  {len(life_params)} frozen (arrangement, ratio, rule) settings -> {len(psets)} distinct parameter sets")

    dev_rows, ext_rows, ent_rows, flow_rows, ref_rows = [], [], [], [], []
    per_cap = {}           # (arr, r, rule) -> list of (cap, cm, cm_state, cm_dev, comp, extra-sums)
    mult_ok_all = True
    for cap in order:
        rows = test[test["scenario"] == cap].sort_values("row_id")
        ids = rows["row_id"].to_numpy()
        if not np.array_equal(ids, np.arange(ids[0], ids[0] + len(ids))):
            raise RuntimeError(f"{cap}: test row_ids are not contiguous")
        path = build_test_capture(cap, int(ids[0]), len(ids), rows, log)
        z = np.load(path)
        mult_ok_all &= bool(z["mult_ok"])
        rank, ts, host = z["rank"], z["ts"], z["host"]
        names = z["host_names"]
        n = len(rank)
        ycap = rows["malicious"].to_numpy(dtype=np.int8)
        if n == 0:
            ext_rows.append({"capture": cap, "raw_flows": int(z["n_raw"]), "internal_device_flows": 0,
                             "internal_devices": 0, "external_originators": int((~z["host_internal"]).sum())})
            log(f"    {cap}: no internal-device flows; excluded from lifecycle metrics")
            continue
        start_mask = np.r_[True, host[1:] != host[:-1]]
        starts = np.flatnonzero(start_mask)
        ends = np.r_[starts[1:], n]
        y = ycap[rank]
        for s, e in zip(starts, ends):
            yy = y[s:e]
            mal = np.flatnonzero(yy > 0)
            dev_rows.append({"capture": cap, "device": str(names[host[s]]), "flows": int(e - s),
                             "benign_flows": int((yy == 0).sum()), "malicious_flows": int((yy == 1).sum()),
                             "onset_position": int(mal[0]) if len(mal) else None,
                             "seconds_before_onset": float(ts[s + mal[0]] - ts[s]) if len(mal) else None,
                             "seconds_total": float(ts[e - 1] - ts[s])})
        ext_rows.append({"capture": cap, "raw_flows": int(z["n_raw"]), "internal_device_flows": n,
                         "internal_devices": int(len(starts)),
                         "external_originators": int((~z["host_internal"]).sum())})
        rid = ids  # rank -> row_id
        for (arr, lam, m, t1, t2), uses in psets.items():
            # as script 10: risk scores stored as float32, clipped to [0, 1]
            rt = np.clip(probs.loc[rid, f"{arr}_risk"].to_numpy(dtype=np.float64), 0, 1).astype(np.float32)
            r = rt[rank]
            p = {"lambda": lam, "probation_m": m, "tau1": t1, "tau2": t2}
            act = contextual_actions(r, starts, ends, p)
            out = lean_lifecycle(y, act, ts, starts, ends)
            if n <= REFERENCE_MAX_FLOWS:
                ref_rows.append(reference_check(cap, y, act, r, ts, starts, ends, arr, p, out))
            ent, extra, cm, cms, cmd, comp = out
            for (r, rule) in uses:
                e2 = pd.concat([ent, extra], axis=1)
                e2.insert(0, "device", [str(names[host[s]]) for s in starts])
                e2.insert(0, "capture", cap)
                e2.insert(0, "rule", rule)
                e2.insert(0, "cost_ratio", r)
                e2.insert(0, "arrangement", arr)
                ent_rows.append(e2)
                per_cap.setdefault((arr, r, rule), []).append(
                    {"capture": cap, "cm": cm, "cms": cms, "cmd": cmd, "comp": comp,
                     "pairs_flow": int(extra["pairs_same_flow_truth"].sum()),
                     "flips_flow": int(extra["flips_same_flow_truth"].sum()),
                     "pairs_state": int(extra["pairs_same_device_state"].sum()),
                     "flips_state": int(extra["flips_same_device_state"].sum())})
            del r, act, out
        del z, rank, ts, host, y
        gc.collect()
        log(f"    {cap}: lifecycle evaluated for {len(psets)} parameter sets")

    # pooled flow metrics + cluster bootstrap over captures
    rng = np.random.default_rng(RANDOM_STATE)
    LMET = ["false_trust_rate_allow_given_malicious", "over_restriction_rate_given_benign",
            "compromise_false_trust_allow_after_onset", "share_Suspicious"]
    for (arr, r, rule), lst in per_cap.items():
        p = life_params[(arr, r, rule)]
        tot = lambda k: sum(x[k] for x in lst)                      # noqa: E731
        comp = {"post_onset_flows": sum(x["comp"]["post_onset_flows"] for x in lst),
                "post_onset_allowed": sum(x["comp"]["post_onset_allowed"] for x in lst)}
        stab = {"decision_flip_rate_same_flow_truth": tot("flips_flow") / tot("pairs_flow") if tot("pairs_flow") else np.nan,
                "decision_flip_rate_same_device_state": tot("flips_state") / tot("pairs_state") if tot("pairs_state") else np.nan}
        boot = {}
        if B > 0 and len(lst) > 1:
            k = len(lst)
            vals = {(t, mname): [] for t in ("per_flow", "device_state") for mname in LMET}
            for _ in range(B):
                c = np.bincount(rng.integers(0, k, k), minlength=k)
                for t, key in (("per_flow", "cm"), ("device_state", "cms")):
                    cmb = sum(c[i] * lst[i][key] for i in range(k))
                    cpb = {"post_onset_flows": sum(c[i] * lst[i]["comp"]["post_onset_flows"] for i in range(k)),
                           "post_onset_allowed": sum(c[i] * lst[i]["comp"]["post_onset_allowed"] for i in range(k))}
                    fm = S10.flow_metrics(cmb, cpb) if cmb.sum() else {}
                    for mname in LMET:
                        vals[(t, mname)].append(fm.get(mname, np.nan))
            boot = {k2: tuple(np.nanpercentile(v, [2.5, 97.5])) for k2, v in vals.items()}
        for truth, key in (("per_flow", "cm"), ("device_state", "cms"), ("device_state_device_averaged", "cmd")):
            mat = tot(key)
            row = {"arrangement": arr, "cost_ratio": r, "rule": rule, **p, "truth": truth,
                   "captures_with_internal_devices": len(lst), **S10.flow_metrics(mat, comp), **stab,
                   "decision_matrix_true_BM_x_action_ARJ": np.asarray(mat).tolist()}
            for mname in LMET:
                if (truth, mname) in boot:
                    row[f"{mname}_ci_low"], row[f"{mname}_ci_high"] = boot[(truth, mname)]
            flow_rows.append(row)

    ent_all = pd.concat(ent_rows, ignore_index=True) if ent_rows else pd.DataFrame()
    summ_rows = []
    for (arr, r, rule), g in ent_all.groupby(["arrangement", "cost_ratio", "rule"]):
        base = S10.entity_summary(g)
        mal = g[g["malicious"]]
        ben = g[~g["malicious"]]
        later = mal[mal["onset_pos"] > 0]
        base.update({
            "benign_share_flows_restricted_or_rejected": float(ben["pre_onset_restricted_or_rejected"].sum() / ben["flows"].sum()) if len(ben) else np.nan,
            "benign_share_flows_rejected": float(ben["pre_onset_rejected"].sum() / ben["flows"].sum()) if len(ben) else np.nan,
            "benign_revocations_total": int(ben["revocations_after_trust"].sum()) if len(ben) else 0,
            "later_compromised_devices": len(later),
            "later_compromised_share_trusted_before_onset": float(later["pre_onset_allowed_ever"].mean()) if len(later) else np.nan,
            "later_compromised_pre_onset_false_revocations": int(later["pre_onset_revocations"].sum()) if len(later) else 0,
            "malicious_median_delay_flows_to_first_nonallow": float(mal["delay_flows_onset_to_first_nonallow"].median()) if len(mal) else np.nan,
            "malicious_median_delay_seconds_to_first_nonallow": float(mal["delay_seconds_onset_to_first_nonallow"].median()) if len(mal) else np.nan,
        })
        summ_rows.append({"arrangement": arr, "cost_ratio": r, "rule": rule, **life_params[(arr, r, rule)], **base})

    life = {"devices": pd.DataFrame(dev_rows), "captures": pd.DataFrame(ext_rows),
            "entities": ent_all, "flows": pd.DataFrame(flow_rows), "summary": pd.DataFrame(summ_rows),
            "reference_checks": pd.DataFrame(ref_rows), "multiplicity_replay_ok": mult_ok_all}
    log(f"  multiplicities reproduced exactly by the raw-flow replay on every held-out capture: {mult_ok_all}")
    if len(ref_rows):
        rc = life["reference_checks"]
        log(f"  reference check vs script 10 on {rc['capture'].nunique()} capture(s), {len(rc)} parameter sets: "
            f"action differences {int(rc['action_differences'].sum())} "
            f"(all explained by rounding at a threshold: {bool((rc['action_differences'] == rc['explained_by_threshold_rounding']).all())})")
    return life


# ======================================================================
# [4] reporting
# ======================================================================

def fmt(v, lo=None, hi=None, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = f"{v:.{d}f}"
    if lo is not None and not (isinstance(lo, float) and np.isnan(lo)):
        s += f" [{lo:.{d}f}, {hi:.{d}f}]"
    return s


def thesis_tables(res, life, run_dir, weights_ok):
    tdir = os.path.join(run_dir, "thesis_tables")
    os.makedirs(tdir, exist_ok=True)
    weightings = ["unweighted"] + (["multiplicity_weighted"] if weights_ok else [])
    for ds, r in res.items():
        main_groups = ["all"] if ds == "rt_iot2022" else ["capture35_test_rows", "heldout_captures", "all"]
        for wt in weightings:
            p1 = r["p1"][(r["p1"]["group"].isin(main_groups)) & (r["p1"]["weighting"] == wt)]
            rows = []
            for _, x in p1.iterrows():
                rows.append({"group": x["group"], "arrangement": x["arrangement"], "n": int(x["n"]) if wt == "unweighted" else x["n"],
                             "average_precision": fmt(x.get("average_precision")),
                             **{m: fmt(x.get(m), x.get(f"{m}_ci_low"), x.get(f"{m}_ci_high"))
                                for m in ["roc_auc", "macro_f1_at_0.5_present_classes", "mcc_at_0.5",
                                          "recall_benign", "recall_malicious", "log_loss", "brier"]}})
            pd.DataFrame(rows).to_csv(os.path.join(tdir, f"T_phase1_{ds}_{wt}.csv"), index=False)
            dec = r["dec"][(r["dec"]["group"].isin(main_groups)) & (r["dec"]["weighting"] == wt)]
            rows = []
            for _, x in dec.iterrows():
                cr = x["cost_ratio"]
                rows.append({"group": x["group"], "arrangement": x["arrangement"],
                             "rule": "no trust layer (r >= 0.5 -> Reject)" if x["decision_rule"] != "cost_minimising_thresholds"
                             else f"thresholds ({x['threshold_rows']})",
                             "cost_ratio": "" if pd.isna(cr) else int(cr),
                             "tau1": x["tau1"], "tau2": x["tau2"],
                             **{m: fmt(x.get(m), x.get(f"{m}_ci_low"), x.get(f"{m}_ci_high"))
                                for m in ["false_trust_rate_allow_given_malicious", "over_restriction_rate_given_benign",
                                          "benign_rejected_rate", "share_Trusted", "share_Suspicious", "share_Untrusted",
                                          "malicious_share_within_Suspicious"]},
                             "mean_cost_at_own_ratio": (fmt(x.get(f"mean_cost_C{int(cr)}"),
                                                            x.get(f"mean_cost_C{int(cr)}_ci_low"),
                                                            x.get(f"mean_cost_C{int(cr)}_ci_high"))
                                                        if not pd.isna(cr) else "")})
            pd.DataFrame(rows).to_csv(os.path.join(tdir, f"T_decisions_{ds}_{wt}.csv"), index=False)
        sens = r["sens"]
        sens.round(4).to_csv(os.path.join(tdir, f"T_sensitivity_{ds}.csv"), index=False)
        prim = "all" if ds == "rt_iot2022" else "heldout_captures"
        sh = sens[(sens["arrangement"] == "hybrid_xgb_rf") & (sens["group"] == prim)]
        for col in ("share_Suspicious", "false_trust_rate_allow_given_malicious", "over_restriction_rate_given_benign"):
            sh.pivot_table(index=["b", "rho"], columns="cost_ratio", values=col).round(4).to_csv(
                os.path.join(tdir, f"T_sensitivity_{ds}_hybrid_{col}.csv"))
        if ds == "iot23":
            pc = r["p1"][r["p1"]["group"].str.startswith("capture:") & (r["p1"]["weighting"] == "unweighted")]
            for col in ("macro_f1_at_0.5_present_classes", "roc_auc"):
                piv = pc.pivot_table(index="group", columns="arrangement", values=col)
                piv.index = [i.split(":", 1)[1] for i in piv.index]
                piv.round(4).to_csv(os.path.join(tdir, f"T_iot23_per_capture_{col}.csv"))
    if life is not None:
        s = life["summary"]
        s.round(4).to_csv(os.path.join(tdir, "T_lifecycle_device_summary.csv"), index=False)
        f = life["flows"]
        f.drop(columns=["decision_matrix_true_BM_x_action_ARJ"]).round(4).to_csv(
            os.path.join(tdir, "T_lifecycle_flow_metrics.csv"), index=False)



def dev_vs_test(res, run_dir):
    rows = []
    gmap = {"heldout_captures": "unseen_capture_rows", "capture35_test_rows": "capture35_rows", "all": "all"}
    for ds, r in res.items():
        dm = os.path.join(RESULTS, "08_risk_models", f"{ds}_metrics.csv")
        dd = os.path.join(RESULTS, "09_decision_layer", f"{ds}_decision_metrics.csv")
        if not (os.path.exists(dm) and os.path.exists(dd)):
            continue
        d8 = pd.read_csv(dm)
        d8 = d8[d8["fold"].astype(str) == "pooled"]
        d9 = pd.read_csv(dd)
        for tg, dg in gmap.items():
            if ds == "rt_iot2022" and tg != "all":
                continue
            for arr in ARRANGEMENTS:
                t = r["p1"][(r["p1"]["arrangement"] == arr) & (r["p1"]["group"] == tg) & (r["p1"]["weighting"] == "unweighted")]
                d = d8[(d8["arrangement"] == arr) & (d8["group"] == dg)]
                if len(t) and len(d):
                    for m in ("roc_auc", "macro_f1_at_0.5_present_classes", "mcc_at_0.5", "log_loss", "recall_malicious"):
                        rows.append({"dataset": ds, "test_group": tg, "development_group": dg, "arrangement": arr,
                                     "rule": "risk score", "cost_ratio": None, "metric": m,
                                     "development_out_of_fold": d[m].iloc[0] if m in d else np.nan,
                                     "test": t[m].iloc[0] if m in t else np.nan})
                for rs in ROWSETS[ds]:
                    for cr in COST_RATIOS:
                        t = r["dec"][(r["dec"]["arrangement"] == arr) & (r["dec"]["group"] == tg) &
                                     (r["dec"]["weighting"] == "unweighted") & (r["dec"]["threshold_rows"] == rs) &
                                     (r["dec"]["cost_ratio"] == cr)]
                        d = d9[(d9["arrangement"] == arr) & (d9["group"] == dg) & (d9["threshold_rows"] == rs) &
                               (d9["cost_ratio"] == cr)]
                        if len(t) and len(d):
                            for m in ("false_trust_rate_allow_given_malicious", "over_restriction_rate_given_benign",
                                      "share_Suspicious", f"mean_cost_C{cr}"):
                                rows.append({"dataset": ds, "test_group": tg, "development_group": dg,
                                             "arrangement": arr, "rule": f"thresholds ({rs})", "cost_ratio": cr,
                                             "metric": m, "development_out_of_fold": d[m].iloc[0],
                                             "test": t[m].iloc[0]})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(run_dir, "development_vs_test.csv"), index=False)
    return out


def figures(res, life, run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fdir = os.path.join(run_dir, "figures")
    os.makedirs(fdir, exist_ok=True)
    # 1. IoT-23 per-capture macro-F1
    if "iot23" in res:
        p1 = res["iot23"]["p1"]
        pc = p1[p1["group"].str.startswith("capture:") & (p1["weighting"] == "unweighted")]
        caps = sorted(pc["group"].unique())
        fig, ax = plt.subplots(figsize=(10, 4.8))
        w = 0.2
        for i, arr in enumerate(ARRANGEMENTS):
            v = [pc[(pc["group"] == c) & (pc["arrangement"] == arr)]["macro_f1_at_0.5_present_classes"].iloc[0] for c in caps]
            ax.bar(np.arange(len(caps)) + (i - 1.5) * w, v, width=w, label=arr)
        c35 = p1[(p1["group"] == "capture35_test_rows") & (p1["weighting"] == "unweighted") &
                 (p1["arrangement"] == "hybrid_xgb_rf")]["macro_f1_at_0.5_present_classes"].iloc[0]
        ax.axhline(c35, ls="--", lw=1, color="grey", label="primary hybrid on Capture-35-1 test rows")
        ax.set_xticks(np.arange(len(caps)))
        ax.set_xticklabels([c.replace("CTU-IoT-Malware-", "").replace("CTU-", "") for c in caps], rotation=45, ha="right")
        ax.set_ylabel("macro-F1 at r >= 0.5 (classes present)")
        ax.set_ylim(0, 1.02)
        ax.set_title("IoT-23 test: macro-F1 (r >= 0.5) on each wholly held-out capture", fontsize=10)
        ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        fig.savefig(os.path.join(fdir, "iot23_per_capture_macro_f1.png"), dpi=200)
        plt.close(fig)
    # 2. trade-off: development frontier vs test operating points
    for ds, r in res.items():
        fr_path = os.path.join(RESULTS, "09_decision_layer", f"{ds}_tradeoff_frontier.csv")
        if not os.path.exists(fr_path):
            continue
        fr = pd.read_csv(fr_path)
        rs = ROWSETS[ds][0]
        tg = "all" if ds == "rt_iot2022" else "heldout_captures"
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for i, arr in enumerate(ARRANGEMENTS):
            f = fr[(fr["arrangement"] == arr) & (fr["threshold_rows"] == rs)]
            line, = ax.plot(f["false_trust_rate"], f["over_restriction_rate"], alpha=0.5,
                            label=f"{arr}: development trade-off")
            d = r["dec"][(r["dec"]["arrangement"] == arr) & (r["dec"]["group"] == tg) &
                         (r["dec"]["weighting"] == "unweighted") &
                         ((r["dec"]["threshold_rows"] == rs) | (r["dec"]["decision_rule"] != "cost_minimising_thresholds"))]
            ax.scatter(d["false_trust_rate_allow_given_malicious"], d["over_restriction_rate_given_benign"],
                       color=line.get_color(), s=25, zorder=3)
        ax.set_xlabel("False-trust rate  P(Allow | malicious)")
        ax.set_ylabel("Over-restriction rate  P(Restrict or Reject | benign)")
        ax.set_title(f"{ds}: development trade-off (lines) vs TEST operating points (dots: no trust layer + 5 cost ratios)\n"
                     f"test group: {tg}", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fdir, f"{ds}_tradeoff_dev_vs_test.png"), dpi=200)
        plt.close(fig)
    # 3. lifecycle: compromise false trust and benign share allowed
    if life is not None and len(life["summary"]):
        s = life["summary"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
        for ax, col, title in ((axes[0], "malicious_flow_weighted_allow_after_onset", "Compromised devices: share of post-onset flows allowed"),
                               (axes[1], "benign_flow_weighted_share_allowed", "Benign devices: share of flows allowed")):
            for j, rule in enumerate(("singular", "contextual")):
                for i, arr in enumerate(ARRANGEMENTS):
                    v = [s[(s["arrangement"] == arr) & (s["rule"] == rule) & (s["cost_ratio"] == c)][col].iloc[0]
                         for c in COST_RATIOS]
                    ax.plot(COST_RATIOS, v, marker="o" if rule == "singular" else "s",
                            ls="-" if rule == "singular" else "--", label=f"{arr} ({rule})")
            ax.set_xscale("log")
            ax.set_xticks(COST_RATIOS)
            ax.set_xticklabels(COST_RATIOS)
            ax.set_xlabel("assumed cost ratio C")
            ax.set_title(title, fontsize=9)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
        axes[1].legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(fdir, "iot23_lifecycle_heldout.png"), dpi=200)
        plt.close(fig)


DISCLOSURE = """# Disclosure: prior examination of the test partitions

This run is a SUBSEQUENT, DISCLOSED evaluation of test data that had already been
examined. It is not an untouched or independent confirmatory test.

What was examined before this run:
1. An earlier version of the method (three researcher-defined trust levels assigned to
   attack types, the same four model arrangements) was evaluated on test partitions built
   from the same cleaned data, including one completed final evaluation with this script.
2. Earlier diagnostics broke those test results down by action, true class, capture group
   and risk-score distribution.

How this affects interpretation:
- The gap between Capture-35-1 and unseen captures was seen on test data and motivated
  the scenario-aware split, the partial-grouping validation design and the unseen-capture
  selection rows. These choices are supported by the literature independently, but their
  adoption was test-informed.
- The change of target to the datasets' own benign/malicious label, with trust levels
  derived from the risk score by cost-based thresholds, was made after that evaluation.
  With the new target the split selected partly different held-out captures.
- No model, feature, hyperparameter, threshold or lifecycle parameter used in this run
  was fitted or selected on test data; all were frozen on development data before this
  script ran (see manifest.json for fingerprints).
- Results here are a frozen-protocol measurement, not independent confirmation of the
  design. Freezing the protocol does not reverse the earlier exposure.
"""

METRICS_MD = """# Metric definitions (final evaluation)

Unit of analysis: a distinct flow record (after identifier removal and deduplication).
Secondary: the same metric weighted by row_multiplicity (raw flows represented), reported
only if the weights reconcile exactly with the raw counts (checks in manifest.json).
Truth: the datasets' own label, benign = 0, malicious = 1.
Risk score: r = P(malicious) from each arrangement.
Trust levels (derived, never labelled): r < tau1 -> Trusted (Allow);
tau1 <= r < tau2 -> Suspicious (Restrict); r >= tau2 -> Untrusted (Reject).

Phase 1 (risk score)
- ROC-AUC; average precision; log-loss; Brier score; mean risk.
- At r >= 0.5 (reference only): accuracy, per-class precision / recall / F1, macro-F1 over
  classes present in the group, MCC.

Phase 2 (per request)
- Cost model (three-way decisions, Yao 2010): benign Allow 0, Restrict b, Reject 1;
  malicious Allow C, Restrict rho*C, Reject 0; primary b = rho = 1/3, C in {1, 2, 5, 10, 20}.
  (tau1, tau2) frozen from development (cost-minimising on out-of-fold scores).
- FTR (false-trust rate) = P(Allow | malicious)
- ORR (over-restriction rate) = P(Restrict or Reject | benign); P(Reject | benign)
- share of flows in each trust level; share of malicious flows within each level
- mean cost at each C (primary b, rho)
- reference without a trust layer: r >= 0.5 -> Reject, else Allow (no intermediate level)
- sensitivity: frozen thresholds of every (b, rho) in {0.1, 0.2, 1/3, 0.5}^2 and every C,
  applied to the test data (point estimates).

Lifecycle (IoT-23, wholly held-out captures; subjects = internal devices: RFC 1918,
IPv6 link-local fe80::/10, unique-local fc00::/7; flows ordered by device then timestamp)
- contextual risk: Beta reputation of the risk score with forgetting factor lambda.
- onset = a device's first malicious flow; device state = benign before onset,
  compromised from onset (no remediation exists in the captures).
- trust establishment: share of benign devices ever Allowed; median flows / seconds from
  the first request to the first Allow; for later-compromised devices (onset > 0), share
  Allowed at least once before onset.
- false trust: FTR per flow (per-flow truth); compromise false trust = P(Allow | flow at or
  after the device's onset).
- revocation delay: flows and seconds from onset to the first Reject (and to the first
  Restrict-or-Reject); share of compromised devices ever Rejected after onset.
- false revocation: benign devices' share of flows Restricted/Rejected and Allow->(Restrict
  or Reject) transitions; for later-compromised devices, such transitions before onset.
- renewed access after compromise (re-trust): share of flows after the first post-onset
  Reject that are Allowed. With no remediation in the data, every re-trust is an error.
- decision stability: share of consecutive same-device flow pairs whose action changes
  while the truth is unchanged (per-flow truth, and device-state truth).

Uncertainty: 95% percentile bootstrap, B=1000, seed 42. One joint resample per replicate.
RT-IoT2022 and Capture-35-1: row resampling (ignores within-capture dependence; intervals
too narrow). Held-out captures: cluster resampling of whole captures (approximate with few
clusters). IoT-23 all rows: both, same replicate. ROC-AUC intervals from 1,000-bin score
histograms. Lifecycle: cluster resampling over captures that contain internal devices.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["rt_iot2022", "iot23"], choices=["rt_iot2022", "iot23"])
    ap.add_argument("--verify-only", action="store_true", help="integrity checks + manifest; no test labels opened")
    ap.add_argument("--skip-lifecycle", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--confirm-rerun", default=None, metavar="REASON")
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    runlog = os.path.join(OUT_ROOT, "run_log.jsonl")
    completed = sorted(d for d in os.listdir(OUT_ROOT)
                       if d.startswith("run_") and os.path.exists(os.path.join(OUT_ROOT, d, "COMPLETED.json")))
    if completed and not args.verify_only and not args.confirm_rerun:
        print(f"A completed final evaluation already exists: {completed}.\n"
              f"Re-running is a second look at the test set. If it is genuinely needed (for example a "
              f"crash fix that changes no frozen input), run again with --confirm-rerun \"<reason>\".\n"
              f"Nothing was evaluated and nothing was written.")
        sys.exit(2)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    kind = "verify" if args.verify_only else "run"
    run_dir = os.path.join(OUT_ROOT, f"{kind}_{stamp}")
    os.makedirs(run_dir, exist_ok=False)
    log = Log(os.path.join(run_dir, "console_log.txt"))

    log("=" * 70 + f"\nFINAL EVALUATION ({kind}) started {now()}\nrun folder: {rel(run_dir)}\n" + "=" * 70)
    log("Disclosure: this is a subsequent, disclosed evaluation of previously examined test partitions.")
    with open(os.path.join(run_dir, "DISCLOSURE.md"), "w", encoding="utf-8") as f:
        f.write(DISCLOSURE)
    with open(os.path.join(run_dir, "METRIC_DEFINITIONS.md"), "w", encoding="utf-8") as f:
        f.write(METRICS_MD)

    manifest = {"script": "11_final_evaluation.py", "started": now(), "kind": kind, "args": vars(args),
                "previous_completed_runs": completed, "rerun_reason": args.confirm_rerun,
                "bootstrap_B": args.bootstrap, "random_state": RANDOM_STATE}
    manifest.update(build_manifest(args.datasets, log))
    with open(runlog, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "start", "time": now(), "run": rel(run_dir), "kind": kind,
                            "rerun_reason": args.confirm_rerun, "previous_completed_runs": completed,
                            "inputs_sha256": {r["file"]: r["sha256"] for r in manifest["files"]}}) + "\n")

    log("\n[0] frozen-input checks")
    specs, models, thr, sens_thr = {}, {}, {}, {}
    for ds in args.datasets:
        specs[ds] = load_spec(ds)
        models[ds] = load_models(ds, len(specs[ds]["selected_columns"]))
        thr[ds] = load_thresholds(ds)
        sens_thr[ds] = load_sensitivity(ds)
        log(f"  {ds}: 4 arrangements loaded (class order and input width verified); "
            f"{len(thr[ds])} frozen threshold pairs; {len(sens_thr[ds])} sensitivity settings")
        check_encoder(ds, specs[ds], log)
    life_params = load_lifecycle_params() if "iot23" in args.datasets else None
    if life_params:
        log(f"  iot23: {len(life_params)} frozen lifecycle settings")
    split_summary = None
    checks = {}
    for ds in args.datasets:
        train_ids = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ds}_train.csv"),
                                usecols=["row_id", "row_multiplicity"] + (["scenario"] if ds == "iot23" else []),
                                low_memory=False)
        test_meta = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ds}_test.csv"),
                                usecols=["row_id", "row_multiplicity"] + (["scenario"] if ds == "iot23" else []),
                                low_memory=False)
        inter = np.intersect1d(train_ids["row_id"], test_meta["row_id"])
        allids = np.sort(np.r_[train_ids["row_id"].to_numpy(), test_meta["row_id"].to_numpy()])
        complete = np.array_equal(allids, np.arange(len(allids)))
        if len(inter) or not complete:
            raise SystemExit(f"{ds}: partitions overlap ({len(inter)}) or are incomplete")
        c = {"development_rows": len(train_ids), "test_rows": len(test_meta), "disjoint_and_complete": True}
        if ds == "iot23":
            with open(os.path.join(RESULTS, "03_split", "iot23_split_summary.json")) as f:
                split_summary = json.load(f)
            held = sorted(set(test_meta["scenario"]) - {CARVE})
            if held != sorted(split_summary["whole_scenarios_to_test"]):
                raise SystemExit("iot23: held-out test captures differ from the split summary")
            if set(train_ids["scenario"]) & set(held):
                raise SystemExit("iot23: a held-out test capture also appears in development")
            c["heldout_test_captures"] = held
        ok, det = verify_multiplicity(ds, train_ids, test_meta, log)
        c["multiplicity_reconciles_with_cleaning"] = ok
        c["multiplicity_details"] = det
        checks[ds] = c
        log(f"  {ds}: partitions disjoint and complete ({len(train_ids):,} + {len(test_meta):,})")
        del train_ids, test_meta
    manifest["checks"] = checks

    if args.verify_only:
        manifest["finished"] = now()
        with open(os.path.join(run_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        with open(runlog, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "verify_only_finished", "time": now(), "run": rel(run_dir)}) + "\n")
        log(f"\nVERIFY-ONLY finished: every frozen-input check passed. No test labels were opened.\n"
            f"Archive the project now (see instructions), then run without --verify-only.")
        return

    # ---------------- evaluation ----------------
    res = {}
    for ds in args.datasets:
        res[ds] = per_request(ds, models[ds], specs[ds], thr[ds], sens_thr[ds], log, run_dir, args.bootstrap, None)
        with open(os.path.join(run_dir, f"{ds}_confusion_matrices.json"), "w") as f:
            json.dump(res[ds]["cms"], f, indent=1)

    life = None
    if "iot23" in args.datasets and not args.skip_lifecycle:
        probs = pd.read_csv(os.path.join(run_dir, "iot23_test_probabilities.csv"), low_memory=False)
        life = lifecycle_stage(probs, life_params, split_summary, log, run_dir, args.bootstrap)
        del probs
        life["devices"].to_csv(os.path.join(run_dir, "lifecycle_internal_devices.csv"), index=False)
        life["captures"].to_csv(os.path.join(run_dir, "lifecycle_captures.csv"), index=False)
        life["entities"].to_csv(os.path.join(run_dir, "lifecycle_device_detail.csv"), index=False)
        life["flows"].drop(columns=["decision_matrix_true_BM_x_action_ARJ"]).to_csv(
            os.path.join(run_dir, "lifecycle_flow_metrics.csv"), index=False)
        with open(os.path.join(run_dir, "lifecycle_decision_matrices.json"), "w") as f:
            json.dump(life["flows"][["arrangement", "cost_ratio", "rule", "truth",
                                     "decision_matrix_true_BM_x_action_ARJ"]].to_dict(orient="records"), f, indent=1)
        life["summary"].to_csv(os.path.join(run_dir, "lifecycle_summary.csv"), index=False)
        life["reference_checks"].to_csv(os.path.join(run_dir, "lifecycle_reference_checks.csv"), index=False)

    # weighting verdict
    weights_ok = all(checks[ds]["multiplicity_reconciles_with_cleaning"] for ds in args.datasets)
    replay_note = "not run"
    if life is not None:
        weights_ok &= bool(life["multiplicity_replay_ok"])
        replay_note = bool(life["multiplicity_replay_ok"])
    manifest["multiplicity_weighting_verified"] = {"verdict": weights_ok,
                                                   "cleaning_reconciliation": {ds: checks[ds]["multiplicity_reconciles_with_cleaning"] for ds in args.datasets},
                                                   "raw_flow_replay_of_heldout_captures": replay_note}
    log(f"\nMultiplicity weighting verified: {weights_ok} -> weighted results "
        f"{'WRITTEN' if weights_ok else 'NOT written'}")
    for ds, r in res.items():
        keep = ["unweighted"] + (["multiplicity_weighted"] if weights_ok else [])
        r["p1"] = r["p1"][r["p1"]["weighting"].isin(keep)]
        r["dec"] = r["dec"][r["dec"]["weighting"].isin(keep)]
        r["sub"] = r["sub"][r["sub"]["weighting"].isin(keep)]
        r["p1"].to_csv(os.path.join(run_dir, f"{ds}_phase1_metrics.csv"), index=False)
        r["dec"].to_csv(os.path.join(run_dir, f"{ds}_decision_metrics.csv"), index=False)
        r["sub"].to_csv(os.path.join(run_dir, f"{ds}_trust_level_by_attack_type.csv"), index=False)
        r["sens"].to_csv(os.path.join(run_dir, f"{ds}_sensitivity.csv"), index=False)

    thesis_tables(res, life, run_dir, weights_ok)
    dvt = dev_vs_test(res, run_dir)
    figures(res, life, run_dir)

    # console summary
    log("\n" + "=" * 70 + "\nSUMMARY (unweighted; full tables in the run folder)\n" + "=" * 70)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 400):
        for ds, r in res.items():
            groups = ["all"] if ds == "rt_iot2022" else ["capture35_test_rows", "heldout_captures", "all"]
            p1 = r["p1"][(r["p1"]["weighting"] == "unweighted") & r["p1"]["group"].isin(groups)]
            cols = ["arrangement", "group", "n", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high",
                    "average_precision", "macro_f1_at_0.5_present_classes", "recall_benign", "recall_malicious",
                    "log_loss"]
            log(f"\n{ds}: Phase 1 (risk score)\n" + p1[[c for c in cols if c in p1]].round(4).to_string(index=False))
            dec = r["dec"][(r["dec"]["weighting"] == "unweighted") & r["dec"]["group"].isin(groups) &
                           ((r["dec"]["threshold_rows"] == ROWSETS[ds][0]) |
                            (r["dec"]["decision_rule"] != "cost_minimising_thresholds"))]
            cols = ["arrangement", "group", "decision_rule", "cost_ratio", "tau1", "tau2",
                    "false_trust_rate_allow_given_malicious", "false_trust_rate_allow_given_malicious_ci_low",
                    "false_trust_rate_allow_given_malicious_ci_high", "over_restriction_rate_given_benign",
                    "share_Suspicious", "malicious_share_within_Suspicious"]
            log(f"\n{ds}: decisions (primary thresholds)\n" + dec[[c for c in cols if c in dec]].round(4).to_string(index=False))
        if life is not None:
            cols = ["arrangement", "cost_ratio", "rule", "lambda", "probation_m", "benign_devices", "malicious_devices",
                    "benign_share_ever_trusted", "benign_flow_weighted_share_allowed",
                    "malicious_share_ever_rejected", "malicious_median_delay_flows_to_reject",
                    "malicious_flow_weighted_allow_after_onset", "malicious_retrust_share_after_reject"]
            log("\nLifecycle (held-out captures, internal devices)\n" +
                life["summary"][[c for c in cols if c in life["summary"]]].round(4).to_string(index=False))

    # completion
    outputs = {}
    for root, _, files in os.walk(run_dir):
        for name in files:
            p = os.path.join(root, name)
            if name not in ("COMPLETED.json", "console_log.txt"):
                outputs[rel(p)] = sha256(p)
    manifest["finished"] = now()
    manifest["outputs_sha256"] = outputs
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    with open(os.path.join(run_dir, "COMPLETED.json"), "w") as f:
        json.dump({"finished": manifest["finished"], "manifest_sha256": sha256(os.path.join(run_dir, "manifest.json"))}, f, indent=2)
    with open(runlog, "a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "completed", "time": now(), "run": rel(run_dir),
                            "manifest_sha256": sha256(os.path.join(run_dir, "manifest.json"))}) + "\n")
    log(f"\nDone {now()}. Outputs: {rel(run_dir)}/  (send back console_log.txt and the thesis_tables folder)")


if __name__ == "__main__":
    main()