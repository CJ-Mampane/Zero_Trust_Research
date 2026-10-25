"""
5_feature_selection.py

Performs feature filtering, ranking, feature-count evaluation, and final
feature selection for RT-IoT2022 and IoT-23 using the binary target
'malicious' (0 = benign, 1 = malicious).

The script follows the frozen protocol in
config/feature_selection_protocol.json. The protocol's SHA-256 hash is
recorded in results/05_feature_selection/protocol_lock.json and verified
on subsequent runs. A changed protocol causes execution to stop.

Only development data is loaded. Test partitions are not opened.

Feature selection pipeline:
    1. Apply unsupervised filters to development data:
       - Remove constant numeric variables.
       - Filter rare one-hot categorical levels using the protocol's
         specified rarity populations.
       - Remove highly correlated numeric variables using absolute
         Spearman correlation and the configured tie-break rule.
    2. Rank variables using grouped permutation importance from
       Random Forest and XGBoost models, evaluated with binary macro-F1.
    3. Generate nested rankings so each fold's feature-count evaluation
       uses rankings that exclude that fold from ranking construction.
    4. Evaluate candidate feature counts using cross-validation and apply
       the protocol's configured selection rule, including the
       one-standard-error rule where specified.
    5. Calculate diagnostic measures, including feature-selection
       stability, agreement between rankings, mutual-information rankings,
       and destination-port importance. Diagnostics do not determine
       feature selection.

Model fits are checkpointed so completed computations can be reused
when the script is restarted.

Usage:
    python src/5_feature_selection.py rt_iot2022
    python src/5_feature_selection.py iot23
    python src/5_feature_selection.py
"""

import os
import sys
import gc
import json
import time
import hashlib
import datetime

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PROTOCOL_PATH = os.path.join(PROJECT_ROOT, "config", "feature_selection_protocol.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "05_feature_selection")
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET = "malicious"                  # 0 = benign, 1 = malicious
ROW_ID = "row_id"
MODELS = ["random_forest", "xgboost"]


def rel(p):
    return os.path.relpath(p, PROJECT_ROOT).replace("\\", "/")


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


# ----------------------------------------------------------------------
# protocol lock
# ----------------------------------------------------------------------

def load_protocol():
    if not os.path.exists(PROTOCOL_PATH):
        raise SystemExit(f"Protocol file not found: {rel(PROTOCOL_PATH)}")
    with open(PROTOCOL_PATH, "rb") as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()
    proto = json.loads(raw)
    lock_path = os.path.join(RESULTS_DIR, "protocol_lock.json")
    if os.path.exists(lock_path):
        with open(lock_path) as f:
            lock = json.load(f)
        if lock["sha256"] != digest:
            raise SystemExit(
                "STOP: config/feature_selection_protocol.json has changed since it was locked "
                f"on {lock['locked_at']}. The protocol was frozen before feature selection ran and "
                "must not be edited afterwards. If a change is genuinely required, record it as a "
                "new protocol version with its reason, delete the lock deliberately, and disclose it.")
        print(f"Protocol {proto['protocol_id']} verified against lock (locked {lock['locked_at']}).")
    else:
        with open(lock_path, "w") as f:
            json.dump({"protocol_id": proto["protocol_id"], "protocol_file": rel(PROTOCOL_PATH),
                       "sha256": digest, "locked_at": now()}, f, indent=2)
        print(f"Protocol {proto['protocol_id']} LOCKED now (sha256 {digest[:16]}...).")
    return proto, digest

# checkpointing
class Checkpoint:
    def __init__(self, path):
        self.path = path
        self.done = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self.done[rec["key"]] = rec
        print(f"  checkpoint: {len(self.done)} completed fits found in {rel(path)}")

    def get(self, key):
        return self.done.get(key)

    def put(self, key, payload):
        rec = {"key": key, **to_py(payload)}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.done[key] = rec

# data
def load_development(ds, proto):
    cfg = proto["datasets"][ds]
    dev = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ds}_train.csv"), low_memory=False)
    folds = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["cv_folds_file"]))
    dev = dev.merge(folds, on=ROW_ID, how="left", validate="one_to_one")
    if dev[cfg["fold_column"]].isna().any():
        raise ValueError("Some development rows have no fold assignment.")
    fold = dev[cfg["fold_column"]].to_numpy(dtype=np.int64)
    if cfg["selection_rows"] == "all":
        sel = np.ones(len(dev), dtype=bool)
    else:
        sel = (dev["scenario_group"] == cfg["selection_rows"]).to_numpy()
    carve = ((dev["scenario"] == cfg["carve_out_scenario"]).to_numpy()
             if cfg["carve_out_scenario"] else None)
    y = dev[TARGET].to_numpy(dtype=np.int64)
    if not set(np.unique(y)) <= set(proto["class_codes"].values()):
        raise ValueError("Target values outside the protocol's class codes.")
    meta = set(proto["metadata_columns_never_features"])
    candidates = [c for c in dev.columns if c not in meta]
    print(f"  development rows: {len(dev):,} | folds: {sorted(set(fold))} | "
          f"selection rows: {sel.sum():,} | candidate variables: {len(candidates)}")
    return dev, candidates, fold, sel, carve, y


def find_correlation(frame, threshold):
    corr = frame.corr(method="spearman").abs().fillna(0.0)
    cols = list(corr.columns)
    removed = []
    while len(cols) > 1:
        sub = corr.loc[cols, cols].to_numpy(copy=True)
        np.fill_diagonal(sub, 0.0)
        if sub.max() <= threshold:
            break
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        a, b = cols[i], cols[j]
        mean_a, mean_b = sub[i].mean(), sub[j].mean()
        drop, keep = (a, b) if mean_a > mean_b else (b, a)
        removed.append({"dropped": drop, "kept": keep, "spearman_abs": float(sub[i, j]),
                        "mean_abs_corr_dropped": float(max(mean_a, mean_b)),
                        "mean_abs_corr_kept": float(min(mean_a, mean_b))})
        cols.remove(drop)
    return removed, corr


def fit_filters(dev, candidates, carve, ds, proto):
    flt = proto["filters_unsupervised_fit_on_all_development_rows"]
    cfg = proto["datasets"][ds]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(dev[c])]
    categorical = [c for c in candidates if c not in numeric]

    for c in numeric:
        if not np.isfinite(dev[c].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Non-finite values in numeric variable '{c}'.")
    constant = [c for c in numeric if dev[c].min() == dev[c].max()]
    numeric = [c for c in numeric if c not in constant]

    populations = {"all_development": np.ones(len(dev), dtype=bool)}
    if "non_carveout_development" in cfg["rarity_populations"] and carve is not None:
        populations["non_carveout_development"] = ~carve
    thr = flt["variance_threshold_onehot"]
    kept_levels, level_log, missing_cat = {}, {}, {}
    for c in categorical:
        s = dev[c].astype(object).where(dev[c].notna(), "<missing>").astype(str)
        missing_cat[c] = int(dev[c].isna().sum())
        keep, per_pop = set(), {}
        for pname, mask in populations.items():
            p = s[mask].value_counts(normalize=True)
            passing = set(p[(p * (1 - p)) > thr].index)
            per_pop[pname] = sorted(passing)
            keep |= passing
        kept_levels[c] = sorted(keep)
        only_extra = sorted(set(keep) - set(per_pop["all_development"]))
        level_log[c] = {"n_levels": int(s.nunique()), "kept": kept_levels[c],
                        "kept_only_because_of_non_carveout_population": only_extra}
    removed_cat = [c for c in categorical if not kept_levels[c]]
    categorical = [c for c in categorical if kept_levels[c]]

    excl = flt["correlation"]["exclude_variables"]
    corr_cols = [c for c in numeric if c not in excl]
    removed_corr, spearman = find_correlation(dev[corr_cols], flt["correlation"]["threshold"]) \
        if len(corr_cols) > 1 else ([], pd.DataFrame())
    dropped_corr = [r["dropped"] for r in removed_corr]
    numeric = [c for c in numeric if c not in dropped_corr]

    # Pearson comparison (reporting only)
    pearson_pairs = []
    if len(corr_cols) > 1:
        pc = dev[corr_cols].corr(method="pearson").abs()
        for i, a in enumerate(corr_cols):
            for b in corr_cols[i + 1:]:
                if pc.loc[a, b] > flt["correlation"]["threshold"] or spearman.loc[a, b] > flt["correlation"]["threshold"]:
                    pearson_pairs.append({"a": a, "b": b, "pearson_abs": float(pc.loc[a, b]),
                                          "spearman_abs": float(spearman.loc[a, b])})

    variables = [c for c in candidates if c in numeric or c in categorical]
    log = {"candidate_variables": candidates, "constant_numeric_removed": constant,
           "categorical_levels": level_log, "categorical_removed_no_level_survived": removed_cat,
           "categorical_missing_values_coded_as_<missing>": missing_cat,
           "correlation_removed": removed_corr, "pairs_above_threshold_pearson_vs_spearman": pearson_pairs,
           "surviving_variables": variables,
           "surviving_numeric": [c for c in variables if c in numeric],
           "surviving_categorical": [c for c in variables if c in categorical]}
    print(f"  filters: {len(candidates)} candidates -> {len(variables)} variables "
          f"(constant: {len(constant)}, categorical without surviving level: {len(removed_cat)}, "
          f"correlation: {len(dropped_corr)} {dropped_corr})")
    return log, kept_levels


def encode(dev, variables, numeric, kept_levels):
    arrays, names, blocks = [], [], {}
    for v in variables:
        start = len(names)
        if v in numeric:
            arrays.append(dev[v].to_numpy(dtype=np.float32))
            names.append(v)
        else:
            s = dev[v].astype(object).where(dev[v].notna(), "<missing>").astype(str).to_numpy()
            for lvl in kept_levels[v]:
                arrays.append((s == lvl).astype(np.float32))
                names.append(f"{v}={lvl}")
        blocks[v] = list(range(start, len(names)))
    X = np.column_stack(arrays).astype(np.float32, copy=False)
    del arrays
    return X, names, blocks

# models and scoring
def macro_f1(y, p):
    return float(f1_score(y, p, labels=[0, 1], average="macro", zero_division=0))


def per_class_f1(y, p):
    return [float(v) for v in f1_score(y, p, labels=[0, 1], average=None, zero_division=0)]


def fit_model(name, proto, X, y):
    rs = proto["random_state"]
    inst = proto["ranking"]["instruments"][name]
    if set(np.unique(y)) != {0, 1}:
        raise RuntimeError("A training set is missing a class; fold design violated.")
    if name == "random_forest":
        m = RandomForestClassifier(n_estimators=inst["n_estimators"], class_weight="balanced",
                                   random_state=rs, n_jobs=-1)
        m.fit(X, y)
    else:
        m = XGBClassifier(n_estimators=inst["n_estimators"], tree_method="hist",
                          eval_metric=inst["eval_metric"], random_state=rs, n_jobs=-1)
        m.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return m


def grouped_permutation_importance(model, X_eval, y_eval, blocks, repeats, seed):
    base = macro_f1(y_eval, model.predict(X_eval))
    rng = np.random.default_rng(seed)
    out = {}
    for v, idx in blocks.items():
        drops = []
        for _ in range(repeats):
            perm = rng.permutation(len(X_eval))
            Xp = X_eval.copy()
            Xp[:, idx] = X_eval[np.ix_(perm, idx)]
            drops.append(base - macro_f1(y_eval, model.predict(Xp)))
        out[v] = [float(np.mean(drops)), float(np.std(drops))]
    return base, out

# ranking
def importance_runs(ds, proto, X, y, fold, sel, blocks, ck):
    K = len(np.unique(fold))
    jobs = [("final", -1, g) for g in range(K)]
    jobs += [("nested", f, g) for f in range(K) for g in range(K) if g != f]
    rows = []
    for stage, f, g in jobs:
        train = (fold != g) if stage == "final" else ((fold != f) & (fold != g))
        evalm = (fold == g) & sel
        for mi, m in enumerate(MODELS):
            key = f"importance|stage={stage}|outer={f}|inner={g}|model={m}"
            rec = ck.get(key)
            if rec is None:
                t0 = time.perf_counter()
                model = fit_model(m, proto, X[train], y[train])
                base, imp = grouped_permutation_importance(
                    model, X[evalm], y[evalm], blocks, proto["ranking"]["repeats"],
                    seed=[proto["random_state"], f + 1, g, mi])
                del model
                gc.collect()
                rec = {"stage": stage, "outer": f, "inner": g, "model": m,
                       "train_rows": int(train.sum()), "eval_rows": int(evalm.sum()),
                       "baseline_macro_f1": base, "importance": imp,
                       "seconds": time.perf_counter() - t0}
                ck.put(key, rec)
                print(f"    {stage:6s} outer={f:>2} inner={g} {m:13s} baseline F1={base:.4f} "
                      f"({rec['seconds']:.0f}s)")
            for v, (mean, sd) in rec["importance"].items():
                rows.append({"stage": stage, "outer": f, "inner": g, "model": m, "variable": v,
                             "importance_mean": mean, "importance_sd": sd,
                             "baseline_macro_f1": rec["baseline_macro_f1"]})
    return pd.DataFrame(rows)


def aggregate_ranking(runs, variables):
    per_model = runs.groupby(["model", "variable"])["importance_mean"].mean().unstack("model")
    per_model = per_model.reindex(index=variables)
    per_model["average_importance"] = per_model[MODELS].mean(axis=1)
    per_model["order"] = range(len(variables))
    per_model = per_model.sort_values(["average_importance", "order"], ascending=[False, True])
    per_model["rank"] = range(1, len(per_model) + 1)
    return per_model.drop(columns="order")


# count curve + one-SE rule
def candidate_counts(n):
    if n <= 10:
        return list(range(1, n + 1))
    step = max(1, round(n / 10))
    counts = list(range(step, n + 1, step))
    if counts[-1] != n:
        counts.append(n)
    return counts


def count_curve(proto, X, y, fold, sel, carve, blocks, nested_rankings, ck):
    K = len(np.unique(fold))
    counts = candidate_counts(len(blocks))
    print(f"  candidate counts: {counts}")
    rows = []
    for n in counts:
        for f in range(K):
            top = nested_rankings[f].index[:n].tolist()
            cols = sorted(c for v in top for c in blocks[v])
            train, val = fold != f, fold == f
            for m in MODELS:
                key = f"curve|n={n}|fold={f}|model={m}"
                rec = ck.get(key)
                if rec is None:
                    t0 = time.perf_counter()
                    model = fit_model(m, proto, X[train][:, cols], y[train])
                    pred = model.predict(X[val][:, cols])
                    del model
                    gc.collect()
                    yv, sv = y[val], sel[val]
                    rec = {"n_variables": n, "fold": f, "model": m, "variables": top,
                           "f1_selection_rows": macro_f1(yv[sv], pred[sv]),
                           "per_class_f1_selection_rows": per_class_f1(yv[sv], pred[sv]),
                           "f1_all_validation_rows": macro_f1(yv, pred),
                           "seconds": time.perf_counter() - t0}
                    if carve is not None:
                        cv = carve[val]
                        rec["f1_carveout_rows"] = macro_f1(yv[cv], pred[cv])
                    ck.put(key, rec)
                    print(f"    n={n:>3} fold={f} {m:13s} F1(selection)={rec['f1_selection_rows']:.4f} "
                          f"F1(all)={rec['f1_all_validation_rows']:.4f} ({rec['seconds']:.0f}s)")
                rows.append({k: v for k, v in rec.items() if k not in ("key", "variables")})
    return pd.DataFrame(rows)


def one_standard_error(curve):
    per_fold = curve.groupby(["n_variables", "fold"])["f1_selection_rows"].mean().unstack("fold")
    K = per_fold.shape[1]
    summary = pd.DataFrame({
        "mean_f1_selection_rows": per_fold.mean(axis=1),
        "sd_across_folds": per_fold.std(axis=1, ddof=1),
    })
    summary["se"] = summary["sd_across_folds"] / np.sqrt(K)
    for col in ("f1_all_validation_rows", "f1_carveout_rows"):
        if col in curve.columns:
            summary[f"mean_{col}"] = curve.groupby("n_variables")[col].mean()
    for m in MODELS:
        summary[f"mean_f1_selection_rows_{m}"] = curve[curve["model"] == m] \
            .groupby("n_variables")["f1_selection_rows"].mean()
    for f in per_fold.columns:
        summary[f"fold{f}_f1_selection_rows"] = per_fold[f]
    best_n = int(summary["mean_f1_selection_rows"].idxmax())
    se = float(summary.loc[best_n, "se"])
    threshold = float(summary.loc[best_n, "mean_f1_selection_rows"] - se)
    selected_n = int(min(summary.index[summary["mean_f1_selection_rows"] >= threshold]))
    summary["is_best"] = summary.index == best_n
    summary["is_selected"] = summary.index == selected_n
    return summary, best_n, se, threshold, selected_n


# diagnostics

def nogueira_stability(sets, d):
    M = len(sets)
    if M < 2 or d == 0:
        return None
    Z = np.zeros((M, d))
    for i, s in enumerate(sets):
        Z[i, list(s)] = 1
    k_bar = Z.sum(axis=1).mean()
    if k_bar in (0, d):
        return None
    p = Z.mean(axis=0)
    s2 = (M / (M - 1)) * p * (1 - p)
    return float(1 - s2.mean() / ((k_bar / d) * (1 - k_bar / d)))


def mutual_information_ranking(dev, variables, numeric, y, proto):
    n_max = proto["mi_sample_rows"]
    idx = np.arange(len(dev))
    if len(idx) > n_max:
        idx, _ = train_test_split(idx, train_size=n_max, stratify=y, random_state=proto["random_state"])
    cols, discrete = [], []
    for v in variables:
        if v in numeric:
            cols.append(dev[v].to_numpy(dtype=np.float64)[idx])
            discrete.append(False)
        else:
            cols.append(pd.Categorical(dev[v].astype(str)).codes[idx].astype(np.float64))
            discrete.append(True)
    Xmi = np.column_stack(cols)
    mi = mutual_info_classif(Xmi, y[idx], discrete_features=np.array(discrete),
                             random_state=proto["random_state"])
    return pd.Series(mi, index=variables).sort_values(ascending=False), int(len(idx))


# main per dataset

def run_dataset(ds, proto, digest):
    print("=" * 70 + f"\nFEATURE SELECTION ({proto['protocol_id']}): {ds}\n" + "=" * 70)
    t_start = time.perf_counter()
    dev, candidates, fold, sel, carve, y = load_development(ds, proto)
    K = len(np.unique(fold))

    log, kept_levels = fit_filters(dev, candidates, carve, ds, proto)
    variables, numeric = log["surviving_variables"], log["surviving_numeric"]
    with open(os.path.join(RESULTS_DIR, f"{ds}_filter_log.json"), "w") as f:
        json.dump(to_py(log), f, indent=2)

    X, names, blocks = encode(dev, variables, numeric, kept_levels)
    print(f"  encoded matrix: {X.shape[0]:,} rows x {X.shape[1]} columns "
          f"({len(variables)} variables)")
    mi, mi_rows = mutual_information_ranking(dev, variables, numeric, y, proto)
    row_ids = dev[ROW_ID].to_numpy()
    trust = dev[TARGET].to_numpy()
    del dev
    gc.collect()

    ck = Checkpoint(os.path.join(RESULTS_DIR, f"{ds}_checkpoint.jsonl"))

    print("\n  [ranking] grouped permutation importance")
    runs = importance_runs(ds, proto, X, y, fold, sel, blocks, ck)
    runs.to_csv(os.path.join(RESULTS_DIR, f"{ds}_importance_runs.csv"), index=False)
    final_rank = aggregate_ranking(runs[runs["stage"] == "final"], variables)
    nested = {f: aggregate_ranking(runs[(runs["stage"] == "nested") & (runs["outer"] == f)], variables)
              for f in range(K)}
    final_rank.to_csv(os.path.join(RESULTS_DIR, f"{ds}_ranking_final.csv"))
    pd.concat([r.assign(outer_fold=f) for f, r in nested.items()]) \
        .to_csv(os.path.join(RESULTS_DIR, f"{ds}_rankings_nested.csv"))

    print("\n  [curve] feature-count curve on nested rankings")
    curve = count_curve(proto, X, y, fold, sel, carve, blocks, nested, ck)
    curve.to_csv(os.path.join(RESULTS_DIR, f"{ds}_count_curve_by_fold.csv"), index=False)
    summary, best_n, se, threshold, selected_n = one_standard_error(curve)
    summary.to_csv(os.path.join(RESULTS_DIR, f"{ds}_count_curve_summary.csv"))

    one_se_n = selected_n
    count_rule = proto["datasets"][ds].get("count_rule", "one_standard_error")
    if count_rule == "all_filtered_variables":
        selected_n = len(variables)
    elif count_rule != "one_standard_error":
        raise SystemExit(f"Unknown count_rule '{count_rule}' in the protocol.")
    selected_vars = final_rank.index[:selected_n].tolist()
    selected_cols = [names[c] for v in selected_vars for c in blocks[v]]
    col_idx = [c for v in selected_vars for c in blocks[v]]

    # diagnostics
    var_index = {v: i for i, v in enumerate(variables)}
    sets = [{var_index[v] for v in nested[f].index[:selected_n]} for f in range(K)]
    taus = []
    for a in range(K):
        for b in range(a + 1, K):
            ra = nested[a]["rank"].reindex(variables).to_numpy()
            rb = nested[b]["rank"].reindex(variables).to_numpy()
            taus.append({"folds": [a, b], "kendall_tau": float(kendalltau(ra, rb).statistic)})
    mi_vs_perm = spearmanr(mi.reindex(variables).to_numpy(),
                           final_rank["average_importance"].reindex(variables).to_numpy()).statistic
    port = "id_resp_p"
    port_info = None
    if port in variables:
        port_info = {"final_rank": int(final_rank.loc[port, "rank"]),
                     "final_average_importance": float(final_rank.loc[port, "average_importance"]),
                     "nested_ranks": {f: int(nested[f].loc[port, "rank"]) for f in range(K)},
                     "selected": port in selected_vars}
    diag = {
        "protocol_id": proto["protocol_id"], "protocol_sha256": digest,
        "K_folds": K, "n_variables_after_filters": len(variables),
        "best_n": best_n, "best_mean_f1_selection_rows": float(summary.loc[best_n, "mean_f1_selection_rows"]),
        "se_of_best": se, "one_se_threshold": threshold, "one_se_rule_n": one_se_n,
        "count_rule_applied": count_rule, "selected_n": selected_n,
        "selected_variables": selected_vars,
        "nogueira_stability_of_top_n_across_nested_rankings": nogueira_stability(sets, len(variables)),
        "kendall_tau_between_nested_rankings": taus,
        "mutual_information_ranking": mi.to_dict(), "mi_sample_rows": mi_rows,
        "spearman_mi_vs_permutation_importance": None if np.isnan(mi_vs_perm) else float(mi_vs_perm),
        "destination_port": port_info,
        "runtime_seconds_this_run": time.perf_counter() - t_start,
    }
    with open(os.path.join(RESULTS_DIR, f"{ds}_diagnostics.json"), "w") as f:
        json.dump(to_py(diag), f, indent=2)

    spec = {
        "protocol_id": proto["protocol_id"], "protocol_sha256": digest,
        "selected_variables": selected_vars, "selected_columns": selected_cols,
        "numeric_variables": [v for v in selected_vars if v in numeric],
        "categorical_levels": {v: kept_levels[v] for v in selected_vars if v not in numeric},
        "encoding_rule": "numeric: value as float32; categorical: one 0/1 column '<var>=<level>' per "
                         "kept level, missing coded '<missing>', unseen or dropped levels -> all zeros",
        "class_codes": proto["class_codes"],
        "created": now(),
    }
    with open(os.path.join(RESULTS_DIR, f"{ds}_selection.json"), "w") as f:
        json.dump(spec, f, indent=2)

    out = pd.DataFrame(X[:, col_idx], columns=selected_cols)
    out.insert(0, ROW_ID, row_ids)
    out[TARGET] = trust
    out_path = os.path.join(PROCESSED_DIR, f"{ds}_train_selected.csv")
    out.to_csv(out_path, index=False)

    # console report
    print("\n  Final ranking (grouped permutation importance, selection rows):")
    print(final_rank.round(5).to_string())
    print("\n  Count curve (decision column: mean_f1_selection_rows):")
    print(summary.round(4).to_string())
    print(f"\n  Best n = {best_n} (mean {summary.loc[best_n, 'mean_f1_selection_rows']:.4f}); "
          f"SE = {se:.4f}; threshold = {threshold:.4f}; one-SE rule n = {one_se_n}; "
          f"rule applied: {count_rule}; SELECTED n = {selected_n}")
    print(f"  Selected variables: {selected_vars}")
    print(f"  Encoded columns ({len(selected_cols)}): {selected_cols}")
    print(f"  Nogueira stability (top-{selected_n} across nested rankings): "
          f"{diag['nogueira_stability_of_top_n_across_nested_rankings']}")
    print(f"  Kendall tau between nested rankings: {taus}")
    print(f"  Spearman(MI ranking, permutation ranking): {diag['spearman_mi_vs_permutation_importance']}")
    if port_info:
        print(f"  Destination port: {port_info}")
    print(f"  Saved: {rel(out_path)}  shape={out.shape}\n")


if __name__ == "__main__":
    proto, digest = load_protocol()
    targets = sys.argv[1:] or ["rt_iot2022", "iot23"]
    for ds in targets:
        if ds not in proto["datasets"]:
            raise SystemExit(f"Unknown dataset '{ds}'.")
        run_dataset(ds, proto, digest)
    print("Done. Outputs: data/processed/*_train_selected.csv, results/05_feature_selection/")