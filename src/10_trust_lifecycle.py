"""
10_trust_lifecycle.py

Evaluates whether contextual, history-based trust decisions improve on
singular per-flow decisions for IoT-23 using development data only. It
replays raw flows in timestamp order, maps them to the deduplicated rows
with out-of-fold machine-learning risk scores, and evaluates trust
decisions for devices originating traffic from monitored internal networks.
Capture-35-1 and all test captures are excluded.

The script evaluates four risk-model arrangements:
    - hybrid_xgb_rf
    - xgboost_alone
    - random_forest_alone
    - hybrid_rf_xgb

Contextual trust uses a Beta reputation model with exponential forgetting.
For each device, the model updates its benign and malicious evidence using
the flow's risk score, r = P(malicious), and a forgetting factor lambda.
A uniform prior is used at the start of each device's sequence. The resulting
contextual risk is converted into Trusted, Suspicious, or Untrusted decisions
using two thresholds:

    - Trusted:     risk < tau1          -> Allow
    - Suspicious:  tau1 <= risk < tau2  -> Restrict
    - Untrusted:   risk >= tau2         -> Reject

During probation, the first m requests from each device cannot be Allowed
and are instead Restricted. Setting lambda = 0 and m = 0 reproduces the
singular per-flow risk rule, providing a direct comparator.

Parameter selection minimises a three-way decision cost using b = rho = 1/3
and cost ratios C in {1, 2, 5, 10, 20}. The search evaluates forgetting
factors, probation lengths, and ordered threshold pairs on a 0.01 grid,
including a threshold representing 'never'. The objective is class-balanced
and device-equal within each device state: each device's flows within a
state sum to one, and benign and compromised states receive equal weight
when both are present. This reduces the influence of the development
captures' highly imbalanced flow counts.

The script compares contextual and singular rules using the same selection
and evaluation procedure. For cross-validated evaluation, parameters are
selected on all folds except the held-out fold and applied to that fold.
Final parameter estimates are also selected using all development folds
for subsequent test evaluation. Selection and evaluation use device-state
truth: a device is considered benign until its first malicious flow and
compromised thereafter, with no return to the benign state because the
captures contain no remediation events.

The analysis includes:
    - Per-capture device trajectories and descriptive summaries of external
      originators, which are excluded from the lifecycle analysis.
    - Contextual-risk replay and cached flow-to-row mappings, with checks
      against deduplicated row counts, durations, packet counts, labels,
      scenario assignments, and fold assignments.
    - Selected contextual and singular parameters for each model arrangement
      and cost ratio.
    - Cross-validated flow-level decision metrics under per-flow truth,
      device-state truth, and device-state device-averaged weighting.
    - False-trust, over-restriction, rejection, restriction, and trust-level
      rates, along with mean and class-balanced decision costs.
    - Device lifecycle metrics, including time and number of flows until
      first Allow, the share of benign devices ever Trusted, revocations
      after trust, compromise-to-Reject delay, the share of compromised
      devices ever rejected, Allows after compromise, and re-trust after
      the first post-compromise Reject.
    - Decision matrices for each model arrangement, cost ratio, rule, and
      evaluation truth definition.

Subjects are originating devices whose addresses fall within explicit
RFC 1918 IPv4, IPv6 link-local, or IPv6 unique-local ranges. External
originators are summarised descriptively but excluded from continuous
authentication lifecycle metrics.

Usage:
    python src/10_trust_lifecycle.py
"""

import os
import gc
import json
import time

import ipaddress

import numpy as np
import pandas as pd
from scipy.signal import lfilter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INTERIM_SCEN = os.path.join(PROJECT_ROOT, "data", "interim", "iot23_scenarios")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OOF_PATH = os.path.join(PROJECT_ROOT, "results", "08_risk_models", "iot23_oof_probabilities.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "10_trust_lifecycle")
CACHE = os.path.join(OUT_DIR, "replay_cache")
os.makedirs(CACHE, exist_ok=True)

CARVE = "CTU-IoT-Malware-Capture-35-1"
LEVELS = ["Trusted", "Suspicious", "Untrusted"]     # derived from the risk score, never labels
B_RESTRICT = 1 / 3          # cost model of 9_decision_layer.py
RHO_RESTRICT = 1 / 3
ARRANGEMENTS = ["hybrid_xgb_rf", "xgboost_alone", "random_forest_alone", "hybrid_rf_xgb"]
COST_RATIOS = [1, 2, 5, 10, 20]
LAMBDAS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
PROBATION = [0, 1, 5, 10, 25, 50, 100, 250, 1000]
GRID = np.round(np.linspace(0.0, 1.0, 101), 2)
G = len(GRID)
NEVER = G                   # threshold index G = "above 1": never reached
IDENT = ["ts", "id_orig_h", "id_resp_h", "id_orig_p"]
DASH = ["duration", "orig_bytes", "resp_bytes"]
BIG = np.iinfo(np.int64).max // 4
CHUNK = 300_000


def standardize(df):
    df.columns = (df.columns.str.strip().str.lower().str.replace(" ", "_")
                  .str.replace(".", "_", regex=False).str.replace("-", "_", regex=False))
    return df


# ======================================================================
# Stage A: raw flows -> deduplicated rows -> out-of-fold risks (cached)
# ======================================================================

def load_risk_tables():
    oof = pd.read_csv(OOF_PATH, low_memory=False)
    n = int(oof["row_id"].max()) + 1
    tables = {}
    for arr in ARRANGEMENTS:
        r = np.full(n, np.nan, dtype=np.float32)
        r[oof["row_id"].to_numpy()] = np.clip(oof[f"{arr}_risk"].to_numpy(dtype=np.float64), 0, 1).astype(np.float32)
        tables[arr] = r
    meta = {}
    for col in ("true_class", "fold"):
        a = np.full(n, -1, dtype=np.int64)
        a[oof["row_id"].to_numpy()] = oof[col].to_numpy()
        meta[col] = a
    lab = np.empty(n, dtype=object)
    lab[oof["row_id"].to_numpy()] = oof["detailed_label"].astype(str).to_numpy()
    meta["detailed_label"] = lab
    grp = np.empty(n, dtype=object)
    grp[oof["row_id"].to_numpy()] = oof["scenario_group"].astype(str).to_numpy()
    meta["scenario_group"] = grp
    return tables, meta


INTERNAL_NETS = [ipaddress.ip_network(n) for n in
                 ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",   # RFC 1918
                  "fe80::/10", "fc00::/7")]                        # IPv6 link-local, unique-local (RFC 4193)


def is_internal(addr):
    """Subjects of continuous authentication = devices on the monitored network.
    Explicit RFC 1918 / IPv6 link-local / ULA ranges only (Python's is_private also
    covers reserved, documentation and benchmarking ranges, which are not devices)."""
    try:
        ip = ipaddress.ip_address(str(addr))
    except ValueError:
        return False
    return any(ip.version == n.version and ip in n for n in INTERNAL_NETS)


def device_state(y, pos):
    """Compromise-based truth: per device, running maximum of the binary label
    (benign until the first malicious flow, then compromised; never decreases
    because no remediation occurs in the captures)."""
    if len(y) == 0:
        return y.astype(np.int64)
    seg = np.cumsum(pos == 0) - 1
    z = y.astype(np.int64) + 3 * seg
    return np.maximum.accumulate(z) - 3 * seg


def build_capture(cap, offset, n_unique, dev_dur, dev_pkts, tables, meta):
    path = os.path.join(CACHE, f"{cap}_v3_binary.npz")
    if os.path.exists(path):
        return path
    t0 = time.perf_counter()
    rank_of, host_of = {}, {}
    ts_l, host_l, rank_l, dur_l, pkt_l, lab_l = [], [], [], [], [], []
    for chunk in pd.read_csv(os.path.join(INTERIM_SCEN, f"{cap}.csv"), dtype=str,
                             keep_default_na=False, chunksize=CHUNK):
        chunk = standardize(chunk)
        for c in ("label", "detailed_label"):
            chunk[c] = chunk[c].str.strip().str.lower()
        for c in DASH:
            is_dash = chunk[c].eq("-")
            chunk[c] = pd.to_numeric(chunk[c].where(~is_dash), errors="coerce").fillna(0.0).astype("float64")
        key_cols = [c for c in chunk.columns if c not in IDENT]
        h = pd.util.hash_pandas_object(chunk[key_cols], index=False).to_numpy()
        uniq, first = np.unique(h, return_index=True)
        for u in uniq[np.argsort(first)].tolist():
            if u not in rank_of:
                rank_of[u] = len(rank_of)
        rank_l.append(pd.Series(h).map(rank_of).to_numpy(dtype=np.int64))
        hosts = chunk["id_orig_h"].to_numpy()
        for hst in pd.unique(hosts):
            if hst not in host_of:
                host_of[hst] = len(host_of)
        host_l.append(pd.Series(hosts).map(host_of).to_numpy(dtype=np.int32))
        ts_l.append(pd.to_numeric(chunk["ts"], errors="coerce").to_numpy(dtype=np.float64))
        dur_l.append(chunk["duration"].to_numpy(dtype=np.float64))
        pkt_l.append(pd.to_numeric(chunk["orig_pkts"], errors="coerce").to_numpy(dtype=np.float64))
        lab_l.append(chunk["detailed_label"].to_numpy(dtype=object))
    if len(rank_of) != n_unique:
        raise RuntimeError(f"{cap}: {len(rank_of)} unique keys, cleaned file has {n_unique} rows")
    row = offset + np.concatenate(rank_l)
    ts, host = np.concatenate(ts_l), np.concatenate(host_l)
    # verification
    bad_d = int((np.concatenate(dur_l) != dev_dur[row]).sum())
    bad_p = int((np.concatenate(pkt_l) != dev_pkts[row]).sum())
    bad_l = int((np.concatenate(lab_l) != meta["detailed_label"][row]).sum())
    if bad_d or bad_p or bad_l or np.isnan(ts).any():
        raise RuntimeError(f"{cap}: mapping verification failed (duration {bad_d}, orig_pkts {bad_p}, "
                           f"label {bad_l}, NaN ts {int(np.isnan(ts).sum())})")
    if (meta["scenario_group"][row] != "whole_scenario").any():
        raise RuntimeError(f"{cap}: rows outside the whole-scenario group")
    folds = np.unique(meta["fold"][row])
    if len(folds) != 1:
        raise RuntimeError(f"{cap}: capture spans folds {folds}")
    order = np.lexsort((ts, host))                      # device primary, time secondary (stable)
    row, ts, host = row[order], ts[order], host[order]
    start = np.r_[True, host[1:] != host[:-1]]
    seg = np.cumsum(start) - 1
    pos = np.arange(len(row)) - np.flatnonzero(start)[seg]
    names = np.array([str(h) for h in host_of.keys()])          # index = host code
    arrays = {"row": row, "ts": ts, "host": host, "pos": pos,
              "y": meta["true_class"][row].astype(np.int8), "fold": np.int64(folds[0]),
              "host_names": names, "host_internal": np.array([is_internal(h) for h in names])}
    for arr in ARRANGEMENTS:
        arrays[f"{arr}_r"] = tables[arr][row]
    np.savez(path, **arrays)
    print(f"  {cap}: {len(row):,} flows, {len(host_of):,} devices, {n_unique:,} unique rows "
          f"-> verified ({time.perf_counter() - t0:.0f}s)")
    return path


def build_all():
    tables, meta = load_risk_tables()
    dev = pd.read_csv(os.path.join(PROCESSED_DIR, "iot23_train.csv"),
                      usecols=["row_id", "scenario", "duration", "orig_pkts"], low_memory=False)
    n = len(meta["fold"])
    dev_dur = np.full(n, np.nan)
    dev_pkts = np.full(n, np.nan)
    dev_dur[dev["row_id"].to_numpy()] = dev["duration"].to_numpy(dtype=np.float64)
    dev_pkts[dev["row_id"].to_numpy()] = dev["orig_pkts"].to_numpy(dtype=np.float64)
    caps = {}
    for cap, g in dev.groupby("scenario"):
        if cap == CARVE:
            continue
        ids = np.sort(g["row_id"].to_numpy())
        if not np.array_equal(ids, np.arange(ids[0], ids[0] + len(ids))):
            raise RuntimeError(f"{cap}: row_ids are not contiguous")
        caps[cap] = build_capture(cap, int(ids[0]), len(ids), dev_dur, dev_pkts, tables, meta)
    del tables, meta, dev
    gc.collect()
    return caps


def internal(path):
    """Load a capture and keep only flows originated by internal devices.
    Arrays are sorted by device then time, so filtering whole devices keeps
    each device's sequence and 'pos' intact."""
    d = np.load(path)
    flags = np.array([is_internal(h) for h in d["host_names"]])
    keep = flags[d["host"]]
    out = {k: d[k][keep] for k in d.files if k not in ("fold", "host_names", "host_internal")}
    out["fold"] = int(d["fold"])
    out["host_names"] = d["host_names"]
    out["y_state"] = device_state(out["y"], out["pos"])
    return out


def device_tables(caps):
    """Descriptive: every internal device's trajectory; external originators summarised."""
    dev_rows, ext_rows = [], []
    for cap, path in caps.items():
        d = np.load(path)
        y, host, pos, ts = d["y"].astype(np.int64), d["host"], d["pos"], d["ts"]
        names = d["host_names"]
        internal_flag = np.array([is_internal(h) for h in names])
        for h in np.unique(host):
            m = host == h
            yy, tt = y[m], ts[m]
            mal = np.flatnonzero(yy > 0)
            row = {"capture": cap, "fold": int(d["fold"]), "device": str(names[h]),
                   "internal": bool(internal_flag[h]), "flows": int(m.sum()),
                   "benign_flows": int((yy == 0).sum()), "malicious_flows": int((yy == 1).sum()),
                   "onset_position": int(mal[0]) if len(mal) else None,
                   "seconds_before_onset": float(tt[mal[0]] - tt[0]) if len(mal) else None,
                   "seconds_total": float(tt[-1] - tt[0]),
                   "benign_flows_after_onset": int((yy[mal[0]:] == 0).sum()) if len(mal) else None}
            (dev_rows if internal_flag[h] else ext_rows).append(row)
    dev = pd.DataFrame(dev_rows)
    ext = pd.DataFrame(ext_rows)
    ext_sum = ext.groupby("capture").agg(external_originators=("device", "size"), flows=("flows", "sum"),
                                         benign=("benign_flows", "sum"),
                                         malicious=("malicious_flows", "sum")) if len(ext) else pd.DataFrame()
    return dev, ext_sum


# ======================================================================
# Stage B: contextual risks, histograms, selection
# ======================================================================

def contextual(r, pos, lam):
    r = r.astype(np.float64)
    if lam == 0.0:
        return r
    decay = lam ** (pos + 1.0)
    out = []
    for x in (1.0 - r, r):
        y = lfilter([1.0], [1.0, -lam], x)
        start_idx = np.flatnonzero(pos == 0)
        s = start_idx[np.cumsum(pos == 0) - 1]
        prev = np.where(s > 0, y[np.maximum(s - 1, 0)], 0.0)
        out.append(y - prev * decay + decay)          # remove carry-over, add prior (1 each)
    alpha, beta = out
    return beta / (alpha + beta)


def grid_index(r):
    return np.clip(np.searchsorted(GRID, r, side="right") - 1, 0, G - 1)


def outcome_from_hist(H):
    """H: (2, G) counts over (true class, k) -> counts[c, action, a1, a2] for
    tau1 = GRID[a1], tau2 = GRID[a2] (index G = never); meaningful for a1 <= a2."""
    below = np.concatenate([np.zeros((2, 1)), np.cumsum(H, axis=1)], axis=1)   # (2, G+1): rows with k < a
    total = H.sum(axis=1)
    allow = np.broadcast_to(below[:, :, None], (2, G + 1, G + 1))
    rej = np.broadcast_to((total[:, None] - below)[:, None, :], (2, G + 1, G + 1))
    res = np.broadcast_to(below[:, None, :], (2, G + 1, G + 1)) - allow
    return np.stack([allow, res, rej], axis=1)


VALID = np.triu(np.ones((G + 1, G + 1), dtype=bool))     # tau1 <= tau2


def tau_value(a):
    return float(GRID[a]) if a < G else float("inf")


def combined_counts(H2):
    """H2: (2, G, 2) with last axis = [in probation, established]."""
    est = outcome_from_hist(H2[..., 1])
    pro = outcome_from_hist(H2[..., 0]).copy()
    pro[:, 1] += pro[:, 0]          # probation: Allow -> Restrict
    pro[:, 0] = 0
    return est + pro


def cost_matrix(ratio, b=B_RESTRICT, rho=RHO_RESTRICT):
    """rows: true benign (0), malicious (1); columns: Allow, Restrict, Reject."""
    return np.array([[0.0, b, 1.0], [ratio, rho * ratio, 0.0]])


def accumulate_histograms(caps, arr):
    """H[fold][li, mi] -> (2, G, 2) counts."""
    H = {}
    for cap, path in caps.items():
        d = internal(path)
        f = d["fold"]
        H.setdefault(f, np.zeros((len(LAMBDAS), len(PROBATION), 2, G, 2), dtype=np.float64))
        if len(d["y"]) == 0:
            continue
        y, pos = d["y_state"].astype(np.int64), d["pos"]       # selection truth: device state
        w = device_class_weights(y, pos)                       # each (device, state) unit sums to 1
        for li, lam in enumerate(LAMBDAS):
            k = grid_index(contextual(d[f"{arr}_r"], pos, lam))
            base = y * G + k
            for mi, m in enumerate(PROBATION):
                idx = base * 2 + (pos >= m)
                H[f][li, mi] += np.bincount(idx, weights=w, minlength=2 * G * 2).reshape(2, G, 2)
    return H


def device_class_weights(y_state, pos):
    """Weight 1/n for each flow of a (device, true state) unit, so every device
    counts equally within its state (one capture's millions of flows cannot
    decide the parameters)."""
    seg = np.cumsum(pos == 0) - 1
    key = seg * 2 + y_state.astype(np.int64)
    return 1.0 / np.bincount(key)[key]


def select(Hsum, ratio, singular_only=False):
    """Class-balanced objective: mean over true states present of the
    device-averaged cost (equal priors; development base rates are an
    artefact of how the captures were recorded)."""
    cm = cost_matrix(ratio)
    best = (np.inf, None)
    lis = [0] if singular_only else range(len(LAMBDAS))
    for li in lis:
        mis = [0] if singular_only else range(len(PROBATION))
        for mi in mis:
            counts = combined_counts(Hsum[li, mi].astype(np.float64))
            units = counts[:, :, 0, 0].sum(axis=1)
            present = units > 0
            norm = counts[present] / units[present][:, None, None, None]
            mc = np.einsum("ca,caij->ij", cm[present], norm) / present.sum()
            mc = np.where(VALID, mc, np.inf)
            idx = int(np.argmin(mc.T))
            a2, a1 = divmod(idx, G + 1)
            if mc[a1, a2] < best[0] - 1e-12:
                best = (float(mc[a1, a2]), {"lambda": LAMBDAS[li], "probation_m": PROBATION[mi],
                                            "tau1": tau_value(a1), "tau2": tau_value(a2)})
    return best[1], best[0]


# ======================================================================
# Stage C: evaluation with chosen parameters (decisions + lifecycle)
# ======================================================================

def decide(d, arr, p):
    pos = d["pos"]
    cr = contextual(d[f"{arr}_r"], pos, p["lambda"])
    act = np.where(cr >= p["tau2"], 2, np.where(cr >= p["tau1"], 1, 0))
    act = np.where((act == 0) & (pos < p["probation_m"]), 1, act)
    return act


def lifecycle(d, act):
    y, pos, ts = d["y"].astype(np.int64), d["pos"], d["ts"]
    starts = np.flatnonzero(pos == 0)
    seg = np.cumsum(pos == 0) - 1
    n = np.diff(np.r_[starts, len(y)])
    mal = y > 0
    onset = np.minimum.reduceat(np.where(mal, pos, BIG), starts)
    first_allow = np.minimum.reduceat(np.where(act == 0, pos, BIG), starts)
    allowed = np.add.reduceat((act == 0).astype(np.int64), starts)
    post = pos >= onset[seg]
    first_rej = np.minimum.reduceat(np.where((act == 2) & post, pos, BIG), starts)
    allow_post = np.add.reduceat(((act == 0) & post).astype(np.int64), starts)
    after_rej = pos > first_rej[seg]
    retrust = np.add.reduceat(((act == 0) & after_rej).astype(np.int64), starts)
    n_after_rej = np.add.reduceat(after_rej.astype(np.int64), starts)
    revoke = np.r_[False, (act[1:] > 0) & (act[:-1] == 0) & (pos[1:] > 0)]
    revocations = np.add.reduceat(revoke.astype(np.int64), starts)
    t0 = ts[starts]
    def tsec(p):
        ok = p < BIG
        out = np.full(len(p), np.nan)
        out[ok] = ts[starts[ok] + p[ok]] - t0[ok]
        return out
    ent = pd.DataFrame({
        "flows": n, "malicious": onset < BIG,
        "onset_pos": np.where(onset < BIG, onset, np.nan),
        "flows_to_first_allow": np.where(first_allow < BIG, first_allow, np.nan),
        "seconds_to_first_allow": tsec(first_allow),
        "share_allowed": allowed / n,
        "revocations_after_trust": revocations,
        "allows_after_onset": np.where(onset < BIG, allow_post, np.nan),
        "flows_after_onset": np.where(onset < BIG, n - onset, np.nan),
        "delay_flows_onset_to_reject": np.where((onset < BIG) & (first_rej < BIG), first_rej - onset, np.nan),
        "delay_seconds_onset_to_reject": np.where((onset < BIG) & (first_rej < BIG),
                                                  tsec(first_rej) - tsec(np.where(onset < BIG, onset, BIG)), np.nan),
        "ever_rejected_after_onset": first_rej < BIG,
        "allows_after_first_reject": np.where(first_rej < BIG, retrust, np.nan),
        "flows_after_first_reject": np.where(first_rej < BIG, n_after_rej, np.nan),
    })
    cm = np.zeros((2, 3), dtype=np.int64)
    np.add.at(cm, (y, act), 1)
    ys = d["y_state"].astype(np.int64)
    cm_state = np.zeros((2, 3), dtype=np.int64)
    np.add.at(cm_state, (ys, act), 1)
    cm_dev = np.zeros((2, 3), dtype=np.float64)
    np.add.at(cm_dev, (ys, act), device_class_weights(ys, pos))
    comp = {"post_onset_flows": int(post.sum()), "post_onset_allowed": int(((act == 0) & post).sum())}
    return ent, cm, cm_state, cm_dev, comp


def flow_metrics(cm, comp):
    tot = cm.sum(axis=1)
    out = {"flows": float(cm.sum()),
           "false_trust_rate_allow_given_malicious": cm[1, 0] / tot[1] if tot[1] else np.nan,
           "malicious_restricted_rate": cm[1, 1] / tot[1] if tot[1] else np.nan,
           "over_restriction_rate_given_benign": (cm[0, 1] + cm[0, 2]) / tot[0] if tot[0] else np.nan,
           "reject_given_benign": cm[0, 2] / tot[0] if tot[0] else np.nan,
           "compromise_false_trust_allow_after_onset": (comp["post_onset_allowed"] / comp["post_onset_flows"]
                                                        if comp["post_onset_flows"] else np.nan)}
    for k, lvl in enumerate(LEVELS):
        out[f"share_{lvl}"] = float(cm[:, k].sum() / cm.sum()) if cm.sum() else np.nan
    for r in COST_RATIOS:
        cmat = cost_matrix(r)
        out[f"mean_cost_ratio{r}"] = float((cmat * cm).sum() / cm.sum())
        out[f"class_balanced_cost_ratio{r}"] = float(np.mean(
            [(cmat[c] * cm[c]).sum() / tot[c] for c in range(2) if tot[c] > 0]))
    return out


def entity_summary(ent):
    ben, mal = ent[~ent["malicious"]], ent[ent["malicious"]]
    return {
        "benign_devices": len(ben), "malicious_devices": len(mal),
        "benign_share_ever_trusted": float(ben["flows_to_first_allow"].notna().mean()) if len(ben) else np.nan,
        "benign_median_flows_to_first_allow": float(ben["flows_to_first_allow"].median()) if len(ben) else np.nan,
        "benign_median_seconds_to_first_allow": float(ben["seconds_to_first_allow"].median()) if len(ben) else np.nan,
        "benign_flow_weighted_share_allowed": float((ben["share_allowed"] * ben["flows"]).sum() / ben["flows"].sum()) if len(ben) else np.nan,
        "benign_devices_with_revocation": float((ben["revocations_after_trust"] > 0).mean()) if len(ben) else np.nan,
        "malicious_share_ever_rejected": float(mal["ever_rejected_after_onset"].mean()) if len(mal) else np.nan,
        "malicious_median_delay_flows_to_reject": float(mal["delay_flows_onset_to_reject"].median()) if len(mal) else np.nan,
        "malicious_median_delay_seconds_to_reject": float(mal["delay_seconds_onset_to_reject"].median()) if len(mal) else np.nan,
        "malicious_flow_weighted_allow_after_onset": float(mal["allows_after_onset"].sum() / mal["flows_after_onset"].sum()) if len(mal) else np.nan,
        "malicious_retrust_share_after_reject": float(mal["allows_after_first_reject"].sum() / mal["flows_after_first_reject"].sum())
            if len(mal) and mal["flows_after_first_reject"].sum() else np.nan,
    }


# ======================================================================

def main():
    t_all = time.perf_counter()
    print("=" * 70 + "\nTRUST LIFECYCLE (contextual trust), IoT-23 development captures only\n" + "=" * 70)
    print("\n[A] mapping raw flows to out-of-fold risks (cached after first run)")
    caps = build_all()
    cap_fold = {c: int(np.load(p)["fold"]) for c, p in caps.items()}
    folds = sorted(set(cap_fold.values()))
    print(f"  captures by fold: { {f: [c for c in caps if cap_fold[c] == f] for f in folds} }")

    dev_tab, ext_tab = device_tables(caps)
    dev_tab.to_csv(os.path.join(OUT_DIR, "iot23_internal_device_trajectories.csv"), index=False)
    ext_tab.to_csv(os.path.join(OUT_DIR, "iot23_external_originators_summary.csv"))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        print("\nINTERNAL devices (subjects of continuous authentication), per capture:")
        print(dev_tab.to_string(index=False))
        print("\nEXTERNAL originators (excluded from lifecycle analysis), per capture:")
        print(ext_tab.to_string())
    print("\n(Analysis below uses flows originated by internal devices only.)")

    params_rows, flow_rows, ent_rows, ent_detail = [], [], [], []
    for arr in ARRANGEMENTS:
        print(f"\n[B] {arr}: contextual risks and histograms over {len(LAMBDAS)} x {len(PROBATION)} configurations")
        t0 = time.perf_counter()
        H = accumulate_histograms(caps, arr)
        print(f"  done in {time.perf_counter() - t0:.0f}s")
        total = sum(H.values())
        for ratio in COST_RATIOS:
            for kind in ("contextual", "singular"):
                final_p, final_c = select(total, ratio, singular_only=(kind == "singular"))
                params_rows.append({"arrangement": arr, "cost_ratio": ratio, "rule": kind,
                                    "chosen_on": "all development folds", "selection_mean_cost": final_c, **final_p})
                # cross-validated evaluation
                cm_all = np.zeros((2, 3), dtype=np.int64)
                cms_all = np.zeros((2, 3), dtype=np.int64)
                cmd_all = np.zeros((2, 3), dtype=np.float64)
                comp_all = {"post_onset_flows": 0, "post_onset_allowed": 0}
                ents = []
                for f in folds:
                    p, _ = select(sum(H[g] for g in folds if g != f), ratio, singular_only=(kind == "singular"))
                    params_rows.append({"arrangement": arr, "cost_ratio": ratio, "rule": kind,
                                        "chosen_on": f"folds other than {f} (applied to fold {f})", **p})
                    for cap in [c for c in caps if cap_fold[c] == f]:
                        d = internal(caps[cap])
                        if len(d["y"]) == 0:
                            continue
                        act = decide(d, arr, p)
                        ent, cm, cm_s, cm_d, comp = lifecycle(d, act)
                        ent.insert(0, "capture", cap)
                        ents.append(ent)
                        cm_all += cm
                        cms_all += cm_s
                        cmd_all += cm_d
                        for k in comp_all:
                            comp_all[k] += comp[k]
                ent = pd.concat(ents, ignore_index=True)
                ent.insert(0, "rule", kind)
                ent.insert(0, "cost_ratio", ratio)
                ent.insert(0, "arrangement", arr)
                ent_detail.append(ent)
                for truth, mat in (("device_state_device_averaged", cmd_all), ("device_state", cms_all),
                                   ("per_flow", cm_all)):
                    flow_rows.append({"arrangement": arr, "cost_ratio": ratio, "rule": kind, "truth": truth,
                                      "evaluation": "cross-validated", **flow_metrics(mat, comp_all),
                                      "decision_matrix_true_BM_x_action_ARJ": mat.tolist()})
                ent_rows.append({"arrangement": arr, "cost_ratio": ratio, "rule": kind,
                                 "evaluation": "cross-validated", **entity_summary(ent)})
        del H
        gc.collect()

    params = pd.DataFrame(params_rows)
    flows = pd.DataFrame(flow_rows)
    ents = pd.DataFrame(ent_rows)
    params.to_csv(os.path.join(OUT_DIR, "iot23_lifecycle_parameters.csv"), index=False)
    flows.drop(columns=["decision_matrix_true_BM_x_action_ARJ"]).to_csv(
        os.path.join(OUT_DIR, "iot23_lifecycle_flow_metrics.csv"), index=False)
    with open(os.path.join(OUT_DIR, "iot23_lifecycle_decision_matrices.json"), "w") as f:
        json.dump(flows[["arrangement", "cost_ratio", "rule", "truth", "decision_matrix_true_BM_x_action_ARJ"]]
                  .to_dict(orient="records"), f, indent=1)
    ents.to_csv(os.path.join(OUT_DIR, "iot23_lifecycle_device_summary.csv"), index=False)
    pd.concat(ent_detail, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "iot23_lifecycle_device_detail.csv"), index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("\nFinal parameters (chosen on all development folds):")
        print(params[params["chosen_on"] == "all development folds"][
            ["arrangement", "cost_ratio", "rule", "lambda", "probation_m", "tau1", "tau2",
             "selection_mean_cost"]].to_string(index=False))
        cols = ["arrangement", "cost_ratio", "rule", "false_trust_rate_allow_given_malicious",
                "over_restriction_rate_given_benign", "share_Suspicious",
                "compromise_false_trust_allow_after_onset",
                "class_balanced_cost_ratio1", "class_balanced_cost_ratio10"]
        for truth in ("device_state_device_averaged", "device_state", "per_flow"):
            print(f"\nPer-flow decisions (cross-validated), truth = {truth}:")
            print(flows[flows["truth"] == truth][cols].round(4).to_string(index=False))
        print("\nDevice lifecycle (cross-validated):")
        print(ents[["arrangement", "cost_ratio", "rule", "benign_devices", "malicious_devices",
                    "benign_share_ever_trusted", "benign_median_flows_to_first_allow",
                    "benign_flow_weighted_share_allowed", "malicious_share_ever_rejected",
                    "malicious_median_delay_flows_to_reject", "malicious_flow_weighted_allow_after_onset",
                    "malicious_retrust_share_after_reject"]].round(4).to_string(index=False))
    print(f"\nDone in {(time.perf_counter() - t_all) / 60:.0f} min. Outputs: results/10_trust_lifecycle/")


if __name__ == "__main__":
    main()