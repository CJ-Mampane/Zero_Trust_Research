"""
2_data_cleaning.py

Cleans the outputs of 1_data_loading.py and creates the binary target
'malicious' (0 = benign, 1 = malicious), taken directly from each dataset's
own labels. No trust levels are assigned here: Trusted / Suspicious /
Untrusted are derived later from the model's risk score and two
cost-based thresholds (9_decision_layer.py). Every content-based
decision in the project is made HERE, visibly, with its justification
written next to it and its effect logged per class (and, for IoT-23,
per scenario).


DECISIONS (what, why, evidence)

D1. BINARY TARGET 'malicious' FROM THE DATASETS' OWN LABELS
    IoT-23: the dataset's 'label' column (benign / malicious) is used
    as it is: malicious = 1 if label == 'malicious'. Stratosphere counts
    every non-benign flow, including horizontal port scans, as malicious.
    RT-IoT2022: the file has no benign/malicious column, so the split
    published on the UCI repository page is used: 'Normal Patterns'
    (MQTT, Thing_speak, Wipro_bulb) -> 0; 'Attack patterns' (DoS SYN,
    ARP poisoning, the five Nmap scans, DDoS Slowloris, Metasploit SSH
    brute force) -> 1.
    No attack type is assigned to an intermediate level by the
    researcher. The original attack names (attack_type, detailed_label)
    are kept as METADATA for reporting only, never as features.
    Any label value not covered is a hard error (not dropped).
    The label tables are saved to results/02_cleaning/ for the appendix.
    Labels are PER FLOW, not per device.

D2. IDENTIFIER REMOVAL BEFORE DEDUPLICATION  (leakage principle)
    IoT-23: ts, id.orig_h, id.resp_h, id.orig_p (timestamp, IPs,
    ephemeral source port). RT-IoT2022: 'Unnamed: 0' (row index),
    source_file (provenance), id.orig_p (ephemeral source port).
    The RT-IoT2022 raw file contains no IP, timestamp or session column
    (verified by 0_raw_data_inventory.py). Destination port (id.resp_p)
    is KEPT as a candidate feature; its use is tested later (ablation).
    Rationale: a model conditioned on an IP/timestamp/source port does
    not generalise to unseen devices/times; rows differing only in these
    fields are the same recorded behaviour.

D3. DEDUPLICATION  (leakage principle; unit of analysis changes)
    Key = every remaining column after D2 (no hand-picked subset).
    A duplicate split across train/test lets a model be tested on a row
    it memorised. CONSEQUENCE, stated in the thesis: the unit of analysis
    becomes "distinct flow records", not "flows as they occurred".
    To keep the traffic-volume view recoverable, each retained row gets
    row_multiplicity = how many raw rows it stands for. This is METADATA
    and must NEVER be used as a model feature.
    IoT-23 deduplication is PER SCENARIO: 'scenario' is part of the key,
    so identical flows in two different captures are both kept (two
    independent captures observing the same behaviour). This matches
    what the historical script actually did (its docstring wrongly said
    "global"). How many rows a global dedup would ALSO remove is
    reported as a diagnostic, not applied.

D4. IoT-23 '-' -> 0 for duration, orig_bytes, resp_bytes  (data type fix)
    Zeek writes '-' when a value was not measured (e.g. a connection
    with no reply). 0 is used as the numeric encoding. Disclosed
    limitation: this merges "not measured" with "zero". Counts of
    replaced values are logged per scenario.

D5. IoT-23 label vs detailed-label consistency  (verify, don't assume)
    'benign' must pair with '-', 'malicious' must not. Violations are
    written to results/02_cleaning/label_violations/ and excluded.
    Label text is lower-cased first ('Benign' vs 'benign' both occur).

D6. RT-IoT2022 missing values / infinities
    The raw file has none (inventory). This script RAISES if any appear
    rather than silently dropping rows.

CHANGES FROM THE HISTORICAL VERSION
 a. IoT-23 numeric columns from D4 are always cast to float64 before
    hashing. Historically pd.to_numeric returned int64 in a chunk with no
    '-' and float64 in a chunk with one; int 5 and float 5.0 hash
    differently, so the same row could escape deduplication depending on
    chunk boundaries. Deduplication is now deterministic. The retained
    row count may therefore be slightly LOWER than the historical ~2.13M.
 b. IoT-23 is processed one scenario at a time (dedup is per scenario,
    so this is exactly equivalent) with restart support.
 c. row_multiplicity metadata column added (D3).
 d. Per-step, per-class (and per-scenario) counts are logged.
 e. Unmapped labels and missing values raise errors instead of being
    dropped with only a printed warning.
 f. Diagnostics: identical feature vectors carrying DIFFERENT trust
    states (label conflicts), and feature vectors shared across
    scenarios. Reported only; nothing is removed on their basis.

Hash note: rows are compared via 64-bit hashes. With ~2M distinct rows
the probability of any collision is ~1e-7 (negligible; stated for
completeness).

Usage (from the project root or src/):
    python src/2_data_cleaning.py                # both datasets
    python src/2_data_cleaning.py rt_iot2022     # RT only (seconds)
    python src/2_data_cleaning.py iot23          # IoT-23 only (hours; restartable)

"""

import os
import sys
import glob
import json
import time
import datetime
from collections import Counter

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INTERIM_DIR = os.path.join(PROJECT_ROOT, "data", "interim")
IOT23_SCENARIO_DIR = os.path.join(INTERIM_DIR, "iot23_scenarios")
IOT23_CLEAN_SCEN_DIR = os.path.join(INTERIM_DIR, "iot23_clean_scenarios")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "02_cleaning")
VIOLATIONS_DIR = os.path.join(RESULTS_DIR, "label_violations")
for d in (IOT23_CLEAN_SCEN_DIR, PROCESSED_DIR, RESULTS_DIR, VIOLATIONS_DIR):
    os.makedirs(d, exist_ok=True)

CHUNK_SIZE = 300_000
TARGET = "malicious"               # 0 = benign, 1 = malicious (dataset labels)
MULT = "row_multiplicity"          # metadata, never a feature

# ---------------------------------------------------------------- RT
RT_LABEL_COLUMN = "attack_type"
RT_INDEX_COLUMNS = ["unnamed:_0"]
RT_PROVENANCE_COLUMNS = ["source_file"]
RT_IDENTIFIER_COLUMNS = ["id_orig_p"]
# Normal vs attack split as published on the UCI repository page
# (https://archive.ics.uci.edu/dataset/942/rt-iot2022).
RT_NORMAL_PATTERNS = ["mqtt_publish", "thing_speak", "wipro_bulb"]
RT_ATTACK_PATTERNS = ["dos_syn_hping", "arp_poisioning", "nmap_udp_scan",
                      "nmap_xmas_tree_scan", "nmap_os_detection", "nmap_tcp_scan",
                      "nmap_fin_scan", "ddos_slowloris", "metasploit_brute_force_ssh"]
RT_MALICIOUS_MAP = {**{k: 0 for k in RT_NORMAL_PATTERNS}, **{k: 1 for k in RT_ATTACK_PATTERNS}}

# ---------------------------------------------------------------- IoT-23
IOT23_IDENTIFIER_RAW = ["ts", "id.orig_h", "id.resp_h", "id.orig_p"]   # names in interim files
IOT23_MALICIOUS_MAP = {"benign": 0, "malicious": 1}   # the dataset's own 'label' column
DASH_TO_ZERO = ["duration", "orig_bytes", "resp_bytes"]
OTHER_NUMERIC = ["id_resp_p", "missed_bytes", "orig_pkts", "orig_ip_bytes",
                 "resp_pkts", "resp_ip_bytes"]
LABEL_COLS = ["label", "detailed_label"]


def rel(path):
    return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")


def standardize_columns(df):
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_").str.replace(".", "_", regex=False)
                  .str.replace("-", "_", regex=False))
    return df


def save_mapping_tables():
    pd.DataFrame({"attack_type_lowercase": list(RT_MALICIOUS_MAP),
                  "uci_page_group": ["Normal Patterns" if v == 0 else "Attack patterns"
                                     for v in RT_MALICIOUS_MAP.values()],
                  TARGET: list(RT_MALICIOUS_MAP.values())}) \
        .to_csv(os.path.join(RESULTS_DIR, "rt_iot2022_label_mapping.csv"), index=False)
    pd.DataFrame({"label_lowercase": list(IOT23_MALICIOUS_MAP),
                  TARGET: list(IOT23_MALICIOUS_MAP.values())}) \
        .to_csv(os.path.join(RESULTS_DIR, "iot23_label_mapping.csv"), index=False)
    for old in ("rt_iot2022_trust_state_before_after.csv", "iot23_trust_state_before_after.csv",
                "iot23_scenario_trust_state_crosstab.csv"):
        if os.path.exists(os.path.join(RESULTS_DIR, old)):
            os.remove(os.path.join(RESULTS_DIR, old))


# RT-IoT2022 (123,117 rows -- fits in memory)

def clean_rt_iot2022():
    print("=" * 70 + "\nRT-IoT2022\n" + "=" * 70)
    df = pd.read_csv(os.path.join(INTERIM_DIR, "rt_iot2022_merged.csv"), low_memory=False)
    df = standardize_columns(df)
    n_raw = len(df)

    required = RT_INDEX_COLUMNS + RT_PROVENANCE_COLUMNS + RT_IDENTIFIER_COLUMNS + [RT_LABEL_COLUMN]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Expected columns not found: {missing_cols}")

    # D1 -- mapping first (it changes no rows, and lets every step be logged per class)
    norm = df[RT_LABEL_COLUMN].astype(str).str.strip().str.lower()
    unmapped = sorted(set(norm) - set(RT_MALICIOUS_MAP))
    if unmapped:
        raise ValueError(f"attack_type values not on the UCI normal/attack lists: {unmapped}")
    df[TARGET] = norm.map(RT_MALICIOUS_MAP).astype("int64")

    # D6 -- verify, never silently drop
    n_missing = int(df.isna().sum().sum())
    num = df.select_dtypes("number")
    n_inf = int(np.isinf(num.to_numpy(dtype="float64")).sum()) if num.shape[1] else 0
    if n_missing or n_inf:
        raise ValueError(f"{n_missing} missing / {n_inf} infinite values found -- a "
                         f"handling decision is needed before cleaning continues.")
    print(f"Loaded {n_raw:,} rows; 0 missing, 0 infinite values (verified)")

    # D2 -- index + provenance columns (no row changes)
    df = df.drop(columns=RT_INDEX_COLUMNS + RT_PROVENANCE_COLUMNS)

    # Diagnostic: what dedup would remove if the source port were KEPT
    key_with_port = [c for c in df.columns if c != TARGET]
    dup_if_port_kept = df.duplicated(subset=key_with_port, keep="first")

    # D2 -- source port removed BEFORE dedup
    df = df.drop(columns=RT_IDENTIFIER_COLUMNS)

    # D3 -- dedup on every remaining column, with multiplicity
    key = [c for c in df.columns if c != TARGET]
    h = pd.util.hash_pandas_object(df[key], index=False)
    multiplicity = h.map(h.value_counts())
    keep = ~h.duplicated(keep="first")
    clean = df.loc[keep].copy()
    clean[MULT] = multiplicity[keep].astype("int64").to_numpy()
    assert int(clean[MULT].sum()) == n_raw, "multiplicities do not sum to the raw row count"

    # Per-class effect of each step
    steps = pd.DataFrame({
        "raw_rows": df.groupby(RT_LABEL_COLUMN).size(),
        "unique_rows_if_source_port_kept": df.loc[~dup_if_port_kept.to_numpy()].groupby(RT_LABEL_COLUMN).size(),
        "unique_rows_primary": clean.groupby(RT_LABEL_COLUMN).size(),
    }).fillna(0).astype("int64")
    steps.insert(0, TARGET, [RT_MALICIOUS_MAP[str(i).strip().lower()] for i in steps.index])
    steps["share_retained_primary"] = (steps["unique_rows_primary"] / steps["raw_rows"]).round(4)
    steps = steps.sort_values("raw_rows", ascending=False)
    total = steps.drop(columns=[TARGET, "share_retained_primary"]).sum()
    steps.loc["TOTAL"] = [""] + total.tolist() + [round(total["unique_rows_primary"] / total["raw_rows"], 4)]
    steps.to_csv(os.path.join(RESULTS_DIR, "rt_iot2022_cleaning_steps_by_class.csv"))

    by_state = pd.DataFrame({
        "raw_rows": df.groupby(TARGET).size(),
        "unique_rows_primary": clean.groupby(TARGET).size()})
    by_state["raw_share"] = (by_state["raw_rows"] / by_state["raw_rows"].sum()).round(4)
    by_state["unique_share"] = (by_state["unique_rows_primary"] / by_state["unique_rows_primary"].sum()).round(4)
    by_state.index.name = TARGET
    by_state.to_csv(os.path.join(RESULTS_DIR, "rt_iot2022_malicious_before_after.csv"))

    # Diagnostic f: identical feature vectors with different labels
    feat = [c for c in clean.columns if c not in (TARGET, MULT, RT_LABEL_COLUMN)]
    hf = pd.util.hash_pandas_object(clean[feat], index=False)
    n_states = clean.groupby(hf.to_numpy())[TARGET].transform("nunique")
    n_attacks = clean.groupby(hf.to_numpy())[RT_LABEL_COLUMN].transform("nunique")
    conflict_rows = clean.loc[(n_attacks > 1).to_numpy()].copy()
    conflict_rows.insert(0, "feature_vector_id", hf[(n_attacks > 1).to_numpy()].to_numpy())
    conflict_rows.sort_values("feature_vector_id").to_csv(
        os.path.join(RESULTS_DIR, "rt_iot2022_conflicting_label_rows.csv"), index=False)

    out = os.path.join(PROCESSED_DIR, "rt_iot2022_clean.csv")
    clean.to_csv(out, index=False)

    diag = {
        "rows_raw": n_raw,
        "columns_removed": RT_INDEX_COLUMNS + RT_PROVENANCE_COLUMNS + RT_IDENTIFIER_COLUMNS,
        "duplicates_removed_if_source_port_kept": int(dup_if_port_kept.sum()),
        "duplicates_removed_primary": int(n_raw - len(clean)),
        "rows_retained_primary": int(len(clean)),
        "feature_vectors_with_more_than_one_attack_type": int(hf[(n_attacks > 1).to_numpy()].nunique()),
        "feature_vectors_with_both_benign_and_malicious": int(hf[(n_states > 1).to_numpy()].nunique()),
        "rows_involved_in_benign_malicious_conflicts": int((n_states > 1).sum()),
        "output_columns": int(clean.shape[1]),
    }
    with open(os.path.join(RESULTS_DIR, "rt_iot2022_cleaning_summary.json"), "w") as f:
        json.dump(diag, f, indent=2)

    print("\nPer attack type (raw -> if source port kept -> primary):")
    print(steps.to_string())
    print("\nBenign (0) / malicious (1) before/after:")
    print(by_state.to_string())
    print("\nSummary:")
    for k, v in diag.items():
        print(f"  {k}: {v}")
    print(f"Saved: {rel(out)}  shape={clean.shape}\n")


# ======================================================================
# IoT-23 stage 1: one scenario at a time (restartable)
# ======================================================================

def convert_old_scenario_file(final, stats_path):
    """A per-scenario file cleaned earlier carries a 'trust_state' column
    instead of 'malicious'. Deduplication never used the target (key = all
    columns except the target), so the retained rows and multiplicities are
    exactly those this script would produce now. The file is therefore
    converted in place (trust_state -> malicious from the dataset 'label')
    instead of re-reading the raw scenario, and its stats are recomputed."""
    print("  cleaned earlier with 3-level target -> converting to binary 'malicious' (same rows)")
    tmp = final + ".conv"
    if os.path.exists(tmp):
        os.remove(tmp)
    raw_by, uniq_by = Counter(), Counter()
    hdr = False
    n_unique = 0
    for part in pd.read_csv(final, dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE):
        lab = part["label"].str.strip().str.lower().map(IOT23_MALICIOUS_MAP)
        if lab.isna().any():
            raise ValueError(f"{final}: label values other than benign/malicious")
        cols = part.columns.tolist()
        part.insert(cols.index("trust_state"), TARGET, lab.astype("int64"))
        part = part.drop(columns=["trust_state"])
        for k, v in part.groupby([TARGET, "detailed_label"]).size().items():
            uniq_by[k] += int(v)
        for k, v in part.groupby([TARGET, "detailed_label"])[MULT].apply(
                lambda s: s.astype("int64").sum()).items():
            raw_by[k] += int(v)
        n_unique += len(part)
        part.to_csv(tmp, mode="a", header=not hdr, index=False)
        hdr = True
    with open(stats_path) as f:
        stats = json.load(f)
    if n_unique != stats["rows_unique"] or sum(raw_by.values()) != stats["rows_raw"] - stats["rows_label_violations"]:
        raise RuntimeError(f"{final}: converted counts do not match the earlier stats -- refusing to save")
    os.replace(tmp, final)
    stats["by_class"] = [{TARGET: int(k[0]), "detailed_label": k[1],
                          "raw_rows": raw_by[k], "unique_rows": uniq_by[k]} for k in sorted(raw_by)]
    with open(stats_path, "w") as f:
        json.dump(stats, f)
    print(f"  {n_unique:,} unique rows converted")
    return stats


def clean_iot23_scenario(path):
    scen = os.path.basename(path)[:-4]
    final = os.path.join(IOT23_CLEAN_SCEN_DIR, f"{scen}.csv")
    stats_path = os.path.join(IOT23_CLEAN_SCEN_DIR, f"{scen}.stats.json")
    tmp1, tmp2 = final + ".tmp", final + ".tmp2"
    viol_path = os.path.join(VIOLATIONS_DIR, f"{scen}.csv")

    if os.path.exists(final) and os.path.exists(stats_path):
        if "trust_state" in pd.read_csv(final, nrows=0).columns:
            return convert_old_scenario_file(final, stats_path)
        print("  already done -> skipped")
        with open(stats_path) as f:
            return json.load(f)
    for p in (tmp1, tmp2, viol_path):
        if os.path.exists(p):
            os.remove(p)

    header = pd.read_csv(path, nrows=0).columns.tolist()
    absent = [c for c in IOT23_IDENTIFIER_RAW if c not in header]
    if absent:
        raise ValueError(f"{scen}: expected identifier columns missing: {absent}")
    usecols = [c for c in header if c not in IOT23_IDENTIFIER_RAW]     # D2 at read time

    counts = {}                         # row hash -> raw rows it represents
    raw_by, uniq_by = Counter(), Counter()
    dash, other_nonnum, nonnum_examples = Counter(), Counter(), {}
    n_raw = n_viol = 0
    written = viol_written = False

    reader = pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False,
                         chunksize=CHUNK_SIZE)
    for chunk in reader:
        chunk = standardize_columns(chunk)
        n_raw += len(chunk)
        for c in LABEL_COLS:
            chunk[c] = chunk[c].str.strip().str.lower()

        # D5
        bad = ((chunk["label"] == "benign") & (chunk["detailed_label"] != "-")) | \
              ((chunk["label"] == "malicious") & (chunk["detailed_label"] == "-"))
        if bad.any():
            chunk.loc[bad].to_csv(viol_path, mode="a", header=not viol_written, index=False)
            viol_written = True
            n_viol += int(bad.sum())
            chunk = chunk.loc[~bad]
        if chunk.empty:
            continue

        # D1 -- the dataset's own benign/malicious label
        chunk[TARGET] = chunk["label"].map(IOT23_MALICIOUS_MAP)
        if chunk[TARGET].isna().any():
            raise ValueError(f"{scen}: label values other than benign/malicious: "
                             f"{sorted(chunk.loc[chunk[TARGET].isna(), 'label'].unique())}")
        chunk[TARGET] = chunk[TARGET].astype("int64")

        # D4 (+ change a: always float64)
        for c in DASH_TO_ZERO:
            is_dash = chunk[c].eq("-")
            dash[c] += int(is_dash.sum())
            numeric = pd.to_numeric(chunk[c].where(~is_dash), errors="coerce")
            other = numeric.isna() & ~is_dash
            if other.any():
                other_nonnum[c] += int(other.sum())
                nonnum_examples.setdefault(c, chunk.loc[other, c].unique()[:5].tolist())
            chunk[c] = numeric.fillna(0.0).astype("float64")
        for c in OTHER_NUMERIC:         # check only; values left as text
            notnum = pd.to_numeric(chunk[c], errors="coerce").isna()
            if notnum.any():
                other_nonnum[c] += int(notnum.sum())
                nonnum_examples.setdefault(c, chunk.loc[notnum, c].unique()[:5].tolist())

        for k, v in chunk.groupby([TARGET, "detailed_label"]).size().items():
            raw_by[k] += int(v)

        # D3 -- key = every column except the derived target
        key_cols = [c for c in chunk.columns if c != TARGET]
        hashes = pd.util.hash_pandas_object(chunk[key_cols], index=False).to_numpy()
        uniq, first_idx, cnt = np.unique(hashes, return_index=True, return_counts=True)
        new_pos = []
        for hval, idx, c in zip(uniq.tolist(), first_idx.tolist(), cnt.tolist()):
            prev = counts.get(hval)
            if prev is None:
                counts[hval] = c
                new_pos.append(idx)
            else:
                counts[hval] = prev + c
        if new_pos:
            new_pos.sort()
            kept = chunk.iloc[new_pos].copy()
            kept["_row_key"] = hashes[new_pos].astype(str)
            kept.to_csv(tmp1, mode="a", header=not written, index=False)
            written = True

    if not written:
        print("  WARNING: no rows retained for this scenario")
        return None

    # second pass: attach multiplicity
    str_counts = {str(k): v for k, v in counts.items()}
    hdr = False
    n_unique = 0
    mult_by = Counter()
    for part in pd.read_csv(tmp1, dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE):
        m = part["_row_key"].map(str_counts)
        if m.isna().any():
            raise RuntimeError(f"{scen}: multiplicity lookup failed")
        part[MULT] = m.astype("int64")
        part = part.drop(columns=["_row_key"])
        part[TARGET] = part[TARGET].astype("int64")
        for k, v in part.groupby([TARGET, "detailed_label"]).size().items():
            uniq_by[k] += int(v)
        for k, v in part.groupby([TARGET, "detailed_label"])[MULT].sum().items():
            mult_by[k] += int(v)
        n_unique += len(part)
        part.to_csv(tmp2, mode="a", header=not hdr, index=False)
        hdr = True

    # verification: multiplicities reconstruct the raw counts exactly
    if dict(mult_by) != dict(raw_by) or sum(counts.values()) != n_raw - n_viol:
        raise RuntimeError(f"{scen}: multiplicity check failed -- refusing to save")

    os.replace(tmp2, final)
    os.remove(tmp1)
    stats = {
        "scenario": scen, "rows_raw": n_raw, "rows_label_violations": n_viol,
        "rows_unique": n_unique, "duplicates_removed": n_raw - n_viol - n_unique,
        "dash_replaced_by_zero": dict(dash),
        "other_non_numeric_values": dict(other_nonnum),
        "non_numeric_examples": nonnum_examples,
        "by_class": [{TARGET: int(k[0]), "detailed_label": k[1],
                      "raw_rows": raw_by[k], "unique_rows": uniq_by.get(k, 0)}
                     for k in sorted(raw_by)],
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f)
    print(f"  {n_raw:,} raw -> {n_viol:,} label violations -> {n_unique:,} unique rows")
    if other_nonnum:
        print(f"  NOTE non-numeric values found: {dict(other_nonnum)} e.g. {nonnum_examples}")
    return stats


# ======================================================================
# IoT-23 stage 2: combine + diagnostics
# ======================================================================

def combine_iot23(all_stats):
    print("\nCombining per-scenario files into data/processed/iot23_clean.csv ...")
    files = sorted(glob.glob(os.path.join(IOT23_CLEAN_SCEN_DIR, "*.csv")))
    out = os.path.join(PROCESSED_DIR, "iot23_clean.csv")
    tmp = out + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)

    columns = None
    det_codes = {}
    h_parts, s_parts, c_parts, d_parts, m_parts = [], [], [], [], []
    scen_names = []
    for si, f in enumerate(files):
        scen_names.append(os.path.basename(f)[:-4])
        for part in pd.read_csv(f, dtype=str, keep_default_na=False, chunksize=CHUNK_SIZE):
            if columns is None:
                columns = part.columns.tolist()
            elif part.columns.tolist() != columns:
                raise ValueError(f"{f}: column order differs from the first scenario file")
            part.to_csv(tmp, mode="a", header=(si == 0 and not h_parts), index=False)
            feat = [c for c in columns if c not in (TARGET, MULT, "scenario", *LABEL_COLS)]
            h_parts.append(pd.util.hash_pandas_object(part[feat], index=False).to_numpy())
            s_parts.append(np.full(len(part), si, dtype=np.int16))
            c_parts.append(part[TARGET].astype("int64").to_numpy(dtype=np.int8))
            for v in part["detailed_label"].unique():
                det_codes.setdefault(v, len(det_codes))
            d_parts.append(part["detailed_label"].map(det_codes).to_numpy(dtype=np.int16))
            m_parts.append(part[MULT].astype("int64").to_numpy())
    os.replace(tmp, out)

    diag = pd.DataFrame({"h": np.concatenate(h_parts), "scen": np.concatenate(s_parts),
                         "cls": np.concatenate(c_parts), "det": np.concatenate(d_parts),
                         "mult": np.concatenate(m_parts)})
    n = len(diag)
    diag["cls_in_scen"] = diag.groupby(["scen", "h"])["cls"].transform("nunique")
    diag["n_scen"] = diag.groupby("h")["scen"].transform("nunique")
    diag["cls_global"] = diag.groupby("h")["cls"].transform("nunique")
    rows_if_global_dedup = int(diag[["h", "det"]].drop_duplicates().shape[0])

    per_scen = diag.groupby("scen").agg(
        unique_rows=("h", "size"), raw_rows_represented=("mult", "sum"),
        rows_with_benign_malicious_conflict_in_scenario=("cls_in_scen", lambda s: int((s > 1).sum())),
        rows_whose_features_also_occur_in_another_scenario=("n_scen", lambda s: int((s > 1).sum())))
    per_scen.index = [scen_names[i] for i in per_scen.index]
    per_scen["share_of_clean_dataset"] = (per_scen["unique_rows"] / n).round(4)
    per_scen = per_scen.sort_values("unique_rows", ascending=False)
    per_scen.to_csv(os.path.join(RESULTS_DIR, "iot23_per_scenario_diagnostics.csv"))

    summary = {
        "rows_raw": int(sum(s["rows_raw"] for s in all_stats)),
        "rows_label_violations": int(sum(s["rows_label_violations"] for s in all_stats)),
        "rows_retained_primary_per_scenario_dedup": n,
        "rows_if_global_dedup_were_used_instead": rows_if_global_dedup,
        "rows_in_benign_malicious_conflicts_within_scenario": int((diag["cls_in_scen"] > 1).sum()),
        "rows_whose_feature_vector_occurs_in_another_scenario": int((diag["n_scen"] > 1).sum()),
        "rows_in_benign_malicious_conflicts_across_scenarios": int((diag["cls_global"] > 1).sum()),
        "note": "Diagnostics compare all non-label, non-metadata columns. Reported only; "
                "nothing was removed on their basis.",
    }
    with open(os.path.join(RESULTS_DIR, "iot23_cleaning_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return per_scen, summary


def clean_iot23():
    print("=" * 70 + "\nIoT-23 (per scenario; restartable)\n" + "=" * 70)
    files = sorted(glob.glob(os.path.join(IOT23_SCENARIO_DIR, "*.csv")))
    if not files:
        raise ValueError(f"No per-scenario CSVs in {rel(IOT23_SCENARIO_DIR)} -- run 1_data_loading.py first")
    print(f"Found {len(files)} scenario files")

    all_stats = []
    t0 = time.perf_counter()
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {os.path.basename(f)[:-4]}")
        t1 = time.perf_counter()
        st = clean_iot23_scenario(f)
        print(f"  took {time.perf_counter() - t1:.0f}s (total {time.perf_counter() - t0:.0f}s)")
        if st:
            all_stats.append(st)

    by_class = pd.DataFrame([{"scenario": s["scenario"], **r} for s in all_stats for r in s["by_class"]])
    by_class.to_csv(os.path.join(RESULTS_DIR, "iot23_cleaning_by_scenario_class.csv"), index=False)
    pd.DataFrame([{k: v for k, v in s.items() if k != "by_class"} for s in all_stats]) \
        .to_csv(os.path.join(RESULTS_DIR, "iot23_cleaning_by_scenario.csv"), index=False)

    crosstab = by_class.pivot_table(index="scenario", columns=TARGET, values="unique_rows",
                                    aggfunc="sum", fill_value=0)
    for c in (0, 1):
        if c not in crosstab.columns:
            crosstab[c] = 0
    crosstab = crosstab[[0, 1]].rename(columns={0: "benign", 1: "malicious"})
    crosstab["scenario_total"] = crosstab.sum(axis=1)
    crosstab = crosstab.sort_values("scenario_total", ascending=False)
    crosstab.to_csv(os.path.join(RESULTS_DIR, "iot23_scenario_malicious_crosstab.csv"))

    per_scen, summary = combine_iot23(all_stats)

    state = by_class.groupby(TARGET)[["raw_rows", "unique_rows"]].sum()
    state["raw_share"] = (state["raw_rows"] / state["raw_rows"].sum()).round(4)
    state["unique_share"] = (state["unique_rows"] / state["unique_rows"].sum()).round(4)
    state.to_csv(os.path.join(RESULTS_DIR, "iot23_malicious_before_after.csv"))

    print("\nBenign (0) / malicious (1) before/after deduplication:")
    print(state.to_string())
    print("\nScenario x benign/malicious (unique rows):")
    print(crosstab.to_string())
    print("\nPer-scenario diagnostics:")
    print(per_scen.to_string())
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["rt_iot2022", "iot23"]
    started = datetime.datetime.now().isoformat(timespec="seconds")
    save_mapping_tables()
    for name in targets:
        if name == "rt_iot2022":
            clean_rt_iot2022()
        elif name == "iot23":
            clean_iot23()
        else:
            raise SystemExit(f"Unknown dataset '{name}'. Use rt_iot2022 and/or iot23.")
    with open(os.path.join(RESULTS_DIR, "run_manifest.json"), "w") as f:
        json.dump({"script": "2_data_cleaning.py", "targets": targets, "started": started,
                   "finished": datetime.datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    print("\nDone. Outputs: data/processed/, results/02_cleaning/")