
"""
1_data_loading.py

Loads the raw RT-IoT2022 and IoT-23 datasets into data/interim/
for subsequent preprocessing and analysis.

RT-IoT2022:
    - Locates CSV files in the raw dataset directory, including files
      without a .csv extension where possible.
    - Loads and combines the available files into a single dataset.
    - Adds a source_file column to identify the origin of each row.
    - Saves the combined dataset as:
      data/interim/rt_iot2022_merged.csv

IoT-23:
    - Locates Zeek conn.log.labeled files in the raw dataset directory.
    - Reads the files in chunks, processing one scenario at a time.
    - Separates the label and detailed-label fields when they are
      combined in the original file.
    - Excludes structurally malformed records that cannot be parsed
      as valid records.
    - Retains the selected traffic features, labels, timestamps,
      IP addresses, port numbers, and scenario identifiers.
    - Saves each processed scenario as a separate CSV file in:
      data/interim/iot23_scenarios/
    - Records loading statistics and label counts for each scenario.

The script also generates a loading log, label-count report, and run
manifest in results/01_loading/ to document the processing outcomes.

Data loading is limited to reading, combining, and structurally
validating records. Deduplication, feature exclusion, label
transformation, and other content-based preprocessing are handled
in subsequent pipeline stages.

Usage on the terminal under the project folder:
    python 1_data_loading.py
    python 1_data_loading.py --only-first 1
    python 1_data_loading.py --skip-rt
"""

import os
import gc
import glob
import json
import time
import argparse
import datetime
from collections import Counter

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_RT_IOT2022_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "rt_iot2022")
RAW_IOT23_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "iot23")
INTERIM_DIR = os.path.join(PROJECT_ROOT, "data", "interim")
IOT23_SCENARIO_DIR = os.path.join(INTERIM_DIR, "iot23_scenarios")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "01_loading")

for d in (INTERIM_DIR, IOT23_SCENARIO_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

CHUNK_SIZE = 200_000  # rows held in memory at a time


def rel(path):
    return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")


# ----------------------------------------------------------------------
# RT-IoT2022
# ----------------------------------------------------------------------

def find_all_csvs(root_dir):
    files = glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)
    print(f"Found {len(files)} .csv files under {rel(root_dir)}")
    if not files:
        all_files = [f for f in glob.glob(os.path.join(root_dir, "**", "*"), recursive=True)
                     if os.path.isfile(f)]
        print(f"  Trying {len(all_files)} extension-less file(s) as CSV:")
        for f in all_files:
            try:
                test_df = pd.read_csv(f, low_memory=False, nrows=5)
                if test_df.shape[1] > 1:
                    print(f"    {rel(f)} -> valid CSV ({test_df.shape[1]} columns)")
                    files.append(f)
            except Exception as e:
                print(f"    {rel(f)} -> not a CSV ({type(e).__name__})")
    return files


def load_rt_iot2022():
    files = find_all_csvs(RAW_RT_IOT2022_DIR)
    if not files:
        raise ValueError("No usable RT-IoT2022 file found")
    if len(files) > 1:
        print(f"  NOTE: {len(files)} files will be concatenated: {[rel(f) for f in files]}")
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df["source_file"] = rel(f)
        frames.append(df)
        print(f"  Loaded {rel(f)} -> {df.shape}")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    out = os.path.join(INTERIM_DIR, "rt_iot2022_merged.csv")
    combined.to_csv(out, index=False)
    print(f"RT-IoT2022 saved: {combined.shape} -> {rel(out)}")
    return {"rt_files": [rel(f) for f in files], "rt_rows": int(len(combined)),
            "rt_columns": int(combined.shape[1])}


# ----------------------------------------------------------------------
# IoT-23
# ----------------------------------------------------------------------

# Structural, not content: 'uid' is a random per-flow ID; local_orig /
# local_resp are always empty in IoT-23.
DROP_AT_READ = ["uid", "local_orig", "local_resp"]

# Columns written to disk. IPs, timestamp and both ports are KEPT here so
# every exclusion decision is made visibly in the cleaning stage.
IOT23_KEEP = ["ts", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
              "proto", "service", "duration", "orig_bytes", "resp_bytes",
              "conn_state", "missed_bytes", "history", "orig_pkts",
              "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
              "label", "detailed-label"]


def find_all_labeled_logs(root_dir):
    files = sorted(glob.glob(os.path.join(root_dir, "**", "conn.log.labeled"), recursive=True))
    print(f"Found {len(files)} conn.log.labeled files under {rel(root_dir)}")
    return files


def scenario_name(filepath, fallback_index):
    for part in filepath.replace("\\", "/").split("/"):
        if part.startswith("CTU-"):
            return part
    return f"scenario_{fallback_index:02d}"


def read_field_names(filepath):
    """Column names exactly as the file's tab-separated fields (the last
    entry may be Zeek's glued 'tunnel_parents   label   detailed-label')."""
    with open(filepath, "r", errors="replace") as f:
        for line in f:
            if line.startswith("#fields"):
                return line.rstrip("\n").split("\t")[1:]
    return None


def count_lines(filepath):
    """
    Returns (data_lines, blank_lines, data_lines_containing_hash).
    data_lines = non-comment, non-blank lines -- what the parser should read.
    """
    data = blank = with_hash = 0
    with open(filepath, "rb") as f:
        for line in f:
            if line.startswith(b"#"):
                continue
            if not line.strip():
                blank += 1
                continue
            data += 1
            if b"#" in line:
                with_hash += 1
    return data, blank, with_hash


def split_glued_column(chunk, glued_col):
    if glued_col is None:
        return chunk
    parts = chunk[glued_col].str.split(r"\s+", n=2, expand=True)
    parts = parts.reindex(columns=[0, 1, 2])
    chunk["label"] = parts[1]
    chunk["detailed-label"] = parts[2]
    return chunk.drop(columns=[glued_col])


def add_label_counts(counter, chunk):
    for k, v in chunk.groupby(["label", "detailed-label"], dropna=False).size().items():
        counter[(str(k[0]), str(k[1]))] += int(v)


def process_scenario(filepath, scen):
    final_path = os.path.join(IOT23_SCENARIO_DIR, f"{scen}.csv")
    tmp_path = final_path + ".tmp"
    stats_path = os.path.join(IOT23_SCENARIO_DIR, f"{scen}.stats.json")

    if os.path.exists(final_path) and os.path.exists(stats_path):
        print("  already done -> skipped")
        with open(stats_path) as f:
            return json.load(f)

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    names = read_field_names(filepath)
    if names is None:
        print("  WARNING: no #fields header, skipping file")
        return None

    glued_col = names[-1] if (len(names[-1].split()) == 3) else None
    if glued_col is None and "label" not in names:
        raise ValueError(f"{filepath}: cannot find label column(s) in header: {names[-3:]}")

    expected, blank_lines, lines_with_hash = count_lines(filepath)
    if lines_with_hash:
        print(f"  WARNING: {lines_with_hash:,} data line(s) contain '#'; pandas' "
              f"comment='#' truncates them -- they will show up as malformed below.")
    usecols = [c for c in names if c not in DROP_AT_READ]

    reader = pd.read_csv(filepath, sep="\t", comment="#", header=None, names=names,
                         usecols=usecols, dtype=str, keep_default_na=False,
                         chunksize=CHUNK_SIZE, on_bad_lines="skip")

    rows_read = rows_kept = malformed_short_lines = 0
    raw_counts, kept_counts = Counter(), Counter()
    header_written = False

    for chunk in reader:
        chunk = split_glued_column(chunk, glued_col)
        rows_read += len(chunk)
        add_label_counts(raw_counts, chunk)          # BEFORE the structural check

        # STRUCTURAL VALIDITY, not content filtering. on_bad_lines="skip"
        # only catches lines with TOO MANY fields; a truncated line (too
        # FEW) is padded with NaN instead. A real Zeek record always has
        # id.resp_h and a label, so a row missing either was never a
        # parseable record.
        resp_h, label = chunk["id.resp_h"], chunk["label"]
        well_formed = resp_h.notna() & resp_h.ne("") & label.notna() & label.ne("")
        n_bad = int((~well_formed).sum())
        if n_bad:
            chunk = chunk.loc[well_formed]
        malformed_short_lines += n_bad

        # Every remaining row is written: no dedup, no sampling, no content filter.
        chunk = chunk.copy()
        chunk["scenario"] = scen
        chunk = chunk[IOT23_KEEP + ["scenario"]]
        add_label_counts(kept_counts, chunk)         # AFTER the structural check
        chunk.to_csv(tmp_path, mode="a", header=not header_written, index=False)
        header_written = True
        rows_kept += len(chunk)

    del reader
    gc.collect()

    if not header_written:
        print("  WARNING: nothing written for this scenario")
        return None

    os.replace(tmp_path, final_path)
    stats = {
        "scenario": scen, "source": rel(filepath),
        "data_lines_in_file": expected,
        "blank_lines_in_file": blank_lines,
        "data_lines_containing_hash": lines_with_hash,
        "rows_read_by_parser": rows_read,
        "rows_skipped_too_many_fields": expected - rows_read,
        "rows_dropped_too_few_fields_or_missing_key": malformed_short_lines,
        "rows_written": rows_kept,
        "label_counts": [{"label": k[0], "detailed_label": k[1],
                          "rows_before_structural_check": raw_counts[k],
                          "rows_written": kept_counts.get(k, 0)}
                         for k in raw_counts],
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f)
    print(f"  {expected:,} data lines -> parser read {rows_read:,} "
          f"({expected - rows_read:,} skipped: too many fields); "
          f"{malformed_short_lines:,} dropped: too few fields / missing key "
          f"-> wrote {rows_kept:,} rows")
    return stats


def build_iot23(only_first=None):
    logs = find_all_labeled_logs(RAW_IOT23_DIR)
    if not logs:
        raise ValueError(f"No conn.log.labeled files under {RAW_IOT23_DIR}")
    if only_first:
        logs = logs[:only_first]

    all_stats = []
    t0 = time.perf_counter()
    for i, fp in enumerate(logs, 1):
        scen = scenario_name(fp, i)
        print(f"[{i}/{len(logs)}] {scen}")
        t1 = time.perf_counter()
        st = process_scenario(fp, scen)
        print(f"  took {time.perf_counter() - t1:.0f}s (total {time.perf_counter() - t0:.0f}s)")
        if st:
            all_stats.append(st)

    log_df = pd.DataFrame([{k: v for k, v in s.items() if k != "label_counts"} for s in all_stats])
    log_df.to_csv(os.path.join(RESULTS_DIR, "iot23_loading_log.csv"), index=False)

    rows = [{"scenario": s["scenario"], **lc} for s in all_stats for lc in s["label_counts"]]
    label_df = pd.DataFrame(rows)
    label_df.to_csv(os.path.join(RESULTS_DIR, "iot23_label_counts_before_vs_after_check.csv"),
                    index=False)

    print("\n=== label x detailed-label, all scenarios ===")
    print(label_df.groupby(["label", "detailed_label"])
          [["rows_before_structural_check", "rows_written"]].sum()
          .sort_values("rows_before_structural_check", ascending=False).to_string())
    print("\n=== totals ===")
    print(f"Data lines in files        : {log_df['data_lines_in_file'].sum():,}")
    print(f"Skipped (too many fields)  : {log_df['rows_skipped_too_many_fields'].sum():,}")
    print(f"Dropped (too few / no key) : {log_df['rows_dropped_too_few_fields_or_missing_key'].sum():,}")
    print(f"Rows written               : {log_df['rows_written'].sum():,}")
    print(f"Data lines containing '#'  : {log_df['data_lines_containing_hash'].sum():,}")
    return {"iot23_logs": len(logs), "iot23_rows_written": int(log_df["rows_written"].sum())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-first", type=int, default=None,
                    help="process only the first N IoT-23 scenarios (timing test)")
    ap.add_argument("--skip-rt", action="store_true")
    args = ap.parse_args()

    manifest = {"script": "1_data_loading.py", "args": vars(args),
                "started": datetime.datetime.now().isoformat(timespec="seconds")}

    if not args.skip_rt:
        print("=== RT-IoT2022 ===")
        manifest.update(load_rt_iot2022())
    print("\n=== IoT-23 (streamed, full data, per-scenario) ===")
    manifest.update(build_iot23(args.only_first))

    manifest["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(RESULTS_DIR, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nDone. Outputs: data/interim/, results/01_loading/")