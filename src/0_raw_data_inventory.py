"""
00_raw_data_inventory.py

Inventories the raw datasets and records the software environment
before data loading and preprocessing.

The script performs the following tasks:

1. Software environment
   - Records the Python version, operating system, processor, CPU count,
     installed package versions, and available system RAM.
   - Saves the environment details to results/00_inventory/environment.txt.

2. Raw file inventory
   - Locates files under data/raw/.
   - Records each file's relative path and size.
   - Calculates SHA-256 fingerprints for files, with optional hashing
     of the large IoT-23 log files.
   - Saves the inventory to results/00_inventory/raw_file_inventory.csv.

3. RT-IoT2022 dataset profile
   - Identifies files that can be read as CSV.
   - Records dataset dimensions, column names, data types, missing values,
     unique values, example values, and potential identifier columns.
   - Summarises label distributions, numeric infinite values, and
     exact duplicate-row counts.
   - Saves the column profile, label counts, and summary report
     in results/00_inventory/.

4. IoT-23 dataset structure
   - Identifies scenario folders and conn.log.labeled files.
   - Inspects Zeek headers and records file sizes, field counts,
     and the structure of the label fields.
   - Identifies missing log files, missing headers, and duplicate
     scenario names.
   - Saves the log inventory to results/00_inventory/iot23_log_inventory.csv.

The script is read-only with respect to the raw datasets. It does not
modify, move, delete, clean, or deduplicate files under data/.
It creates reports in results/00_inventory/ for data provenance,
dataset inspection, and reproducibility.

Usage:
    python 00_raw_data_inventory.py
    python 00_raw_data_inventory.py --hash-iot23
"""

import os
import sys
import glob
import hashlib
import platform
import argparse
import datetime
import importlib

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
RAW_RT_DIR = os.path.join(RAW_DIR, "rt_iot2022")
RAW_IOT23_DIR = os.path.join(RAW_DIR, "iot23")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "00_inventory")
os.makedirs(OUT_DIR, exist_ok=True)

# Column-name fragments that suggest an identifier (IP, time, flow/session
# ID, MAC). Only used to FLAG columns for manual inspection -- nothing is
# dropped on the basis of this list.
IDENTIFIER_HINTS = ["ip", "addr", "_h", ".h", "ts", "time", "flow_id", "uid",
                    "session", "mac", "src", "dst", "host", "device"]
LABEL_CANDIDATES = ["attack_type", "attack type", "label", "class"]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def rel(path):
    return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")


def sha256_of(path, block=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def scenario_name(filepath, fallback_index):
    """Identical rule to 1_data_loading.py, so names match across scripts."""
    for part in filepath.replace("\\", "/").split("/"):
        if part.startswith("CTU-"):
            return part
    return f"scenario_{fallback_index:02d}"


# ----------------------------------------------------------------------
# 1. environment
# ----------------------------------------------------------------------

def record_environment():
    lines = [
        f"run_time_local: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"python: {sys.version.split()[0]}",
        f"platform: {platform.platform()}",
        f"processor: {platform.processor()}",
        f"cpu_count: {os.cpu_count()}",
    ]
    for pkg in ["pandas", "numpy", "sklearn", "xgboost", "scipy", "joblib", "matplotlib"]:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
        except ImportError:
            version = "NOT INSTALLED"
        lines.append(f"{pkg}: {version}")
    try:
        import psutil
        lines.append(f"total_ram_gb: {psutil.virtual_memory().total / 1e9:.1f}")
    except ImportError:
        lines.append("total_ram_gb: unknown (psutil not installed)")

    path = os.path.join(OUT_DIR, "environment.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("=== Environment ===")
    print("\n".join("  " + l for l in lines))
    print(f"  saved: {rel(path)}\n")


# ----------------------------------------------------------------------
# 2. every raw file: path, size, fingerprint
# ----------------------------------------------------------------------

def inventory_raw_files(hash_iot23):
    print("=== Raw file inventory ===")
    rows = []
    for root, _, files in os.walk(RAW_DIR):
        for name in sorted(files):
            path = os.path.join(root, name)
            relpath = rel(path)
            dataset = relpath.split("/")[2] if relpath.count("/") >= 2 else "?"
            size = os.path.getsize(path)
            do_hash = (dataset != "iot23") or hash_iot23
            if do_hash:
                print(f"  hashing {relpath} ({size / 1e6:,.1f} MB)...")
            rows.append({
                "dataset_folder": dataset,
                "relative_path": relpath,
                "size_bytes": size,
                "size_mb": round(size / 1e6, 2),
                "sha256": sha256_of(path) if do_hash else "not computed (use --hash-iot23)",
            })
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "raw_file_inventory.csv")
    df.to_csv(path, index=False)
    if len(df):
        print(df.groupby("dataset_folder")["size_mb"].agg(["count", "sum"])
              .rename(columns={"count": "n_files", "sum": "total_mb"}).to_string())
    print(f"  saved: {rel(path)}\n")


# ----------------------------------------------------------------------
# 3. RT-IoT2022 profile
# ----------------------------------------------------------------------

def find_rt_files():
    candidates = []
    for path in sorted(glob.glob(os.path.join(RAW_RT_DIR, "**", "*"), recursive=True)):
        if not os.path.isfile(path):
            continue
        try:
            probe = pd.read_csv(path, nrows=5, low_memory=False)
        except Exception as e:
            print(f"  {rel(path)} -> not readable as CSV ({type(e).__name__})")
            continue
        if probe.shape[1] > 1:
            candidates.append(path)
            print(f"  {rel(path)} -> readable as CSV ({probe.shape[1]} columns)")
    return candidates


def profile_rt():
    print("=== RT-IoT2022 ===")
    files = find_rt_files()
    if not files:
        print("  WARNING: no CSV-readable file found under", rel(RAW_RT_DIR))
        return
    if len(files) > 1:
        print(f"  NOTE: {len(files)} CSV-readable files found -- each is profiled; "
              f"tell me which one is the dataset.")

    summary_lines, profile_frames, label_frames = [], [], []
    for path in files:
        df = pd.read_csv(path, low_memory=False)
        tag = rel(path)
        summary_lines.append(f"file: {tag}")
        summary_lines.append(f"  shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

        prof = pd.DataFrame({
            "file": tag,
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "n_missing": df.isna().sum().values,
            "n_unique": [df[c].nunique(dropna=False) for c in df.columns],
            "example_value": [df[c].iloc[0] if len(df) else None for c in df.columns],
        })
        prof["identifier_hint"] = [
            any(h in c.lower() for h in IDENTIFIER_HINTS) for c in df.columns]
        profile_frames.append(prof)

        # numeric infinities (data-integrity fact, not a decision)
        num = df.select_dtypes("number")
        n_inf = int((num.abs() == float("inf")).sum().sum()) if num.shape[1] else 0
        summary_lines.append(f"  infinite numeric values: {n_inf:,}")
        summary_lines.append(f"  total missing cells: {int(df.isna().sum().sum()):,}")

        label_col = next((c for c in df.columns if c.strip().lower() in LABEL_CANDIDATES), None)
        if label_col:
            vc = df[label_col].value_counts(dropna=False)
            label_frames.append(pd.DataFrame({
                "file": tag, "label_column": label_col,
                "label_value": vc.index.astype(str), "rows": vc.values,
                "share": (vc.values / len(df)).round(6)}))
            summary_lines.append(f"  label column: '{label_col}' ({vc.shape[0]} distinct values)")
        else:
            summary_lines.append("  WARNING: no label column recognised")

        # DESCRIPTIVE duplicate counts only -- no rows are removed here.
        index_like = [c for c in df.columns if c.lower().startswith("unnamed")]
        no_index = df.drop(columns=index_like)
        summary_lines.append(f"  index-like columns: {index_like or 'none'}")
        summary_lines.append(f"  exact duplicate rows, all columns: {int(df.duplicated().sum()):,}")
        summary_lines.append(f"  exact duplicate rows, excluding index-like columns: "
                             f"{int(no_index.duplicated().sum()):,}")

        hinted = prof.loc[prof["identifier_hint"], "column"].tolist()
        summary_lines.append(f"  columns flagged for manual identifier check: {hinted}")

    for l in summary_lines:
        print(" ", l)

    pd.concat(profile_frames).to_csv(os.path.join(OUT_DIR, "rt_iot2022_column_profile.csv"), index=False)
    if label_frames:
        lab = pd.concat(label_frames)
        lab.to_csv(os.path.join(OUT_DIR, "rt_iot2022_label_counts.csv"), index=False)
        print("\n  Label counts:")
        print(lab[["label_value", "rows", "share"]].to_string(index=False))
    with open(os.path.join(OUT_DIR, "rt_iot2022_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"  saved: {rel(OUT_DIR)}/rt_iot2022_*\n")


# ----------------------------------------------------------------------
# 4. IoT-23 structure
# ----------------------------------------------------------------------

def read_zeek_header(path, max_lines=50):
    """Reads only the leading '#' header lines plus the first data line."""
    header, first_data = {}, None
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.rstrip("\n")
            if line.startswith("#"):
                key, _, value = line[1:].partition("\t") if "\t" in line else line[1:].partition(" ")
                header[key.strip()] = value
            elif line.strip():
                first_data = line
                break
    return header, first_data


def inventory_iot23():
    print("=== IoT-23 ===")
    if not os.path.isdir(RAW_IOT23_DIR):
        print("  WARNING: folder not found:", rel(RAW_IOT23_DIR))
        return

    top_folders = sorted(d for d in os.listdir(RAW_IOT23_DIR)
                         if os.path.isdir(os.path.join(RAW_IOT23_DIR, d)))
    logs = sorted(glob.glob(os.path.join(RAW_IOT23_DIR, "**", "conn.log.labeled"), recursive=True))
    print(f"  top-level sub-folders: {len(top_folders)}")
    print(f"  conn.log.labeled files found: {len(logs)}")

    rows = []
    for i, path in enumerate(logs, 1):
        header, first_data = read_zeek_header(path)
        fields = header.get("fields", "").split("\t") if "fields" in header else []
        last_field = fields[-1] if fields else ""
        glued = len(last_field.split()) == 3
        rows.append({
            "scenario": scenario_name(path, i),
            "relative_path": rel(path),
            "size_mb": round(os.path.getsize(path) / 1e6, 2),
            "zeek_path": header.get("path", ""),
            "zeek_open": header.get("open", ""),
            "n_header_fields": len(fields),
            "label_field_glued": glued,
            "last_header_field": last_field,
            "first_data_line_n_fields": len(first_data.split("\t")) if first_data else 0,
            "has_fields_header": bool(fields),
        })
    df = pd.DataFrame(rows)
    path_out = os.path.join(OUT_DIR, "iot23_log_inventory.csv")
    df.to_csv(path_out, index=False)
    if len(df):
        print(df[["scenario", "size_mb", "n_header_fields", "label_field_glued",
                  "first_data_line_n_fields"]].to_string(index=False))
        print(f"  total size: {df['size_mb'].sum():,.1f} MB")

    # problems worth knowing about BEFORE loading
    dup_names = df["scenario"][df["scenario"].duplicated()].tolist() if len(df) else []
    if dup_names:
        print(f"  WARNING: duplicate scenario names: {dup_names}")
    if len(df) and (~df["has_fields_header"]).any():
        print("  WARNING: some logs have no #fields header:",
              df.loc[~df["has_fields_header"], "relative_path"].tolist())

    folders_with_log = {rel(p).split("/")[3] for p in logs if rel(p).count("/") >= 3}
    missing = [d for d in top_folders if d not in folders_with_log]
    if missing:
        pd.DataFrame({"folder_without_conn_log_labeled": missing}).to_csv(
            os.path.join(OUT_DIR, "iot23_folders_without_log.csv"), index=False)
        print(f"  WARNING: {len(missing)} sub-folder(s) contain no conn.log.labeled: {missing}")
    else:
        print("  every top-level sub-folder contains a conn.log.labeled file")
    print(f"  saved: {rel(path_out)}\n")


# ----------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash-iot23", action="store_true",
                    help="also compute SHA-256 of the large IoT-23 logs (slow)")
    args = ap.parse_args()

    print("=" * 70)
    print("RAW DATA INVENTORY (read-only -- nothing under data/ is modified)")
    print(f"project root: {PROJECT_ROOT}")
    print("=" * 70 + "\n")

    record_environment()
    inventory_raw_files(args.hash_iot23)
    profile_rt()
    inventory_iot23()

    print("Done. Send back the full console output (or the files in results/00_inventory/).")