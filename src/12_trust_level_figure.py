"""
12_trust_level_figure.py

Creates thesis Figure 5.2 from the completed results of script 11. It
compares the distribution of benign and malicious test flows across the
Trusted, Suspicious, and Untrusted levels for RT-IoT2022 and IoT-23.

The script only reads existing results and creates the figure, data CSV,
and source hashes. It does not train models or calculate new results.

Outputs:
    results/12_trust_level_figure/<run name>/
        hybrid_trust_levels_by_class.png
        hybrid_trust_levels_by_class_data.csv
        sources.json

Usage:
    python src/12_trust_level_figure.py
    python src/12_trust_level_figure.py --run run_20261024_203525
"""

import os
import sys
import json
import hashlib
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS = os.path.join(PROJECT_ROOT, "results")
EVAL_ROOT = os.path.join(RESULTS, "11_final_evaluation")
OUT_ROOT = os.path.join(RESULTS, "12_trust_level_figure")

ARRANGEMENT = "hybrid_xgb_rf"
COST_RATIOS = [1, 2, 5, 10, 20]
PANELS = [  # dataset, group, primary threshold rows, panel title
    ("rt_iot2022", "all", "all", "RT-IoT2022 test data"),
    ("iot23", "heldout_captures", "unseen_capture_rows", "IoT-23: 13 held-out captures"),
]
COLOURS = {"Trusted": "#4caf50", "Suspicious": "#f2b233", "Untrusted": "#d9534f"}

# Fixed style so the image is the same on every computer: DejaVu Sans ships with
# matplotlib, so no system font is used.
STYLE = {"font.family": "DejaVu Sans", "font.size": 10}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pick_run(name):
    if name:
        run = os.path.join(EVAL_ROOT, name)
        if not os.path.exists(os.path.join(run, "COMPLETED.json")):
            raise SystemExit(f"{run} is not a completed run of script 11.")
        return run
    runs = sorted(d for d in os.listdir(EVAL_ROOT)
                  if d.startswith("run_") and os.path.exists(os.path.join(EVAL_ROOT, d, "COMPLETED.json")))
    if not runs:
        raise SystemExit("No completed run of script 11 found in results/11_final_evaluation/.")
    return os.path.join(EVAL_ROOT, runs[-1])


def one_row(d, **cond):
    q = d
    for k, v in cond.items():
        q = q[q[k] == v]
    if len(q) != 1:
        raise SystemExit(f"Expected exactly one row for {cond}, found {len(q)}.")
    return q.iloc[0]


def shares(r):
    benign = [1 - r["over_restriction_rate_given_benign"], r["benign_restricted_rate"], r["benign_rejected_rate"]]
    malicious = [r["false_trust_rate_allow_given_malicious"], r["malicious_restricted_rate"],
                 r["malicious_rejected_rate"]]
    for name, v in (("benign", benign), ("malicious", malicious)):
        if abs(sum(v) - 1) > 1e-6:
            raise SystemExit(f"{name} shares do not add up to 1: {v}")
    return benign, malicious


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="run folder name in results/11_final_evaluation/ (default: latest completed)")
    args = ap.parse_args()

    run = pick_run(args.run)
    out_dir = os.path.join(OUT_ROOT, os.path.basename(run))
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update(STYLE)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
    rows, sources = [], {}
    for ax, (ds, group, thr_rows, title) in zip(axes, PANELS):
        path = os.path.join(run, f"{ds}_decision_metrics.csv")
        sources[os.path.relpath(path, PROJECT_ROOT)] = sha256(path)
        d = pd.read_csv(path)
        d = d[(d["arrangement"] == ARRANGEMENT) & (d["weighting"] == "unweighted") & (d["group"] == group)]
        settings = [("No trust\nlayer", one_row(d, decision_rule="no_trust_layer_r_ge_0.5_reject"))]
        for c in COST_RATIOS:
            settings.append((f"C = {c}", one_row(d, decision_rule="cost_minimising_thresholds",
                                                  threshold_rows=thr_rows, cost_ratio=float(c))))
        n = int(settings[0][1]["n"])
        x = np.arange(len(settings))
        w = 0.38
        for k, (label, r) in enumerate(settings):
            benign, malicious = shares(r)
            rows.append([ds, group, label.replace("\n", " "), n, *benign, *malicious])
            for off, vals in ((-w / 2 - 0.01, benign), (w / 2 + 0.01, malicious)):
                bottom = 0.0
                for v, level in zip(vals, COLOURS):
                    ax.bar(x[k] + off, v, w, bottom=bottom, color=COLOURS[level], edgecolor="white", linewidth=0.6)
                    bottom += v
            ax.text(x[k] - w / 2 - 0.01, 1.015, "B", ha="center", fontsize=10)
            ax.text(x[k] + w / 2 + 0.01, 1.015, "M", ha="center", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([s[0] for s in settings], fontsize=11)
        ax.set_title(f"{title} ({n:,} flows)", fontsize=12)
        ax.set_ylim(0, 1.07)
        ax.tick_params(labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Share of flows in each trust level", fontsize=11)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    fig.legend([Patch(color=c) for c in COLOURS.values()],
               ["Trusted (Allow)", "Suspicious (Restrict)", "Untrusted (Reject)"],
               loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.text(0.5, 0.085, "B = benign flows, M = malicious flows (labels used only to mark the decisions afterwards)",
             ha="center", fontsize=10, color="#444444")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    png = os.path.join(out_dir, "hybrid_trust_levels_by_class.png")
    fig.savefig(png, dpi=150, metadata={"Software": None})
    plt.close(fig)

    cols = ["dataset", "group", "setting", "n_flows", "benign_trusted", "benign_suspicious", "benign_untrusted",
            "malicious_trusted", "malicious_suspicious", "malicious_untrusted"]
    data = pd.DataFrame(rows, columns=cols)
    data.to_csv(os.path.join(out_dir, "hybrid_trust_levels_by_class_data.csv"), index=False, float_format="%.6f")
    with open(os.path.join(out_dir, "sources.json"), "w") as f:
        json.dump({"evaluation_run": os.path.basename(run), "inputs_sha256": sources,
                   "figure_sha256": sha256(png), "matplotlib": matplotlib.__version__}, f, indent=2)
    print(f"Figure written to {os.path.relpath(png, PROJECT_ROOT)}")
    print(data.round(4).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())