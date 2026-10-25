"""
6_eda.py

Performs exploratory data analysis (EDA) on the feature-selected
development data for RT-IoT2022 and IoT-23.

The analysis is descriptive and does not influence feature selection,
model training, or other pipeline decisions. The test partitions are
never opened.

The binary target is represented using the class labels 'benign' and
'malicious'. Analysis covers the selected numeric and categorical
variables specified in each dataset's feature-selection configuration.

Analysis includes:
    - Class distributions, including distinct rows and the raw traffic
      represented by row_multiplicity.
    - Distributions of attack types or detailed labels.
    - Numeric descriptive statistics by class and group, including
      means, standard deviations, percentiles, and zero-value shares.
    - Boxplots comparing numeric variables between classes.
    - Categorical-level shares within each class.
    - Spearman correlation analysis of selected numeric variables.

For IoT-23, tables and plots are additionally separated into Capture-35-1
rows and rows from other captures to examine differences between these
groups. RT-IoT2022 is analysed as a single development-data group.

Usage:
    python src/6_eda.py rt_iot2022
    python src/6_eda.py iot23
    python src/6_eda.py
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FS_DIR = os.path.join(PROJECT_ROOT, "results", "05_feature_selection")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "06_eda")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TARGET, ROW_ID, MULT = "class", "row_id", "row_multiplicity"     # 'class' = named copy of 'malicious'
CLASSES = ["benign", "malicious"]
COLORS = {"benign": "#4C72B0", "malicious": "#C44E52"}
CFG = {"rt_iot2022": {"fine": "attack_type", "grouped": False},
       "iot23": {"fine": "detailed_label", "grouped": True}}
GROUP_LABEL = {"capture_35_1_rows": "Capture-35-1", "whole_scenario": "Other captures"}


def safe(name):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def load(ds):
    with open(os.path.join(FS_DIR, f"{ds}_selection.json")) as f:
        sel = json.load(f)
    dev = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ds}_train.csv"), low_memory=False)
    dev[TARGET] = dev["malicious"].map({0: "benign", 1: "malicious"})
    if CFG[ds]["grouped"]:
        folds = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ds}_cv_folds.csv"),
                            usecols=[ROW_ID, "scenario_group"])
        dev = dev.merge(folds, on=ROW_ID, how="left", validate="one_to_one")
        dev["group"] = dev["scenario_group"].map(GROUP_LABEL)
    else:
        dev["group"] = "All development rows"
    return dev, sel


def class_tables(dev, ds):
    fine = CFG[ds]["fine"]
    dist = pd.DataFrame({
        "distinct_rows": dev[TARGET].value_counts(),
        "raw_rows_represented": dev.groupby(TARGET)[MULT].sum(),
    }).reindex(CLASSES)
    dist["distinct_share"] = (dist["distinct_rows"] / dist["distinct_rows"].sum()).round(4)
    dist["raw_share"] = (dist["raw_rows_represented"] / dist["raw_rows_represented"].sum()).round(4)
    fine_t = dev.groupby([TARGET, fine]).agg(distinct_rows=(ROW_ID, "size"),
                                             raw_rows_represented=(MULT, "sum")).reset_index()
    fine_t = fine_t.sort_values("distinct_rows", ascending=False)
    dist.to_csv(os.path.join(OUT_DIR, f"{ds}_class_distribution.csv"))
    fine_t.to_csv(os.path.join(OUT_DIR, f"{ds}_fine_label_distribution.csv"), index=False)
    print("\nClass distribution (development):\n" + dist.to_string())
    print("\nBy fine-grained label:\n" + fine_t.to_string(index=False))


def numeric_descriptives(dev, ds, numeric):
    rows = []
    for v in numeric:
        for g, gdf in dev.groupby("group"):
            for c in CLASSES:
                s = gdf.loc[gdf[TARGET] == c, v]
                if s.empty:
                    continue
                q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
                rows.append({"variable": v, "group": g, TARGET: c, "n": len(s),
                             "mean": s.mean(), "sd": s.std(), "min": s.min(),
                             "p05": q[0.05], "p25": q[0.25], "median": q[0.5],
                             "p75": q[0.75], "p95": q[0.95], "max": s.max(),
                             "share_zero": float((s == 0).mean())})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT_DIR, f"{ds}_numeric_descriptives.csv"), index=False)
    print("\nNumeric descriptives (median / p95 / share zero):")
    print(t[["variable", "group", TARGET, "n", "median", "p95", "share_zero"]]
          .round(4).to_string(index=False))


def boxplots(dev, ds, numeric):
    groups = list(dev["group"].dropna().unique())
    for v in numeric:
        fig, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 4.5), squeeze=False)
        for ax, g in zip(axes[0], groups):
            gdf = dev[dev["group"] == g]
            data, labels, cols = [], [], []
            for c in CLASSES:
                s = gdf.loc[gdf[TARGET] == c, v].to_numpy()
                if len(s):
                    data.append(s)
                    labels.append(f"{c}\n(n={len(s):,})")
                    cols.append(COLORS[c])
            bp = ax.boxplot(data, patch_artist=True, showfliers=False)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, fontsize=8)
            for patch, col in zip(bp["boxes"], cols):
                patch.set_facecolor(col)
                patch.set_alpha(0.7)
            ax.set_yscale("symlog", linthresh=1)
            ax.set_title(g, fontsize=10)
            ax.set_ylabel(f"{v} (symlog scale)")
        fig.suptitle(f"{ds}: {v} by class (development; outliers not drawn)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"{ds}_box_{safe(v)}.png"), dpi=200)
        plt.close(fig)


def categorical_shares(dev, ds, levels):
    rows = []
    for v, kept in levels.items():
        s = dev[v].astype(object).where(dev[v].notna(), "<missing>").astype(str)
        s = s.where(s.isin(kept), "other (dropped levels)")
        for g, gdf in dev.assign(_lvl=s).groupby("group"):
            ct = pd.crosstab(gdf["_lvl"], gdf[TARGET], normalize="columns").reindex(columns=CLASSES)
            for lvl in ct.index:
                for c in CLASSES:
                    rows.append({"variable": v, "group": g, "level": lvl, TARGET: c,
                                 "share_within_class": ct.loc[lvl, c]})
        groups = list(dev["group"].dropna().unique())
        fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4.5), squeeze=False)
        for ax, g in zip(axes[0], groups):
            gdf = dev.assign(_lvl=s)[dev["group"] == g]
            ct = pd.crosstab(gdf["_lvl"], gdf[TARGET], normalize="columns").reindex(columns=CLASSES).fillna(0)
            ct.plot(kind="bar", ax=ax, color=[COLORS[c] for c in CLASSES], alpha=0.8, width=0.8)
            ax.set_title(g, fontsize=10)
            ax.set_ylabel("share of class rows")
            ax.set_xlabel(v)
            ax.tick_params(axis="x", labelsize=8, rotation=45)
        fig.suptitle(f"{ds}: {v} level shares within each class (development)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"{ds}_bar_{safe(v)}.png"), dpi=200)
        plt.close(fig)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT_DIR, f"{ds}_categorical_shares.csv"), index=False)


def spearman_heatmap(dev, ds, numeric):
    if len(numeric) < 2:
        return
    corr = dev[numeric].corr(method="spearman")
    corr.to_csv(os.path.join(OUT_DIR, f"{ds}_spearman_selected_numeric.csv"))
    fig, ax = plt.subplots(figsize=(1.1 * len(numeric) + 3, 1.0 * len(numeric) + 2.5))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(numeric)))
    ax.set_xticklabels(numeric, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(numeric)))
    ax.set_yticklabels(numeric, fontsize=8)
    for i in range(len(numeric)):
        for j in range(len(numeric)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"{ds}: Spearman correlation, selected numeric variables (development)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{ds}_spearman_heatmap.png"), dpi=200)
    plt.close(fig)
    print("\nSpearman correlation of selected numeric variables:\n" + corr.round(3).to_string())


def run(ds):
    print("=" * 70 + f"\nEDA: {ds} (development data only)\n" + "=" * 70)
    dev, sel = load(ds)
    numeric = sel["numeric_variables"]
    levels = sel["categorical_levels"]
    print(f"Rows: {len(dev):,} | numeric: {numeric} | categorical: {list(levels)}")
    class_tables(dev, ds)
    numeric_descriptives(dev, ds, numeric)
    boxplots(dev, ds, numeric)
    categorical_shares(dev, ds, levels)
    spearman_heatmap(dev, ds, numeric)


if __name__ == "__main__":
    import sys
    for ds in sys.argv[1:] or ["rt_iot2022", "iot23"]:
        run(ds)
    print("\nDone. Outputs: results/06_eda/ (tables) and results/06_eda/figures/ (PNG)")