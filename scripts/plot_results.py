#!/usr/bin/env python
"""Plot comparison results from CSV files in a test set directory.

Usage:
    python scripts/plot_results.py

Configuration:
    Edit FOLDER (test set path) below.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.stats import ttest_rel

# ── Configuration ─────────────────────────────────────────────────────────────
FOLDER = 'data/testSet_20A_50T_CONDET'
# ──────────────────────────────────────────────────────────────────────────────

LABELS = [
    'Success Rate', 'Makespan', 'Time Cost',
    'Average Waiting Time', 'Sum Traveling Distance', 'Efficiency',
]
LABELS_IN_CSV = [
    'success_rate', 'makespan', 'time_cost',
    'waiting_time', 'travel_dist', 'efficiency',
]

metrics_dir = os.path.join(FOLDER, 'metrics')
os.makedirs(metrics_dir, exist_ok=True)

# Collect all CSV result files
csv_files = natsorted(
    glob.glob(os.path.join(FOLDER, 'results', '*.csv')),
    key=lambda y: y.lower(),
)

if not csv_files:
    # Fallback: try root of folder
    csv_files = natsorted(
        glob.glob(os.path.join(FOLDER, '*.csv')),
        key=lambda y: y.lower(),
    )

dfs = []
file_names = []
for file in csv_files:
    if file.endswith('.csv'):
        file_names.append(os.path.basename(file).replace('.csv', ''))
        dfs.append(pd.read_csv(file))

if not dfs:
    print(f"No CSV files found in {FOLDER}/results/")
    exit(1)

# ── Pairwise statistical tests ────────────────────────────────────────────────
p_metrics = pd.DataFrame(columns=['Method'] + file_names)
for m, label_csv in enumerate(LABELS_IN_CSV):
    for i, df_i in enumerate(dfs):
        p = {}
        for j, df_j in enumerate(dfs):
            if df_i is not df_j:
                result = ttest_rel(df_i[label_csv].values, df_j[label_csv].values)
                p[file_names[j]] = (
                    np.format_float_scientific(result.statistic, 2)
                    + ', ' + np.format_float_scientific(result.pvalue, 2)
                )
            else:
                p[file_names[j]] = '0, 0'
        p['Method'] = file_names[i] + ' ' + LABELS[m]
        p = pd.DataFrame(p, index=[file_names[i]])
        p_metrics = pd.concat([p_metrics, p])
p_metrics.to_csv(os.path.join(metrics_dir, 'p_metrics.csv'), index=False)

# ── Summary metrics CSV ───────────────────────────────────────────────────────
metrics_csv = pd.DataFrame(columns=['Method'] + LABELS)
for i, df in enumerate(dfs):
    metrics = {}
    for j, label in enumerate(LABELS_IN_CSV):
        if label == 'success_rate':
            metrics[LABELS[j]] = (
                str(round(np.sum(df[label]) / len(df[label]), 3))
                + ' (+- ' + str(round(np.nanstd(df[label]), 3)) + ')'
            )
        else:
            metrics[LABELS[j]] = (
                str(round(np.nanmean(df[label]), 3))
                + ' (+- ' + str(round(np.nanstd(df[label]), 3)) + ')'
            )
    metrics['Method'] = file_names[i]
    metrics = pd.DataFrame(metrics, index=[file_names[i]])
    metrics_csv = pd.concat([metrics_csv, metrics])
metrics_csv.to_csv(os.path.join(metrics_dir, 'metrics.csv'), index=False)

# ── Per-episode line plots ────────────────────────────────────────────────────
for m, label in enumerate(LABELS_IN_CSV):
    plt.figure(dpi=300)
    for idx, df in enumerate(dfs):
        plt.plot(df[label], label=file_names[idx])
    plt.legend()
    plt.title(LABELS[m])
    plt.savefig(os.path.join(metrics_dir, f'{LABELS[m]}.png'))
    plt.close()

# ── Average bar plots with error bars ─────────────────────────────────────────
for m, label in enumerate(LABELS_IN_CSV):
    plt.figure(dpi=300)
    for idx, df in enumerate(dfs):
        mean = np.nanmean(df[label])
        std = np.nanstd(df[label])
        min_ = np.min(df[label])
        max_ = np.max(df[label])
        plt.errorbar(idx, mean, std, fmt='b', lw=3, alpha=0.5)
        plt.errorbar(
            idx, mean,
            np.array([[round(mean - min_, 4)], [round(max_ - mean, 4)]]),
            fmt='.', lw=1, label=file_names[idx],
        )
    plt.legend(fontsize="7")
    plt.xticks([])
    plt.title(LABELS[m])
    plt.savefig(os.path.join(metrics_dir, f'{LABELS[m]} Average.png'))
    plt.close()

print(f"Plots saved to {metrics_dir}/")
