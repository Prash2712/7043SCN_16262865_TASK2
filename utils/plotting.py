_# -*- coding: utf-8 -*-
"""
Plotting utilities for visualizing training and evaluation results.

This module provides functions to generate and save various plots, such as
reward curves, win rate trends, and performance comparison charts.
"""

import os
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set a professional plot style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["font.family"] = "sans-serif"


def plot_training_results(log_file: str, plot_dir: str):
    """Generates and saves plots from a training log file.

    Args:
        log_file: Path to the training_log.csv file.
        plot_dir: Directory to save the generated plots.
    """
    if not os.path.exists(log_file):
        print(f"[Plotting] Log file not found: {log_file}")
        return

    os.makedirs(plot_dir, exist_ok=True)
    df = pd.read_csv(log_file)

    # --- 1. Reward Curve ---
    plt.figure()
    sns.lineplot(data=df, x=\"Match\", y=\"Score\", label=\"Per-Match Score\")
    plt.plot(df[\"Match\"], df[\"Score\"].rolling(window=50).mean(), label=\"50-Match Avg. Score\", color=\"red\")
    plt.title(\"Training: Reward Curve\", fontsize=16, weight=\"bold\")
    plt.xlabel(\"Match Number\")
    plt.ylabel(\"Final Score\")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, \"reward_curve.png\"))
    plt.close()

    # --- 2. Win Rate Curve ---
    df[\"Win\"] = (df[\"Position\"] == 1).astype(int)
    df[\"Win Rate\"] = df[\"Win\"].expanding().mean()
    plt.figure()
    sns.lineplot(data=df, x=\"Match\", y=\"Win Rate\", color=\"green\")
    plt.title(\"Training: Cumulative Win Rate\", fontsize=16, weight=\"bold\")
    plt.xlabel(\"Match Number\")
    plt.ylabel(\"Win Rate\")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(plot_dir, \"win_rate_curve.png\"))
    plt.close()

    print(f\"[Plotting] Training plots saved to {plot_dir}\")


def plot_evaluation_comparison(eval_results: Dict[str, Dict], plot_dir: str):
    """Generates bar plots comparing evaluation results across different runs.

    Args:
        eval_results: A dictionary where keys are run names (e.g., \"vs_Random\")
            and values are dictionaries with \"Win Rate\" and \"Avg Score\".
        plot_dir: Directory to save the generated plots.
    """
    os.makedirs(plot_dir, exist_ok=True)
    df = pd.DataFrame.from_dict(eval_results, orient=\"index\")

    # --- 1. Win Rate Comparison ---
    plt.figure()
    sns.barplot(x=df.index, y=df[\"Win Rate\"], palette=\"viridis\")
    plt.title(\"Evaluation: Win Rate Comparison\", fontsize=16, weight=\"bold\")
    plt.xlabel(\"Opponent Type\")
    plt.ylabel(\"Win Rate\")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(plot_dir, \"comparison_win_rate.png\"))
    plt.close()

    # --- 2. Average Score Comparison ---
    plt.figure()
    sns.barplot(x=df.index, y=df[\"Avg Score\"], palette=\"plasma\")
    plt.title(\"Evaluation: Average Score Comparison\", fontsize=16, weight=\"bold\")
    plt.xlabel(\"Opponent Type\")
    plt.ylabel(\"Average Score\")
    plt.savefig(os.path.join(plot_dir, \"comparison_avg_score.png\"))
    plt.close()

    print(f\"[Plotting] Evaluation comparison plots saved to {plot_dir}\")
