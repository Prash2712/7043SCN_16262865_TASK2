_# -*- coding: utf-8 -*-
"""
Main entry point for running the training process for Variant 3.

This script loads the configuration, initializes the appropriate trainer
(one for the baseline sparse-reward agent and one for the shaped-reward agent),
and runs the training loop. It also generates the final plots.
"""

import yaml
import os

from training.trainer import Trainer
from utils.plotting import plot_training_results


def main():
    """Main function to orchestrate the training runs."""
    # --- Load Base Configuration ---
    with open("configs/config.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    # --- 1. Train the Baseline (Sparse Reward) Agent ---
    print("\n" + "="*60)
    print("  RUNNING: BASELINE AGENT (SPARSE REWARDS)")
    print("="*60 + "\n")
    baseline_config = base_config.copy()
    baseline_config["reward_shaping"]["enabled"] = False
    baseline_config["project_name"] = "ppo_variant3_baseline"
    
    baseline_trainer = Trainer(config=baseline_config)
    baseline_trainer.train()

    # --- 2. Train the Variant 3 (Shaped Reward) Agent ---
    print("\n" + "="*60)
    print("  RUNNING: VARIANT 3 AGENT (SHAPED REWARDS)")
    print("="*60 + "\n")
    shaped_config = base_config.copy()
    shaped_config["reward_shaping"]["enabled"] = True
    shaped_config["project_name"] = "ppo_variant3_reward_shaping"

    shaped_trainer = Trainer(config=shaped_config)
    shaped_trainer.train()

    # --- 3. Generate Plots for Both Runs ---
    print("\n" + "="*60)
    print("  GENERATING PLOTS")
    print("="*60 + "\n")
    
    # Plots for baseline
    baseline_log = os.path.join("outputs", baseline_config["project_name"], "logs", "training_log.csv")
    baseline_plot_dir = os.path.join("outputs", baseline_config["project_name"], "plots")
    plot_training_results(baseline_log, baseline_plot_dir)

    # Plots for shaped agent
    shaped_log = os.path.join("outputs", shaped_config["project_name"], "logs", "training_log.csv")
    shaped_plot_dir = os.path.join("outputs", shaped_config["project_name"], "plots")
    plot_training_results(shaped_log, shaped_plot_dir)

    print("\nAll training runs and plotting complete.")


if __name__ == "__main__":
    main()
