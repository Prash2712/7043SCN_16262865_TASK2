_# -*- coding: utf-8 -*-
"""
Main entry point for running the evaluation process for Variant 3.

This script evaluates the trained reward-shaped agent against multiple
opponent types (Random, Heuristic_V1, Heuristic_V2) and generates
comparison plots.
"""

import yaml
import os
from typing import Dict

from evaluation.evaluator import Evaluator
from utils.plotting import plot_evaluation_comparison


def main():
    """Main function to orchestrate the evaluation runs."""
    # --- Load Base Configuration ---
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # --- Define Paths for the Trained Model ---
    # This assumes the training script was run and the final shaped model exists
    trained_model_dir = os.path.join("outputs", "ppo_variant3_reward_shaping", "models")
    actor_path = os.path.join(trained_model_dir, "actor_final.pt")
    critic_path = os.path.join(trained_model_dir, "critic_final.pt")

    if not (os.path.exists(actor_path) and os.path.exists(critic_path)):
        print(f"ERROR: Trained models not found at {trained_model_dir}")
        print("Please run the training script (run_train.py) first.")
        return

    config["evaluation"]["actor_model_path"] = actor_path
    config["evaluation"]["critic_model_path"] = critic_path

    # --- Run Evaluation Against Different Opponents ---
    opponents_to_test = ["Random", "Heuristic_V1", "Heuristic_V2"]
    eval_results: Dict[str, Dict] = {}

    for opponent in opponents_to_test:
        print("\n" + "="*60)
        print(f"  EVALUATING AGENT vs: {opponent}")
        print("="*60 + "\n")

        run_name = f"eval_vs_{opponent}"
        evaluator = Evaluator(config=config, opponent_type=opponent, run_name=run_name)
        evaluator.evaluate()
        
        # Store results for final comparison plot
        avg_score, win_rate = evaluator.metrics.get_summary()
        eval_results[opponent] = {"Win Rate": win_rate, "Avg Score": avg_score}

    # --- Generate Final Comparison Plots ---
    print("\n" + "="*60)
    print("  GENERATING EVALUATION COMPARISON PLOTS")
    print("="*60 + "\n")
    
    summary_plot_dir = os.path.join("outputs", "evaluation_summary_plots")
    plot_evaluation_comparison(eval_results, summary_plot_dir)

    print("\nAll evaluation runs and plotting complete.")


if __name__ == "__main__":
    main()
