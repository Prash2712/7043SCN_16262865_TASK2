_# -*- coding: utf-8 -*-
"""
Self-check script to verify the project structure and outputs.

This script checks for the presence of all required files and directories
to ensure the project is complete and ready for submission.
"""

import os


def main():
    """Runs the self-check process."""
    print("--- Running Project Self-Check ---")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    required_files = [
        "configs/config.yaml",
        "ppo/__init__.py",
        "ppo/model.py",
        "ppo/ppo_agent.py",
        "ppo/replay_buffer.py",
        "reward_shaping/__init__.py",
        "reward_shaping/shapers.py",
        "training/__init__.py",
        "training/trainer.py",
        "evaluation/__init__.py",
        "evaluation/evaluator.py",
        "utils/__init__.py",
        "utils/plotting.py",
        "utils/metrics.py",
        "run_train.py",
        "run_eval.py",
        "requirements.txt",
        "README.md",
        "scripts/self_check.py"
    ]
    
    all_ok = True
    
    print("\n[1] Checking for required source files...")
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"  [OK] Found: {file_path}")
        else:
            print(f"  [FAIL] Missing: {file_path}")
            all_ok = False
            
    print("\n[2] Checking for generated output directories (after running run_train.py)..." )
    output_dirs = [
        "outputs/ppo_variant3_baseline/logs",
        "outputs/ppo_variant3_baseline/models",
        "outputs/ppo_variant3_baseline/plots",
        "outputs/ppo_variant3_reward_shaping/logs",
        "outputs/ppo_variant3_reward_shaping/models",
        "outputs/ppo_variant3_reward_shaping/plots",
    ]
    for dir_path in output_dirs:
        full_path = os.path.join(base_dir, dir_path)
        if os.path.isdir(full_path):
            print(f"  [OK] Found directory: {dir_path}")
        else:
            print(f"  [INFO] Not found (expected after training): {dir_path}")

    print("\n[3] Checking for generated evaluation directories (after running run_eval.py)..." )
    eval_dirs = [
        "outputs/eval_vs_Random/logs",
        "outputs/eval_vs_Heuristic_V1/logs",
        "outputs/eval_vs_Heuristic_V2/logs",
        "outputs/evaluation_summary_plots"
    ]
    for dir_path in eval_dirs:
        full_path = os.path.join(base_dir, dir_path)
        if os.path.isdir(full_path):
            print(f"  [OK] Found directory: {dir_path}")
        else:
            print(f"  [INFO] Not found (expected after evaluation): {dir_path}")
            
    print("\n--- Self-Check Complete ---")
    if all_ok:
        print("Result: All required source files are present. Good to go!")
    else:
        print("Result: One or more required files are missing. Please check the list above.")

if __name__ == "__main__":
    main()
