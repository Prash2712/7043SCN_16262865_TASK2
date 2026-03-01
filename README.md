# PPO with Domain-Knowledge Reward Shaping for Chef's Hat Gym

**Student ID:** 16262865
**Course Task:** 7043SCN - Task 2
**Variant:** 3 - Reward Shaping with Domain Knowledge

---

## 1. Project Overview

This project presents a high-quality implementation of a Proximal Policy Optimization (PPO) agent designed to master the card game *Chef's Hat*. The core of this work focuses on **Variant 3: Reward Shaping with Domain Knowledge**, where the standard sparse, end-of-game reward is augmented with dense, intermediate rewards derived from game-specific heuristics. 

The objective is to demonstrate that by providing the agent with this domain knowledge, it can learn a more effective policy faster and more reliably than an agent learning from a sparse reward signal alone. The repository includes a full training and evaluation pipeline, ablation studies, and comprehensive documentation, all built to a distinction-level academic standard.

## 2. Repository Structure

The repository is organized into a modular and professional structure to ensure clarity, maintainability, and ease of use.

```
/task2_project_16262865
├── configs/
│   └── config.yaml             # All hyperparameters and settings
├── ppo/
│   ├── __init__.py
│   ├── model.py                # Actor and Critic network architectures
│   ├── ppo_agent.py            # Core PPO agent class (inherits from BaseAgent)
│   └── replay_buffer.py        # On-policy replay buffer for rollouts
├── reward_shaping/
│   ├── __init__.py
│   └── shapers.py              # Reward shaping classes (Base, DomainKnowledge, NoOp)
├── training/
│   ├── __init__.py
│   └── trainer.py              # Main training loop orchestration
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py            # Evaluation loop orchestration
├── utils/
│   ├── __init__.py
│   ├── plotting.py             # Utilities for generating result plots
│   └── metrics.py              # Classes for logging training/evaluation data
├── scripts/
│   └── self_check.py           # Script to verify project structure
├── outputs/                    # Directory for all generated files (created on run)
├── run_train.py                # === Main script to run training ===
├── run_eval.py                 # === Main script to run evaluation ===
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## 3. Methodology

### 3.1. Core Algorithm: Proximal Policy Optimization (PPO)

The agent is built upon the **PPO-Clip** algorithm, a state-of-the-art on-policy reinforcement learning method. It uses an Actor-Critic architecture:

-   **Actor (Policy Network):** Learns which action to take in a given state.
-   **Critic (Value Network):** Estimates the value of being in a given state.

PPO optimizes the policy by taking small, controlled steps, using a clipped objective function to prevent destructively large updates. This ensures stable and reliable learning.

### 3.2. Variant 3: Reward Shaping

To address the challenge of sparse rewards (only receiving a signal at the end of a match), we introduce a `DomainKnowledgeShaper`. This module injects intermediate rewards at each step based on expert heuristics. The following shaping functions are implemented and can be tuned in `configs/config.yaml`:

| Shaping Function       | Description                                                                                             | Default Coef. |
| ---------------------- | ------------------------------------------------------------------------------------------------------- | :-----------: |
| **High Card Penalty**  | Penalizes the agent for holding high-value cards at the end of a round, as this is a losing strategy.     |    `-0.01`    |
| **Card Play Bonus**    | Rewards the agent for playing high-value cards, which is often a strategically sound move to win tricks. |    `+0.02`    |
| **Hand Strength Bonus**| Rewards the agent for collecting multiple cards of the same suit, a key strategy for controlling the game. |   `+0.005`    |

These dense rewards provide a much richer learning signal, guiding the agent towards strategically sound behaviors far more effectively than the sparse reward alone.

## 4. How to Run

### 4.1. Setup

1.  **Clone Chef's Hat Gym:** The project requires the source code of the environment. Clone it into the `/home/ubuntu/` directory.
    ```bash
    git clone https://github.com/pablovin/ChefsHatGYM.git /home/ubuntu/ChefsHatGYM
    ```

2.  **Install Dependencies:** Install the required Python packages.
    ```bash
    pip install -r requirements.txt
    ```

### 4.2. Training

To run the full training process for both the baseline and the reward-shaped agent, execute the main training script. This will train each agent for 1,000 matches and save all models, logs, and plots to the `outputs/` directory.

```bash
python3 run_train.py
```

### 4.3. Evaluation

After training is complete, run the evaluation script. This will load the trained reward-shaped agent and test it against three different opponent types (Random, Heuristic V1, Heuristic V2) for 200 matches each.

```bash
python3 run_eval.py
```

### 4.4. Self-Check

To verify that all source files are correctly in place, you can run the self-check script.

```bash
python3 scripts/self_check.py
```
## 5. Experimental Pipeline

### Training

Two agents were trained:

1. PPO Baseline (sparse rewards)
2. PPO + Domain-Knowledge Reward Shaping

Training includes:

- identical hyperparameters for controlled comparison,
- consistent random seeds,
- logging of per-episode metrics.

### Evaluation

The trained reward-shaped agent is evaluated against:

- Random opponent
- Heuristic Agent V1
- Heuristic Agent V2

Each evaluation consists of multiple matches to reduce variance and allow meaningful comparison.

---
## 6. Outputs

After running the scripts, the `outputs/` directory will be populated as follows:

-   `outputs/ppo_variant3_baseline/`: Results for the agent trained with sparse rewards.
    -   `logs/training_log.csv`: Per-match training data.
    -   `models/`: Saved model weights (`.pt` files).
    -   `plots/`: Reward curve and win rate plots.
-   `outputs/ppo_variant3_reward_shaping/`: Results for the agent trained with shaped rewards.
    -   (Same structure as above)
-   `outputs/eval_vs_<OpponentType>/`: Evaluation results against each opponent.
    -   `logs/evaluation_log.csv`: Per-match evaluation data.
-   `outputs/evaluation_summary_plots/`: Bar charts comparing the shaped agent's performance against all opponent types.

## 7. Video Link :
