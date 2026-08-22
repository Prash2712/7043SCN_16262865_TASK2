# PPO with Domain-Knowledge Reward Shaping

A reinforcement-learning project investigating whether **domain-informed reward shaping** can provide a richer learning signal than sparse end-of-game rewards in the multi-agent Chef's Hat environment.

The implementation uses **Proximal Policy Optimization (PPO)** with an Actor–Critic architecture and separates reward shaping, training, evaluation, configuration and metrics into reusable modules.

## Technical focus

- Proximal Policy Optimization (PPO-Clip)
- Actor–Critic policy/value networks
- Sparse-reward reinforcement learning
- Domain-knowledge reward shaping
- Controlled baseline-vs-shaped comparisons
- Config-driven experiments
- Multi-opponent evaluation
- Reproducible random seeds and logging

## Repository structure

```text
.
├── configs/
│   └── config.yaml
├── ppo/
│   ├── model.py
│   ├── ppo_agent.py
│   └── replay_buffer.py
├── reward_shaping/
│   └── shapers.py
├── training/
│   └── trainer.py
├── evaluation/
│   └── evaluator.py
├── utils/
│   ├── plotting.py
│   └── metrics.py
├── scripts/
│   └── self_check.py
├── run_train.py
├── run_eval.py
├── requirements.txt
└── README.md
```

## Approach

### Baseline

The baseline agent learns from the environment's sparse outcome signal without additional domain-informed shaping.

### Domain-knowledge shaping

The shaped agent augments the environment reward with intermediate signals designed around game-state information. The implementation exposes these coefficients through configuration so the shaping design can be varied without rewriting the learning loop.

Examples include incentives or penalties associated with card-play efficiency and hand state.

The purpose is not to assume that hand-crafted shaping is automatically superior. The experiment is structured so the shaped agent can be compared against the sparse-reward baseline under otherwise controlled settings.

## PPO design

PPO updates the policy using a clipped surrogate objective to constrain destructive policy changes. The implementation follows the standard Actor–Critic pattern:

- **Actor:** produces the action policy
- **Critic:** estimates state value
- **Rollout buffer:** stores on-policy trajectories for PPO updates

This architecture is appropriate for comparing learning behaviour under different reward definitions while keeping the core policy optimiser fixed.

## Running the project

Install the project dependencies and the Chef's Hat Gym environment required by the implementation, then run:

```bash
python run_train.py
```

For evaluation:

```bash
python run_eval.py
```

A structural self-check is also provided:

```bash
python scripts/self_check.py
```

## Evaluation design

The project is designed to compare learned behaviour across several opponent types, including random and heuristic agents. Training and evaluation outputs are written to the generated `outputs/` structure when experiments are run.

This README intentionally does **not** publish performance figures that are not committed as independently inspectable result artifacts in the repository.

## Why this project matters

Sparse and delayed rewards are a central reinforcement-learning difficulty: a terminal outcome may provide too little information about which earlier decisions were useful. Reward shaping can improve the density of the learning signal, but poorly designed shaping can also bias behaviour.

This project provides a modular setting for examining that trade-off with a modern policy-gradient method.

## Technical stack

`Python` · `PyTorch` · `PPO` · `Actor–Critic` · `YAML configuration` · `NumPy` · `pandas` · `Matplotlib`

## Academic provenance

Originally developed for Coventry University module **7043SCN — Generative AI and Reinforcement Learning**, using the assigned reward-shaping variant.

A demonstration video associated with the original coursework is available in the project history/documentation.

## Author

**Prasanth Balisetty**  
Data Science & Machine Learning

[GitHub](https://github.com/Prash2712)
