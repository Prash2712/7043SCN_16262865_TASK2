_# -*- coding: utf-8 -*-
"""
Evaluation loop for the trained PPO agent.

This module handles the evaluation of a trained agent against different
opponent types to assess its performance and generalization.
"""

import os
import sys
from typing import Dict, Any

from tqdm import tqdm

# Add the Chef's Hat Gym source to the Python path
sys.path.insert(0, '/home/ubuntu/ChefsHatGYM/src')

from ppo.ppo_agent import PPOAgent
from rooms.room import Room
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgentV1, HeuristicAgentV2
from reward_shaping.shapers import NoOpShaper # Evaluation uses sparse rewards
from utils.metrics import EvaluationMetrics


class Evaluator:
    """Manages the evaluation process for a trained PPO agent."""

    def __init__(self, config: Dict[str, Any], opponent_type: str, run_name: str):
        """Initializes the Evaluator.

        Args:
            config: A dictionary containing the full project configuration.
            opponent_type: The type of opponent to evaluate against ('Random', etc.).
            run_name: A unique name for this evaluation run (e.g., 'eval_vs_random').
        """
        self.config = config
        self.opponent_type = opponent_type
        self.output_dir = os.path.join(config["outputs"]["output_dir"], run_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize metrics logger for this specific run
        self.metrics = EvaluationMetrics(log_dir=os.path.join(self.output_dir, "logs"))

    def evaluate(self):
        """Executes the main evaluation loop."""
        print(f"[Evaluator] Starting evaluation against: {self.opponent_type}")

        # --- 1. Initialize Environment and Agents ---
        opponent_map = {
            "Random": RandomAgent,
            "Heuristic_V1": HeuristicAgentV1,
            "Heuristic_V2": HeuristicAgentV2
        }
        opponent_class = opponent_map.get(self.opponent_type, RandomAgent)

        # Initialize the PPO agent
        state_dim = 28
        action_dim = 200
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            p_hyperparameters=self.config["ppo"],
            reward_shaper=NoOpShaper()  # IMPORTANT: No reward shaping during evaluation
        )

        # Load the trained model weights
        actor_path = self.config["evaluation"]["actor_model_path"]
        critic_path = self.config["evaluation"]["critic_model_path"]
        if actor_path and critic_path and os.path.exists(actor_path) and os.path.exists(critic_path):
            agent.load_models(actor_path, critic_path)
            print(f"[Evaluator] Loaded trained models for evaluation.")
        else:
            print("[Evaluator] WARNING: No trained models found. Evaluating an untrained agent.")

        # --- 2. Run Evaluation Matches ---
        total_matches = self.config["evaluation"]["total_matches"]
        progress_bar = tqdm(range(total_matches), desc=f"Evaluating vs {self.opponent_type}")

        for match_num in progress_bar:
            room = Room(room_name=f"PPO Eval vs {self.opponent_type}", num_players=4, verbose=False)
            room.add_player(agent, "PPO_Agent")
            room.add_player(opponent_class("Opponent_1"), "Opponent_1")
            room.add_player(opponent_class("Opponent_2"), "Opponent_2")
            room.add_player(opponent_class("Opponent_3"), "Opponent_3")

            match_results = room.run_game()

            # --- 3. Log Metrics ---
            final_scores = match_results["final_scores"]
            ppo_score = final_scores.get("PPO_Agent", 0)
            ppo_position = match_results["finishing_order"].index("PPO_Agent") + 1
            self.metrics.log_match(match_num, ppo_score, ppo_position)

            avg_score, win_rate = self.metrics.get_summary()
            progress_bar.set_postfix({"Win Rate": f"{win_rate:.2%}", "Avg Score": f"{avg_score:.2f}"})

        # --- 4. Finalize Evaluation ---
        self.metrics.save_log()
        print(f"[Evaluator] Evaluation against {self.opponent_type} finished.")
        avg_score, win_rate = self.metrics.get_summary()
        print(f"[Evaluator] Final Results - Win Rate: {win_rate:.2%}, Average Score: {avg_score:.2f}")
