_# -*- coding: utf-8 -*-
"""
Training loop for the PPO agent in the Chef's Hat environment.

This module orchestrates the training process. It initializes the environment,
agent, and opponents, then runs a specified number of matches, collecting
data and triggering the agent's learning process.
"""

import os
import sys
from typing import Dict, Any

import pandas as pd
from tqdm import tqdm

# Add the Chef's Hat Gym source to the Python path
# This is a common workaround for environments not installed as a package
sys.path.insert(0, '/home/ubuntu/ChefsHatGYM/src')

from ppo.ppo_agent import PPOAgent
from rooms.room import Room
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgentV1, HeuristicAgentV2
from reward_shaping.shapers import DomainKnowledgeShaper, NoOpShaper
from utils.metrics import TrainingMetrics


class Trainer:
    """Manages the entire training process for the PPO agent."""

    def __init__(self, config: Dict[str, Any]):
        """Initializes the Trainer.

        Args:
            config: A dictionary containing the full project configuration.
        """
        self.config = config
        self.output_dir = os.path.join(config["outputs"]["output_dir"], config["project_name"])
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize metrics logger
        self.metrics = TrainingMetrics(log_dir=os.path.join(self.output_dir, "logs"))

    def train(self):
        """Executes the main training loop."""
        print("[Trainer] Starting training...")

        # --- 1. Initialize Environment and Agents ---
        # Get opponent class based on config
        opponent_map = {
            "Random": RandomAgent,
            "Heuristic_V1": HeuristicAgentV1,
            "Heuristic_V2": HeuristicAgentV2
        }
        opponent_class = opponent_map.get(self.config["env"]["opponent_type"], RandomAgent)
        print(f"[Trainer] Training against opponent: {self.config['env']['opponent_type']}")

        # Initialize the reward shaper based on config
        if self.config["reward_shaping"]["enabled"]:
            reward_shaper = DomainKnowledgeShaper(self.config["reward_shaping"])
            print("[Trainer] Reward Shaping: ENABLED")
        else:
            reward_shaper = NoOpShaper()
            print("[Trainer] Reward Shaping: DISABLED (Sparse Rewards)")

        # Initialize the PPO agent
        # These dimensions are specific to Chef's Hat
        state_dim = 28  # 17 hand + 11 board
        action_dim = 200
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            p_hyperparameters=self.config["ppo"],
            reward_shaper=reward_shaper
        )

        # --- 2. Run Training Matches ---
        total_matches = self.config["training"]["total_matches"]
        rollout_steps = self.config["ppo"]["rollout_steps"]
        
        progress_bar = tqdm(range(total_matches), desc="Training Matches")
        
        for match_num in progress_bar:
            # Setup the room for a new match
            room = Room(room_name="PPO Training Room", num_players=4, verbose=False)
            room.add_player(agent, "PPO_Agent")
            room.add_player(opponent_class("Opponent_1"), "Opponent_1")
            room.add_player(opponent_class("Opponent_2"), "Opponent_2")
            room.add_player(opponent_class("Opponent_3"), "Opponent_3")

            # Run the match and get results
            match_results = room.run_game()
            
            # --- 3. Log Metrics ---
            final_scores = match_results["final_scores"]
            ppo_score = final_scores.get("PPO_Agent", 0)
            ppo_position = match_results["finishing_order"].index("PPO_Agent") + 1
            self.metrics.log_match(match_num, ppo_score, ppo_position)

            # Update progress bar
            if match_num % self.config["training"]["log_interval"] == 0:
                avg_score, win_rate = self.metrics.get_summary()
                progress_bar.set_postfix({"Win Rate": f"{win_rate:.2%}", "Avg Score": f"{avg_score:.2f}"})

            # --- 4. Trigger Learning Step ---
            if len(agent.p_buffer) >= rollout_steps:
                agent.learn()

            # --- 5. Save Model Checkpoints ---
            if match_num > 0 and match_num % self.config["training"]["save_interval"] == 0:
                agent.save_models(os.path.join(self.output_dir, "models"), f"match_{match_num}")

        # --- 6. Finalize Training ---
        agent.save_models(os.path.join(self.output_dir, "models"), "final")
        self.metrics.save_log()
        print("[Trainer] Training finished.")
        avg_score, win_rate = self.metrics.get_summary()
        print(f"[Trainer] Final Results - Win Rate: {win_rate:.2%}, Average Score: {avg_score:.2f}")
