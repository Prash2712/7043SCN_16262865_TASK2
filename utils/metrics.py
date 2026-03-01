_# -*- coding: utf-8 -*-
"""
Metrics logging utilities for training and evaluation.

This module provides classes to handle the logging of performance metrics
to CSV files in a structured format.
"""

import os
from typing import List, Tuple

import pandas as pd


class TrainingMetrics:
    """Logs and manages metrics during the training process."""

    def __init__(self, log_dir: str):
        """Initializes the TrainingMetrics logger.

        Args:
            log_dir: The directory to save the log file in.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "training_log.csv")
        self.metrics = []

    def log_match(self, match_num: int, score: float, position: int):
        """Logs the results of a single training match.

        Args:
            match_num: The number of the match.
            score: The final score of the PPO agent.
            position: The finishing position of the PPO agent (1-4).
        """
        self.metrics.append({"Match": match_num, "Score": score, "Position": position})

    def get_summary(self) -> Tuple[float, float]:
        """Calculates and returns a summary of the performance so far.

        Returns:
            A tuple containing the average score and the win rate.
        """
        if not self.metrics:
            return 0.0, 0.0
        df = pd.DataFrame(self.metrics)
        avg_score = df["Score"].mean()
        win_rate = (df["Position"] == 1).mean()
        return avg_score, win_rate

    def save_log(self):
        """Saves all collected metrics to a CSV file."""
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.log_file, index=False)
        print(f"[Metrics] Training log saved to {self.log_file}")


class EvaluationMetrics(TrainingMetrics):
    """Logs and manages metrics during the evaluation process.

    Inherits from TrainingMetrics but saves to a different default filename.
    """

    def __init__(self, log_dir: str):
        """Initializes the EvaluationMetrics logger.

        Args:
            log_dir: The directory to save the log file in.
        """
        super().__init__(log_dir)
        self.log_file = os.path.join(self.log_dir, "evaluation_log.csv")

    def save_log(self):
        """Saves all collected metrics to the evaluation CSV file."""
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.log_file, index=False)
        print(f"[Metrics] Evaluation log saved to {self.log_file}")
