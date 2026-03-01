import abc
from typing import Dict, Any, List

import numpy as np


class BaseRewardShaper(abc.ABC):
    """Abstract base class for all reward shaping strategies."""

    @abc.abstractmethod
    def shape(self, observation: Dict[str, Any], reward: float, done: bool, info: Dict[str, Any]) -> float:
        """Shapes the reward based on the given observation and game state.

        Args:
            observation: The environment observation dictionary.
            reward: The original reward from the environment.
            done: A flag indicating if the episode has terminated.
            info: A dictionary containing auxiliary diagnostic information.

        Returns:
            The shaped reward.
        """
        pass


class DomainKnowledgeShaper(BaseRewardShaper):
    """Implements reward shaping based on pre-defined game-specific domain knowledge.

    This shaper applies a series of bonuses and penalties based on the agent's
    actions and hand composition, designed to guide the learning process more
    effectively than a sparse terminal reward alone.

    Attributes:
        shaping_coefficients (Dict[str, float]): A dictionary holding the coefficients
            for each shaping strategy, loaded from the project configuration.
    """

    def __init__(self, shaping_coefficients: Dict[str, float]):
        """Initializes the DomainKnowledgeShaper.

        Args:
            shaping_coefficients: A dictionary with keys for each shaping strategy
                (e.g., 'high_card_penalty_coef') and their corresponding float values.
        """
        self.shaping_coefficients = shaping_coefficients
        print(f"[DomainKnowledgeShaper] Initialized with coefficients: {self.shaping_coefficients}")

    def shape(self, observation: Dict[str, Any], reward: float, done: bool, info: Dict[str, Any]) -> float:
        """Calculates and applies the domain knowledge-based reward adjustments.

        Args:
            observation: The environment observation, expected to contain 'player_hand'.
            reward: The original reward from the environment.
            done: A flag indicating if the episode has terminated.
            info: A dictionary containing auxiliary diagnostic information.

        Returns:
            The adjusted reward after applying all shaping rules.
        """
        shaped_reward = reward

        # --- 1. High Card Penalty (applied at the end of a round) ---
        # This penalty discourages holding onto high-value cards unnecessarily.
        if done:
            penalty_coef = self.shaping_coefficients.get("high_card_penalty_coef", 0.0)
            if penalty_coef != 0:
                player_hand = observation.get("player_hand", [])
                high_cards = [card for card in player_hand if card[0] > 10] # Card value is the first element
                penalty = penalty_coef * sum([card[0] for card in high_cards])
                shaped_reward += penalty

        # --- 2. Card Play Bonus (applied during the game) ---
        # This bonus encourages playing high-value cards, which is often a good move.
        bonus_coef = self.shaping_coefficients.get("card_play_bonus_coef", 0.0)
        if bonus_coef != 0 and "action_info" in info:
            action = info["action_info"].get("action")
            if action and action[0] == 1: # Action type 1 is 'play card'
                card_played_value = action[1]
                if card_played_value > 8:
                    bonus = bonus_coef * card_played_value
                    shaped_reward += bonus

        # --- 3. Hand Strength Bonus (applied during the game) ---
        # This bonus encourages collecting cards of the same suit, a key strategy.
        strength_coef = self.shaping_coefficients.get("hand_strength_bonus_coef", 0.0)
        if strength_coef != 0:
            player_hand = observation.get("player_hand", [])
            if player_hand:
                suits = [card[1] for card in player_hand] # Suit is the second element
                if suits:
                    max_suit_count = np.max(np.bincount(suits))
                    # Reward having more than 2 cards of the same suit
                    if max_suit_count > 2:
                        strength_bonus = strength_coef * (max_suit_count - 2)
                        shaped_reward += strength_bonus

        return shaped_reward


class NoOpShaper(BaseRewardShaper):
    """A shaper that performs no operation, returning the original reward.

    This is used for baseline comparisons against shaped reward strategies.
    """

    def shape(self, observation: Dict[str, Any], reward: float, done: bool, info: Dict[str, Any]) -> float:
        """Returns the original, unmodified reward.

        Args:
            observation: The environment observation (ignored).
            reward: The original reward from the environment.
            done: A flag indicating if the episode has terminated (ignored).
            info: A dictionary containing auxiliary diagnostic information (ignored).

        Returns:
            The original reward.
        """
        return reward
