import gym
from gym import spaces
import numpy as np
from transformers import PreTrainedTokenizerFast

from ppo_reward_function import MultiObjectiveRewardFunction
from CodonTransformer.CodonEvaluation import get_GC_content


class CodonOptimizationEnv(gym.Env):
    """
    Custom Gym environment for codon optimization, implementing blueprint recommendations.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        reward_function: MultiObjectiveRewardFunction,
        protein: str,
        organism: int,
        max_len: int = 1024,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.reward_function = reward_function
        self.protein = protein
        self.organism = organism
        self.max_len = max_len

        self.action_space = spaces.Discrete(self.tokenizer.vocab_size)
        self.observation_space = spaces.Box(
            low=0, high=self.tokenizer.vocab_size, shape=(self.max_len,), dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_sequence_tokens = [self.tokenizer.cls_token_id]
        self.current_codons = []
        return self._get_obs(), {}

    def _get_rolling_gc_content(self, window_size=50):
        """Calculates the GC content of the last `window_size` codons."""
        if not self.current_codons:
            return 50.0
        
        codons_to_check = "".join(self.current_codons[-window_size:])
        return get_GC_content(codons_to_check)

    def _is_high_cai_codon(self, codon: str) -> bool:
        """Checks if a codon is in the top quintile of CAI values for its amino acid."""
        if codon not in self.reward_function.cai_quintiles:
            return False
        
        quintile_threshold = self.reward_function.cai_quintiles[codon]
        return self.reward_function.cai_weights.get(codon, 0.0) >= quintile_threshold

    def step(self, action):
        # Decode and validate the action
        decoded_action = self.tokenizer.decode([action], skip_special_tokens=True).strip()
        is_valid_codon = len(decoded_action) == 3 and decoded_action in self.reward_function.cai_weights

        if is_valid_codon:
            self.current_sequence_tokens.append(action)
            self.current_codons.append(decoded_action)

        done = len(self.current_codons) >= self.max_len or action == self.tokenizer.sep_token_id
        obs = self._get_obs()
        
        # --- Dense Reward Shaping (Blueprint §2.1) ---
        reward = -0.001  # 1. Time penalty to encourage shorter valid sequences

        if not done and is_valid_codon:
            # 2. GC content bonus for staying in the target range
            gc_content = self._get_rolling_gc_content()
            if self.reward_function.gc_min <= gc_content <= self.reward_function.gc_max:
                reward += 0.01
            
            # 3. High-CAI codon bonus
            if self._is_high_cai_codon(decoded_action):
                reward += 0.005
        
        # --- Terminal Reward ---
        if done:
            final_sequence = "".join(self.current_codons)
            if not final_sequence or len(final_sequence) % 3 != 0:
                reward = -1.0  # Severe penalty for invalid final sequence
            else:
                # Add the composite score from the main reward function
                reward += self.reward_function([final_sequence])[0].item()

        return obs, reward, done, False, {}

    def _get_obs(self):
        """The observation is the current sequence of token IDs, padded to max_len."""
        seq = np.array(self.current_sequence_tokens, dtype=np.int32)
        padded_obs = np.zeros(self.max_len, dtype=np.int32)
        padded_obs[: len(seq)] = seq
        return padded_obs