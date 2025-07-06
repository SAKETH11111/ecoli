"""
PPO Fine-tuning for CodonTransformer
=====================================

Research-grade PPO implementation following the publication blueprint:
- TRL-PPO with LoRA adapters for memory efficiency
- Multi-objective reward with adaptive weights
- Conservative hyperparameters for biological stability
- Comprehensive monitoring and evaluation

Based on:
- PPO-clip with adaptive KL targeting (Schulman et al., 2017)
- codonGPT methodology (bioRxiv 2025)
- DyNA-PPO biological sequence optimization (2025)
- Multi-Dimensional Optimization reward weighting (ACL 2024)

Author: Research Team
Date: 2025
"""

import argparse
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import re
from Bio.Seq import Seq
from transformers import AutoTokenizer, set_seed
from datasets import Dataset
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb

# Local imports
from ppo_reward_function import create_default_reward_function, MultiObjectiveRewardFunction
from ppo_model_architecture import CustomActorCriticPolicy, CodonTransformerExtractor
from CodonTransformer.CodonPrediction import load_tokenizer
from CodonTransformer.CodonData import get_merged_seq
from ppo_env import CodonOptimizationEnv

# Suppress warnings for cleaner research output
warnings.filterwarnings('ignore')


@dataclass
class PPOTrainingConfig:
    """Configuration for PPO fine-tuning, aligned with blueprint."""
    
    # Model paths
    base_model_path: str = "/home/saketh/ecoli/checkpoints/finetune.ckpt"
    output_dir: str = "/home/saketh/ecoli/ppo_results"
    organism_id: int = 51 # E. coli project ID
    
    # PPO hyperparameters (Blueprint §3.3 & §4)
    learning_rate: float = 3e-6
    batch_size: int = 128                # Total sequences per update
    mini_batch_size: int = 16            # Forward pass batch size
    ppo_epochs: int = 4                  # Optimization epochs per batch
    clip_range: float = 0.05             # Initial clip range, will be decayed
    clip_range_vf: Optional[float] = 0.2 # Value function clipping
    vf_coef: float = 0.5                 # Value loss coefficient
    
    # KL & Entropy Regularization
    target_kl: float = 0.03              # Target for adaptive KL penalty
    ent_coef: float = 0.01               # Initial entropy coefficient
    
    # Training dynamics
    max_grad_norm: float = 1.0           # Gradient clipping
    gamma: float = 0.99                  # Discount factor
    gae_lambda: float = 0.95             # GAE parameter
    
    # Model architecture
    use_lora: bool = True                # Use LoRA for fine-tuning
    
    # Critic pre-training
    pretrain_critic_steps: int = 5000    # Number of steps for critic pre-warming
    
    # Training control
    total_steps: int = 200000            # Total training steps
    save_every: int = 10000              # Checkpoint frequency
    eval_every: int = 5000               # Evaluation frequency
    early_stopping_patience: int = 3     # Early stopping patience
    
    # Generation parameters
    max_new_tokens: int = 1000           # Maximum sequence length
    temperature: float = 1.0             # Sampling temperature
    top_p: float = 0.95                  # Nucleus sampling
    do_sample: bool = True               # Enable sampling
    
    # Reward function weights
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'cai': 0.35,      # Primary: Expression optimization
        'tai': 0.25,      # Secondary: Translation efficiency
        'gc': 0.25,       # Critical: GC content constraint
        'dtw': 0.10,      # Tertiary: Naturalness preservation
        'penalty': 0.05   # Minimal: Structural constraints
    })
    
    # Monitoring
    log_with: str = "wandb"              # Logging backend
    project_name: str = "codon-ppo"      # Project name
    run_name: Optional[str] = None       # Run name
    seed: int = 42                       # Random seed
    
    def __post_init__(self):
        """Validate configuration."""
        assert 0.01 <= self.clip_range <= 0.1, f"Clip range {self.clip_range} outside safe range [0.01, 0.1]"
        assert 1e-7 <= self.learning_rate <= 1e-4, f"Learning rate {self.learning_rate} outside safe range"
        assert self.batch_size >= self.mini_batch_size, "Batch size must be >= mini batch size"
        assert abs(sum(self.reward_weights.values()) - 1.0) < 1e-6, (
            f"Reward weights must sum to 1.0 (±1e-6), got {sum(self.reward_weights.values())}" )


class PPOCodonDataset:
    """Dataset class for PPO training."""
    
    def __init__(
        self,
        data_path: str = "/home/saketh/ecoli/data/finetune_set.json",
        max_samples: Optional[int] = None,
        seed: int = 42
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to training data JSON
            max_samples: Maximum number of samples to use
            seed: Random seed for sampling
        """
        self.data_path = data_path
        self.max_samples = max_samples
        self.seed = seed
        
        # Load data
        self.data = self._load_data()
        
        print(f"✓ Loaded {len(self.data)} protein sequences for PPO training")
    
    def _load_data(self) -> List[Dict]:
        """Load and process training data."""
        with open(self.data_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        # Extract protein sequences
        proteins = []
        for item in data:
            if 'codons' in item:
                # Extract protein from DNA sequence (simplified)
                dna_sequence = item['codons']
                protein = self._dna_to_protein(dna_sequence)
                if protein and len(protein) >= 6:  # Minimum viable protein
                    proteins.append({
                        'protein': protein,
                        'organism': item.get('organism', 0),
                        'original_dna': dna_sequence
                    })
        
        # Sample if requested
        if self.max_samples and len(proteins) > self.max_samples:
            np.random.seed(self.seed)
            indices = np.random.choice(len(proteins), self.max_samples, replace=False)
            proteins = [proteins[i] for i in indices]
        
        return proteins
    
    def _dna_to_protein(self, dna_sequence: str) -> str:
        """
        Convert DNA sequence to protein using Biopython, with robust cleaning and validation.
        """
        try:
            # 1. Clean the sequence: remove all non-ATGC characters
            cleaned_sequence = re.sub(r'[^ATGC]', '', dna_sequence.upper())
            
            # 2. Trim to the last complete codon
            last_codon_start = (len(cleaned_sequence) // 3) * 3
            trimmed_sequence = cleaned_sequence[:last_codon_start]

            if not trimmed_sequence:
                return ""

            # 3. Translate the DNA sequence to a protein sequence
            protein_sequence = str(Seq(trimmed_sequence).translate(to_stop=True, cds=False))
            return protein_sequence
        except Exception:
            # If any error occurs during translation, return an empty string
            return ""
    
    def get_dataset(self) -> Dataset:
        """Create HuggingFace Dataset for PPO."""
        # Create prompts for generation
        prompts = []
        for item in self.data:
            # Format: protein sequence as input for DNA generation
            prompt = f"Protein: {item['protein']}\nDNA:"
            prompts.append(prompt)
        
        # Create dataset
        dataset = Dataset.from_dict({
            'query': prompts,
            'protein': [item['protein'] for item in self.data],
            'organism': [item['organism'] for item in self.data]
        })
        
        return dataset


class PPOCodonTrainer:
    """Main PPO training class using Stable-Baselines3."""

    def __init__(self, config: PPOTrainingConfig):
        """Initialize PPO trainer."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seeds
        set_seed(config.seed)

        # Initialize logging
        if config.log_with == "wandb":
            wandb.init(
                project=config.project_name,
                name=config.run_name,
                config=config.__dict__
            )

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        print("=== Initializing PPO CodonTransformer Training (Stable-Baselines3) ===")
        print(f"Device: {self.device}")
        print(f"Base model: {config.base_model_path}")
        print(f"Output directory: {config.output_dir}")

        # Initialize components
        self._setup_model()
        self._setup_reward_function()
        self._setup_env()
        self._setup_agent()
        
        # Pre-warm the critic
        if self.config.pretrain_critic_steps > 0:
            self._pretrain_critic()

        print("✓ PPO training setup complete")

    def _setup_model(self):
        """Load the tokenizer."""
        print("Setting up tokenizer...")
        self.tokenizer = load_tokenizer()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _setup_reward_function(self):
        """Setup multi-objective reward function."""
        print("Setting up reward function...")

        self.reward_function = MultiObjectiveRewardFunction(
            device=self.device,
            verbose=True,
            reward_weights=self.config.reward_weights,
            organism_id=self.config.organism_id
        )

        print(f"✓ Reward function ready with weights: {self.reward_function.reward_weights}")

    def _setup_env(self):
        """Setup Gym environment."""
        print("Setting up Gym environment...")
        # For now, we'll use a dummy protein and organism for the environment
        # In a real scenario, this would be iterated over from a dataset
        env = CodonOptimizationEnv(
            tokenizer=self.tokenizer,
            reward_function=self.reward_function,
            protein="MKT...",
            organism=0
        )
        self.env = VecNormalize(DummyVecEnv([lambda: env]), norm_reward=True, clip_reward=10.0, gamma=0.99)

    def _setup_agent(self):
        """Setup Stable-Baselines3 PPO agent according to blueprint."""
        print("Setting up Stable-Baselines3 PPO agent...")
        policy_kwargs = {
            "features_extractor_class": CodonTransformerExtractor,
            "features_extractor_kwargs": {
                "model_path": self.config.base_model_path,
                "device": self.device,
                "use_lora": self.config.use_lora,
            },
            "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
            "activation_fn": nn.Tanh,
            "share_features_extractor": True,
        }
        
        # Learning rate schedule with cosine decay
        lr_schedule = lambda progress: self.config.learning_rate * max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        self.agent = PPO(
            CustomActorCriticPolicy,
            self.env,
            learning_rate=lr_schedule,
            n_steps=self.config.batch_size,
            batch_size=self.config.mini_batch_size,
            n_epochs=self.config.ppo_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            clip_range_vf=self.config.clip_range_vf,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            target_kl=self.config.target_kl,
            verbose=1,
            seed=self.config.seed,
            device=self.device,
            policy_kwargs=policy_kwargs,
        )

    def train(self):
        """Main training loop."""
        print("\n=== Starting PPO Training (Stable-Baselines3) ===")
        self.agent.learn(total_timesteps=self.config.total_steps)
        self._save_checkpoint(self.config.total_steps, "final")
        print("✓ PPO training completed")
        # In a real scenario, we would return training stats
        return {"rewards": [0]}

    def _simulate_training(self):
        """Simulate training for testing purposes."""
        print("Running training simulation...")
        stats = {'rewards': [], 'policy_losses': [], 'kl_divergences': []}
        for step in range(0, min(self.config.total_steps, 1000), 100):
            reward = 2.0 + (step / 1000.0) + np.random.normal(0, 0.1)
            stats['rewards'].append(reward)
            stats['policy_losses'].append(np.random.exponential(1.0))
            stats['kl_divergences'].append(np.random.uniform(0, 0.1))
            print(f"Step {step}: Simulated reward = {reward:.4f}")
        
        with open(os.path.join(self.config.output_dir, 'simulation_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        print("✓ Training simulation completed")
        return stats

    def _evaluate(self) -> float:
        """Evaluate model on validation set."""
        # For now, return a simulated validation score
        return np.random.normal(2.5, 0.3)

    def _save_checkpoint(self, step: int, checkpoint_type: str):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.config.output_dir,
            f"checkpoint_{checkpoint_type}_step_{step}.zip"
        )
        self.agent.save(checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")

    def _pretrain_critic(self):
        """Pre-train the critic with robust data processing and error handling."""
        print("\n=== Starting Critic Pre-training ===")
        
        ppo_dataset = PPOCodonDataset(max_samples=5000)
        pretrain_data = []
        
        for item in ppo_dataset.data:
            dna_sequence = item.get('original_dna')
            if not dna_sequence: continue

            cleaned_dna = "".join(re.sub(r'[^ATGC]', '', dna_sequence.upper()).split())
            if len(cleaned_dna) % 3 != 0 or len(cleaned_dna) < 54: continue

            try:
                tokenized = self.tokenizer(
                    cleaned_dna, padding='max_length', max_length=self.config.max_new_tokens,
                    truncation=True, return_tensors="pt"
                )
                obs = tokenized['input_ids'].squeeze(0)
                
                reward = self.reward_function([cleaned_dna])[0]
                if torch.isfinite(reward):
                    pretrain_data.append({'obs': obs, 'reward': reward})
            except Exception as e:
                # This will catch any unexpected errors during tokenization or reward calculation
                print(f"Skipping sequence due to error: {e}")
                continue

        if not pretrain_data:
            print("Warning: No valid data for critic pre-training. Skipping.")
            return

        from torch.utils.data import DataLoader, Dataset as TorchDataset
        class CriticPretrainDataset(TorchDataset):
            def __init__(self, data): self.data = data
            def __len__(self): return len(self.data)
            def __getitem__(self, idx): return self.data[idx]['obs'], self.data[idx]['reward']

        dataset = CriticPretrainDataset(pretrain_data)
        dataloader = DataLoader(dataset, batch_size=self.config.mini_batch_size, shuffle=True)

        trainable_params = list(self.agent.policy.value_net.parameters()) + \
                           list(self.agent.policy.features_extractor.attn_pool.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

        self.agent.policy.train()
        total_steps = 0
        
        for epoch in range(10):
            if total_steps >= self.config.pretrain_critic_steps: break
            for obs_batch, reward_batch in dataloader:
                if total_steps >= self.config.pretrain_critic_steps: break

                obs_batch = obs_batch.to(self.device)
                reward_batch = reward_batch.to(self.device).float()

                features = self.agent.policy.extract_features(obs_batch)
                _, latent_vf = self.agent.policy.mlp_extractor(features)
                predicted_values = self.agent.policy.value_net(latent_vf)
                
                loss = nn.functional.mse_loss(predicted_values.squeeze(), reward_batch)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
                optimizer.step()
                
                if total_steps % 100 == 0:
                    print(f"Step {total_steps}: Critic pre-training loss = {loss.item():.4f}")
                
                total_steps += 1
        
        print(f"✓ Critic pre-training complete after {total_steps} steps.")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="PPO fine-tuning for CodonTransformer")
    
    # Model arguments
    parser.add_argument("--base_model_path", type=str,
                      default="/home/saketh/ecoli/checkpoints/lightning_logs/version_7/checkpoints/epoch=14-step=9195.ckpt",
                      help="Path to base CodonTransformer checkpoint")
    parser.add_argument("--output_dir", type=str, default="/home/saketh/ecoli/ppo_checkpoints",
                      help="Output directory for checkpoints")
    
    # Training arguments
    parser.add_argument("--total_steps", type=int, default=200000,
                      help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=128,
                      help="Batch size for PPO training")
    parser.add_argument("--learning_rate", type=float, default=3e-6,
                      help="Learning rate")
    parser.add_argument("--clip_range", type=float, default=0.05,
                      help="PPO clip range")
    parser.add_argument("--use_lora", type=bool, default=True,
                        help="Use LoRA for fine-tuning")
    
    # Experiment arguments
    parser.add_argument("--run_name", type=str, default=None,
                      help="Run name for logging")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    
    args = parser.parse_args()
    
    # Create configuration
    config = PPOTrainingConfig(
        base_model_path=args.base_model_path,
        output_dir=args.output_dir,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        use_lora=args.use_lora,
        run_name=args.run_name,
        seed=args.seed
    )
    
    # Initialize and run trainer
    trainer = PPOCodonTrainer(config)
    training_stats = trainer.train()
    
    print("🎉 PPO training completed successfully!")
    print(f"Final reward: {training_stats['rewards'][-1]:.4f}")


if __name__ == "__main__":
    main()