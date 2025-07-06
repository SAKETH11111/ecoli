"""
PPO CodonTransformer Main Entry Point
======================================

Complete integration script for PPO enhancement of CodonTransformer following
the publication-grade blueprint for state-of-the-art codon optimization.

This script orchestrates the entire PPO pipeline:
1. Setup and validation
2. Hyperparameter optimization (optional)
3. Full PPO training
4. Comprehensive evaluation
5. Model deployment

Based on the 7-day sprint methodology from the research blueprint.

Author: Research Team
Date: 2025
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import wandb

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Local imports
from ppo_finetune import PPOTrainingConfig, PPOCodonTrainer
from ppo_hyperparameter_sweep import run_hyperparameter_sweep

# Suppress warnings for cleaner research output
warnings.filterwarnings('ignore')


class PPOCodonPipeline:
    """
    Complete PPO enhancement pipeline for CodonTransformer.
    
    Implements the full research workflow from setup to deployment.
    """
    
    def __init__(
        self,
        base_model_path: str,
        output_dir: str = "/home/saketh/ecoli/ppo_production",
        device: Optional[torch.device] = None,
        verbose: bool = True
    ):
        """
        Initialize PPO pipeline.
        
        Args:
            base_model_path: Path to base CodonTransformer checkpoint
            output_dir: Output directory for all pipeline results
            device: Computation device (auto-detected if None)
            verbose: Enable detailed logging
        """
        self.base_model_path = base_model_path
        self.output_dir = Path(output_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        # Create directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sweep_dir = self.output_dir / "hyperparameter_sweep"
        self.training_dir = self.output_dir / "training"
        self.evaluation_dir = self.output_dir / "evaluation"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.reports_dir = self.output_dir / "reports"
        
        for dir_path in [self.sweep_dir, self.training_dir, self.evaluation_dir, 
                        self.checkpoints_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)
        
        
        if verbose:
            print("🧬 PPO CodonTransformer Pipeline Initialized")
            print(f"  Base model: {base_model_path}")
            print(f"  Output directory: {output_dir}")
            print(f"  Device: {self.device}")
            print(f"  GPU available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  GPU count: {torch.cuda.device_count()}")
                print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def validate_setup(self) -> bool:
        """
        Validate pipeline setup and dependencies.
        
        Returns:
            True if setup is valid, False otherwise
        """
        if self.verbose:
            print("\n=== Validating Pipeline Setup ===")
        
        validation_results = {
            'base_model_exists': False,
            'dependencies_available': False,
            'data_available': False,
            'gpu_available': torch.cuda.is_available()
        }
        
        # Check base model
        if os.path.exists(self.base_model_path):
            validation_results['base_model_exists'] = True
            if self.verbose:
                print("✓ Base model found")
        else:
            if self.verbose:
                print(f"✗ Base model not found: {self.base_model_path}")
        
        # Check dependencies
        try:
            import stable_baselines3
            import gym
            import peft
            import optuna
            validation_results['dependencies_available'] = True
            if self.verbose:
                print("✓ PPO dependencies available")
        except ImportError as e:
            if self.verbose:
                print(f"✗ Missing dependencies: {e}")
        
        # Check data availability
        data_paths = [
            "/home/saketh/ecoli/data/test_set.json",
            "/home/saketh/ecoli/data/finetune_set.json",
            "/home/saketh/ecoli/data/ecoli_processed_genes.csv"
        ]
        
        if all(os.path.exists(path) for path in data_paths):
            validation_results['data_available'] = True
            if self.verbose:
                print("✓ Training and evaluation data available")
        else:
            if self.verbose:
                print("✗ Some data files missing")
        
        # Overall validation
        all_valid = all(validation_results.values())
        
        if self.verbose:
            if all_valid:
                print("✅ Pipeline setup validation successful")
            else:
                print("❌ Pipeline setup validation failed")
                print("  Issues found:")
                for check, status in validation_results.items():
                    if not status:
                        print(f"    - {check}")
        
        return all_valid
    
    def run_hyperparameter_optimization(
        self,
        n_trials: int = 20,
        timeout_hours: Optional[int] = None
    ) -> Dict:
        """
        Run hyperparameter optimization sweep.
        
        Args:
            n_trials: Number of optimization trials
            timeout_hours: Maximum time in hours (None for unlimited)
            
        Returns:
            Optimization results
        """
        if self.verbose:
            print(f"\n=== Phase 1: Hyperparameter Optimization ===")
            print(f"Trials: {n_trials}")
            if timeout_hours:
                print(f"Timeout: {timeout_hours} hours")
        
        # Run sweep
        timeout_seconds = timeout_hours * 3600 if timeout_hours else None
        
        sweep_results = run_hyperparameter_sweep(
            base_model_path=self.base_model_path,
            n_trials=n_trials,
            output_dir=str(self.sweep_dir)
        )
        
        # Save results
        results_path = self.sweep_dir / "final_results.json"
        with open(results_path, 'w') as f:
            json.dump(sweep_results, f, indent=2)
        
        if self.verbose:
            print("✅ Hyperparameter optimization completed")
            print(f"  Best trial: {sweep_results['best_trial_number']}")
            print(f"  Best objective: {sweep_results['best_objective']:.4f}")
        
        return sweep_results
    
    def run_full_training(
        self,
        config: Optional[PPOTrainingConfig] = None,
        use_wandb: bool = True,
        run_name: Optional[str] = None
    ) -> Dict:
        """
        Run full PPO training with best hyperparameters.
        
        Args:
            config: Training configuration (auto-created if None)
            use_wandb: Enable Weights & Biases logging
            run_name: Custom run name
            
        Returns:
            Training results
        """
        if self.verbose:
            print(f"\n=== Phase 2: Full PPO Training ===")
        
        # Create or use provided configuration
        if config is None:
            # Try to load best hyperparameters from sweep
            best_params_path = self.sweep_dir / "study_results" / "best_parameters.json"
            if best_params_path.exists():
                with open(best_params_path, 'r') as f:
                    best_params = json.load(f)
                
                if self.verbose:
                    print("Using optimized hyperparameters from sweep")
                
                # Create config with best parameters
                config = self._create_config_from_sweep(best_params)
            else:
                if self.verbose:
                    print("Using default configuration (no sweep results found)")
                
                config = PPOTrainingConfig(
                    base_model_path=self.base_model_path,
                    output_dir=str(self.training_dir),
                    total_steps=200000,
                    log_with="wandb" if use_wandb else None,
                    run_name=run_name or f"ppo_codon_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    # Default Stable-Baselines3 parameters
                    learning_rate=3e-6,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    max_grad_norm=0.5
                )
        
        # Update paths
        config.output_dir = str(self.training_dir)
        if run_name:
            config.run_name = run_name
        
        if self.verbose:
            print(f"  Total steps: {config.total_steps:,}")
            print(f"  SB3 n_steps: {config.n_steps}")
            print(f"  SB3 batch_size: {config.batch_size}")
            print(f"  SB3 n_epochs: {config.n_epochs}")
            print(f"  Learning rate: {config.learning_rate:.2e}")
            print(f"  Clip range: {config.clip_range}")
        
        # Initialize trainer
        trainer = PPOCodonTrainer(config)
        
        # Run training
        training_stats = trainer.train()
        
        # Save training results
        results_path = self.training_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(training_stats, f, indent=2)
        
        if self.verbose:
            print("✅ PPO training completed")
            if training_stats.get('rewards'):
                print(f"  Final reward: {training_stats['rewards'][-1]:.4f}")
                print(f"  Training steps: {len(training_stats['rewards'])}")
        
        return training_stats
    
    
    def generate_final_report(
        self,
        sweep_results: Optional[Dict] = None,
        training_results: Optional[Dict] = None
    ) -> str:
        """
        Generate comprehensive final report.
        
        Args:
            sweep_results: Hyperparameter optimization results
            training_results: Training results
            evaluation_results: Evaluation results
            
        Returns:
            Report content as string
        """
        if self.verbose:
            print(f"\n=== Generating Final Report ===")
        
        report_lines = [
            "# PPO CodonTransformer Enhancement - Final Report",
            "=" * 60,
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Base model: {self.base_model_path}",
            f"Output directory: {self.output_dir}",
            f"Device: {self.device}",
            "",
            "## Executive Summary",
            "",
            "This report summarizes the PPO enhancement of CodonTransformer for",
            "state-of-the-art codon optimization following the publication-grade",
            "research blueprint.",
            ""
        ]
        
        # Hyperparameter optimization summary
        if sweep_results:
            report_lines.extend([
                "## Hyperparameter Optimization",
                "",
                f"- Total trials: {sweep_results.get('total_trials', 'N/A')}",
                f"- Best trial: {sweep_results.get('best_trial_number', 'N/A')}",
                f"- Best objective: {sweep_results.get('best_objective', 'N/A'):.4f}",
                "",
                "### Optimized Parameters:",
            ])
            
            best_params = sweep_results.get('best_parameters', {})
            for param, value in best_params.items():
                if isinstance(value, float):
                    report_lines.append(f"- {param}: {value:.6f}")
                else:
                    report_lines.append(f"- {param}: {value}")
            
            report_lines.append("")
        
        # Training summary
        if training_results:
            report_lines.extend([
                "## Training Results",
                "",
                f"- Training completed successfully",
                f"- Total training steps: {len(training_results.get('rewards', []))}",
            ])
            
            if training_results.get('rewards'):
                final_reward = training_results['rewards'][-1]
                initial_reward = training_results['rewards'][0]
                improvement = final_reward - initial_reward
                
                report_lines.extend([
                    f"- Initial reward: {initial_reward:.4f}",
                    f"- Final reward: {final_reward:.4f}",
                    f"- Total improvement: {improvement:.4f}",
                ])
            
            report_lines.append("")
        
        
        # Conclusions and next steps
        report_lines.extend([
            "## Conclusions",
            "",
            "The PPO enhancement pipeline has been successfully implemented",
            "following the research-grade blueprint. Key achievements:",
            "",
            "✅ Multi-objective reward function with normalized components",
            "✅ LoRA-enhanced PPO architecture for memory efficiency", 
            "✅ Conservative hyperparameters for biological stability",
            "✅ Production-ready checkpointing and model management",
            "",
            "## Next Steps",
            "",
            "1. **Wet-lab Validation**: Test top sequences experimentally",
            "2. **Publication Preparation**: Compile results for submission",
            "3. **Model Deployment**: Deploy best model for production use",
            "4. **Community Release**: Open-source the enhanced system",
            "",
            "## Technical Specifications",
            "",
            f"- Framework: PyTorch + Stable-Baselines3 + PEFT",
            f"- Model architecture: BigBird + LoRA adapters",
            f"- Training approach: PPO with clipped objective",
            f"- Evaluation metrics: CAI, tAI, GC content, DTW distance",
            f"- Target organism: E. coli",
            "",
            "---",
            "*Report generated by PPO CodonTransformer Pipeline*"
        ])
        
        # Save report
        report_content = "\n".join(report_lines)
        report_path = self.reports_dir / "final_report.md"
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        if self.verbose:
            print(f"✅ Final report generated: {report_path}")
        
        return report_content
    
    def _create_config_from_sweep(self, best_params: Dict) -> PPOTrainingConfig:
        """Create training config from sweep results."""
        params = best_params.get('best_parameters', {})
        
        # Extract reward weights if available
        reward_weights = {}
        if 'cai_weight' in params:
            reward_weights = {
                'cai': params.get('cai_weight', 0.35),
                'tai': params.get('tai_weight', 0.25),
                'gc': params.get('gc_weight', 0.25),
                'dtw': params.get('dtw_weight', 0.10),
                'penalty': params.get('penalty_weight', 0.05)
            }
        else:
            reward_weights = {
                'cai': 0.35,
                'tai': 0.25,
                'gc': 0.25,
                'dtw': 0.10,
                'penalty': 0.05
            }
        
        return PPOTrainingConfig(
            base_model_path=self.base_model_path,
            output_dir=str(self.training_dir),
            learning_rate=params.get('learning_rate', 3e-6),
            n_steps=params.get('n_steps', 2048),
            batch_size=params.get('batch_size', 64),
            n_epochs=params.get('n_epochs', 10),
            gamma=params.get('gamma', 0.99),
            gae_lambda=params.get('gae_lambda', 0.95),
            clip_range=params.get('clip_range', 0.2),
            ent_coef=params.get('ent_coef', 0.01),
            vf_coef=params.get('vf_coef', 0.5),
            max_grad_norm=params.get('max_grad_norm', 0.5),
            lora_rank=params.get('lora_rank', 16),
            lora_alpha=params.get('lora_alpha', 32),
            lora_dropout=params.get('lora_dropout', 0.1),
            reward_weights=reward_weights,
            total_steps=200000,  # Full training
            log_with="wandb",
            run_name=f"ppo_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )


def main():
    """Main entry point for PPO CodonTransformer pipeline."""
    parser = argparse.ArgumentParser(
        description="PPO Enhancement Pipeline for CodonTransformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with hyperparameter optimization
  python ppo_main.py --base-model /path/to/model.ckpt --full-pipeline
  
  # Training only with default parameters
  python ppo_main.py --base-model /path/to/model.ckpt --train-only
  
  # Hyperparameter sweep only
  python ppo_main.py --base-model /path/to/model.ckpt --sweep-only --n-trials 50
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--base-model", 
        type=str, 
        required=True,
        help="Path to base CodonTransformer checkpoint"
    )
    
    # Pipeline control
    parser.add_argument(
        "--full-pipeline", 
        action="store_true",
        help="Run complete pipeline (sweep + training)"
    )
    parser.add_argument(
        "--sweep-only", 
        action="store_true",
        help="Run hyperparameter optimization only"
    )
    parser.add_argument(
        "--train-only", 
        action="store_true",
        help="Run training only (skip hyperparameter optimization)"
    )
    
    # Configuration
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="/home/saketh/ecoli/ppo_production",
        help="Output directory for all results"
    )
    parser.add_argument(
        "--n-trials", 
        type=int, 
        default=20,
        help="Number of hyperparameter optimization trials"
    )
    parser.add_argument(
        "--timeout-hours", 
        type=int,
        help="Maximum time for hyperparameter optimization (hours)"
    )
    parser.add_argument(
        "--run-name", 
        type=str,
        help="Custom run name for tracking"
    )
    parser.add_argument(
        "--no-wandb", 
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--device", 
        type=str,
        help="Computation device (cuda/cpu, auto-detected if not specified)"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Minimize output (quiet mode)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.full_pipeline, args.sweep_only, args.train_only]):
        parser.error("Must specify one of: --full-pipeline, --sweep-only, --train-only")
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize pipeline
    pipeline = PPOCodonPipeline(
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        device=device,
        verbose=not args.quiet
    )
    
    # Validate setup
    if not pipeline.validate_setup():
        print("❌ Pipeline validation failed. Please check setup and try again.")
        return 1
    
    # Run requested operations
    sweep_results = None
    training_results = None
    
    try:
        if args.full_pipeline or args.sweep_only:
            # Hyperparameter optimization
            sweep_results = pipeline.run_hyperparameter_optimization(
                n_trials=args.n_trials,
                timeout_hours=args.timeout_hours
            )
        
        if args.full_pipeline or args.train_only:
            # Training
            training_results = pipeline.run_full_training(
                use_wandb=not args.no_wandb,
                run_name=args.run_name
            )
        
        # Generate final report
        if args.full_pipeline:
            pipeline.generate_final_report(
                sweep_results=sweep_results,
                training_results=training_results
            )
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📁 Results available in: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())