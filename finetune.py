"""
File: finetune.py
-------------------
Finetune the CodonTransformer model.

The pretrained model is loaded directly from Hugging Face.
The dataset is a JSON file. You can use prepare_training_data from CodonData to
prepare the dataset. The repository README has a guide on how to prepare the
dataset and use this script.
"""

import argparse
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BigBirdForMaskedLM, logging as hf_logging

import torch.nn.functional as F
from CodonTransformer.CodonUtils import (
    C_indices,
    G_indices,
    MAX_LEN,
    TOKEN2MASK,
    IterableJSONData,
)

# Suppress excessive INFO logs from transformers (e.g., BigBird attention fall-backs)
hf_logging.set_verbosity_warning()

class MaskedTokenizerCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        tokenized = self.tokenizer(
            [ex["codons"] for ex in examples],
            return_attention_mask=True,
            return_token_type_ids=True,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        seq_len = tokenized["input_ids"].shape[-1]
        species_index = torch.tensor([[ex["organism"]] for ex in examples])
        tokenized["token_type_ids"] = species_index.repeat(1, seq_len)

        inputs = tokenized["input_ids"]
        targets = tokenized["input_ids"].clone()

        prob_matrix = torch.full(inputs.shape, 0.15)
        prob_matrix[torch.where(inputs < 5)] = 0.0
        selected = torch.bernoulli(prob_matrix).bool()

        # 80% of the time, replace masked input tokens with respective mask tokens
        replaced = torch.bernoulli(torch.full(selected.shape, 0.8)).bool() & selected
        inputs[replaced] = torch.tensor(
            list((map(TOKEN2MASK.__getitem__, inputs[replaced].numpy())))
        ).long()

        # 10% of the time, we replace masked input tokens with random vector.
        randomized = (
            torch.bernoulli(torch.full(selected.shape, 0.1)).bool()
            & selected
            & ~replaced
        )
        random_idx = torch.randint(26, 90, prob_matrix.shape, dtype=torch.long)
        inputs[randomized] = random_idx[randomized]

        tokenized["input_ids"] = inputs
        tokenized["labels"] = torch.where(selected, targets, -100)

        return tokenized


class plTrainHarness(pl.LightningModule):
    def __init__(self, model, learning_rate, warmup_fraction, gc_penalty_weight, tokenizer, 
                 gc_target=0.52, use_lagrangian=False, lagrangian_rho=10.0, curriculum_epochs=3,
                 alm_tolerance=1e-5, alm_dual_tolerance=1e-5, alm_penalty_update_factor=10.0,
                 alm_initial_penalty_factor=20.0, alm_tolerance_update_factor=0.1,
                 alm_rel_penalty_increase_threshold=0.1, alm_max_penalty=1e6, alm_min_penalty=1e-6):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_fraction = warmup_fraction
        self.gc_penalty_weight = gc_penalty_weight
        self.tokenizer = tokenizer
        
        # Augmented-Lagrangian GC Control parameters
        self.gc_target = gc_target
        self.use_lagrangian = use_lagrangian
        self.lagrangian_rho = lagrangian_rho
        self.curriculum_epochs = curriculum_epochs
        
        # Enhanced ALM parameters (inspired by alpaqa research)
        self.alm_tolerance = alm_tolerance
        self.alm_dual_tolerance = alm_dual_tolerance
        self.alm_penalty_update_factor = alm_penalty_update_factor
        self.alm_initial_penalty_factor = alm_initial_penalty_factor
        self.alm_tolerance_update_factor = alm_tolerance_update_factor
        self.alm_rel_penalty_increase_threshold = alm_rel_penalty_increase_threshold
        self.alm_max_penalty = alm_max_penalty
        self.alm_min_penalty = alm_min_penalty
        
        # Initialize Lagrangian multiplier as buffer (persists across checkpoints)
        self.register_buffer("lambda_gc", torch.tensor(0.0))
        
        # Adaptive penalty coefficient (rho) - starts as parameter, becomes adaptive
        self.register_buffer("rho_adaptive", torch.tensor(self.lagrangian_rho))
        
        # Step counter for periodic lambda updates
        self.register_buffer("step_counter", torch.tensor(0))
        
        # ALM convergence tracking
        self.register_buffer("previous_constraint_violation", torch.tensor(float('inf')))
        self.register_buffer("constraint_violation_history", torch.zeros(10))  # Track last 10 values
        self.register_buffer("alm_iteration_counter", torch.tensor(0))
        
        # Configure BigBird to use sparse attention (set once to avoid per-step prints)
        if hasattr(self.model, 'bert') and hasattr(self.model.bert, 'set_attention_type'):
            try:
                self.model.bert.set_attention_type("block_sparse")
            except Exception:
                # Fallback silently if method missing (future-proof)
                pass
        
        # Create GC lookup table for codons
        self._create_gc_lookup_table()

    def _create_gc_lookup_table(self):
        """Create a lookup tensor that maps each token index to its GC content fraction."""
        from CodonTransformer.CodonUtils import TOKEN2INDEX
        
        # Initialize GC lookup tensor for all tokens
        vocab_size = len(TOKEN2INDEX)
        gc_lookup = torch.zeros(vocab_size)
        
        # Calculate GC content for each codon token
        for token, idx in TOKEN2INDEX.items():
            if "_" in token and len(token.split("_")) == 2:
                # Extract codon sequence (e.g., "k_aaa" -> "aaa")
                codon = token.split("_")[-1].upper()
                if len(codon) == 3:  # Valid codon
                    # Count G and C nucleotides
                    gc_count = codon.count('G') + codon.count('C')
                    gc_content = gc_count / 3.0  # Fraction of GC content
                    gc_lookup[idx] = gc_content
        
        # Register as buffer so it moves with the model to GPU
        self.register_buffer("gc_lookup_tensor", gc_lookup)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        
        # Enhanced scheduler: CosineAnnealingWarmRestarts for better sequence task performance
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(self.trainer.estimated_stepping_batches // 4),  # First restart after 1/4 of training
                T_mult=2,  # Double the restart period each time
                eta_min=self.learning_rate * 0.01,  # Minimum learning rate (1% of max)
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        # Forward pass
        outputs = self.model(**batch)
        mlm_loss = outputs.loss

        # Increment step counter
        self.step_counter += 1
        
        # Enhanced Augmented-Lagrangian GC Control with Self-Tuning
        gc_loss = 0
        if self.use_lagrangian or self.gc_penalty_weight > 0:
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            # Calculate expected GC content per position using differentiable approach
            # g_i = Σ_j P_ij · gc(j) where gc(j) is GC content of codon j
            expected_gc = torch.matmul(probs, self.gc_lookup_tensor)
            
            # Apply 1D convolution with uniform kernel size 50 for local GC smoothing
            window_size = 50
            expected_gc_unsqueezed = expected_gc.unsqueeze(1)  # Add channel dimension
            conv_weight = torch.ones(1, 1, window_size, device=self.device) / window_size
            gc_window = F.conv1d(expected_gc_unsqueezed, conv_weight, padding="same").squeeze(1)
            
            # Mask out padding positions
            active_positions = batch["labels"] != -100
            gc_window_active = gc_window[active_positions]
            
            if gc_window_active.numel() > 0:
                mean_gc = gc_window_active.mean()
                
                # Log current GC content
                self.log("mean_gc_window", mean_gc, on_step=True, prog_bar=True)
                
                # Apply curriculum learning - only enforce GC constraint after warm-up
                current_epoch = self.current_epoch
                if current_epoch >= self.curriculum_epochs:
                    
                    if self.use_lagrangian:
                        # Enhanced Self-Tuning Augmented-Lagrangian approach
                        gc_deviation = mean_gc - self.gc_target
                        current_violation = torch.abs(gc_deviation)
                        
                        # Update constraint violation history for adaptive penalty adjustment
                        new_history = torch.zeros_like(self.constraint_violation_history)
                        new_history[:-1] = self.constraint_violation_history[1:]
                        new_history[-1] = current_violation
                        self.constraint_violation_history = new_history
                        
                        # Self-tuning penalty coefficient (rho) update - inspired by alpaqa
                        if self.step_counter % 20 == 0 and self.step_counter > 0:
                            # Check if constraint violation is improving
                            violation_improvement = self.previous_constraint_violation - current_violation
                            relative_improvement = violation_improvement / max(self.previous_constraint_violation, 1e-8)
                            
                            # Adaptive rho update based on constraint violation progress
                            if current_violation > self.alm_dual_tolerance:
                                # If violation is still too high, check if we're making progress
                                if relative_improvement < self.alm_rel_penalty_increase_threshold:
                                    # Not improving fast enough, increase penalty
                                    new_rho = self.rho_adaptive * self.alm_penalty_update_factor
                                    self.rho_adaptive = torch.clamp(new_rho, self.alm_min_penalty, self.alm_max_penalty)
                                    
                                    # Update Lagrangian multiplier
                                    self.lambda_gc = self.lambda_gc + self.rho_adaptive * gc_deviation.detach()
                                else:
                                    # Making good progress, just update multiplier
                                    self.lambda_gc = self.lambda_gc + self.rho_adaptive * gc_deviation.detach()
                            else:
                                # Violation is acceptable, just update multiplier
                                self.lambda_gc = self.lambda_gc + self.rho_adaptive * gc_deviation.detach()
                            
                            # Update previous violation for next iteration
                            self.previous_constraint_violation = current_violation
                            self.alm_iteration_counter += 1
                            
                        # Augmented-Lagrangian loss: λ·(mean_gc - μ) + (ρ/2)(mean_gc - μ)²
                        lagrangian_term = self.lambda_gc * gc_deviation
                        penalty_term = (self.rho_adaptive / 2) * (gc_deviation ** 2)
                        gc_loss = lagrangian_term + penalty_term
                        
                        # Enhanced logging for ALM system monitoring
                        self.log("lambda_gc", self.lambda_gc, on_step=True, prog_bar=True)
                        self.log("rho_adaptive", self.rho_adaptive, on_step=True, prog_bar=True)
                        self.log("gc_deviation", gc_deviation, on_step=True, prog_bar=True)
                        self.log("constraint_violation", current_violation, on_step=True, prog_bar=False)
                        self.log("alm_iteration", self.alm_iteration_counter, on_step=True, prog_bar=False)
                        
                    else:
                        # Fallback to old penalty approach if not using Lagrangian
                        gc_dev = F.relu(torch.abs(mean_gc - self.gc_target) - 0.02)  # 2% tolerance
                        gc_loss = gc_dev
                        
                    self.log("gc_loss", gc_loss, on_step=True, prog_bar=True)
        
        # Combine losses
        if self.use_lagrangian:
            total_loss = mlm_loss + gc_loss
        else:
            total_loss = mlm_loss + self.gc_penalty_weight * gc_loss
            
        self.log_dict(
            dictionary={
                "loss": total_loss,
                "mlm_loss": mlm_loss,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            },
            on_step=True,
            prog_bar=True,
        )
        return total_loss


class DumpStateDict(pl.Callback):
    def __init__(self, checkpoint_dir, checkpoint_filename, every_n_train_steps):
        super().__init__()
        self.dirpath = checkpoint_dir
        self.every_n_train_steps = every_n_train_steps
        self.checkpoint_filename = checkpoint_filename

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        model = pl_module.model
        torch.save(
            model.state_dict(), os.path.join(self.dirpath, self.checkpoint_filename)
        )


class ALMMonitoringCallback(pl.Callback):
    """
    Enhanced PyTorch Lightning callback for monitoring ALM system behavior during training.
    
    Provides comprehensive logging and analysis of:
    - Lambda (Lagrangian multiplier) evolution
    - Rho (penalty coefficient) adaptive updates
    - Constraint violation progress
    - ALM convergence metrics
    
    Inspired by research-level ALM implementations for optimal constraint handling.
    """
    
    def __init__(self, log_every_n_steps=20, convergence_window=50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.convergence_window = convergence_window
        
        # Track ALM convergence metrics
        self.lambda_history = []
        self.rho_history = []
        self.violation_history = []
        self.step_history = []
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Monitor ALM system on each training step."""
        if hasattr(pl_module, 'use_lagrangian') and pl_module.use_lagrangian:
            current_step = trainer.global_step
            
            # Log ALM metrics every N steps
            if current_step % self.log_every_n_steps == 0:
                # Extract ALM state from logged metrics
                lambda_gc = trainer.logged_metrics.get('lambda_gc', 0)
                rho_adaptive = trainer.logged_metrics.get('rho_adaptive', 0)
                constraint_violation = trainer.logged_metrics.get('constraint_violation', 0)
                gc_deviation = trainer.logged_metrics.get('gc_deviation', 0)
                
                # Store history for convergence analysis
                self.lambda_history.append(float(lambda_gc))
                self.rho_history.append(float(rho_adaptive))
                self.violation_history.append(float(constraint_violation))
                self.step_history.append(current_step)
                
                # Keep only recent history for efficiency
                if len(self.lambda_history) > self.convergence_window:
                    self.lambda_history = self.lambda_history[-self.convergence_window:]
                    self.rho_history = self.rho_history[-self.convergence_window:]
                    self.violation_history = self.violation_history[-self.convergence_window:]
                    self.step_history = self.step_history[-self.convergence_window:]
                
                # Log comprehensive ALM metrics to TensorBoard
                if trainer.logger is not None:
                    # Primary ALM metrics
                    trainer.logger.log_metrics({
                        'alm/lambda_gc': float(lambda_gc),
                        'alm/rho_adaptive': float(rho_adaptive),
                        'alm/constraint_violation': float(constraint_violation),
                        'alm/gc_deviation': float(gc_deviation),
                    }, step=current_step)
                    
                    # Convergence analysis (if we have sufficient history)
                    if len(self.violation_history) >= 10:
                        recent_violations = self.violation_history[-10:]
                        violation_trend = recent_violations[-1] - recent_violations[0]
                        violation_stability = max(recent_violations) - min(recent_violations)
                        
                        trainer.logger.log_metrics({
                            'alm/violation_trend': violation_trend,
                            'alm/violation_stability': violation_stability,
                            'alm/rho_growth_rate': self.rho_history[-1] / max(self.rho_history[0], 1e-8),
                        }, step=current_step)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Comprehensive ALM system analysis at epoch end."""
        if hasattr(pl_module, 'use_lagrangian') and pl_module.use_lagrangian:
            current_epoch = trainer.current_epoch
            
            # Only analyze after curriculum warm-up period
            if current_epoch >= pl_module.curriculum_epochs:
                # Get current ALM state
                lambda_gc = trainer.logged_metrics.get('lambda_gc', 0)
                rho_adaptive = trainer.logged_metrics.get('rho_adaptive', 0)
                constraint_violation = trainer.logged_metrics.get('constraint_violation', 0)
                mean_gc = trainer.logged_metrics.get('mean_gc_window', 0)
                
                # ALM convergence assessment
                converged = float(constraint_violation) <= pl_module.alm_dual_tolerance
                
                # Detailed epoch summary
                print(f"\n{'='*60}")
                print(f"🔍 ALM System Analysis - Epoch {current_epoch}")
                print(f"{'='*60}")
                print(f"📊 Current State:")
                print(f"   • GC Content: {float(mean_gc):.4f} (target: {pl_module.gc_target:.4f})")
                print(f"   • Constraint Violation: {float(constraint_violation):.2e}")
                print(f"   • Lambda (Multiplier): {float(lambda_gc):.4f}")
                print(f"   • Rho (Penalty): {float(rho_adaptive):.2e}")
                print(f"   • Converged: {'✅ Yes' if converged else '❌ No'}")
                
                # Convergence diagnostics
                if len(self.violation_history) >= 5:
                    recent_violations = self.violation_history[-5:]
                    improvement_rate = (recent_violations[0] - recent_violations[-1]) / max(recent_violations[0], 1e-8)
                    
                    print(f"📈 Convergence Diagnostics:")
                    print(f"   • Recent Improvement Rate: {improvement_rate:.2%}")
                    print(f"   • Penalty Growth: {self.rho_history[-1] / max(self.rho_history[0], 1e-8):.2f}x")
                    print(f"   • Stability: {'Good' if max(recent_violations) - min(recent_violations) < 1e-3 else 'Improving'}")
                
                print(f"{'='*60}\n")
                
                # TensorBoard epoch summary
                if trainer.logger is not None:
                    trainer.logger.log_metrics({
                        'alm_epoch/converged': 1.0 if converged else 0.0,
                        'alm_epoch/final_lambda': float(lambda_gc),
                        'alm_epoch/final_rho': float(rho_adaptive),
                        'alm_epoch/final_violation': float(constraint_violation),
                    }, step=current_epoch)


class GCValidationHook(pl.Callback):
    """Validation hook to monitor GC content during training."""
    
    def __init__(self, gc_target=0.52, tolerance=0.02):
        super().__init__()
        self.gc_target = gc_target
        self.tolerance = tolerance
        self.gc_target_min = gc_target - tolerance
        self.gc_target_max = gc_target + tolerance
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Check GC content at the end of each epoch."""
        if hasattr(pl_module, 'use_lagrangian') and pl_module.use_lagrangian:
            current_epoch = trainer.current_epoch
            
            # Only validate after curriculum warm-up period
            if current_epoch >= pl_module.curriculum_epochs:
                # Get the logged mean GC content from the last step
                if 'mean_gc_window' in trainer.logged_metrics:
                    current_gc = trainer.logged_metrics.get('mean_gc_window', None)
                    
                    if current_gc is not None:
                        current_gc_val = float(current_gc)
                        
                        # Log validation status
                        within_target = self.gc_target_min <= current_gc_val <= self.gc_target_max
                        
                        if within_target:
                            print(f"✅ Epoch {current_epoch}: GC content {current_gc_val:.3f} is within target range [{self.gc_target_min:.3f}, {self.gc_target_max:.3f}]")
                        else:
                            print(f"⚠️  Epoch {current_epoch}: GC content {current_gc_val:.3f} is outside target range [{self.gc_target_min:.3f}, {self.gc_target_max:.3f}]")
                            
                        # Log lambda value if available
                        if 'lambda_gc' in trainer.logged_metrics:
                            lambda_val = float(trainer.logged_metrics.get('lambda_gc', 0))
                            print(f"   Lambda: {lambda_val:.4f}")
                            
                        # Assert for development - comment out in production
                        # assert within_target, f"GC content {current_gc_val:.3f} outside acceptable range after curriculum warm-up"


def main(args):
    """Finetune the CodonTransformer model."""
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
    model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer-base")
    harnessed_model = plTrainHarness(
        model, args.learning_rate, args.warmup_fraction, args.gc_penalty_weight, tokenizer,
        gc_target=args.gc_target, use_lagrangian=args.use_lagrangian, 
        lagrangian_rho=args.lagrangian_rho, curriculum_epochs=args.curriculum_epochs,
        alm_tolerance=args.alm_tolerance, alm_dual_tolerance=args.alm_dual_tolerance,
        alm_penalty_update_factor=args.alm_penalty_update_factor,
        alm_initial_penalty_factor=args.alm_initial_penalty_factor,
        alm_tolerance_update_factor=args.alm_tolerance_update_factor,
        alm_rel_penalty_increase_threshold=args.alm_rel_penalty_increase_threshold,
        alm_max_penalty=args.alm_max_penalty, alm_min_penalty=args.alm_min_penalty
    )

    # Load the training data
    train_data = IterableJSONData(args.dataset_dir, dist_env="slurm")
    data_loader = DataLoader(
        dataset=train_data,
        collate_fn=MaskedTokenizerCollator(tokenizer),
        batch_size=args.batch_size,
        num_workers=0 if args.debug else args.num_workers,
        persistent_workers=False if args.debug else True,
    )

    # Setup trainer and callbacks
    save_checkpoint = DumpStateDict(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_filename=args.checkpoint_filename,
        every_n_train_steps=args.save_every_n_steps,
    )
    gc_validation = GCValidationHook(
        gc_target=args.gc_target,
        tolerance=0.02  # 2% tolerance around target
    )
    
    # Enhanced ALM monitoring callback for comprehensive system analysis
    alm_monitor = ALMMonitoringCallback(
        log_every_n_steps=args.log_every_n_steps,
        convergence_window=50  # Track last 50 steps for convergence analysis
    )
    
    callbacks = [save_checkpoint, gc_validation, alm_monitor]
    
    # Determine accelerator and device configuration dynamically
    if args.num_gpus > 0:
        accelerator = "gpu"
        devices = args.num_gpus
    else:
        # Fallback to CPU training when --num_gpus 0
        accelerator = "cpu"
        devices = 1  # Lightning expects at least one device

    trainer = pl.Trainer(
        default_root_dir=args.checkpoint_dir,
        strategy=("ddp_find_unused_parameters_true" if accelerator == "gpu" and devices > 1 else "auto"),
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed" if accelerator == "gpu" else 32,
        max_epochs=args.max_epochs,
        deterministic=False,
        enable_checkpointing=True,
        callbacks=callbacks,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=args.log_every_n_steps,
    )

    # Finetune the model
    trainer.fit(harnessed_model, data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune the CodonTransformer model.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory where checkpoints will be saved",
    )
    parser.add_argument(
        "--checkpoint_filename",
        type=str,
        default="finetune.ckpt",
        help="Filename for the saved checkpoint",
    )
    parser.add_argument(
        "--batch_size", type=int, default=6, help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=15, help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--num_workers", type=int, default=3, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=4, help="Number of GPUs to use for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--warmup_fraction",
        type=float,
        default=0.1,
        help="Fraction of total steps to use for warmup",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=512,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--gc_penalty_weight",
        type=float,
        default=0.0,
        help="Weight for the GC content penalty in the loss function",
    )
    parser.add_argument(
        "--gc_target",
        type=float,
        default=0.52,
        help="Target GC content (default: 0.52 for E. coli)",
    )
    parser.add_argument(
        "--use_lagrangian",
        action="store_true",
        help="Use Augmented-Lagrangian method for GC control",
    )
    parser.add_argument(
        "--lagrangian_rho",
        type=float,
        default=10.0,
        help="Penalty coefficient for Augmented-Lagrangian method",
    )
    parser.add_argument(
        "--curriculum_epochs",
        type=int,
        default=3,
        help="Number of warm-up epochs before enforcing GC constraints",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=20,
        help="How often to log metrics (in training steps)",
    )
    
    # Enhanced ALM parameters for self-tuning GC constraint system
    parser.add_argument(
        "--alm_tolerance",
        type=float,
        default=1e-5,
        help="Primal tolerance for ALM inner solver stopping criterion",
    )
    parser.add_argument(
        "--alm_dual_tolerance",
        type=float,
        default=1e-5,
        help="Dual tolerance for ALM constraint violation",
    )
    parser.add_argument(
        "--alm_penalty_update_factor",
        type=float,
        default=10.0,
        help="Factor for updating ALM penalty parameters (rho_update_factor)",
    )
    parser.add_argument(
        "--alm_initial_penalty_factor",
        type=float,
        default=20.0,
        help="Factor for automatic ALM penalty initialization (init_rho)",
    )
    parser.add_argument(
        "--alm_tolerance_update_factor",
        type=float,
        default=0.1,
        help="Factor for updating ALM primal tolerance",
    )
    parser.add_argument(
        "--alm_rel_penalty_increase_threshold",
        type=float,
        default=0.1,
        help="Relative threshold for ALM penalty increases (gc_tolerance)",
    )
    parser.add_argument(
        "--alm_max_penalty",
        type=float,
        default=1e6,
        help="Maximum ALM penalty value to prevent ill-conditioning",
    )
    parser.add_argument(
        "--alm_min_penalty",
        type=float,
        default=1e-6,
        help="Minimum ALM penalty value",
    )
    
    args = parser.parse_args()
    main(args)
