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
import warnings

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BigBirdForMaskedLM, logging as hf_logging

import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR

# Try to import EMA library
try:
    from ema_pytorch import EMA
    EMA_AVAILABLE = True
except ImportError:
    EMA_AVAILABLE = False
    warnings.warn("ema-pytorch not available. Install with: pip install ema-pytorch")

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
                     gc_target=0.52, use_lagrangian=False, lagrangian_rho=10.0, curriculum_epochs=3, weight_decay=0.01,
                     use_swa=False, swa_start_epoch=10, swa_lr=0.01, use_ema=False, ema_decay=0.9999):
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
        self.weight_decay = weight_decay
        
        # SWA and EMA parameters
        self.use_swa = use_swa
        self.swa_start_epoch = swa_start_epoch
        self.swa_lr = swa_lr
        self.use_ema = use_ema and EMA_AVAILABLE
        self.ema_decay = ema_decay
        
        # Initialize Lagrangian multiplier as buffer (persists across checkpoints)
        self.register_buffer("lambda_gc", torch.tensor(0.0))
        
        # Step counter for periodic lambda updates
        self.register_buffer("step_counter", torch.tensor(0))
        
        # Initialize SWA and EMA models
        self.swa_model = None
        self.swa_scheduler = None
        self.ema_model = None
        
        if self.use_ema:
            self.ema_model = EMA(
                self.model,
                beta=self.ema_decay,
                update_every=1,
                update_after_step=100,  # Start EMA after 100 steps
            )
            print(f"✅ EMA initialized with decay={self.ema_decay}")
        
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
            weight_decay=self.weight_decay
        )
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=int(self.trainer.estimated_stepping_batches),
                pct_start=self.warmup_fraction,
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
        
        # Augmented-Lagrangian GC Control
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
                        # Augmented-Lagrangian approach
                        gc_deviation = mean_gc - self.gc_target
                        
                        # Update lambda every 20 steps
                        if self.step_counter % 20 == 0:
                            self.lambda_gc = self.lambda_gc + self.lagrangian_rho * gc_deviation.detach()
                            
                        # Augmented-Lagrangian loss: λ·(mean_gc - μ) + (ρ/2)(mean_gc - μ)²
                        lagrangian_term = self.lambda_gc * gc_deviation
                        penalty_term = (self.lagrangian_rho / 2) * (gc_deviation ** 2)
                        gc_loss = lagrangian_term + penalty_term
                        
                        self.log("lambda_gc", self.lambda_gc, on_step=True, prog_bar=True)
                        self.log("gc_deviation", gc_deviation, on_step=True, prog_bar=True)
                        
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
        
        # Update EMA model if enabled
        if self.use_ema and self.ema_model is not None:
            self.ema_model.update()
        
        return total_loss
    
    def on_train_epoch_end(self):
        """Handle SWA model initialization and updates."""
        if self.use_swa and self.current_epoch >= self.swa_start_epoch:
            if self.swa_model is None:
                # Initialize SWA model
                print(f"🔄 Initializing SWA model at epoch {self.current_epoch}")
                self.swa_model = AveragedModel(self.model)
                self.swa_scheduler = SWALR(
                    self.trainer.optimizers[0],
                    swa_lr=self.swa_lr,
                    anneal_epochs=5,
                    anneal_strategy="cos"
                )
                print(f"✅ SWA initialized with lr={self.swa_lr}")
            else:
                # Update SWA model
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                print(f"📊 SWA model updated at epoch {self.current_epoch}")
    
    def on_train_end(self):
        """Finalize training with SWA batch normalization update."""
        if self.use_swa and self.swa_model is not None:
            print("🔧 Updating SWA batch normalization statistics...")
            # Update batch normalization statistics for SWA model
            torch.optim.swa_utils.update_bn(
                self.trainer.train_dataloader, self.swa_model, device=self.device
            )
            print("✅ SWA batch normalization updated")
        
        # Finalize EMA model
        if self.use_ema and self.ema_model is not None:
            print("✅ EMA model finalized")
    
    def get_swa_model(self):
        """Get the SWA model for checkpointing."""
        return self.swa_model
    
    def get_ema_model(self):
        """Get the EMA model for checkpointing."""
        if self.use_ema and self.ema_model is not None:
            return self.ema_model.ema_model
        return None


class DumpStateDict(pl.Callback):
    def __init__(self, checkpoint_dir, checkpoint_filename, every_n_train_steps):
        super().__init__()
        self.dirpath = checkpoint_dir
        self.every_n_train_steps = every_n_train_steps
        self.checkpoint_filename = checkpoint_filename

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Save regular model
        model = pl_module.model
        torch.save(
            model.state_dict(), os.path.join(self.dirpath, self.checkpoint_filename)
        )
        
        # Save SWA model if available
        if hasattr(pl_module, 'swa_model') and pl_module.swa_model is not None:
            swa_filename = self.checkpoint_filename.replace('.ckpt', '_swa.ckpt')
            torch.save(
                pl_module.swa_model.state_dict(), 
                os.path.join(self.dirpath, swa_filename)
            )
            print(f"💾 SWA model saved: {swa_filename}")
        
        # Save EMA model if available
        if hasattr(pl_module, 'ema_model') and pl_module.ema_model is not None:
            ema_filename = self.checkpoint_filename.replace('.ckpt', '_ema.ckpt')
            torch.save(
                pl_module.ema_model.state_dict(), 
                os.path.join(self.dirpath, ema_filename)
            )
            print(f"💾 EMA model saved: {ema_filename}")


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
    model = BigBirdForMaskedLM.from_pretrained("adibvafa/CodonTransformer")
    harnessed_model = plTrainHarness(
        model, args.learning_rate, args.warmup_fraction, args.gc_penalty_weight, tokenizer,
        gc_target=args.gc_target, use_lagrangian=args.use_lagrangian,
        lagrangian_rho=args.lagrangian_rho, curriculum_epochs=args.curriculum_epochs,
        weight_decay=args.weight_decay, use_swa=args.use_swa, swa_start_epoch=args.swa_start_epoch,
        swa_lr=args.swa_lr, use_ema=args.use_ema, ema_decay=args.ema_decay
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
    callbacks = [save_checkpoint, gc_validation]
    
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
        "--batch_size", type=int, default=2, help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=20, help="Maximum number of epochs to train"
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
        "--num_gpus", type=int, default=1, help="Number of GPUs to use for training"
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
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for the optimizer",
    )
    # SWA parameters
    parser.add_argument(
        "--use_swa",
        action="store_true",
        help="Use Stochastic Weight Averaging for better generalization",
    )
    parser.add_argument(
        "--swa_start_epoch",
        type=int,
        default=10,
        help="Epoch to start SWA (default: 10)",
    )
    parser.add_argument(
        "--swa_lr",
        type=float,
        default=0.01,
        help="Learning rate for SWA (default: 0.01)",
    )
    # EMA parameters  
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use Exponential Moving Average for model stability",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA decay rate (default: 0.9999)",
    )
    args = parser.parse_args()
    main(args)
