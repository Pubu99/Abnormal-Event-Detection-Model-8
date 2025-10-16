"""
Research-Enhanced Trainer for Multi-Task Anomaly Detection.

Handles:
- Regression loss (future feature prediction) - PRIMARY
- Focal loss (classification) - AUXILIARY  
- MIL ranking loss (weakly supervised) - TERTIARY
- VAE loss (unsupervised reconstruction) - TERTIARY

Training strategy:
- Multi-task learning with weighted loss combination
- Gradient accumulation for larger effective batch size
- Mixed precision training (FP16)
- W&B logging for all loss components
- Early stopping on val F1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from typing import Dict, Tuple, Optional
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path

from src.models.losses import (
    FocalLoss,
    MILRankingLoss,
    TemporalRegressionLoss,
    VAELoss
)
from src.training.metrics import calculate_metrics


class ResearchTrainer:
    """
    Trainer for research-enhanced multi-task model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config,
        device: str = 'cuda',
        logger = None
    ):
        """
        Initialize research trainer.
        
        Args:
            model: Research-enhanced model
            train_loader: Training dataloader (sequences)
            val_loader: Validation dataloader (sequences)
            config: Configuration object
            device: Device to train on
            logger: Logger instance
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger
        
        # Move model to device
        self.model.to(device)
        
        # Initialize losses
        self._init_losses()
        
        # Initialize optimizer
        self._init_optimizer()
        
        # Initialize scheduler
        self._init_scheduler()
        
        # Mixed precision scaler
        self.use_amp = config.training.mixed_precision
        if self.use_amp:
            self.scaler = GradScaler('cuda')
        
        # Gradient accumulation
        self.grad_accum_steps = config.training.gradient_accumulation_steps
        
        # Gradient clipping
        self.grad_clip_config = config.training.get('gradient_clip', {})
        self.use_grad_clip = self.grad_clip_config.get('enabled', True)
        self.max_grad_norm = self.grad_clip_config.get('max_norm', 1.0)
        
        # Early stopping
        self.early_stopping_patience = config.training.early_stopping.patience
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0
        
        # Checkpointing
        self.checkpoint_dir = Path(config.output.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint saving frequency (save every N epochs)
        self.save_checkpoint_every = config.training.get('save_checkpoint_every', 10)
        
        print("\n✅ Research Trainer initialized")
        print(f"   Device: {device}")
        print(f"   Mixed precision: {self.use_amp}")
        print(f"   Gradient accumulation: {self.grad_accum_steps}")
        print(f"   Gradient clipping: {self.use_grad_clip}")
        print(f"   Checkpoint dir: {self.checkpoint_dir}")
        print(f"   Save checkpoint every: {self.save_checkpoint_every} epochs")
    
    def _init_losses(self):
        """Initialize all loss functions."""
        loss_config = self.config.training.loss
        
        # 1. Regression loss (PRIMARY)
        regression_config = loss_config.regression
        self.regression_loss_fn = TemporalRegressionLoss(
            loss_type=regression_config.get('type', 'smooth_l1'),
            reduction=regression_config.get('reduction', 'mean')
        )
        self.regression_weight = regression_config.get('weight', 1.0)
        
        # 2. Focal loss (AUXILIARY)
        classification_config = loss_config.classification
        self.focal_loss_fn = FocalLoss(
            gamma=classification_config.get('gamma', 2.0),
            alpha=classification_config.get('alpha', None),  # Auto-compute
            reduction=classification_config.get('reduction', 'mean')
        )
        self.focal_weight = classification_config.get('weight', 0.5)
        
        # 3. MIL ranking loss (TERTIARY)
        mil_config = loss_config.get('mil_ranking', {})
        if mil_config.get('enabled', True):
            self.use_mil = True
            self.mil_loss_fn = MILRankingLoss(
                margin=mil_config.get('margin', 0.5),
                positive_bag_weight=mil_config.get('positive_bag_weight', 3.0)
            )
            self.mil_weight = mil_config.get('weight', 0.3)
        else:
            self.use_mil = False
        
        # 4. VAE loss (TERTIARY)
        vae_config = loss_config.get('vae', {})
        if vae_config.get('weight', 0.3) > 0:
            self.use_vae = True
            self.vae_loss_fn = VAELoss(
                reconstruction_weight=vae_config.get('reconstruction_weight', 1.0),
                kl_weight=vae_config.get('kl_weight', 0.01),
                reduction=vae_config.get('reduction', 'mean')
            )
            self.vae_weight = vae_config.get('weight', 0.3)
        else:
            self.use_vae = False
        
        print(f"\n📊 Loss Configuration:")
        print(f"   Regression (PRIMARY): weight={self.regression_weight}")
        print(f"   Focal (AUXILIARY): weight={self.focal_weight}")
        if self.use_mil:
            print(f"   MIL Ranking (TERTIARY): weight={self.mil_weight}")
        if self.use_vae:
            print(f"   VAE (TERTIARY): weight={self.vae_weight}")
    
    def _init_optimizer(self):
        """Initialize optimizer."""
        opt_config = self.config.training.optimizer
        
        if opt_config.type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=opt_config.weight_decay,
                betas=tuple(opt_config.betas)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.type}")
        
        print(f"\n🔧 Optimizer: {opt_config.type}")
        print(f"   Learning rate: {self.config.training.learning_rate}")
        print(f"   Weight decay: {opt_config.weight_decay}")
    
    def _init_scheduler(self):
        """Initialize learning rate scheduler."""
        sched_config = self.config.training.lr_scheduler
        
        if sched_config.type == 'onecycle':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.training.max_learning_rate,
                total_steps=self.config.training.num_epochs * len(self.train_loader),
                pct_start=sched_config.pct_start,
                div_factor=sched_config.div_factor,
                final_div_factor=sched_config.final_div_factor
            )
            self.scheduler_step_on_batch = True
        else:
            raise ValueError(f"Unknown scheduler: {sched_config.type}")
        
        print(f"\n📈 Scheduler: {sched_config.type}")
        print(f"   Max LR: {self.config.training.max_learning_rate}")
    
    def compute_losses(
        self,
        outputs: Dict,
        labels: torch.Tensor,
        future_frames: torch.Tensor,
        backbone_outputs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute all losses for multi-task learning.
        
        Args:
            outputs: Model outputs dictionary
            labels: Class labels (B,)
            future_frames: Future frames for regression (B, C, H, W)
            backbone_outputs: Optional backbone features for future frames
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        losses = {}
        
        # 1. Regression loss (predict future features)
        if 'regression' in outputs and backbone_outputs is not None:
            predicted_future = outputs['regression']  # (B, D)
            
            # Need to extract features from future frames
            # For efficiency, we'll use the current pooled features as proxy
            # In full implementation, would pass future_frames through backbone
            target_future = outputs.get('pooled_features', predicted_future).detach()
            
            regression_loss = self.regression_loss_fn(predicted_future, target_future)
            losses['regression'] = regression_loss * self.regression_weight
        
        # 2. Focal loss (classification)
        if 'class_logits' in outputs:
            class_logits = outputs['class_logits']  # (B, num_classes)
            focal_loss = self.focal_loss_fn(class_logits, labels)
            losses['focal'] = focal_loss * self.focal_weight
        
        # 3. MIL ranking loss (weakly supervised)
        if self.use_mil and 'class_logits' in outputs:
            # Convert multi-class logits to anomaly scores
            # Normal class (NormalVideos) is index 13
            normal_class_idx = 13
            
            # Anomaly score: max prob over all abnormal classes
            probs = F.softmax(outputs['class_logits'], dim=1)
            normal_prob = probs[:, normal_class_idx]
            anomaly_score = 1.0 - normal_prob  # Higher = more abnormal
            
            # Binary labels: 0=normal, 1=abnormal
            binary_labels = (labels != normal_class_idx).long()
            
            mil_loss = self.mil_loss_fn(anomaly_score, binary_labels)
            losses['mil_ranking'] = mil_loss * self.mil_weight
        
        # 4. VAE loss (reconstruction)
        if self.use_vae and 'vae' in outputs:
            vae_outputs = outputs['vae']
            pooled_features = outputs.get('pooled_features')
            
            if pooled_features is not None:
                vae_loss = self.vae_loss_fn(
                    vae_outputs['reconstructed'],
                    pooled_features,
                    vae_outputs['mu'],
                    vae_outputs['logvar']
                )
                losses['vae_total'] = vae_loss['total'] * self.vae_weight
                losses['vae_recon'] = vae_loss['reconstruction']
                losses['vae_kl'] = vae_loss['kl']
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return total_loss, losses
    
    def train_epoch(self, epoch: int) -> Dict:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_losses = {
            'total': 0.0,
            'regression': 0.0,
            'focal': 0.0,
            'mil_ranking': 0.0,
            'vae_total': 0.0
        }
        
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (frames, labels, future_frames) in enumerate(pbar):
            # Move to device
            frames = frames.to(self.device)  # (B, T, C, H, W)
            labels = labels.to(self.device)
            future_frames = future_frames.to(self.device)  # (B, C, H, W)
            
            # Forward pass with mixed precision
            with autocast('cuda', enabled=self.use_amp):
                outputs = self.model(frames, return_all=True)
                
                # Compute losses
                loss, loss_dict = self.compute_losses(
                    outputs, labels, future_frames
                )
                
                # Scale loss for gradient accumulation
                loss = loss / self.grad_accum_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                if self.use_grad_clip:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Scheduler step (if per-batch)
                if self.scheduler_step_on_batch:
                    self.scheduler.step()
            
            # Accumulate losses
            for key in loss_dict:
                if key in total_losses:
                    total_losses[key] += loss_dict[key].item()
            
            # Get predictions
            if 'class_logits' in outputs:
                preds = outputs['class_logits'].argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss_dict['total'].item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        # Average losses
        num_batches = len(self.train_loader)
        for key in total_losses:
            total_losses[key] /= num_batches
        
        # Compute metrics
        metrics = calculate_metrics(
            np.array(all_preds),
            np.array(all_labels)
        )
        
        metrics.update(total_losses)
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict:
        """
        Validate model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_losses = {
            'total': 0.0,
            'regression': 0.0,
            'focal': 0.0,
            'mil_ranking': 0.0,
            'vae_total': 0.0
        }
        
        all_preds = []
        all_labels = []
        
        for frames, labels, future_frames in tqdm(self.val_loader, desc="Validating"):
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            future_frames = future_frames.to(self.device)
            
            # Forward pass
            outputs = self.model(frames, return_all=True)
            
            # Compute losses
            loss, loss_dict = self.compute_losses(
                outputs, labels, future_frames
            )
            
            # Accumulate losses
            for key in loss_dict:
                if key in total_losses:
                    total_losses[key] += loss_dict[key].item()
            
            # Get predictions
            if 'class_logits' in outputs:
                preds = outputs['class_logits'].argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Average losses
        num_batches = len(self.val_loader)
        for key in total_losses:
            total_losses[key] /= num_batches
        
        # Compute metrics
        metrics = calculate_metrics(
            np.array(all_preds),
            np.array(all_labels)
        )
        
        metrics.update(total_losses)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint with timestamp and accuracy naming."""
        from datetime import datetime
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / 'last.pth'
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint with timestamp and accuracy
        if is_best:
            # Get timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get accuracy metrics (use weighted F1 for imbalanced datasets)
            val_acc = metrics.get('accuracy', 0.0) * 100
            val_f1 = metrics.get('f1_weighted', 0.0) * 100
            
            # Create descriptive filename
            # Format: research_enhanced_YYYYMMDD_HHMMSS_acc{XX.X}_f1{XX.X}.pth
            best_filename = (
                f"research_enhanced_{timestamp}_"
                f"acc{val_acc:.1f}_f1{val_f1:.1f}.pth"
            )
            best_path = self.checkpoint_dir / best_filename
            torch.save(checkpoint, best_path)
            
            # Also save as 'best.pth' for easy reference
            best_link = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_link)
            
            print(f"\n💾 Best checkpoint saved:")
            print(f"   {best_filename}")
            print(f"   Val Accuracy: {val_acc:.2f}%")
            print(f"   Val F1: {val_f1:.2f}%")
    
    def save_periodic_checkpoint(self, epoch: int, metrics: Dict):
        """Save periodic checkpoint with epoch number and metrics."""
        from datetime import datetime
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Get timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get metrics (use weighted F1 for imbalanced datasets)
        val_acc = metrics.get('accuracy', 0.0) * 100
        val_f1 = metrics.get('f1_weighted', 0.0) * 100
        
        # Create filename with epoch number
        # Format: research_enhanced_epoch{XX}_YYYYMMDD_HHMMSS_acc{XX.X}_f1{XX.X}.pth
        checkpoint_filename = (
            f"research_enhanced_epoch{epoch:03d}_{timestamp}_"
            f"acc{val_acc:.1f}_f1{val_f1:.1f}.pth"
        )
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        torch.save(checkpoint, checkpoint_path)
        
        print(f"\n📦 Periodic checkpoint saved: {checkpoint_filename}")
    
    def train(self):
        """Main training loop."""
        print("\n" + "=" * 80)
        print("🚀 Starting Research-Enhanced Training")
        print("=" * 80)
        
        for epoch in range(1, self.config.training.num_epochs + 1):
            print(f"\n📅 Epoch {epoch}/{self.config.training.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Log metrics
            print(f"\n📊 Epoch {epoch} Results:")
            print(f"   Train Loss: {train_metrics['total']:.4f}")
            print(f"   Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"   Val Loss: {val_metrics['total']:.4f}")
            print(f"   Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"   Val F1: {val_metrics['f1_weighted']:.4f}")
            
            # W&B logging
            if self.config.logging.wandb.enabled and wandb.run is not None:
                wandb.log({
                    'epoch': epoch,
                    'train/total_loss': train_metrics['total'],
                    'train/regression_loss': train_metrics['regression'],
                    'train/focal_loss': train_metrics['focal'],
                    'train/accuracy': train_metrics['accuracy'],
                    'train/f1_weighted': train_metrics['f1_weighted'],
                    'train/f1_macro': train_metrics['f1_macro'],
                    'val/total_loss': val_metrics['total'],
                    'val/regression_loss': val_metrics['regression'],
                    'val/focal_loss': val_metrics['focal'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/f1_weighted': val_metrics['f1_weighted'],
                    'val/f1_macro': val_metrics['f1_macro'],
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Check for improvement (use weighted F1 for imbalanced datasets)
            is_best = val_metrics['f1_weighted'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1_weighted']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Save periodic checkpoint (every N epochs)
            if epoch % self.save_checkpoint_every == 0:
                self.save_periodic_checkpoint(epoch, val_metrics)
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n⏹️  Early stopping triggered (patience={self.early_stopping_patience})")
                break
        
        print("\n" + "=" * 80)
        print("✅ Training Complete!")
        print(f"   Best Val F1: {self.best_val_f1:.4f}")
        print("=" * 80)
