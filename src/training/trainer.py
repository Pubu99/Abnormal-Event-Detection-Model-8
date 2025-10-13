"""
Main trainer class for anomaly detection model.
Handles training loop, validation, checkpointing, and logging.
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from src.models import create_model, create_loss_function
from src.data import create_dataloaders, create_test_dataloader
from src.training.metrics import MetricsCalculator
from src.utils import (
    ConfigManager,
    ExperimentLogger,
    set_seed,
    get_gpu_memory_info,
    format_time,
    AverageMeter,
    EarlyStopping,
    get_learning_rate
)


class AnomalyDetectionTrainer:
    """Trainer for anomaly detection model."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Set random seed
        set_seed(self.config.project.seed)
        
        # Setup device
        self.device = torch.device(self.config.hardware.device)
        print(f"\n🖥️  Device: {self.device}")
        if self.device.type == 'cuda':
            gpu_info = get_gpu_memory_info()
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Total memory: {gpu_info['total_gb']:.2f} GB")
            print(f"   Free memory: {gpu_info['free_gb']:.2f} GB")
        
        # Setup logging
        self.logger = ExperimentLogger(self.config)
        
        # Create dataloaders
        print("\n📦 Creating dataloaders...")
        self.train_loader, self.val_loader, self.train_dataset, self.val_dataset = \
            create_dataloaders(self.config, use_weighted_sampling=True)
        
        # Create model
        print("\n🤖 Creating model...")
        self.model = create_model(self.config, device=self.device)
        
        # Log model graph
        self.logger.log_model_graph(self.model, (1, 3, 64, 64))
        self.logger.watch_model(self.model)
        
        # Create loss function
        class_weights = None
        if self.config.training.class_balance.use_class_weights:
            class_weights = self.train_dataset.get_class_weights().to(self.device)
            print(f"\n⚖️  Using class weights for imbalanced data")
        
        self.criterion = create_loss_function(self.config, class_weights)
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if self.config.training.mixed_precision else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.training.early_stopping.patience,
            min_delta=self.config.training.early_stopping.min_delta,
            mode=self.config.training.early_stopping.mode
        )
        
        # Tracking
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.global_step = 0
        
        # Initialize SVDD center
        self._initialize_svdd_center()
        
        print("\n✅ Trainer initialized successfully!")
    
    def _create_optimizer(self):
        """Create optimizer."""
        if self.config.training.optimizer.type == 'adamw':
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.optimizer.weight_decay,
                betas=self.config.training.optimizer.betas
            )
        elif self.config.training.optimizer.type == 'sgd':
            optimizer = SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.optimizer.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer.type}")
        
        print(f"   Optimizer: {self.config.training.optimizer.type}")
        print(f"   Learning rate: {self.config.training.learning_rate}")
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.training.lr_scheduler.type == 'cosine_annealing':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.lr_scheduler.T_max,
                eta_min=self.config.training.lr_scheduler.eta_min
            )
        elif self.config.training.lr_scheduler.type == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            scheduler = None
        
        if scheduler:
            print(f"   LR Scheduler: {self.config.training.lr_scheduler.type}")
        
        return scheduler
    
    def _initialize_svdd_center(self):
        """Initialize Deep SVDD center using network forward pass."""
        print("\n🎯 Initializing SVDD center...")
        self.model.eval()
        centers = []
        
        with torch.no_grad():
            for i, (images, labels, is_anomaly) in enumerate(self.train_loader):
                if i >= 10:  # Use first 10 batches
                    break
                
                images = images.to(self.device)
                outputs = self.model(images, return_embeddings=True)
                
                # Only use normal samples for center initialization
                normal_mask = (is_anomaly == 0)
                if normal_mask.sum() > 0:
                    centers.append(outputs['embeddings'][normal_mask].mean(0))
        
        if centers:
            center = torch.stack(centers).mean(0)
            self.model.anomaly_head.center.data = center
            print(f"   ✅ SVDD center initialized from {len(centers)} batches")
        
        self.model.train()
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        # Metrics tracking
        losses = {
            'total': AverageMeter('Total Loss'),
            'classification': AverageMeter('Classification Loss'),
            'binary': AverageMeter('Binary Loss'),
            'svdd': AverageMeter('SVDD Loss')
        }
        
        metrics_calc = MetricsCalculator(
            self.config.data.num_classes,
            self.config.data.classes
        )
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, labels, is_anomaly) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            is_anomaly = is_anomaly.to(self.device)
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    outputs = self.model(images, return_embeddings=True)
                    loss_dict = self.criterion(
                        outputs, labels, is_anomaly,
                        self.model.anomaly_head.center
                    )
                    loss = loss_dict['total']
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, return_embeddings=True)
                loss_dict = self.criterion(
                    outputs, labels, is_anomaly,
                    self.model.anomaly_head.center
                )
                loss = loss_dict['total']
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Update metrics
            for key in losses:
                if key in loss_dict:
                    losses[key].update(loss_dict[key].item(), images.size(0))
            
            # Get predictions
            with torch.no_grad():
                class_probs = F.softmax(outputs['class_logits'], dim=1)
                class_preds = torch.argmax(class_probs, dim=1)
                
                binary_probs = F.softmax(outputs['binary_logits'], dim=1)
                binary_preds = torch.argmax(binary_probs, dim=1)
                
                metrics_calc.update(
                    class_preds, labels, class_probs,
                    binary_preds, is_anomaly, binary_probs[:, 1]
                )
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].avg:.4f}",
                'lr': f"{get_learning_rate(self.optimizer):.6f}"
            })
            
            # Log to tensorboard
            if self.global_step % self.config.logging.log_every_n_steps == 0:
                log_metrics = {
                    'train/loss_total': losses['total'].avg,
                    'train/loss_classification': losses['classification'].avg,
                    'train/loss_binary': losses['binary'].avg,
                    'train/loss_svdd': losses['svdd'].avg,
                    'train/learning_rate': get_learning_rate(self.optimizer)
                }
                self.logger.log_metrics(log_metrics, step=self.global_step)
            
            self.global_step += 1
        
        # Compute epoch metrics
        epoch_metrics = metrics_calc.compute_all_metrics()
        
        return losses, epoch_metrics
    
    @torch.no_grad()
    def validate(self):
        """Validate model."""
        self.model.eval()
        
        losses = {
            'total': AverageMeter('Total Loss'),
            'classification': AverageMeter('Classification Loss'),
            'binary': AverageMeter('Binary Loss'),
            'svdd': AverageMeter('SVDD Loss')
        }
        
        metrics_calc = MetricsCalculator(
            self.config.data.num_classes,
            self.config.data.classes
        )
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for images, labels, is_anomaly in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            is_anomaly = is_anomaly.to(self.device)
            
            # Forward pass
            outputs = self.model(images, return_embeddings=True)
            loss_dict = self.criterion(
                outputs, labels, is_anomaly,
                self.model.anomaly_head.center
            )
            
            # Update losses
            for key in losses:
                if key in loss_dict:
                    losses[key].update(loss_dict[key].item(), images.size(0))
            
            # Get predictions
            class_probs = F.softmax(outputs['class_logits'], dim=1)
            class_preds = torch.argmax(class_probs, dim=1)
            
            binary_probs = F.softmax(outputs['binary_logits'], dim=1)
            binary_preds = torch.argmax(binary_probs, dim=1)
            
            metrics_calc.update(
                class_preds, labels, class_probs,
                binary_preds, is_anomaly, binary_probs[:, 1]
            )
        
        # Compute metrics
        val_metrics = metrics_calc.compute_all_metrics()
        
        # Log confusion matrix
        cm = metrics_calc.compute_confusion_matrix()
        self.logger.log_confusion_matrix(
            cm, self.config.data.classes, step=self.global_step
        )
        
        return losses, val_metrics
    
    def train(self):
        """Main training loop."""
        print("\n🚀 Starting training...")
        print(f"   Epochs: {self.config.training.epochs}")
        print(f"   Batch size: {self.config.training.batch_size}")
        print(f"   Mixed precision: {self.config.training.mixed_precision}")
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_losses, train_metrics = self.train_epoch()
            
            # Validate
            val_losses, val_metrics = self.validate()
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('f1_macro', 0))
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Log epoch metrics
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{self.config.training.epochs} - {format_time(epoch_time)}")
            print(f"{'='*70}")
            print(f"Train Loss: {train_losses['total'].avg:.4f} | Val Loss: {val_losses['total'].avg:.4f}")
            print(f"Train Acc:  {train_metrics.get('accuracy', 0):.4f} | Val Acc:  {val_metrics.get('accuracy', 0):.4f}")
            print(f"Train F1:   {train_metrics.get('f1_macro', 0):.4f} | Val F1:   {val_metrics.get('f1_macro', 0):.4f}")
            
            # Log to experiment tracker
            log_dict = {
                'epoch': epoch + 1,
                'train/loss': train_losses['total'].avg,
                'train/accuracy': train_metrics.get('accuracy', 0),
                'train/f1_score': train_metrics.get('f1_macro', 0),
                'val/loss': val_losses['total'].avg,
                'val/accuracy': val_metrics.get('accuracy', 0),
                'val/f1_score': val_metrics.get('f1_macro', 0),
            }
            
            if 'binary_f1' in val_metrics:
                log_dict['val/binary_f1'] = val_metrics['binary_f1']
            
            self.logger.log_metrics(log_dict, step=self.global_step)
            
            # Save checkpoint
            monitor_metric = val_metrics.get(
                self.config.training.checkpoint.monitor.replace('val_', ''),
                0
            )
            
            if monitor_metric > self.best_val_metric:
                self.best_val_metric = monitor_metric
                self.save_checkpoint(is_best=True, metrics=val_metrics)
                print(f"✅ New best model! {self.config.training.checkpoint.monitor}: {monitor_metric:.4f}")
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(is_best=False, metrics=val_metrics)
            
            # Early stopping check
            if self.early_stopping(monitor_metric):
                print(f"\n⏹️  Early stopping triggered after {epoch + 1} epochs")
                break
            
            # GPU memory check
            if self.device.type == 'cuda' and (epoch + 1) % 5 == 0:
                gpu_info = get_gpu_memory_info()
                print(f"\nGPU Memory: {gpu_info['allocated_gb']:.2f}/{gpu_info['total_gb']:.2f} GB")
        
        total_time = time.time() - start_time
        print(f"\n🏁 Training completed in {format_time(total_time)}")
        print(f"   Best {self.config.training.checkpoint.monitor}: {self.best_val_metric:.4f}")
        
        # Close logger
        self.logger.close()
    
    def save_checkpoint(self, is_best: bool = False, metrics: Dict = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_metric': self.best_val_metric,
            'config': self.config_manager.to_dict(),
            'metrics': metrics
        }
        
        if is_best:
            filename = "best_model.pth"
        else:
            filename = f"checkpoint_epoch_{self.current_epoch + 1}.pth"
        
        self.logger.save_checkpoint(
            self.model, self.optimizer,
            self.current_epoch, metrics or {},
            filename
        )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_metric = checkpoint.get('best_val_metric', 0.0)
        
        print(f"✅ Checkpoint loaded from {checkpoint_path}")
        print(f"   Resuming from epoch {self.current_epoch + 1}")


if __name__ == "__main__":
    # Create and run trainer
    trainer = AnomalyDetectionTrainer()
    trainer.train()
