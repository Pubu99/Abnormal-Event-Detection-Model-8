"""
Main trainer class for anomaly detection model.
Handles training loop, validation, checkpointing, and logging.
Implements advanced techniques: SAM, SWA, Mixup/CutMix, TTA for better generalization.
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.amp import GradScaler
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm

from src.models import create_model, create_loss_function
from src.data import create_dataloaders, create_test_dataloader, MixupCutmixWrapper, mixup_criterion
from src.training.metrics import MetricsCalculator
from src.training.sam_optimizer import SAM, ASAM
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
        
        # Model Compilation (PyTorch 2.0+) for SPEED - NEW!
        # Note: Disabled when using SAM due to CUDA graph conflicts
        compile_config = self.config.training.get('compile_model', {})
        sam_enabled = self.config.training.get('sam', {}).get('enabled', False)
        
        if compile_config.get('enabled', False) and not sam_enabled:
            try:
                print(f"\n⚡ Compiling model with torch.compile() for speedup...")
                print(f"   Mode: {compile_config.get('mode', 'reduce-overhead')}")
                self.model = torch.compile(
                    self.model,
                    mode=compile_config.get('mode', 'reduce-overhead')
                )
                print(f"   ✅ Model compiled! Expected 30-50% speedup")
            except Exception as e:
                print(f"   ⚠️  Compilation failed: {e}")
                print(f"   Continuing without compilation...")
        elif compile_config.get('enabled', False) and sam_enabled:
            print(f"\n⚠️  torch.compile() disabled because SAM is enabled")
            print(f"   (CUDA graphs conflict with SAM's dual forward passes)")
        
        # Log model graph
        self.logger.log_model_graph(self.model, (1, 3, 64, 64))
        self.logger.watch_model(self.model)
        
        # Create loss function
        class_weights = None
        if self.config.training.class_balance.use_class_weights:
            class_weights = self.train_dataset.get_class_weights().to(self.device)
            print(f"\n⚖️  Using class weights for imbalanced data")
        
        self.criterion = create_loss_function(self.config, class_weights)
        
        # Create optimizer (with SAM if enabled)
        self.use_sam = self.config.training.get('sam', {}).get('enabled', False)
        self.optimizer = self._create_optimizer()
        
        # Create learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Stochastic Weight Averaging (SWA) - NEW!
        self.use_swa = self.config.training.get('swa', {}).get('enabled', False)
        if self.use_swa:
            self.swa_model = AveragedModel(self.model)
            self.swa_start = self.config.training.swa.get('start_epoch', 75)
            self.swa_scheduler = SWALR(
                self.optimizer if not self.use_sam else self.optimizer.base_optimizer,
                swa_lr=self.config.training.swa.get('lr', 0.00005),
                anneal_epochs=self.config.training.swa.get('anneal_epochs', 10)
            )
            print(f"\n📊 SWA enabled - will start at epoch {self.swa_start}")
        else:
            self.swa_model = None
            self.swa_scheduler = None
        
        # Mixup/CutMix Augmentation - NEW!
        mixup_config = self.config.training.get('mixup_cutmix', {})
        if mixup_config.get('enabled', False):
            self.mixup_cutmix = MixupCutmixWrapper(
                mixup_alpha=mixup_config.get('mixup_alpha', 0.2),
                cutmix_alpha=mixup_config.get('cutmix_alpha', 1.0),
                prob=mixup_config.get('prob', 0.5),
                switch_prob=mixup_config.get('switch_prob', 0.5)
            )
            print(f"\n🎲 Mixup/CutMix enabled - alpha={mixup_config.get('mixup_alpha', 0.2)}")
        else:
            self.mixup_cutmix = None
        
        # Mixed precision training
        self.scaler = GradScaler('cuda') if self.config.training.mixed_precision else None
        
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
    
    def _get_actual_model(self):
        """Get the actual model, unwrapping SWA if needed."""
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model
    
    def _get_svdd_center(self):
        """Get SVDD center from the actual model."""
        return self._get_actual_model().anomaly_head.center
    
    def _create_optimizer(self):
        """Create optimizer with optional SAM wrapper."""
        opt_config = self.config.training.optimizer
        
        # Base optimizer parameters
        params = self.model.parameters()
        lr = self.config.training.learning_rate
        weight_decay = opt_config.weight_decay
        
        # Check if SAM is enabled
        sam_config = self.config.training.get('sam', {})
        use_sam = sam_config.get('enabled', False)
        
        if use_sam:
            # Create SAM optimizer
            print(f"\n🔍 Using SAM Optimizer for better generalization")
            print(f"   Rho: {sam_config.get('rho', 0.05)}")
            print(f"   Adaptive: {sam_config.get('adaptive', False)}")
            
            if opt_config.type == 'adamw':
                base_optimizer = AdamW
                optimizer = (ASAM if sam_config.get('adaptive', False) else SAM)(
                    params,
                    base_optimizer,
                    rho=sam_config.get('rho', 0.05),
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=opt_config.betas
                )
            elif opt_config.type == 'sgd':
                base_optimizer = SGD
                optimizer = (ASAM if sam_config.get('adaptive', False) else SAM)(
                    params,
                    base_optimizer,
                    rho=sam_config.get('rho', 0.2),
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=0.9
                )
            else:
                raise ValueError(f"Unknown optimizer: {opt_config.type}")
        else:
            # Standard optimizer
            if opt_config.type == 'adamw':
                optimizer = AdamW(
                    params,
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=opt_config.betas
                )
            elif opt_config.type == 'sgd':
                optimizer = SGD(
                    params,
                    lr=lr,
                    momentum=0.9,
                    weight_decay=weight_decay
                )
            else:
                raise ValueError(f"Unknown optimizer: {opt_config.type}")
        
        print(f"   Base Optimizer: {opt_config.type}")
        print(f"   Learning rate: {lr}")
        print(f"   Weight decay: {weight_decay}")
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_type = self.config.training.lr_scheduler.type
        
        if scheduler_type == 'onecycle':
            # OneCycleLR - FASTEST TRAINING (50% time savings!)
            # Calculate total steps
            steps_per_epoch = len(self.train_loader)
            total_steps = steps_per_epoch * self.config.training.epochs
            
            scheduler = OneCycleLR(
                self.optimizer if not self.use_sam else self.optimizer.base_optimizer,
                max_lr=self.config.training.get('max_learning_rate', self.config.training.learning_rate * 10),
                total_steps=total_steps,
                pct_start=self.config.training.lr_scheduler.get('pct_start', 0.3),
                div_factor=self.config.training.lr_scheduler.get('div_factor', 25),
                final_div_factor=self.config.training.lr_scheduler.get('final_div_factor', 10000),
                anneal_strategy='cos'
            )
            print(f"\n⚡ OneCycleLR Scheduler - FAST TRAINING MODE")
            print(f"   Max LR: {self.config.training.get('max_learning_rate', self.config.training.learning_rate * 10)}")
            print(f"   Total steps: {total_steps:,}")
            print(f"   Expected: 50% faster convergence!")
            
        elif scheduler_type == 'cosine_annealing':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.lr_scheduler.T_max,
                eta_min=self.config.training.lr_scheduler.eta_min
            )
            
        elif scheduler_type == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            scheduler = None
        
        if scheduler and scheduler_type != 'onecycle':
            print(f"   LR Scheduler: {scheduler_type}")
        
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
            self._get_svdd_center().data = center
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
            
            # Apply Mixup/CutMix augmentation - NEW!
            if self.mixup_cutmix is not None:
                images, labels_a, labels_b, lam = self.mixup_cutmix(images, labels)
                use_mixup = True
            else:
                labels_a, labels_b, lam = labels, labels, 1.0
                use_mixup = False
            
            # Define closure for SAM if needed
            def closure():
                """Closure function for SAM optimizer."""
                if self.scaler:
                    with autocast(device_type="cuda"):
                        outputs = self.model(images, return_embeddings=True)
                        if use_mixup:
                            # Compute mixup loss
                            loss_a = self.criterion(outputs, labels_a, is_anomaly, self._get_svdd_center())['total']
                            loss_b = self.criterion(outputs, labels_b, is_anomaly, self._get_svdd_center())['total']
                            loss = lam * loss_a + (1 - lam) * loss_b
                        else:
                            loss_dict = self.criterion(outputs, labels, is_anomaly, self._get_svdd_center())
                            loss = loss_dict['total']
                    return loss
                else:
                    outputs = self.model(images, return_embeddings=True)
                    if use_mixup:
                        loss_a = self.criterion(outputs, labels_a, is_anomaly, self._get_svdd_center())['total']
                        loss_b = self.criterion(outputs, labels_b, is_anomaly, self._get_svdd_center())['total']
                        loss = lam * loss_a + (1 - lam) * loss_b
                    else:
                        loss_dict = self.criterion(outputs, labels, is_anomaly, self._get_svdd_center())
                        loss = loss_dict['total']
                    return loss
            
            # Forward and backward pass
            if self.use_sam:
                # SAM requires two forward-backward passes
                if self.scaler:
                    # With mixed precision, we need manual control
                    # First forward-backward pass
                    self.optimizer.zero_grad()
                    with autocast(device_type="cuda"):
                        outputs = self.model(images, return_embeddings=True)
                        if use_mixup:
                            loss_dict_a = self.criterion(outputs, labels_a, is_anomaly, self._get_svdd_center(), self.current_epoch)
                            loss_dict_b = self.criterion(outputs, labels_b, is_anomaly, self._get_svdd_center(), self.current_epoch)
                            loss = lam * loss_dict_a['total'] + (1 - lam) * loss_dict_b['total']
                            # Create averaged loss_dict
                            loss_dict = {k: lam * loss_dict_a[k] + (1 - lam) * loss_dict_b[k] for k in loss_dict_a}
                        else:
                            loss_dict = self.criterion(outputs, labels, is_anomaly, self._get_svdd_center(), self.current_epoch)
                            loss = loss_dict['total']
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    self.optimizer.first_step(zero_grad=True)
                    self.scaler.update()
                    
                    # Second forward-backward pass  
                    # Reset scaler state for second pass
                    self.scaler = GradScaler('cuda')
                    with autocast(device_type="cuda"):
                        outputs2 = self.model(images, return_embeddings=True)
                        if use_mixup:
                            loss_dict2_a = self.criterion(outputs2, labels_a, is_anomaly, self._get_svdd_center(), self.current_epoch)
                            loss_dict2_b = self.criterion(outputs2, labels_b, is_anomaly, self._get_svdd_center(), self.current_epoch)
                            loss2 = lam * loss_dict2_a['total'] + (1 - lam) * loss_dict2_b['total']
                        else:
                            loss_dict2 = self.criterion(outputs2, labels, is_anomaly, self._get_svdd_center(), self.current_epoch)
                            loss2 = loss_dict2['total']
                    
                    self.scaler.scale(loss2).backward()
                    if self.config.training.gradient_clip.enabled:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                       max_norm=self.config.training.gradient_clip.max_norm)
                    else:
                        self.scaler.unscale_(self.optimizer)
                    self.optimizer.second_step(zero_grad=True)
                    self.scaler.update()
                else:
                    # No mixed precision
                    outputs = self.model(images, return_embeddings=True)
                    if use_mixup:
                        loss_dict_a = self.criterion(outputs, labels_a, is_anomaly, self._get_svdd_center(), self.current_epoch)
                        loss_dict_b = self.criterion(outputs, labels_b, is_anomaly, self._get_svdd_center(), self.current_epoch)
                        loss = lam * loss_dict_a['total'] + (1 - lam) * loss_dict_b['total']
                        loss_dict = {k: lam * loss_dict_a[k] + (1 - lam) * loss_dict_b[k] for k in loss_dict_a}
                    else:
                        loss_dict = self.criterion(outputs, labels, is_anomaly, self._get_svdd_center(), self.current_epoch)
                        loss = loss_dict['total']
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.first_step(zero_grad=True)
                    
                    # Second pass
                    outputs2 = self.model(images, return_embeddings=True)
                    if use_mixup:
                        loss_dict2_a = self.criterion(outputs2, labels_a, is_anomaly, self._get_svdd_center(), self.current_epoch)
                        loss_dict2_b = self.criterion(outputs2, labels_b, is_anomaly, self._get_svdd_center(), self.current_epoch)
                        loss2 = lam * loss_dict2_a['total'] + (1 - lam) * loss_dict2_b['total']
                    else:
                        loss_dict2 = self.criterion(outputs2, labels, is_anomaly, self._get_svdd_center(), self.current_epoch)
                        loss2 = loss_dict2['total']
                    
                    loss2.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.second_step(zero_grad=True)
            else:
                # Standard training (no SAM)
                if self.scaler:
                    with autocast(device_type="cuda"):
                        outputs = self.model(images, return_embeddings=True)
                        if use_mixup:
                            loss_dict_a = self.criterion(outputs, labels_a, is_anomaly, self._get_svdd_center(), self.current_epoch)
                            loss_dict_b = self.criterion(outputs, labels_b, is_anomaly, self._get_svdd_center(), self.current_epoch)
                            loss = lam * loss_dict_a['total'] + (1 - lam) * loss_dict_b['total']
                            loss_dict = {k: lam * loss_dict_a[k] + (1 - lam) * loss_dict_b[k] for k in loss_dict_a}
                        else:
                            loss_dict = self.criterion(outputs, labels, is_anomaly, self._get_svdd_center(), self.current_epoch)
                            loss = loss_dict['total']
                    
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images, return_embeddings=True)
                    if use_mixup:
                        loss_dict_a = self.criterion(outputs, labels_a, is_anomaly, self._get_svdd_center(), self.current_epoch)
                        loss_dict_b = self.criterion(outputs, labels_b, is_anomaly, self._get_svdd_center(), self.current_epoch)
                        loss = lam * loss_dict_a['total'] + (1 - lam) * loss_dict_b['total']
                        loss_dict = {k: lam * loss_dict_a[k] + (1 - lam) * loss_dict_b[k] for k in loss_dict_a}
                    else:
                        loss_dict = self.criterion(outputs, labels, is_anomaly, self._get_svdd_center(), self.current_epoch)
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
            
            # OneCycleLR step (per batch) - NEW for FAST TRAINING!
            if self.scheduler and isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
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
            
            # Get the actual model (unwrap SWA if needed)
            actual_model = self.model.module if hasattr(self.model, 'module') else self.model
            
            loss_dict = self.criterion(
                outputs, labels, is_anomaly,
                actual_model.anomaly_head.center,
                self.current_epoch
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
            
            # Stochastic Weight Averaging (SWA) - NEW!
            if self.use_swa and epoch >= self.swa_start:
                self.swa_model.update_parameters(self.model)
                if self.swa_scheduler:
                    self.swa_scheduler.step()
                if epoch == self.swa_start:
                    print(f"\n📊 SWA started at epoch {epoch + 1}")
            
            # Learning rate scheduling (epoch-based schedulers only)
            if self.scheduler and (not self.use_swa or epoch < self.swa_start):
                # OneCycleLR steps per batch, not per epoch
                if isinstance(self.scheduler, OneCycleLR):
                    pass  # Already stepped in train_epoch()
                elif isinstance(self.scheduler, ReduceLROnPlateau):
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
            # Extract the metric name without 'val_' prefix to match metrics dict keys
            monitor_key = self.config.training.checkpoint.monitor.replace('val_', '')
            
            # Try multiple possible key names for F1 score
            monitor_metric = val_metrics.get(monitor_key, 
                             val_metrics.get('f1_score',
                             val_metrics.get('f1_macro', 0)))
            
            # Check if this is the best model
            is_best = monitor_metric > self.best_val_metric
            if is_best:
                self.best_val_metric = monitor_metric
                print(f"✅ New best model! {self.config.training.checkpoint.monitor}: {monitor_metric:.4f}")
            
            # Save checkpoint every epoch if configured
            save_every_epoch = self.config.training.checkpoint.get('save_every_epoch', False)
            if save_every_epoch:
                # Save with epoch number and metric for tracking
                epoch_filename = f"epoch_{epoch+1:03d}_f1_{monitor_metric:.4f}.pth"
                self.save_checkpoint(is_best=is_best, metrics=val_metrics, filename=epoch_filename)
                if self.config.training.checkpoint.get('verbose', False):
                    print(f"💾 Saved checkpoint: {epoch_filename}")
            else:
                # Original behavior: save best and periodic checkpoints
                if is_best:
                    self.save_checkpoint(is_best=True, metrics=val_metrics)
                
                # Regular checkpoint every 10 epochs
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
        
        # Finalize SWA - NEW!
        if self.use_swa:
            print(f"\n📊 Finalizing SWA model...")
            # Update batch normalization statistics
            torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model, device=self.device)
            
            # Evaluate SWA model
            print(f"\n📊 Evaluating SWA model...")
            temp_model = self.model
            self.model = self.swa_model
            val_losses, val_metrics = self.validate()
            swa_metric = val_metrics.get(
                self.config.training.checkpoint.monitor.replace('val_', ''),
                0
            )
            print(f"   SWA {self.config.training.checkpoint.monitor}: {swa_metric:.4f}")
            
            # Save SWA model if better
            if swa_metric > self.best_val_metric:
                print(f"   ✅ SWA model is better! Saving...")
                self.save_checkpoint(is_best=True, metrics=val_metrics, filename='swa_best.pth')
                self.best_val_metric = swa_metric
            else:
                # Restore original model
                self.model = temp_model
                print(f"   Original model is better.")
        
        # Close logger
        self.logger.close()
    
    def save_checkpoint(self, is_best: bool = False, metrics: Dict = None, filename: str = None):
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
        
        # Use provided filename or generate default
        if filename is None:
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
