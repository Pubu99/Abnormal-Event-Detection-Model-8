"""
Logging utilities for experiment tracking and monitoring.
Integrates TensorBoard, Weights & Biases, and file logging.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from datetime import datetime

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from loguru import logger


class ExperimentLogger:
    """Unified logger for experiments with multiple backends."""
    
    def __init__(self, config, experiment_name: Optional[str] = None):
        """
        Initialize experiment logger.
        
        Args:
            config: Configuration object
            experiment_name: Name of the experiment
        """
        self.config = config
        self.experiment_name = experiment_name or config.experiment.name
        
        # Create log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(config.logging.save_dir) / f"{self.experiment_name}_{timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup loggers
        self._setup_file_logging()
        self._setup_tensorboard()
        self._setup_wandb()
        
        self.step = 0
        
        logger.info(f"🚀 Experiment: {self.experiment_name}")
        logger.info(f"📁 Log directory: {self.log_dir}")
    
    def _setup_file_logging(self):
        """Setup file-based logging with loguru."""
        log_file = self.log_dir / "train.log"
        
        # Remove default handler
        logger.remove()
        
        # Add console handler with colors
        logger.add(
            sys.stdout,
            colorize=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
            level=self.config.logging.level
        )
        
        # Add file handler
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level=self.config.logging.level,
            rotation="100 MB"
        )
        
        self.logger = logger
    
    def _setup_tensorboard(self):
        """Setup TensorBoard logging."""
        self.tensorboard = None
        if self.config.logging.tensorboard.enabled:
            tb_dir = self.log_dir / "tensorboard"
            self.tensorboard = SummaryWriter(log_dir=str(tb_dir))
            logger.info(f"📊 TensorBoard enabled: {tb_dir}")
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        self.wandb = None
        if self.config.logging.wandb.enabled:
            try:
                import wandb
                
                # Initialize wandb
                self.wandb = wandb.init(
                    project=self.config.logging.wandb.project,
                    entity=self.config.logging.wandb.entity,
                    name=self.experiment_name,
                    config=self.config.to_dict() if hasattr(self.config, 'to_dict') else dict(self.config),
                    tags=self.config.logging.wandb.tags,
                    dir=str(self.log_dir)
                )
                logger.info("🌐 Weights & Biases enabled")
            except ImportError:
                logger.warning("⚠️  wandb not installed, skipping W&B logging")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize wandb: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        """
        Log metrics to all backends.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Global step (uses internal counter if None)
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        if step is None:
            step = self.step
        
        # Add prefix to metrics
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        
        # Log to file
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {step} | {metric_str}")
        
        # Log to TensorBoard
        if self.tensorboard:
            for name, value in metrics.items():
                self.tensorboard.add_scalar(name, value, step)
        
        # Log to W&B
        if self.wandb:
            self.wandb.log(metrics, step=step)
    
    def log_image(self, tag: str, image: Union[torch.Tensor, np.ndarray], step: Optional[int] = None):
        """
        Log image to TensorBoard and W&B.
        
        Args:
            tag: Image tag/name
            image: Image tensor or array (C, H, W) or (H, W, C)
            step: Global step
        """
        if step is None:
            step = self.step
        
        # Convert to numpy if tensor
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Log to TensorBoard
        if self.tensorboard:
            # Ensure CHW format
            if image.shape[-1] == 3:  # HWC -> CHW
                image = np.transpose(image, (2, 0, 1))
            self.tensorboard.add_image(tag, image, step)
        
        # Log to W&B
        if self.wandb:
            import wandb
            # W&B expects HWC format
            if image.shape[0] == 3:  # CHW -> HWC
                image = np.transpose(image, (1, 2, 0))
            self.wandb.log({tag: wandb.Image(image)}, step=step)
    
    def log_histogram(self, tag: str, values: Union[torch.Tensor, np.ndarray], step: Optional[int] = None):
        """Log histogram to TensorBoard."""
        if step is None:
            step = self.step
        
        if self.tensorboard:
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu()
            self.tensorboard.add_histogram(tag, values, step)
    
    def log_model_graph(self, model: torch.nn.Module, input_shape: tuple):
        """Log model architecture to TensorBoard."""
        if self.tensorboard:
            try:
                dummy_input = torch.randn(input_shape).to(next(model.parameters()).device)
                self.tensorboard.add_graph(model, dummy_input)
                logger.info("✅ Model graph logged to TensorBoard")
            except Exception as e:
                logger.warning(f"⚠️  Failed to log model graph: {e}")
    
    def log_confusion_matrix(self, cm: np.ndarray, class_names: list, step: Optional[int] = None):
        """Log confusion matrix as image."""
        if step is None:
            step = self.step
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            ax.set_title('Confusion Matrix')
            
            # Convert to image
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            self.log_image('confusion_matrix', image, step)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"⚠️  Failed to log confusion matrix: {e}")
    
    def watch_model(self, model: torch.nn.Module):
        """Watch model with W&B."""
        if self.wandb:
            self.wandb.watch(model, log='all', log_freq=100)
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, metrics: Dict[str, float], filename: str = "checkpoint.pth"):
        """
        Save model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            metrics: Current metrics
            filename: Checkpoint filename
        """
        checkpoint_path = self.log_dir / "checkpoints" / filename
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else dict(self.config)
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save to W&B
        if self.wandb:
            import wandb
            self.wandb.save(str(checkpoint_path))
    
    def increment_step(self):
        """Increment global step counter."""
        self.step += 1
    
    def close(self):
        """Close all logging backends."""
        if self.tensorboard:
            self.tensorboard.close()
        if self.wandb:
            self.wandb.finish()
        logger.info("🏁 Experiment logging closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
