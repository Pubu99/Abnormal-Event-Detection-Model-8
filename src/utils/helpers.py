"""
Utility functions for the anomaly detection system.
"""

import random
import numpy as np
import torch
from typing import Dict, List, Tuple
import os


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get GPU memory information.
    
    Returns:
        Dictionary with memory stats in GB
    """
    if not torch.cuda.is_available():
        return {}
    
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'total_gb': total,
        'free_gb': total - allocated
    }


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update meter with new value.
        
        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """Early stopping to stop training when metric stops improving."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/f1
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        self.compare_fn = np.greater if mode == 'max' else np.less
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop training.
        
        Args:
            score: Current metric score
            
        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        # Check if improved
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def compute_class_weights(class_counts: List[int], method: str = 'inverse') -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        class_counts: List of sample counts per class
        method: 'inverse' or 'effective'
        
    Returns:
        Tensor of class weights
    """
    class_counts = np.array(class_counts)
    total_samples = class_counts.sum()
    
    if method == 'inverse':
        # Inverse frequency
        weights = total_samples / (len(class_counts) * class_counts)
    elif method == 'effective':
        # Effective number of samples
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights
    weights = weights / weights.sum() * len(class_counts)
    
    return torch.FloatTensor(weights)


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def save_predictions(predictions: np.ndarray, labels: np.ndarray, 
                     output_path: str, class_names: List[str] = None):
    """
    Save predictions to file.
    
    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        output_path: Output file path
        class_names: List of class names
    """
    import pandas as pd
    
    df = pd.DataFrame({
        'predicted': predictions,
        'actual': labels,
        'correct': predictions == labels
    })
    
    if class_names:
        df['predicted_class'] = df['predicted'].map(lambda x: class_names[x] if x < len(class_names) else 'Unknown')
        df['actual_class'] = df['actual'].map(lambda x: class_names[x] if x < len(class_names) else 'Unknown')
    
    df.to_csv(output_path, index=False)


__all__ = [
    'set_seed',
    'count_parameters',
    'get_gpu_memory_info',
    'format_time',
    'AverageMeter',
    'EarlyStopping',
    'compute_class_weights',
    'get_learning_rate',
    'save_predictions'
]
