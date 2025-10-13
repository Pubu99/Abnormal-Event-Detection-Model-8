"""
Metrics calculation for anomaly detection evaluation.
Includes accuracy, precision, recall, F1, AUC-ROC, and confusion matrix.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple


class MetricsCalculator:
    """Calculate and track metrics for anomaly detection."""
    
    def __init__(self, num_classes: int, class_names: List[str]):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
            class_names: List of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.binary_predictions = []
        self.binary_targets = []
        self.binary_probabilities = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, 
               probs: torch.Tensor = None,
               binary_preds: torch.Tensor = None,
               binary_targets: torch.Tensor = None,
               binary_probs: torch.Tensor = None):
        """
        Update metrics with new batch.
        
        Args:
            preds: Predicted class labels (B,)
            targets: Ground truth class labels (B,)
            probs: Class probabilities (B, C)
            binary_preds: Binary anomaly predictions (B,)
            binary_targets: Binary anomaly targets (B,)
            binary_probs: Binary anomaly probabilities (B,)
        """
        # Convert to numpy
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if probs is not None and isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        
        # Multi-class metrics
        self.predictions.extend(preds.tolist())
        self.targets.extend(targets.tolist())
        if probs is not None:
            self.probabilities.extend(probs.tolist())
        
        # Binary anomaly metrics
        if binary_preds is not None:
            if isinstance(binary_preds, torch.Tensor):
                binary_preds = binary_preds.detach().cpu().numpy()
            self.binary_predictions.extend(binary_preds.tolist())
        
        if binary_targets is not None:
            if isinstance(binary_targets, torch.Tensor):
                binary_targets = binary_targets.detach().cpu().numpy()
            self.binary_targets.extend(binary_targets.tolist())
        
        if binary_probs is not None:
            if isinstance(binary_probs, torch.Tensor):
                binary_probs = binary_probs.detach().cpu().numpy()
            self.binary_probabilities.extend(binary_probs.tolist())
    
    def compute_multiclass_metrics(self) -> Dict[str, float]:
        """
        Compute multi-class classification metrics.
        
        Returns:
            Dictionary of metrics
        """
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {
            'accuracy': accuracy_score(targets, preds),
            'precision_macro': precision_score(targets, preds, average='macro', zero_division=0),
            'recall_macro': recall_score(targets, preds, average='macro', zero_division=0),
            'f1_macro': f1_score(targets, preds, average='macro', zero_division=0),
            'precision_weighted': precision_score(targets, preds, average='weighted', zero_division=0),
            'recall_weighted': recall_score(targets, preds, average='weighted', zero_division=0),
            'f1_weighted': f1_score(targets, preds, average='weighted', zero_division=0),
        }
        
        # Per-class metrics
        precision_per_class = precision_score(targets, preds, average=None, zero_division=0)
        recall_per_class = recall_score(targets, preds, average=None, zero_division=0)
        f1_per_class = f1_score(targets, preds, average=None, zero_division=0)
        
        for idx, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[idx]
            metrics[f'recall_{class_name}'] = recall_per_class[idx]
            metrics[f'f1_{class_name}'] = f1_per_class[idx]
        
        # AUC-ROC (if probabilities available)
        if len(self.probabilities) > 0:
            try:
                probs = np.array(self.probabilities)
                # One-vs-Rest AUC
                auc_scores = []
                for i in range(self.num_classes):
                    if len(np.unique(targets)) > 1:  # Need at least 2 classes
                        binary_targets = (targets == i).astype(int)
                        if len(np.unique(binary_targets)) > 1:
                            auc = roc_auc_score(binary_targets, probs[:, i])
                            auc_scores.append(auc)
                
                if len(auc_scores) > 0:
                    metrics['auc_roc_macro'] = np.mean(auc_scores)
            except Exception as e:
                print(f"Warning: Could not compute AUC-ROC: {e}")
        
        return metrics
    
    def compute_binary_metrics(self) -> Dict[str, float]:
        """
        Compute binary anomaly detection metrics.
        
        Returns:
            Dictionary of metrics
        """
        if len(self.binary_predictions) == 0 or len(self.binary_targets) == 0:
            return {}
        
        preds = np.array(self.binary_predictions)
        targets = np.array(self.binary_targets)
        
        metrics = {
            'binary_accuracy': accuracy_score(targets, preds),
            'binary_precision': precision_score(targets, preds, zero_division=0),
            'binary_recall': recall_score(targets, preds, zero_division=0),
            'binary_f1': f1_score(targets, preds, zero_division=0),
        }
        
        # AUC-ROC for binary classification
        if len(self.binary_probabilities) > 0:
            try:
                probs = np.array(self.binary_probabilities)
                if len(np.unique(targets)) > 1:
                    metrics['binary_auc_roc'] = roc_auc_score(targets, probs)
            except Exception as e:
                print(f"Warning: Could not compute binary AUC-ROC: {e}")
        
        return metrics
    
    def compute_confusion_matrix(self) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Returns:
            Confusion matrix (num_classes, num_classes)
        """
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        return confusion_matrix(targets, preds, labels=range(self.num_classes))
    
    def get_classification_report(self) -> str:
        """
        Get detailed classification report.
        
        Returns:
            Classification report string
        """
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        return classification_report(
            targets, preds,
            target_names=self.class_names,
            zero_division=0
        )
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Multi-class metrics
        if len(self.predictions) > 0:
            metrics.update(self.compute_multiclass_metrics())
        
        # Binary metrics
        binary_metrics = self.compute_binary_metrics()
        if binary_metrics:
            metrics.update(binary_metrics)
        
        return metrics
    
    def get_summary(self) -> str:
        """
        Get summary of metrics.
        
        Returns:
            Summary string
        """
        metrics = self.compute_all_metrics()
        
        summary = "\n" + "="*60 + "\n"
        summary += "METRICS SUMMARY\n"
        summary += "="*60 + "\n\n"
        
        # Main metrics
        summary += "Multi-Class Classification:\n"
        summary += f"  Accuracy:           {metrics.get('accuracy', 0):.4f}\n"
        summary += f"  Precision (macro):  {metrics.get('precision_macro', 0):.4f}\n"
        summary += f"  Recall (macro):     {metrics.get('recall_macro', 0):.4f}\n"
        summary += f"  F1-Score (macro):   {metrics.get('f1_macro', 0):.4f}\n"
        if 'auc_roc_macro' in metrics:
            summary += f"  AUC-ROC (macro):    {metrics.get('auc_roc_macro', 0):.4f}\n"
        
        summary += "\nBinary Anomaly Detection:\n"
        if 'binary_accuracy' in metrics:
            summary += f"  Accuracy:           {metrics.get('binary_accuracy', 0):.4f}\n"
            summary += f"  Precision:          {metrics.get('binary_precision', 0):.4f}\n"
            summary += f"  Recall:             {metrics.get('binary_recall', 0):.4f}\n"
            summary += f"  F1-Score:           {metrics.get('binary_f1', 0):.4f}\n"
            if 'binary_auc_roc' in metrics:
                summary += f"  AUC-ROC:            {metrics.get('binary_auc_roc', 0):.4f}\n"
        else:
            summary += "  No binary metrics available\n"
        
        summary += "\n" + "="*60 + "\n"
        
        return summary


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray,
                     probabilities: np.ndarray = None,
                     class_names: List[str] = None) -> Dict[str, float]:
    """
    Calculate metrics from predictions and targets.
    
    Args:
        predictions: Predicted labels
        targets: Ground truth labels
        probabilities: Class probabilities (optional)
        class_names: List of class names (optional)
        
    Returns:
        Dictionary of metrics
    """
    num_classes = len(np.unique(targets))
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    calculator = MetricsCalculator(num_classes, class_names)
    calculator.update(predictions, targets, probabilities)
    
    return calculator.compute_all_metrics()


if __name__ == "__main__":
    # Test metrics
    num_classes = 14
    class_names = [f"Class_{i}" for i in range(num_classes)]
    
    # Create dummy data
    np.random.seed(42)
    predictions = np.random.randint(0, num_classes, 1000)
    targets = np.random.randint(0, num_classes, 1000)
    probabilities = np.random.rand(1000, num_classes)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    
    binary_preds = (predictions != 7).astype(int)
    binary_targets = (targets != 7).astype(int)
    binary_probs = np.random.rand(1000)
    
    # Calculate metrics
    calculator = MetricsCalculator(num_classes, class_names)
    calculator.update(
        predictions, targets, probabilities,
        binary_preds, binary_targets, binary_probs
    )
    
    metrics = calculator.compute_all_metrics()
    print("✅ Metrics calculated successfully!")
    print(calculator.get_summary())
