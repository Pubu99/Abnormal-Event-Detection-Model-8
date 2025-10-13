"""
Loss functions for anomaly detection.
Includes Focal Loss, Deep SVDD loss, and combined losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Lin et al. "Focal Loss for Dense Object Detection"
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', label_smoothing=0.0):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for class imbalance
            gamma: Focusing parameter (higher gamma focuses more on hard examples)
            reduction: 'mean' or 'sum'
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs: Predictions (B, C)
            targets: Ground truth labels (B,)
            
        Returns:
            Loss value
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            num_classes = inputs.size(1)
            targets_one_hot = F.one_hot(targets, num_classes).float()
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                            self.label_smoothing / num_classes
            
            # Compute cross entropy with smoothed labels
            log_probs = F.log_softmax(inputs, dim=1)
            ce_loss = -(targets_one_hot * log_probs).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal term
        focal_term = (1 - target_probs) ** self.gamma
        
        # Compute focal loss
        loss = self.alpha * focal_term * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DeepSVDDLoss(nn.Module):
    """Deep Support Vector Data Description loss for anomaly detection."""
    
    def __init__(self, nu=0.1, reduction='mean'):
        """
        Initialize Deep SVDD loss.
        
        Args:
            nu: Outlier fraction (controls sphere volume)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.nu = nu
        self.reduction = reduction
        self.R = None  # Radius (learned)
    
    def forward(self, embeddings, center, is_anomaly=None):
        """
        Compute Deep SVDD loss.
        
        Args:
            embeddings: Embedded features (B, D)
            center: Center vector (D,)
            is_anomaly: Binary anomaly labels (B,) - if None, uses soft boundary
            
        Returns:
            Loss value
        """
        # Compute distances from center
        distances = torch.sum((embeddings - center) ** 2, dim=1)
        
        if is_anomaly is not None:
            # Supervised SVDD: push anomalies outside, normals inside
            normal_mask = (is_anomaly == 0)
            anomaly_mask = (is_anomaly == 1)
            
            # Loss for normal samples (minimize distance)
            normal_loss = distances[normal_mask].mean() if normal_mask.sum() > 0 else 0
            
            # Loss for anomalies (maximize distance, or use hinge loss)
            if anomaly_mask.sum() > 0:
                # Hinge loss: max(0, margin - distance)
                margin = distances[normal_mask].mean() + 1.0 if normal_mask.sum() > 0 else 1.0
                anomaly_loss = F.relu(margin - distances[anomaly_mask]).mean()
                loss = normal_loss + anomaly_loss
            else:
                loss = normal_loss
        else:
            # Unsupervised SVDD: minimize distances for all
            loss = distances.mean()
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task learning:
    - Multi-class classification (Focal Loss)
    - Binary anomaly detection
    - Deep SVDD for embedding learning
    """
    
    def __init__(self, config, class_weights=None):
        """
        Initialize combined loss.
        
        Args:
            config: Configuration object
            class_weights: Class weights for imbalanced data
        """
        super().__init__()
        
        self.config = config
        
        # Multi-class classification loss
        if config.training.loss.type == 'focal':
            self.classification_loss = FocalLoss(
                alpha=config.training.loss.alpha,
                gamma=config.training.loss.gamma,
                label_smoothing=config.training.loss.label_smoothing
            )
        else:
            self.classification_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=config.training.loss.label_smoothing
            )
        
        # Binary anomaly detection loss
        self.binary_loss = nn.CrossEntropyLoss()
        
        # Deep SVDD loss
        self.svdd_loss = DeepSVDDLoss(
            nu=config.model.anomaly_head.nu
        )
        
        # Loss weights
        self.weight_classification = 1.0
        self.weight_binary = 0.5
        self.weight_svdd = 0.3
    
    def forward(self, outputs, targets, is_anomaly, center):
        """
        Compute combined loss.
        
        Args:
            outputs: Model outputs dictionary
            targets: Class labels (B,)
            is_anomaly: Binary anomaly labels (B,)
            center: SVDD center vector
            
        Returns:
            Dictionary with individual and total losses
        """
        # Multi-class classification loss
        class_loss = self.classification_loss(outputs['class_logits'], targets)
        
        # Binary anomaly detection loss
        binary_loss = self.binary_loss(outputs['binary_logits'], is_anomaly)
        
        # Deep SVDD loss (if embeddings available)
        if 'embeddings' in outputs:
            svdd_loss = self.svdd_loss(outputs['embeddings'], center, is_anomaly)
        else:
            svdd_loss = torch.tensor(0.0, device=outputs['class_logits'].device)
        
        # Total loss
        total_loss = (
            self.weight_classification * class_loss +
            self.weight_binary * binary_loss +
            self.weight_svdd * svdd_loss
        )
        
        return {
            'total': total_loss,
            'classification': class_loss,
            'binary': binary_loss,
            'svdd': svdd_loss
        }


class ContrastiveLoss(nn.Module):
    """Contrastive loss for metric learning."""
    
    def __init__(self, margin=1.0, reduction='mean'):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for dissimilar pairs
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, embeddings1, embeddings2, labels):
        """
        Compute contrastive loss.
        
        Args:
            embeddings1: First set of embeddings (B, D)
            embeddings2: Second set of embeddings (B, D)
            labels: Binary similarity labels (1 = similar, 0 = dissimilar)
            
        Returns:
            Loss value
        """
        # Euclidean distance
        distances = F.pairwise_distance(embeddings1, embeddings2)
        
        # Contrastive loss
        loss_similar = labels * distances ** 2
        loss_dissimilar = (1 - labels) * F.relu(self.margin - distances) ** 2
        
        loss = loss_similar + loss_dissimilar
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def create_loss_function(config, class_weights=None):
    """
    Create loss function based on configuration.
    
    Args:
        config: Configuration object
        class_weights: Optional class weights
        
    Returns:
        Loss function
    """
    return CombinedLoss(config, class_weights)


if __name__ == "__main__":
    # Test losses
    from src.utils import ConfigManager
    
    config = ConfigManager().config
    
    # Create dummy data
    batch_size = 32
    num_classes = 14
    embedding_dim = 128
    
    class_logits = torch.randn(batch_size, num_classes)
    binary_logits = torch.randn(batch_size, 2)
    embeddings = torch.randn(batch_size, embedding_dim)
    
    targets = torch.randint(0, num_classes, (batch_size,))
    is_anomaly = (targets != 7).long()  # Class 7 is normal
    center = torch.randn(embedding_dim)
    
    outputs = {
        'class_logits': class_logits,
        'binary_logits': binary_logits,
        'embeddings': embeddings
    }
    
    # Test combined loss
    loss_fn = create_loss_function(config)
    losses = loss_fn(outputs, targets, is_anomaly, center)
    
    print("✅ Loss functions tested successfully!")
    print(f"   Total loss: {losses['total']:.4f}")
    print(f"   Classification loss: {losses['classification']:.4f}")
    print(f"   Binary loss: {losses['binary']:.4f}")
    print(f"   SVDD loss: {losses['svdd']:.4f}")
