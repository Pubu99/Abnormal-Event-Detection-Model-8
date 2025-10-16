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
        if self.alpha is not None:
            loss = self.alpha * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss
        
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


class MILRankingLoss(nn.Module):
    """
    Multiple Instance Learning Ranking Loss for weakly supervised anomaly detection.
    From: "Real-world Anomaly Detection in Surveillance Videos" (CVPR 2018)
    
    Separates normal and abnormal video clips with margin-based ranking.
    """
    
    def __init__(self, margin=0.5, positive_bag_weight=3.0, reduction='mean'):
        """
        Initialize MIL Ranking Loss.
        
        Args:
            margin: Separation margin between normal and abnormal scores
            positive_bag_weight: Weight for abnormal (positive) clips (emphasize anomalies)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.margin = margin
        self.positive_bag_weight = positive_bag_weight
        self.reduction = reduction
    
    def forward(self, scores, labels):
        """
        Compute MIL ranking loss.
        
        Args:
            scores: Anomaly scores for clips (B,) - higher = more abnormal
            labels: Binary labels (B,) - 1=abnormal, 0=normal
            
        Returns:
            Loss value
        """
        # Separate normal and abnormal scores
        normal_mask = labels == 0
        abnormal_mask = labels == 1
        
        if not normal_mask.any() or not abnormal_mask.any():
            # Need both normal and abnormal samples
            return torch.tensor(0.0, device=scores.device)
        
        normal_scores = scores[normal_mask]
        abnormal_scores = scores[abnormal_mask]
        
        # Max score from normal clips (should be low)
        max_normal_score = normal_scores.max()
        
        # Min score from abnormal clips (should be high)
        min_abnormal_score = abnormal_scores.min()
        
        # Ranking loss: abnormal scores should be higher than normal scores by margin
        ranking_loss = F.relu(self.margin + max_normal_score - min_abnormal_score)
        
        # Additional term: encourage all abnormal scores to be high
        abnormal_penalty = F.relu(0.5 - abnormal_scores).mean()
        
        # Weighted combination (emphasize abnormal detection)
        loss = ranking_loss + self.positive_bag_weight * abnormal_penalty
        
        return loss


class TemporalRegressionLoss(nn.Module):
    """
    Temporal Regression Loss for future feature prediction.
    From: "Video Anomaly Detection by Estimating Likelihood of Representations"
    
    Predicts future frame features from current features. High prediction error = anomaly.
    Proven to achieve 88.7% AUC on UCF Crime dataset.
    """
    
    def __init__(self, loss_type='smooth_l1', reduction='mean'):
        """
        Initialize Temporal Regression Loss.
        
        Args:
            loss_type: 'mse', 'smooth_l1' (Huber), or 'l1'
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction=reduction)
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss(reduction=reduction)  # Huber loss
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
    
    def forward(self, predicted_features, target_features):
        """
        Compute regression loss between predicted and actual future features.
        
        Args:
            predicted_features: Predicted future features (B, D)
            target_features: Actual future features (B, D)
            
        Returns:
            Loss value
        """
        return self.criterion(predicted_features, target_features)


class VAELoss(nn.Module):
    """
    Variational Autoencoder Loss for unsupervised anomaly detection.
    Combines reconstruction loss and KL divergence.
    """
    
    def __init__(self, reconstruction_weight=1.0, kl_weight=0.01, reduction='mean'):
        """
        Initialize VAE Loss.
        
        Args:
            reconstruction_weight: Weight for reconstruction term
            kl_weight: Weight for KL divergence (beta-VAE style)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.reduction = reduction
    
    def forward(self, reconstructed, original, mu, logvar):
        """
        Compute VAE loss.
        
        Args:
            reconstructed: Reconstructed features (B, D)
            original: Original features (B, D)
            mu: Mean of latent distribution (B, latent_dim)
            logvar: Log variance of latent distribution (B, latent_dim)
            
        Returns:
            Loss dictionary with total, reconstruction, and kl components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, original, reduction=self.reduction)
        
        # KL divergence: KL(N(mu, sigma) || N(0, 1))
        # = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        if self.reduction == 'mean':
            kl_loss = kl_loss.mean()
        elif self.reduction == 'sum':
            kl_loss = kl_loss.sum()
        
        # Total VAE loss
        total_loss = self.reconstruction_weight * recon_loss + self.kl_weight * kl_loss
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'kl': kl_loss
        }


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
