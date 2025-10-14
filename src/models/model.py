"""
Hybrid Anomaly Detection Model Architecture.
Combines CNN backbone + Temporal modeling + Anomaly detection head.
Optimized for real-time performance with high accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm


class EfficientNetBackbone(nn.Module):
    """EfficientNet backbone for feature extraction."""
    
    def __init__(self, model_name: str = 'efficientnet_b0', pretrained: bool = True, 
                 freeze_backbone: bool = False, dropout: float = 0.3):
        """
        Initialize EfficientNet backbone.
        
        Args:
            model_name: EfficientNet variant name
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone weights
            dropout: Dropout rate
        """
        super().__init__()
        
        # Load EfficientNet from timm (more efficient than torchvision)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 64, 64)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
            self.spatial_dim = features.shape[2]  # Should be 2x2 for 64x64 input
        
        # Adaptive pooling to ensure consistent output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Features (B, feature_dim)
        """
        features = self.backbone(x)  # (B, feature_dim, H', W')
        features = self.adaptive_pool(features)  # (B, feature_dim, 1, 1)
        features = features.flatten(1)  # (B, feature_dim)
        features = self.dropout(features)
        return features


class TemporalEncoder(nn.Module):
    """Temporal encoder using LSTM/GRU for sequential modeling."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.3, encoder_type: str = 'lstm'):
        """
        Initialize temporal encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM/GRU layers
            bidirectional: Whether to use bidirectional LSTM/GRU
            dropout: Dropout rate
            encoder_type: 'lstm' or 'gru'
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.encoder_type = encoder_type
        
        # Choose encoder type
        if encoder_type == 'lstm':
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        elif encoder_type == 'gru':
            self.rnn = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Output dimension
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, lengths=None):
        """
        Forward pass with attention.
        
        Args:
            x: Input tensor (B, T, D) or (B, D) for single frame
            lengths: Sequence lengths (optional)
            
        Returns:
            Encoded features (B, output_dim)
        """
        # Handle single frame input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)
        
        # RNN encoding
        if lengths is not None:
            # Pack padded sequence for efficiency
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            rnn_out, _ = self.rnn(x)
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        else:
            rnn_out, _ = self.rnn(x)  # (B, T, hidden_dim * 2)
        
        # Attention weights
        attn_weights = self.attention(rnn_out)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(rnn_out * attn_weights, dim=1)  # (B, output_dim)
        
        return context


class DeepSVDDHead(nn.Module):
    """Deep SVDD head for anomaly detection."""
    
    def __init__(self, input_dim: int, embedding_dim: int = 128):
        """
        Initialize Deep SVDD head.
        
        Args:
            input_dim: Input feature dimension
            embedding_dim: Embedding space dimension
        """
        super().__init__()
        
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, embedding_dim),
        )
        
        # Center vector (learned during training)
        # CRITICAL FIX: Initialize to zeros instead of randn to prevent gradient explosion
        self.center = nn.Parameter(torch.zeros(embedding_dim), requires_grad=False)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features (B, input_dim)
            
        Returns:
            Embeddings (B, embedding_dim)
        """
        embeddings = self.embedding(x)
        return embeddings
    
    def compute_anomaly_score(self, embeddings):
        """
        Compute anomaly score as distance from center.
        
        Args:
            embeddings: Embedded features (B, embedding_dim)
            
        Returns:
            Anomaly scores (B,)
        """
        distances = torch.sum((embeddings - self.center) ** 2, dim=1)
        return distances


class HybridAnomalyDetector(nn.Module):
    """
    Hybrid anomaly detection model combining:
    - EfficientNet backbone for spatial features
    - LSTM/GRU for temporal modeling
    - Deep SVDD for anomaly detection
    - Classification head for multi-class prediction
    """
    
    def __init__(self, config):
        """
        Initialize hybrid anomaly detector.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        self.num_classes = config.data.num_classes
        
        # Spatial feature extractor (CNN backbone)
        self.backbone = EfficientNetBackbone(
            model_name=config.model.backbone.type,
            pretrained=config.model.backbone.pretrained,
            freeze_backbone=config.model.backbone.freeze_backbone,
            dropout=config.model.backbone.dropout
        )
        
        # Temporal encoder (optional, for video sequences)
        if config.model.temporal.type in ['lstm', 'gru']:
            self.temporal_encoder = TemporalEncoder(
                input_dim=self.backbone.feature_dim,
                hidden_dim=config.model.temporal.hidden_dim,
                num_layers=config.model.temporal.num_layers,
                bidirectional=config.model.temporal.bidirectional,
                dropout=config.model.temporal.dropout,
                encoder_type=config.model.temporal.type
            )
            feature_dim = self.temporal_encoder.output_dim
        else:
            self.temporal_encoder = None
            feature_dim = self.backbone.feature_dim
        
        # Projection head for better feature learning
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, config.model.feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.model.feature_dim),
            nn.Dropout(0.3)
        )
        
        # Multi-class classification head
        self.classifier = nn.Linear(config.model.feature_dim, self.num_classes)
        
        # Anomaly detection head (Deep SVDD)
        self.anomaly_head = DeepSVDDHead(
            input_dim=config.model.feature_dim,
            embedding_dim=config.model.embedding_dim
        )
        
        # Binary anomaly classifier (normal vs anomaly)
        self.binary_classifier = nn.Linear(config.model.feature_dim, 2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def extract_features(self, x):
        """
        Extract features from input.
        
        Args:
            x: Input tensor (B, C, H, W) or (B, T, C, H, W) for sequences
            
        Returns:
            Extracted features (B, feature_dim)
        """
        # Handle video sequences
        if x.dim() == 5:  # (B, T, C, H, W)
            batch_size, seq_len = x.shape[:2]
            # Reshape to (B*T, C, H, W)
            x = x.view(-1, *x.shape[2:])
            # Extract spatial features
            spatial_features = self.backbone(x)  # (B*T, backbone_dim)
            # Reshape back to (B, T, backbone_dim)
            spatial_features = spatial_features.view(batch_size, seq_len, -1)
            # Apply temporal encoding
            if self.temporal_encoder is not None:
                features = self.temporal_encoder(spatial_features)
            else:
                features = spatial_features.mean(dim=1)  # Average pooling
        else:
            # Single frame
            spatial_features = self.backbone(x)
            if self.temporal_encoder is not None:
                features = self.temporal_encoder(spatial_features)
            else:
                features = spatial_features
        
        # Project to common feature space
        features = self.projection(features)
        
        return features
    
    def forward(self, x, return_embeddings=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            return_embeddings: Whether to return embeddings
            
        Returns:
            Dictionary with predictions and embeddings
        """
        # Extract features
        features = self.extract_features(x)
        
        # Multi-class classification
        class_logits = self.classifier(features)
        
        # Binary anomaly classification
        binary_logits = self.binary_classifier(features)
        
        # Anomaly detection (Deep SVDD)
        embeddings = self.anomaly_head(features)
        anomaly_scores = self.anomaly_head.compute_anomaly_score(embeddings)
        
        output = {
            'class_logits': class_logits,
            'binary_logits': binary_logits,
            'anomaly_scores': anomaly_scores,
            'features': features
        }
        
        if return_embeddings:
            output['embeddings'] = embeddings
        
        return output
    
    def predict_anomaly(self, x, threshold=0.5):
        """
        Predict if input is anomalous.
        
        Args:
            x: Input tensor
            threshold: Anomaly score threshold
            
        Returns:
            Dictionary with predictions and confidence
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            
            # Get predictions
            class_probs = F.softmax(output['class_logits'], dim=1)
            class_preds = torch.argmax(class_probs, dim=1)
            class_conf = torch.max(class_probs, dim=1)[0]
            
            binary_probs = F.softmax(output['binary_logits'], dim=1)
            is_anomaly = binary_probs[:, 1] > threshold
            
            # Normalize anomaly scores
            anomaly_scores_norm = torch.sigmoid(output['anomaly_scores'])
            
            return {
                'class_predictions': class_preds,
                'class_confidence': class_conf,
                'is_anomaly': is_anomaly,
                'anomaly_probability': binary_probs[:, 1],
                'anomaly_scores': anomaly_scores_norm
            }


def create_model(config, device='cuda'):
    """
    Create and initialize model.
    
    Args:
        config: Configuration object
        device: Device to place model on
        
    Returns:
        Model instance
    """
    model = HybridAnomalyDetector(config)
    model = model.to(device)
    
    # Print model info
    from src.utils import count_parameters
    total_params, trainable_params = count_parameters(model)
    
    print(f"\n🤖 Model: {config.model.name}")
    print(f"   Backbone: {config.model.backbone.type}")
    print(f"   Temporal: {config.model.temporal.type}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024**2:.2f} MB")
    
    return model


if __name__ == "__main__":
    # Test model
    from src.utils import ConfigManager
    
    config = ConfigManager().config
    model = create_model(config, device='cpu')
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 64, 64)
    output = model(dummy_input)
    
    print(f"\n✅ Model forward pass successful!")
    print(f"   Class logits shape: {output['class_logits'].shape}")
    print(f"   Binary logits shape: {output['binary_logits'].shape}")
    print(f"   Anomaly scores shape: {output['anomaly_scores'].shape}")
