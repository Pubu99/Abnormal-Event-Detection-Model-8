"""
Two-Stream I3D Architecture for Weakly-Supervised Video Anomaly Detection.
Implements:
- RGB Stream: Spatial features from video frames
- Optical Flow Stream: Temporal motion features  
- Multiple Instance Learning (MIL): Video-level weak supervision
- 3D Convolutions: Spatio-temporal feature learning

Based on SOTA approach achieving 86% AUC on UCF Crime dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, mc3_18
import timm
import cv2
import numpy as np


class OpticalFlowExtractor:
    """
    Extract dense optical flow using Farneback algorithm.
    Used to compute motion features between consecutive frames.
    """
    
    def __init__(self):
        self.params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    
    def extract(self, frame1, frame2):
        """
        Extract optical flow between two frames.
        
        Args:
            frame1: First frame (H, W, 3) in RGB
            frame2: Second frame (H, W, 3) in RGB
            
        Returns:
            flow: Optical flow (H, W, 2) - x and y components
        """
        # Convert to grayscale
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = frame1
            gray2 = frame2
        
        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, **self.params
        )
        
        return flow
    
    def batch_extract(self, frames):
        """
        Extract optical flow from sequence of frames.
        
        Args:
            frames: Tensor (T, H, W, 3) or numpy array
            
        Returns:
            flows: Tensor (T-1, H, W, 2) - flow between consecutive frames
        """
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
        
        flows = []
        for i in range(len(frames) - 1):
            flow = self.extract(frames[i], frames[i + 1])
            flows.append(flow)
        
        return np.array(flows)


class I3DStream(nn.Module):
    """
    Single stream of I3D (Inflated 3D ConvNet).
    Uses 3D convolutions to learn spatio-temporal features.
    """
    
    def __init__(self, input_channels=3, pretrained=True, freeze_bn=False):
        """
        Initialize I3D stream.
        
        Args:
            input_channels: Number of input channels (3 for RGB, 2 for flow)
            pretrained: Use pretrained weights (only for RGB stream)
            freeze_bn: Freeze batch normalization layers
        """
        super().__init__()
        
        # Use ResNet3D (R3D) as base architecture
        if pretrained and input_channels == 3:
            self.backbone = r3d_18(pretrained=True)
        else:
            self.backbone = r3d_18(pretrained=False)
        
        # Modify first conv layer if needed (for optical flow)
        if input_channels != 3:
            self.backbone.stem[0] = nn.Conv3d(
                input_channels, 64, 
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2), 
                padding=(1, 3, 3), 
                bias=False
            )
        
        # Remove final FC layer
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Freeze batch norm if specified
        if freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, T, H, W) - batch, channels, time, height, width
            
        Returns:
            features: (B, feature_dim)
        """
        features = self.backbone(x)  # (B, 512)
        return features


class TwoStreamI3D(nn.Module):
    """
    Two-Stream Inflated 3D ConvNet for Video Anomaly Detection.
    
    Architecture:
    - RGB Stream: Learns appearance features from raw frames
    - Flow Stream: Learns motion features from optical flow
    - Fusion: Concatenates features from both streams
    - MIL: Treats video as bag of segments for weak supervision
    """
    
    def __init__(self, config):
        """
        Initialize Two-Stream I3D.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        self.num_classes = config.data.num_classes
        
        # RGB Stream (spatial appearance)
        self.rgb_stream = I3DStream(
            input_channels=3,
            pretrained=config.model.backbone.pretrained,
            freeze_bn=False
        )
        
        # Optical Flow Stream (temporal motion)
        self.flow_stream = I3DStream(
            input_channels=2,  # x and y flow components
            pretrained=False,
            freeze_bn=False
        )
        
        # Optical flow extractor
        self.flow_extractor = OpticalFlowExtractor()
        
        # Feature fusion
        combined_dim = self.rgb_stream.feature_dim + self.flow_stream.feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        
        # Multi-class classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, self.num_classes)
        )
        
        # Binary anomaly detection head (for MIL)
        self.anomaly_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Anomaly score per segment
        )
        
        # Segment-level prediction head
        self.segment_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for fusion and classification layers."""
        for m in [self.fusion, self.classifier, self.anomaly_head, self.segment_head]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm1d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
    
    def extract_optical_flow(self, rgb_clips):
        """
        Extract optical flow from RGB clips.
        
        Args:
            rgb_clips: (B, T, C, H, W) RGB frames
            
        Returns:
            flow_clips: (B, 2, T-1, H, W) Optical flow (x, y components)
        """
        batch_size, seq_len, c, h, w = rgb_clips.shape
        device = rgb_clips.device
        
        # Convert to numpy for OpenCV
        rgb_np = rgb_clips.cpu().numpy()
        
        # Denormalize if needed (assuming normalized to [0, 1])
        rgb_np = (rgb_np * 255).astype(np.uint8)
        
        all_flows = []
        for b in range(batch_size):
            flows = self.flow_extractor.batch_extract(rgb_np[b])  # (T-1, H, W, 2)
            
            # Normalize flow to [-1, 1]
            flow_max = np.abs(flows).max() + 1e-5
            flows = flows / flow_max
            
            # Transpose to (2, T-1, H, W)
            flows = flows.transpose(3, 0, 1, 2)  # (2, T-1, H, W)
            all_flows.append(flows)
        
        # Stack and convert to tensor
        flow_clips = torch.from_numpy(np.array(all_flows)).float().to(device)
        
        return flow_clips  # (B, 2, T-1, H, W)
    
    def forward(self, x, extract_flow=True, return_segments=False):
        """
        Forward pass through two-stream network.
        
        Args:
            x: Input RGB clips (B, T, C, H, W) or (B, C, T, H, W)
            extract_flow: Whether to extract optical flow (True for training)
            return_segments: Return per-segment scores for MIL
            
        Returns:
            Dictionary with predictions
        """
        # Ensure format is (B, T, C, H, W)
        if x.shape[2] == 3:  # (B, C, T, H, W)
            x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        
        batch_size, seq_len, c, h, w = x.shape
        
        # Extract optical flow
        if extract_flow:
            flow_clips = self.extract_optical_flow(x)  # (B, 2, T-1, H, W)
        else:
            # Use dummy flow if pre-extracted
            flow_clips = torch.zeros(batch_size, 2, seq_len-1, h, w, device=x.device)
        
        # Prepare RGB for 3D conv: (B, C, T, H, W)
        rgb_input = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        
        # RGB stream features
        rgb_features = self.rgb_stream(rgb_input)  # (B, 512)
        
        # Flow stream features (if we have flow)
        if flow_clips.shape[2] > 0:
            flow_features = self.flow_stream(flow_clips)  # (B, 512)
        else:
            flow_features = torch.zeros_like(rgb_features)
        
        # Fuse features
        combined = torch.cat([rgb_features, flow_features], dim=1)  # (B, 1024)
        fused_features = self.fusion(combined)  # (B, 256)
        
        # Classification
        class_logits = self.classifier(fused_features)  # (B, num_classes)
        
        # Anomaly scores (for MIL - per video score)
        anomaly_scores = self.anomaly_head(fused_features).squeeze(-1)  # (B,)
        
        # Segment-level scores (for localization)
        segment_scores = self.segment_head(fused_features).squeeze(-1)  # (B,)
        
        output = {
            'class_logits': class_logits,
            'anomaly_scores': anomaly_scores,
            'segment_scores': segment_scores,
            'features': fused_features,
            'binary_logits': torch.stack([1 - torch.sigmoid(anomaly_scores), 
                                          torch.sigmoid(anomaly_scores)], dim=1)
        }
        
        return output
    
    def predict_anomaly(self, x, threshold=0.5):
        """
        Predict if video is anomalous.
        
        Args:
            x: Input video clips
            threshold: Anomaly threshold
            
        Returns:
            Dictionary with predictions
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            
            # Class predictions
            class_probs = F.softmax(output['class_logits'], dim=1)
            class_preds = torch.argmax(class_probs, dim=1)
            class_conf = torch.max(class_probs, dim=1)[0]
            
            # Anomaly predictions
            anomaly_probs = torch.sigmoid(output['anomaly_scores'])
            is_anomaly = anomaly_probs > threshold
            
            return {
                'class_predictions': class_preds,
                'class_confidence': class_conf,
                'is_anomaly': is_anomaly,
                'anomaly_probability': anomaly_probs,
                'anomaly_scores': output['anomaly_scores']
            }


# Keep compatibility with old architecture name
class HybridAnomalyDetector(TwoStreamI3D):
    """Alias for backward compatibility."""
    pass


def create_model(config, device='cuda'):
    
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
