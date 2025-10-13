"""
Variational Autoencoder (VAE) for Anomaly Detection.
Learns latent distribution of normal patterns for unsupervised anomaly detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for learning normal pattern distribution.
    Used as auxiliary task for unsupervised anomaly detection.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dims: list = None):
        """
        Initialize VAE.
        
        Args:
            input_dim: Input feature dimension from backbone
            latent_dim: Latent space dimension
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space: mean and log variance
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input features (B, input_dim)
            
        Returns:
            mu: Mean of latent distribution (B, latent_dim)
            logvar: Log variance of latent distribution (B, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        
        Args:
            mu: Mean (B, latent_dim)
            logvar: Log variance (B, latent_dim)
            
        Returns:
            z: Sampled latent vector (B, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent vector to reconstruction.
        
        Args:
            z: Latent vector (B, latent_dim)
            
        Returns:
            reconstruction: Reconstructed features (B, input_dim)
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass through VAE.
        
        Args:
            x: Input features (B, input_dim)
            
        Returns:
            Dictionary with reconstruction, mu, logvar, z
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def compute_loss(self, x, outputs, beta=1.0):
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            x: Original input (B, input_dim)
            outputs: Forward pass outputs
            beta: Weight for KL divergence (β-VAE)
            
        Returns:
            Dictionary with total loss and components
        """
        reconstruction = outputs['reconstruction']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x, reduction='mean')
        
        # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = kl_loss.mean()
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'vae_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def compute_anomaly_score(self, x):
        """
        Compute anomaly score based on reconstruction error.
        
        Args:
            x: Input features (B, input_dim)
            
        Returns:
            Anomaly scores (B,) - higher = more anomalous
        """
        with torch.no_grad():
            outputs = self.forward(x)
            reconstruction = outputs['reconstruction']
            
            # Reconstruction error per sample
            recon_error = F.mse_loss(reconstruction, x, reduction='none').mean(dim=1)
            
            # Can also add KL divergence for anomaly scoring
            mu = outputs['mu']
            logvar = outputs['logvar']
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            
            # Combined anomaly score
            anomaly_score = recon_error + 0.1 * kl_div
            
            return anomaly_score


class ConvolutionalVAE(nn.Module):
    """
    Convolutional VAE for image reconstruction.
    Works directly on images instead of features.
    """
    
    def __init__(self, in_channels: int = 3, latent_dim: int = 128, image_size: int = 64):
        """
        Initialize Convolutional VAE.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            latent_dim: Latent space dimension
            image_size: Input image size (assumes square)
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Encoder
        self.encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # Calculate flattened size
        self.flatten_size = 256 * 4 * 4
        
        # Latent space
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def encode(self, x):
        """Encode image to latent distribution."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to image."""
        h = self.decoder_input(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def compute_loss(self, x, outputs, beta=1.0):
        """Compute VAE loss for images."""
        reconstruction = outputs['reconstruction']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Reconstruction loss (MSE for images)
        recon_loss = F.mse_loss(reconstruction, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = kl_loss.mean()
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'vae_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def compute_anomaly_score(self, x):
        """Compute anomaly score for images."""
        with torch.no_grad():
            outputs = self.forward(x)
            reconstruction = outputs['reconstruction']
            
            # Pixel-wise reconstruction error
            recon_error = F.mse_loss(reconstruction, x, reduction='none')
            recon_error = recon_error.view(recon_error.size(0), -1).mean(dim=1)
            
            # KL divergence
            mu = outputs['mu']
            logvar = outputs['logvar']
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            
            # Combined score
            anomaly_score = recon_error + 0.1 * kl_div
            
            return anomaly_score
