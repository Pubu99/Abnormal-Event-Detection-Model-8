"""
Configuration loader and manager for the anomaly detection system.
Handles YAML config files with validation and environment variable substitution.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from omegaconf import OmegaConf, DictConfig
import torch


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML config file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        self._setup_paths()
    
    def _load_config(self) -> DictConfig:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        # Load with OmegaConf for advanced features
        config = OmegaConf.load(self.config_path)
        
        # Resolve any interpolations
        OmegaConf.resolve(config)
        
        return config
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate required sections
        required_sections = ['project', 'data', 'model', 'training']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate data paths
        if not Path(self.config.data.train_dir).exists():
            raise FileNotFoundError(f"Training directory not found: {self.config.data.train_dir}")
        
        if not Path(self.config.data.test_dir).exists():
            raise FileNotFoundError(f"Test directory not found: {self.config.data.test_dir}")
        
        # Validate device
        if self.config.hardware.device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA not available, switching to CPU")
            self.config.hardware.device = 'cpu'
    
    def _setup_paths(self):
        """Create necessary directories."""
        paths_to_create = [
            self.config.data.processed_dir,
            self.config.logging.save_dir,
            "outputs/models",
            "outputs/results",
        ]
        
        for path in paths_to_create:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'model.backbone.type')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            return OmegaConf.select(self.config, key, default=default)
        except:
            return default
    
    def update(self, key: str, value: Any):
        """
        Update configuration value.
        
        Args:
            key: Configuration key in dot notation
            value: New value
        """
        OmegaConf.update(self.config, key, value)
    
    def save(self, output_path: str):
        """Save current configuration to file."""
        OmegaConf.save(self.config, output_path)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return OmegaConf.to_container(self.config, resolve=True)
    
    @property
    def device(self) -> torch.device:
        """Get PyTorch device."""
        return torch.device(self.config.hardware.device)
    
    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return self.config.data.num_classes
    
    @property
    def class_names(self) -> list:
        """Get class names."""
        return self.config.data.classes
    
    @property
    def normal_class_idx(self) -> int:
        """Get normal class index."""
        return self.config.data.normal_class_idx
    
    def __repr__(self) -> str:
        return f"ConfigManager(config_path='{self.config_path}')"


def load_config(config_path: Optional[str] = None) -> DictConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration object
    """
    manager = ConfigManager(config_path)
    return manager.config


if __name__ == "__main__":
    # Test configuration loading
    config_manager = ConfigManager()
    print("✅ Configuration loaded successfully!")
    print(f"Project: {config_manager.config.project.name}")
    print(f"Device: {config_manager.device}")
    print(f"Classes: {config_manager.num_classes}")
    print(f"Batch size: {config_manager.config.training.batch_size}")
