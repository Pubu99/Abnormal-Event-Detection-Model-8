"""
Test-Time Augmentation (TTA) for improved inference robustness.
Applies multiple augmentations at test time and aggregates predictions.
"""

import torch
import torch.nn.functional as F
from typing import List, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TestTimeAugmentation:
    """
    Test-Time Augmentation wrapper.
    Applies multiple augmentations and aggregates predictions.
    """
    
    def __init__(self, model, augmentations: List[Callable] = None, aggregation='mean'):
        """
        Initialize TTA.
        
        Args:
            model: PyTorch model
            augmentations: List of augmentation transforms
            aggregation: How to aggregate predictions ('mean', 'max', 'voting')
        """
        self.model = model
        self.augmentations = augmentations or self._default_augmentations()
        self.aggregation = aggregation
    
    def _default_augmentations(self):
        """Default TTA augmentations."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        return [
            # Original
            A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()]),
            
            # Horizontal flip
            A.Compose([
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ]),
            
            # Vertical flip
            A.Compose([
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ]),
            
            # Rotation 90
            A.Compose([
                A.Rotate(limit=90, p=1.0),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ]),
            
            # Brightness adjustment
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ]),
        ]
    
    @torch.no_grad()
    def __call__(self, image, return_all=False):
        """
        Apply TTA to a single image.
        
        Args:
            image: PIL Image or numpy array (H, W, C)
            return_all: If True, return all predictions
            
        Returns:
            Aggregated prediction or list of all predictions
        """
        predictions = []
        
        for aug in self.augmentations:
            # Apply augmentation
            if hasattr(image, 'numpy'):
                img_np = image.numpy()
            else:
                img_np = image
            
            augmented = aug(image=img_np)['image']
            
            # Add batch dimension
            if len(augmented.shape) == 3:
                augmented = augmented.unsqueeze(0)
            
            # Move to same device as model
            device = next(self.model.parameters()).device
            augmented = augmented.to(device)
            
            # Get prediction
            output = self.model(augmented)
            predictions.append(output)
        
        if return_all:
            return predictions
        
        # Aggregate predictions
        predictions = torch.stack(predictions)
        
        if self.aggregation == 'mean':
            return predictions.mean(dim=0)
        elif self.aggregation == 'max':
            return predictions.max(dim=0)[0]
        elif self.aggregation == 'voting':
            # For classification: majority voting
            preds_class = predictions.argmax(dim=-1)
            return torch.mode(preds_class, dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
    
    @torch.no_grad()
    def predict_batch(self, images, return_all=False):
        """
        Apply TTA to a batch of images.
        
        Args:
            images: Tensor of shape (B, C, H, W)
            return_all: If True, return all predictions
            
        Returns:
            Aggregated predictions for the batch
        """
        batch_predictions = []
        
        for img in images:
            pred = self(img.cpu().numpy().transpose(1, 2, 0), return_all=return_all)
            batch_predictions.append(pred)
        
        if return_all:
            return batch_predictions
        
        return torch.stack(batch_predictions)


def create_tta_model(model, config):
    """
    Create TTA-wrapped model based on configuration.
    
    Args:
        model: PyTorch model
        config: Configuration object
        
    Returns:
        TTA-wrapped model
    """
    tta_config = config.inference.get('tta', {})
    
    if tta_config.get('enabled', False):
        aggregation = tta_config.get('aggregation', 'mean')
        return TestTimeAugmentation(model, aggregation=aggregation)
    
    return model
