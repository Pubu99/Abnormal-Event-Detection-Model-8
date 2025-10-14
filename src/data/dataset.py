"""
Dataset classes for UCF Crime anomaly detection.
Handles image loading, augmentation, and class imbalance.
Implements advanced augmentation strategies for better generalization.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Callable, List
import numpy as np
from PIL import Image
import random

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


class UCFCrimeDataset(Dataset):
    """UCF Crime dataset for anomaly detection."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        normal_class_idx: int = 7,
        return_paths: bool = False
    ):
        """
        Initialize UCF Crime dataset.
        
        Args:
            root_dir: Root directory containing Train/Test folders
            split: 'train' or 'test'
            transform: Albumentations transform pipeline
            normal_class_idx: Index of normal class (7 = NormalVideos)
            return_paths: Whether to return image paths
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.normal_class_idx = normal_class_idx
        self.return_paths = return_paths
        
        # Get class names from directory structure
        split_dir = self.root_dir / split.capitalize()
        self.classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.samples = self._load_samples()
        
        # Compute class distribution for weighting
        self.class_counts = self._compute_class_counts()
        
        print(f"📊 {split.upper()} Dataset:")
        print(f"   Total samples: {len(self.samples):,}")
        print(f"   Classes: {len(self.classes)}")
        print(f"   Normal class: {self.classes[normal_class_idx]}")
        self._print_class_distribution()
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and labels."""
        samples = []
        split_dir = self.root_dir / self.split.capitalize()
        
        for class_name in self.classes:
            class_dir = split_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            # Get all PNG images in class directory
            image_paths = list(class_dir.glob("*.png"))
            samples.extend([(str(img_path), class_idx) for img_path in image_paths])
        
        return samples
    
    def _compute_class_counts(self) -> np.ndarray:
        """Compute sample count for each class."""
        counts = np.zeros(len(self.classes), dtype=np.int64)
        for _, label in self.samples:
            counts[label] += 1
        return counts
    
    def _print_class_distribution(self):
        """Print class distribution."""
        print("\n   Class distribution:")
        for idx, class_name in enumerate(self.classes):
            count = self.class_counts[idx]
            percentage = (count / len(self.samples)) * 100
            marker = " ⭐" if idx == self.normal_class_idx else ""
            print(f"      {idx:2d}. {class_name:20s}: {count:8,} ({percentage:5.2f}%){marker}")
    
    def get_class_weights(self, method: str = 'inverse') -> torch.Tensor:
        """
        Compute class weights for weighted loss.
        
        Args:
            method: 'inverse' or 'effective'
            
        Returns:
            Tensor of class weights
        """
        from src.utils.helpers import compute_class_weights
        return compute_class_weights(self.class_counts.tolist(), method=method)
    
    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """
        Create weighted random sampler for balanced training.
        
        Returns:
            WeightedRandomSampler instance
        """
        # Compute sample weights
        class_weights = 1.0 / torch.FloatTensor(self.class_counts)
        sample_weights = torch.zeros(len(self.samples))
        
        for idx, (_, label) in enumerate(self.samples):
            sample_weights[idx] = class_weights[label]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        return sampler
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, label) or (image, label, path) if return_paths=True
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Default: convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Binary anomaly label (0 = normal, 1 = anomaly)
        is_anomaly = int(label != self.normal_class_idx)
        
        if self.return_paths:
            return image, label, is_anomaly, img_path
        
        return image, label, is_anomaly


def get_train_transforms(config) -> A.Compose:
    """
    Get training data augmentation pipeline with advanced techniques.
    
    Args:
        config: Configuration object
        
    Returns:
        Albumentations Compose transform
    """
    aug_config = config.augmentation.train
    
    transforms = [
        # Geometric transforms
        A.HorizontalFlip(p=aug_config.random_horizontal_flip),
        A.VerticalFlip(p=aug_config.random_vertical_flip),
        A.Rotate(limit=aug_config.random_rotation, p=0.5),
        A.ShiftScaleRotate(
            shift_limit=aug_config.random_affine.translate[0],
            scale_limit=0.1,
            rotate_limit=aug_config.random_affine.degrees,
            p=0.5
        ),
        
        # Color transforms
        A.ColorJitter(
            brightness=aug_config.color_jitter.brightness,
            contrast=aug_config.color_jitter.contrast,
            saturation=aug_config.color_jitter.saturation,
            hue=aug_config.color_jitter.hue,
            p=0.5
        ),
        
        # Noise and blur
        A.GaussianBlur(
            blur_limit=(3, aug_config.gaussian_blur.kernel_size),
            p=aug_config.gaussian_blur.p
        ),
        A.GaussNoise(
            var_limit=aug_config.gaussian_noise.var_limit,
            p=aug_config.gaussian_noise.p
        ),
        
        # Weather and lighting augmentation for robustness
        A.RandomBrightnessContrast(p=aug_config.random_brightness_contrast),
        A.RandomShadow(p=aug_config.random_shadow),
        A.RandomRain(p=aug_config.random_rain),
        A.RandomFog(p=aug_config.random_fog),
        
        # Advanced occlusion augmentation - NEW!
        # CoarseDropout (similar to Cutout) for occlusion robustness
        A.CoarseDropout(
            max_holes=aug_config.coarse_dropout.max_holes,
            max_height=aug_config.coarse_dropout.max_height,
            max_width=aug_config.coarse_dropout.max_width,
            fill_value=0,
            p=aug_config.coarse_dropout.p
        ) if aug_config.coarse_dropout.enabled else A.NoOp(),
        
        # Normalize to ImageNet stats
        A.Normalize(
            mean=aug_config.normalize.mean,
            std=aug_config.normalize.std
        ),
        
        # Convert to PyTorch tensor
        ToTensorV2()
    ]
    
    return A.Compose(transforms)


def get_val_transforms(config) -> A.Compose:
    """
    Get validation/test data transforms (no augmentation).
    
    Args:
        config: Configuration object
        
    Returns:
        Albumentations Compose transform
    """
    aug_config = config.augmentation.val
    
    transforms = [
        A.Normalize(
            mean=aug_config.normalize.mean,
            std=aug_config.normalize.std
        ),
        ToTensorV2()
    ]
    
    return A.Compose(transforms)


def mixup_data(x, y, alpha=0.2):
    """
    Apply Mixup augmentation.
    Mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
    
    Args:
        x: Input batch (B, C, H, W)
        y: Target labels (B,)
        alpha: Mixup interpolation strength
        
    Returns:
        Mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """
    Apply CutMix augmentation.
    CutMix: Regularization Strategy to Train Strong Classifiers (https://arxiv.org/abs/1905.04899)
    
    Args:
        x: Input batch (B, C, H, W)
        y: Target labels (B,)
        alpha: CutMix interpolation strength
        
    Returns:
        Mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Get image size
    _, _, H, W = x.size()
    
    # Generate random bounding box
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Uniform sampling of center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply cutmix
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    """Generate random bounding box for CutOut/CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class MixupCutmixWrapper:
    """
    Wrapper to apply Mixup/CutMix augmentation with configurable probability.
    """
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5):
        """
        Args:
            mixup_alpha: Mixup alpha parameter
            cutmix_alpha: CutMix alpha parameter
            prob: Probability of applying any mixing
            switch_prob: Probability of choosing mixup vs cutmix
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
    
    def __call__(self, x, y):
        """
        Apply mixup or cutmix with probability.
        
        Returns:
            mixed_x, y_a, y_b, lam
        """
        if random.random() > self.prob:
            # No mixing
            return x, y, y, 1.0
        
        if random.random() < self.switch_prob:
            # Apply Mixup
            return mixup_data(x, y, self.mixup_alpha)
        else:
            # Apply CutMix
            return cutmix_data(x, y, self.cutmix_alpha)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute loss for mixup/cutmix.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixing coefficient
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def create_dataloaders(config, use_weighted_sampling: bool = True):
    """
    Create train and validation dataloaders.
    
    Args:
        config: Configuration object
        use_weighted_sampling: Whether to use weighted sampling for training
        
    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset)
    """
    # Get transforms
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)
    
    # Create full training dataset
    full_train_dataset = UCFCrimeDataset(
        root_dir=config.data.root_dir,
        split='train',
        transform=train_transform,
        normal_class_idx=config.data.normal_class_idx
    )
    
    # Split into train and validation
    val_size = int(len(full_train_dataset) * config.validation.split_ratio)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset_temp = torch.utils.data.random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.project.seed)
    )
    
    # Create validation dataset with validation transforms
    val_dataset = UCFCrimeDataset(
        root_dir=config.data.root_dir,
        split='train',  # Use train data but with val transform
        transform=val_transform,
        normal_class_idx=config.data.normal_class_idx
    )
    # Use same indices as val split
    val_dataset.samples = [full_train_dataset.samples[i] for i in val_dataset_temp.indices]
    
    # Create samplers and loaders
    if use_weighted_sampling and config.training.class_balance.method in ['weighted_sampling', 'both']:
        # ENHANCED: Support for minority class oversampling
        train_indices = train_dataset.indices
        
        # Compute class counts for train split only
        class_counts_train = np.zeros(len(full_train_dataset.classes))
        for idx in train_indices:
            _, label = full_train_dataset.samples[idx]
            class_counts_train[label] += 1
        
        # Determine minority classes
        if config.training.class_balance.oversample_minority:
            minority_threshold = config.training.class_balance.get('minority_threshold', 1000)
            oversample_ratio = config.training.class_balance.get('oversample_ratio', 2.0)
            
            # Oversample minority classes
            oversampled_indices = list(train_indices)
            for idx in train_indices:
                _, label = full_train_dataset.samples[idx]
                if class_counts_train[label] < minority_threshold:
                    # Add this sample multiple times
                    for _ in range(int(oversample_ratio) - 1):
                        oversampled_indices.append(idx)
            
            print(f"\n🔄 Minority Class Oversampling:")
            print(f"   Original samples: {len(train_indices):,}")
            print(f"   After oversampling: {len(oversampled_indices):,}")
            print(f"   Oversampling ratio: {len(oversampled_indices)/len(train_indices):.2f}x")
            train_indices = oversampled_indices
        
        # Compute sample weights (inverse frequency)
        class_weights = 1.0 / (torch.FloatTensor(class_counts_train) + 1e-6)
        
        # Apply class weight scaling if configured
        if hasattr(config.training.loss, 'class_weights_scale'):
            class_weights = class_weights * config.training.loss.class_weights_scale
        
        sample_weights = torch.zeros(len(full_train_dataset))
        for idx in train_indices:
            _, label = full_train_dataset.samples[idx]
            sample_weights[idx] = class_weights[label]
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights[train_indices],
            num_samples=len(train_indices),
            replacement=True
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        shuffle=shuffle if train_sampler is None else False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor,
        persistent_workers=True if config.data.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.validation.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor,
        persistent_workers=True if config.data.num_workers > 0 else False
    )
    
    print(f"\n📦 DataLoaders created:")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Weighted sampling: {use_weighted_sampling}")
    
    return train_loader, val_loader, full_train_dataset, val_dataset


def create_test_dataloader(config):
    """
    Create test dataloader.
    
    Args:
        config: Configuration object
        
    Returns:
        Test DataLoader and dataset
    """
    test_transform = get_val_transforms(config)
    
    test_dataset = UCFCrimeDataset(
        root_dir=config.data.root_dir,
        split='test',
        transform=test_transform,
        normal_class_idx=config.data.normal_class_idx,
        return_paths=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.validation.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    print(f"\n🧪 Test DataLoader created:")
    print(f"   Test batches: {len(test_loader)}")
    
    return test_loader, test_dataset


if __name__ == "__main__":
    # Test dataset loading
    from src.utils import ConfigManager
    
    config = ConfigManager().config
    train_loader, val_loader, train_ds, val_ds = create_dataloaders(config)
    
    # Test one batch
    images, labels, is_anomaly = next(iter(train_loader))
    print(f"\n✅ Batch loaded successfully!")
    print(f"   Images shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Is anomaly shape: {is_anomaly.shape}")
