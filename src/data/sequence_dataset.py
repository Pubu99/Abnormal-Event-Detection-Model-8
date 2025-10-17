"""
Sequence Dataset for Research-Enhanced Model.
Loads temporal clips (sequences of frames) for regression training.

Features:
- Samples consecutive frames from videos
- Provides current frames + future frame targets for regression
- Handles both train/val splits
- Supports strong augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Optional
import random
from collections import defaultdict


class SequenceAnomalyDataset(Dataset):
    """
    Dataset that loads sequences of frames for temporal modeling.
    
    Each sample consists of:
    - Input: Sequence of T frames (e.g., 16 frames)
    - Target: Class label for the sequence
    - Future frames: For regression training (predict t+k from t)
    """
    
    def __init__(
        self,
        root_dir: str,
        classes: List[str],
        sequence_length: int = 16,
        frame_stride: int = 2,
        future_steps: int = 4,
        transform=None,
        is_train: bool = True
    ):
        """
        Initialize sequence dataset.
        
        Args:
            root_dir: Root directory containing class folders
            classes: List of class names
            sequence_length: Number of frames per sequence
            frame_stride: Stride when sampling frames (1=consecutive, 2=every other frame)
            future_steps: How many steps ahead to predict (for regression)
            transform: Augmentation transforms
            is_train: Whether this is training set
        """
        self.root_dir = Path(root_dir)
        self.classes = classes
        self.sequence_length = sequence_length
        self.frame_stride = frame_stride
        self.future_steps = future_steps
        self.transform = transform
        self.is_train = is_train
        
        # Build dataset: group frames by video
        self.video_sequences = self._build_video_sequences()
        
        # Create flat list of valid sequences
        self.sequences = self._create_sequences()
        
        print(f"\n📊 Sequence Dataset Statistics:")
        print(f"   Root: {root_dir}")
        print(f"   Sequence length: {sequence_length}")
        print(f"   Frame stride: {frame_stride}")
        print(f"   Future steps: {future_steps}")
        print(f"   Total sequences: {len(self.sequences):,}")
        print(f"   Videos: {len(self.video_sequences):,}")
    
    def _build_video_sequences(self) -> Dict[str, Dict]:
        """
        Group frames by video for temporal sampling.
        
        Returns:
            Dictionary mapping video_id to {frames, class_idx}
        """
        video_data = defaultdict(lambda: {'frames': [], 'class_idx': None})
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.root_dir / class_name
            
            if not class_dir.exists():
                continue
            
            # Get all frames in this class
            frame_paths = sorted(class_dir.glob('*.png'))
            
            # Group by video ID (extract from filename)
            for frame_path in frame_paths:
                # Filename format: Abuse001_x264_100.png
                # Extract video ID: Abuse001_x264
                filename = frame_path.stem
                parts = filename.rsplit('_', 1)
                
                if len(parts) == 2:
                    video_id = parts[0]  # e.g., "Abuse001_x264"
                    frame_num = int(parts[1])  # e.g., 100
                    
                    video_key = f"{class_name}_{video_id}"
                    video_data[video_key]['frames'].append({
                        'path': frame_path,
                        'frame_num': frame_num
                    })
                    video_data[video_key]['class_idx'] = class_idx
        
        # Sort frames within each video by frame number
        for video_key in video_data:
            video_data[video_key]['frames'].sort(key=lambda x: x['frame_num'])
        
        return dict(video_data)
    
    def _create_sequences(self) -> List[Dict]:
        """
        Create list of valid sequences from videos.
        
        Returns:
            List of sequence metadata (video_key, start_idx, class_idx)
        """
        sequences = []
        
        for video_key, video_info in self.video_sequences.items():
            frames = video_info['frames']
            class_idx = video_info['class_idx']
            
            # Calculate how many frames we need for sequence + future prediction
            frames_needed = self.sequence_length * self.frame_stride + self.future_steps
            
            # Create sequences with sliding window
            num_frames = len(frames)
            if num_frames < frames_needed:
                # Video too short, skip
                continue
            
            # Sliding window with stride
            window_stride = max(1, self.sequence_length // 4)  # 75% overlap
            
            for start_idx in range(0, num_frames - frames_needed + 1, window_stride):
                sequences.append({
                    'video_key': video_key,
                    'start_idx': start_idx,
                    'class_idx': class_idx
                })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Get a sequence sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of:
            - frames: Tensor (T, C, H, W) - sequence of frames
            - label: int - class label
            - future_frame: Tensor (C, H, W) - frame for regression target
        """
        seq_info = self.sequences[idx]
        video_key = seq_info['video_key']
        start_idx = seq_info['start_idx']
        class_idx = seq_info['class_idx']
        
        video_info = self.video_sequences[video_key]
        frames_info = video_info['frames']
        
        # Sample sequence frames
        frame_indices = [
            start_idx + i * self.frame_stride 
            for i in range(self.sequence_length)
        ]
        
        # Sample future frame for regression
        future_frame_idx = start_idx + self.sequence_length * self.frame_stride + self.future_steps - 1
        
        # Load frames
        frames = []
        for frame_idx in frame_indices:
            frame_path = frames_info[frame_idx]['path']
            frame = Image.open(frame_path).convert('RGB')
            frame = np.array(frame)  # Convert to numpy for Albumentations
            
            if self.transform:
                # Albumentations requires named arguments
                transformed = self.transform(image=frame)
                frame = transformed['image']
            
            frames.append(frame)
        
        # Load future frame
        future_frame_path = frames_info[future_frame_idx]['path']
        future_frame = Image.open(future_frame_path).convert('RGB')
        future_frame = np.array(future_frame)
        
        if self.transform:
            transformed = self.transform(image=future_frame)
            future_frame = transformed['image']
        
        # Stack frames into tensor (T, C, H, W)
        frames = torch.stack(frames, dim=0)
        
        return frames, class_idx, future_frame


def create_sequence_dataloaders(
    config,
    use_weighted_sampling: bool = True
):
    """
    Create sequence dataloaders for training and validation.
    
    Args:
        config: Configuration object
        use_weighted_sampling: Whether to use weighted sampling for class balance
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from src.data.dataset import get_train_transforms, get_val_transforms
    
    # Get transforms
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)
    
    # Sequence parameters
    sequence_length = config.data.preprocessing.get('sequence_length', 16)
    frame_stride = config.data.preprocessing.get('frame_stride', 2)
    future_steps = config.data.preprocessing.get('future_prediction_steps', 4)
    
    # Create full dataset
    train_dir = Path(config.data.train_dir)
    
    full_dataset = SequenceAnomalyDataset(
        root_dir=train_dir,
        classes=config.data.classes,
        sequence_length=sequence_length,
        frame_stride=frame_stride,
        future_steps=future_steps,
        transform=train_transform,
        is_train=True
    )
    
    # Split into train/val
    total_sequences = len(full_dataset)
    train_ratio = config.data.split.train_ratio
    train_size = int(total_sequences * train_ratio)
    val_size = total_sequences - train_size
    
    # Random split
    from torch.utils.data import random_split
    generator = torch.Generator().manual_seed(config.data.split.random_seed)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )
    
    # Update val dataset transform
    # Note: Can't directly change transform on Subset, so we'll handle in collate_fn
    
    print(f"\n📊 Sequence Dataset Split:")
    print(f"   Train sequences: {len(train_dataset):,}")
    print(f"   Val sequences: {len(val_dataset):,}")
    
    # Create dataloaders
    batch_size = config.training.batch_size
    num_workers = config.get('num_workers', 8)
    
    # Class weighting for sampling (if enabled)
    if use_weighted_sampling and config.training.class_balance.method in ['weighted_sampling', 'both']:
        # Compute class weights from training sequences
        train_indices = train_dataset.indices
        class_counts = np.zeros(len(config.data.classes))
        
        for idx in train_indices:
            seq_info = full_dataset.sequences[idx]
            class_idx = seq_info['class_idx']
            class_counts[class_idx] += 1
        
        # Compute sample weights
        class_weights = 1.0 / (torch.FloatTensor(class_counts) + 1e-6)
        sample_weights = torch.zeros(len(train_dataset))
        
        for i, idx in enumerate(train_indices):
            seq_info = full_dataset.sequences[idx]
            class_idx = seq_info['class_idx']
            sample_weights[i] = class_weights[class_idx]
        
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        print(f"   Using weighted sampling for class balance")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✅ Sequence dataloaders created")
    print(f"   Train batches: {len(train_loader):,}")
    print(f"   Val batches: {len(val_loader):,}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test sequence dataset
    import sys
    # Avoid hardcoded absolute paths; ensure the repository root is on sys.path when needed
    
    from src.utils.config import ConfigManager
    from torchvision import transforms
    
    config = ConfigManager('configs/config_research_enhanced.yaml').config
    
    # Simple transform for testing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SequenceAnomalyDataset(
        root_dir='data/raw/Train',
        classes=config.data.classes,
        sequence_length=16,
        frame_stride=2,
        future_steps=4,
        transform=transform,
        is_train=True
    )
    
    print(f"\n✅ Dataset created with {len(dataset):,} sequences")
    
    # Test loading
    frames, label, future_frame = dataset[0]
    print(f"\n📊 Sample loaded:")
    print(f"   Frames shape: {frames.shape}")
    print(f"   Label: {label} ({config.data.classes[label]})")
    print(f"   Future frame shape: {future_frame.shape}")
