"""
Main training script for anomaly detection model.
Run this script to start training.

Usage:
    python train.py
    python train.py --config configs/config.yaml
    python train.py --resume outputs/logs/experiment_name/checkpoints/checkpoint.pth
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training import AnomalyDetectionTrainer
from src.utils import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train anomaly detection model for surveillance systems"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use for training (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print("="*70)
    print(" 🚀 MULTI-CAMERA ANOMALY DETECTION TRAINING")
    print("="*70)
    
    # Create trainer
    trainer = AnomalyDetectionTrainer(config_path=args.config)
    
    # Override config with command line arguments
    if args.seed is not None:
        trainer.config.project.seed = args.seed
        set_seed(args.seed)
    
    if args.device is not None:
        trainer.config.hardware.device = args.device
        trainer.device = torch.device(args.device)
    
    if args.batch_size is not None:
        trainer.config.training.batch_size = args.batch_size
        print(f"   Overriding batch size: {args.batch_size}")
    
    if args.epochs is not None:
        trainer.config.training.epochs = args.epochs
        print(f"   Overriding epochs: {args.epochs}")
    
    if args.wandb:
        trainer.config.logging.wandb.enabled = True
        print(f"   Enabling W&B logging")
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    try:
        trainer.train()
        print("\n✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted by user")
        # Save checkpoint on interruption
        trainer.save_checkpoint(is_best=False, metrics={})
        print("   Checkpoint saved")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import torch  # Import here to avoid circular import
    main()
