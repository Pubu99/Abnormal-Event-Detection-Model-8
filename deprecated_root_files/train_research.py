"""
Training Script for Research-Enhanced Model.

Trains the complete multi-task architecture:
- RNN Regression (88.7% AUC)
- Focal Loss (class imbalance)
- Transformer (long-range dependencies)
- VAE (unsupervised anomaly detection)
- MIL Ranking (weakly supervised)

Expected performance: 85-88% test accuracy on UCF Crime
"""

import sys
import argparse
from pathlib import Path
import torch
import wandb

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.utils.config import ConfigManager
from src.utils.logger import ExperimentLogger
from src.models.research_model import create_research_model
from src.data.sequence_dataset import create_sequence_dataloaders
from src.training.research_trainer import ResearchTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Research-Enhanced Model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_research_enhanced.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to train on'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable W&B logging'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    print("=" * 80)
    print("🚀 Research-Enhanced Anomaly Detection Training")
    print("=" * 80)
    
    # Load configuration
    print(f"\n📋 Loading configuration from {args.config}...")
    config = ConfigManager(args.config).config
    
    # Override wandb if disabled
    if args.no_wandb:
        config.logging.wandb.enabled = False
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    
    # Initialize W&B
    if config.logging.wandb.enabled:
        print(f"\n🔗 Initializing Weights & Biases...")
        wandb.init(
            project=config.logging.wandb.project,
            entity=config.logging.wandb.entity,
            config=dict(config),
            tags=config.logging.wandb.tags,
            name=f"research_enhanced_{config.project.version}"
        )
        print(f"   Project: {config.logging.wandb.project}")
        print(f"   Run: {wandb.run.name}")
    
    # Create logger with experiment name
    experiment_name = config.project.name if hasattr(config, 'project') else "research_enhanced_experiment"
    logger = ExperimentLogger(config, experiment_name=experiment_name)
    
    # Create model
    print(f"\n🏗️  Creating research-enhanced model...")
    model = create_research_model(config, device=device)
    
    # Log model architecture
    if config.logging.wandb.enabled:
        wandb.watch(model, log='all', log_freq=100)
    
    # Create dataloaders
    print(f"\n📊 Creating sequence dataloaders...")
    train_loader, val_loader = create_sequence_dataloaders(
        config,
        use_weighted_sampling=True
    )
    
    # Create trainer
    print(f"\n👨‍🏫 Initializing research trainer...")
    trainer = ResearchTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        logger=logger
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n📂 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"   Resuming from epoch {start_epoch}")
    
    # Training summary
    print("\n" + "=" * 80)
    print("📝 Training Configuration Summary")
    print("=" * 80)
    print(f"Model Architecture:")
    print(f"   Backbone: {config.model.backbone.type}")
    print(f"   Temporal: BiLSTM + Transformer")
    print(f"   Heads: Regression + Classification + VAE")
    print(f"   Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nTraining:")
    print(f"   Epochs: {config.training.num_epochs}")
    print(f"   Batch Size: {config.training.batch_size}")
    print(f"   Learning Rate: {config.training.learning_rate} → {config.training.max_learning_rate}")
    print(f"   Optimizer: {config.training.optimizer.type}")
    print(f"   Scheduler: {config.training.lr_scheduler.type}")
    print(f"   Mixed Precision: {config.training.mixed_precision}")
    print(f"\nLosses:")
    print(f"   Regression (PRIMARY): weight={config.training.loss.regression.weight}")
    print(f"   Focal (AUXILIARY): weight={config.training.loss.classification.weight}")
    if config.training.loss.get('mil_ranking', {}).get('enabled', False):
        print(f"   MIL Ranking (TERTIARY): weight={config.training.loss.mil_ranking.weight}")
    if config.training.loss.get('vae', {}).get('weight', 0) > 0:
        print(f"   VAE (TERTIARY): weight={config.training.loss.vae.weight}")
    print(f"\nData:")
    print(f"   Train sequences: {len(train_loader.dataset):,}")
    print(f"   Val sequences: {len(val_loader.dataset):,}")
    print(f"   Train batches: {len(train_loader):,}")
    print(f"   Val batches: {len(val_loader):,}")
    print(f"\nExpected Performance:")
    print(f"   Target Test Accuracy: 85-88%")
    print(f"   Target AUC: 87-89%")
    print(f"   Current Baseline: 54% (to be beaten by 31-34%)")
    print("=" * 80)
    
    # Confirm training
    print(f"\n⏳ Starting training in 3 seconds...")
    import time
    time.sleep(3)
    
    # Train!
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print(f"   Saving checkpoint...")
        trainer.save_checkpoint(
            epoch=trainer.current_epoch if hasattr(trainer, 'current_epoch') else 0,
            metrics={},
            is_best=False
        )
    except Exception as e:
        print(f"\n\n❌ Training failed with error:")
        print(f"   {type(e).__name__}: {e}")
        raise
    finally:
        # Cleanup
        if config.logging.wandb.enabled:
            wandb.finish()
    
    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"\nBest checkpoint saved at:")
    print(f"   {Path(config.output.checkpoint_dir) / 'best.pth'}")
    print(f"\nNext steps:")
    print(f"   1. Evaluate on test set: python evaluate.py --checkpoint outputs/checkpoints/best.pth")
    print(f"   2. Check W&B dashboard for training curves")
    print(f"   3. Analyze per-class performance")
    print("=" * 80)


if __name__ == '__main__':
    main()
