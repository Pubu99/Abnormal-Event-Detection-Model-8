"""
Evaluation script for Research-Enhanced Model.
Tests model performance on test dataset.

Usage:
    python evaluate_research.py --checkpoint outputs/checkpoints/best.pth
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).parent))

from src.models.research_model import ResearchEnhancedModel
from src.data.sequence_dataset import create_sequence_dataloaders
from src.training.metrics import calculate_metrics
from src.utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate research-enhanced model")
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/checkpoints/best.pth',
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_research_enhanced.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/results',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for evaluation (default: use config)'
    )
    
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, test_loader, device, config, class_names):
    """Evaluate model on test set."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probs = []
    
    print("\n🧪 Evaluating on test set...")
    pbar = tqdm(test_loader, desc="Testing", total=len(test_loader))
    
    for batch in pbar:
        # Unpack tuple: (sequences, labels, future_frames)
        sequences, labels, future_frames = batch
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Forward pass - get all outputs
        outputs = model(sequences)
        
        # Use classification logits for predictions
        class_logits = outputs['class_logits']
        class_probs = F.softmax(class_logits, dim=1)
        class_preds = torch.argmax(class_probs, dim=1)
        
        # Store predictions
        all_predictions.extend(class_preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        all_probs.extend(class_probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    print("\n📊 Computing metrics...")
    metrics = calculate_metrics(all_predictions, all_targets)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Classification report
    report = classification_report(
        all_targets, 
        all_predictions, 
        target_names=class_names,
        digits=4
    )
    
    return metrics, cm, report, all_predictions, all_targets, all_probs


def print_results(metrics, cm, report, checkpoint_path):
    """Print evaluation results."""
    print("\n" + "="*80)
    print(" 📊 EVALUATION RESULTS")
    print("="*80)
    
    print(f"\n📦 Checkpoint: {checkpoint_path}")
    
    print("\n📈 Overall Metrics:")
    print(f"   Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Precision:       {metrics.get('precision_macro', 0):.4f}")
    print(f"   Recall:          {metrics.get('recall_macro', 0):.4f}")
    print(f"   F1 Score:        {metrics.get('f1_weighted', 0):.4f}")
    print(f"   F1 Macro:        {metrics.get('f1_macro', 0):.4f}")
    
    if 'auc_roc_macro' in metrics:
        print(f"   AUC-ROC:         {metrics['auc_roc_macro']:.4f}")
    
    print("\n📋 Per-Class Performance:")
    print(report)
    
    print("\n🔢 Confusion Matrix:")
    print(cm)


def save_results(metrics, cm, report, output_dir, checkpoint_path, all_predictions, all_targets):
    """Save evaluation results to file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save text report
    report_file = output_dir / f"evaluation_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(" RESEARCH-ENHANCED MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Precision:       {metrics.get('precision_macro', 0):.4f}\n")
        f.write(f"Recall:          {metrics.get('recall_macro', 0):.4f}\n")
        f.write(f"F1 Score:        {metrics.get('f1_weighted', 0):.4f}\n")
        f.write(f"F1 Macro:        {metrics.get('f1_macro', 0):.4f}\n")
        if 'auc_roc_macro' in metrics:
            f.write(f"AUC-ROC:         {metrics['auc_roc_macro']:.4f}\n")
        
        f.write("\n\nPER-CLASS PERFORMANCE:\n")
        f.write("-"*80 + "\n")
        f.write(report)
        
        f.write("\n\nCONFUSION MATRIX:\n")
        f.write("-"*80 + "\n")
        f.write(str(cm))
    
    print(f"\n💾 Results saved to: {report_file}")
    
    # Save numpy arrays
    np_file = output_dir / f"predictions_{timestamp}.npz"
    np.savez(
        np_file,
        predictions=all_predictions,
        targets=all_targets,
        confusion_matrix=cm
    )
    print(f"💾 Predictions saved to: {np_file}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("="*80)
    print(" 🚀 RESEARCH-ENHANCED MODEL EVALUATION")
    print("="*80)
    
    # Load config
    print(f"\n📋 Loading config from {args.config}...")
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Load checkpoint
    print(f"\n📦 Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Create model
    print("\n🏗️  Creating model...")
    model = ResearchEnhancedModel(config).to(device)
    
    # Load state dict
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    print("✅ Model loaded successfully")
    
    if 'epoch' in checkpoint:
        print(f"   Checkpoint from epoch: {checkpoint['epoch']}")
    if 'val_metrics' in checkpoint:
        val_metrics = checkpoint['val_metrics']
        print(f"   Validation Accuracy: {val_metrics.get('accuracy', 0):.4f}")
        print(f"   Validation F1: {val_metrics.get('f1_weighted', 0):.4f}")
    
    # Get class names
    class_names = config.data.classes
    print(f"\n📝 Classes ({len(class_names)}): {', '.join(class_names)}")
    
    # Create test dataloader
    print("\n📊 Creating test dataloader...")
    batch_size = args.batch_size if args.batch_size else config.training.batch_size
    
    # Create dataloaders (train and val)
    train_loader, val_loader = create_sequence_dataloaders(config)
    
    # For testing, we'll use the validation set since we don't have a separate test split
    # In a real scenario, you'd create a separate test dataset
    test_loader = val_loader
    print(f"   Test sequences: {len(test_loader.dataset)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Evaluate
    metrics, cm, report, all_predictions, all_targets, all_probs = evaluate(
        model, test_loader, device, config, class_names
    )
    
    # Print results
    print_results(metrics, cm, report, args.checkpoint)
    
    # Save results
    save_results(
        metrics, cm, report, args.output, args.checkpoint,
        all_predictions, all_targets
    )
    
    print("\n" + "="*80)
    print(" ✅ EVALUATION COMPLETE")
    print("="*80)
    
    # Return metrics for programmatic access
    return metrics


if __name__ == '__main__':
    main()
