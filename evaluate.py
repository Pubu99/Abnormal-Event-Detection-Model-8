"""
Evaluation script for trained anomaly detection model.
Tests model performance on test dataset.

Usage:
    python evaluate.py --checkpoint outputs/logs/experiment_name/checkpoints/best_model.pth
"""

import argparse
import sys
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.models import create_model
from src.data import create_test_dataloader
from src.training.metrics import MetricsCalculator
from src.utils import ConfigManager, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection model")
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/results/evaluation.txt',
        help='Path to save evaluation results'
    )
    
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save predictions to file'
    )
    
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, test_loader, device, config):
    """Evaluate model on test set."""
    model.eval()
    
    metrics_calc = MetricsCalculator(
        config.data.num_classes,
        config.data.classes
    )
    
    all_predictions = []
    all_targets = []
    all_paths = []
    all_confidences = []
    
    print("\n🧪 Evaluating on test set...")
    pbar = tqdm(test_loader, desc="Testing")
    
    for images, labels, is_anomaly, paths in pbar:
        images = images.to(device)
        labels = labels.to(device)
        is_anomaly = is_anomaly.to(device)
        
        # Forward pass
        outputs = model(images, return_embeddings=True)
        
        # Get predictions
        class_probs = F.softmax(outputs['class_logits'], dim=1)
        class_preds = torch.argmax(class_probs, dim=1)
        class_conf = torch.max(class_probs, dim=1)[0]
        
        binary_probs = F.softmax(outputs['binary_logits'], dim=1)
        binary_preds = torch.argmax(binary_probs, dim=1)
        
        # Update metrics
        metrics_calc.update(
            class_preds, labels, class_probs,
            binary_preds, is_anomaly, binary_probs[:, 1]
        )
        
        # Store predictions
        all_predictions.extend(class_preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        all_paths.extend(paths)
        all_confidences.extend(class_conf.cpu().numpy())
    
    # Compute metrics
    metrics = metrics_calc.compute_all_metrics()
    confusion_matrix = metrics_calc.compute_confusion_matrix()
    classification_report = metrics_calc.get_classification_report()
    
    return metrics, confusion_matrix, classification_report, \
           all_predictions, all_targets, all_paths, all_confidences


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("="*70)
    print(" 🧪 MODEL EVALUATION")
    print("="*70)
    
    # Load config
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    set_seed(config.project.seed)
    
    # Setup device
    device = torch.device(config.hardware.device)
    print(f"\n🖥️  Device: {device}")
    
    # Load model
    print(f"\n📦 Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = create_model(config, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("   ✅ Model loaded successfully")
    
    if 'epoch' in checkpoint:
        print(f"   Checkpoint from epoch: {checkpoint['epoch'] + 1}")
    
    # Create test dataloader
    print("\n📦 Loading test data...")
    test_loader, test_dataset = create_test_dataloader(config)
    
    # Evaluate
    metrics, cm, report, predictions, targets, paths, confidences = \
        evaluate(model, test_loader, device, config)
    
    # Print results
    print("\n" + "="*70)
    print(" 📊 EVALUATION RESULTS")
    print("="*70)
    
    print("\n" + report)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(" EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model checkpoint: {args.checkpoint}\n")
        f.write(f"Test samples: {len(test_dataset)}\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nMetrics:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    print(f"\n💾 Results saved to {output_path}")
    
    # Save predictions
    if args.save_predictions:
        import pandas as pd
        
        pred_df = pd.DataFrame({
            'path': paths,
            'true_label': targets,
            'predicted_label': predictions,
            'confidence': confidences,
            'true_class': [config.data.classes[t] for t in targets],
            'predicted_class': [config.data.classes[p] for p in predictions],
            'correct': np.array(predictions) == np.array(targets)
        })
        
        pred_path = output_path.parent / 'predictions.csv'
        pred_df.to_csv(pred_path, index=False)
        print(f"   Predictions saved to {pred_path}")
    
    # Summary
    print("\n" + "="*70)
    print(" 📈 SUMMARY")
    print("="*70)
    print(f"   Overall Accuracy:    {metrics.get('accuracy', 0):.2%}")
    print(f"   Macro F1-Score:      {metrics.get('f1_macro', 0):.2%}")
    print(f"   Weighted F1-Score:   {metrics.get('f1_weighted', 0):.2%}")
    if 'binary_f1' in metrics:
        print(f"   Binary F1 (Anomaly): {metrics.get('binary_f1', 0):.2%}")
    if 'auc_roc_macro' in metrics:
        print(f"   AUC-ROC (macro):     {metrics.get('auc_roc_macro', 0):.2%}")
    print("="*70)
    
    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
