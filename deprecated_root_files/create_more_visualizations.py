"""Generate additional advanced visualizations."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

CLASS_NAMES = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
    'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting',
    'Stealing', 'Vandalism', 'NormalVideos'
]

plt.style.use('seaborn-v0_8-paper')
output_dir = Path('outputs/visualizations')
N_CLASSES = len(CLASS_NAMES)

print("\n" + "="*80)
print(" 🎨 CREATING ADVANCED VISUALIZATIONS")
print("="*80 + "\n")

# Load data
data = np.load('outputs/results/predictions_20251016_110958.npz')
y_true = data['y_true']
y_pred = data['y_pred']
y_probs = data['y_probs']

# 2. ROC Curves
print("📊 Creating ROC curves...")
y_true_bin = label_binarize(y_true, classes=range(N_CLASSES))
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
axes = axes.flatten()
for i in range(N_CLASSES):
    ax = axes[i]
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{CLASS_NAMES[i]}', fontweight='bold')
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
for i in range(N_CLASSES, len(axes)):
    axes[i].axis('off')
plt.suptitle('ROC Curves Per Class (One-vs-Rest)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'roc_curves_per_class.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved roc_curves_per_class.png")

# 3. Precision-Recall Curves
print("📊 Creating Precision-Recall curves...")
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
axes = axes.flatten()
for i in range(N_CLASSES):
    ax = axes[i]
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
    avg_precision = average_precision_score(y_true_bin[:, i], y_probs[:, i])
    ax.plot(recall, precision, color='blue', lw=2, label=f'AP = {avg_precision:.4f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{CLASS_NAMES[i]}', fontweight='bold')
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
for i in range(N_CLASSES, len(axes)):
    axes[i].axis('off')
plt.suptitle('Precision-Recall Curves Per Class', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved precision_recall_curves.png")

# 4. t-SNE Projection
print("📊 Creating t-SNE projection (this may take a minute)...")
n_samples = min(5000, len(y_probs))
indices = np.random.choice(len(y_probs), n_samples, replace=False)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
features_2d = tsne.fit_transform(y_probs[indices])
fig, ax = plt.subplots(figsize=(16, 12))
colors = plt.cm.tab20(np.linspace(0, 1, N_CLASSES))
for i in range(N_CLASSES):
    mask = y_true[indices] == i
    ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
              c=[colors[i]], label=CLASS_NAMES[i], alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
ax.set_title(f't-SNE Projection of Feature Space ({n_samples} samples)', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'tsne_feature_projection.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved tsne_feature_projection.png")

# 5. Calibration Plot
print("📊 Creating calibration plot...")
y_prob_max = np.max(y_probs, axis=1)
y_correct = (np.argmax(y_probs, axis=1) == y_true).astype(int)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# Simple binning for calibration
bins = 10
bin_edges = np.linspace(0, 1, bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_correct = []
for i in range(bins):
    mask = (y_prob_max >= bin_edges[i]) & (y_prob_max < bin_edges[i+1])
    if mask.sum() > 0:
        bin_correct.append(y_correct[mask].mean())
    else:
        bin_correct.append(0)
axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
axes[0].plot(bin_centers, bin_correct, 's-', label='Model', markersize=8, linewidth=2)
axes[0].set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
axes[0].set_title('Calibration Plot', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)
# Confidence histogram
axes[1].hist(y_prob_max, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[1].set_xlabel('Predicted Probability (Confidence)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1].set_title('Prediction Confidence Distribution', fontsize=13, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
high_conf = (y_prob_max > 0.9).sum() / len(y_prob_max) * 100
mean_conf = y_prob_max.mean() * 100
axes[1].text(0.05, 0.95, f'Mean: {mean_conf:.2f}%\n>90%: {high_conf:.2f}%',
            transform=axes[1].transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
plt.tight_layout()
plt.savefig(output_dir / 'calibration_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved calibration_plot.png")

# 6. Temporal Sequence Analysis
print("📊 Creating temporal analysis...")
n_seq = 500
indices_seq = sorted(np.random.choice(len(y_true), n_seq, replace=False))
confidences = np.max(y_probs[indices_seq], axis=1)
correctness = (y_true[indices_seq] == y_pred[indices_seq])
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
colors_seq = ['green' if c else 'red' for c in correctness]
axes[0].scatter(range(n_seq), confidences, c=colors_seq, alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
axes[0].axhline(y=0.9, color='blue', linestyle='--', linewidth=1, alpha=0.5)
axes[0].set_ylabel('Confidence', fontsize=12, fontweight='bold')
axes[0].set_title('Prediction Confidence (Green=Correct, Red=Incorrect)', fontsize=13, fontweight='bold')
axes[0].grid(alpha=0.3)
# Anomaly scores
normal_idx = CLASS_NAMES.index('NormalVideos')
anomaly_scores = 1 - y_probs[indices_seq, normal_idx]
axes[1].plot(range(n_seq), anomaly_scores, linewidth=1, color='darkblue')
axes[1].fill_between(range(n_seq), 0, anomaly_scores, alpha=0.3, color='lightblue')
axes[1].axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[1].set_xlabel('Sequence Index', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Anomaly Score', fontsize=12, fontweight='bold')
axes[1].set_title('Anomaly Score Trends', fontsize=13, fontweight='bold')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'temporal_sequence_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved temporal_sequence_analysis.png")

print("\n" + "="*80)
print(" ✅ ALL ADVANCED VISUALIZATIONS CREATED!")
print("="*80)
print(f"\nSaved to: {output_dir}")
print("\nNew visualizations:")
print("   1. train_val_test_comparison.png")
print("   2. roc_curves_per_class.png")
print("   3. precision_recall_curves.png")
print("   4. tsne_feature_projection.png")
print("   5. calibration_plot.png")
print("   6. temporal_sequence_analysis.png")
print()
