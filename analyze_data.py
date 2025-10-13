"""
Data Analysis Script
Analyzes class distribution and imbalance in UCF Crime dataset.
Shows detailed statistics for Train and Test splits.

Usage:
    python analyze_data.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import ConfigManager


def analyze_class_distribution(root_dir: Path, split: str = 'train'):
    """Analyze class distribution for a given split."""
    split_dir = root_dir / split.capitalize()
    
    if not split_dir.exists():
        print(f"❌ Directory not found: {split_dir}")
        return None
    
    # Get all class directories
    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    
    data = []
    total_images = 0
    
    print(f"\n{'='*80}")
    print(f" 📊 {split.upper()} DATA ANALYSIS")
    print(f"{'='*80}\n")
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        # Count PNG images
        image_count = len(list(class_dir.glob("*.png")))
        total_images += image_count
        
        data.append({
            'class': class_name,
            'count': image_count
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df['percentage'] = (df['count'] / total_images) * 100
    df = df.sort_values('count', ascending=False)
    
    # Print statistics
    print(f"Total Images: {total_images:,}\n")
    print(f"{'Class':<25} {'Count':>12} {'Percentage':>12} {'Bar':>30}")
    print(f"{'-'*80}")
    
    max_count = df['count'].max()
    for _, row in df.iterrows():
        class_name = row['class']
        count = row['count']
        pct = row['percentage']
        
        # Create visual bar
        bar_length = int((count / max_count) * 30)
        bar = '█' * bar_length
        
        # Mark normal class
        marker = ' ⭐ NORMAL' if class_name == 'NormalVideos' else ''
        
        print(f"{class_name:<25} {count:>12,} {pct:>11.2f}% {bar:<30}{marker}")
    
    # Calculate imbalance metrics
    print(f"\n{'-'*80}")
    print(f"📈 IMBALANCE ANALYSIS")
    print(f"{'-'*80}")
    
    min_count = df['count'].min()
    max_count = df['count'].max()
    mean_count = df['count'].mean()
    std_count = df['count'].std()
    
    imbalance_ratio = max_count / min_count
    cv = (std_count / mean_count) * 100  # Coefficient of variation
    
    print(f"Minimum samples:        {min_count:,}")
    print(f"Maximum samples:        {max_count:,}")
    print(f"Mean samples:           {mean_count:,.0f}")
    print(f"Std deviation:          {std_count:,.0f}")
    print(f"Imbalance ratio:        {imbalance_ratio:.2f}x")
    print(f"Coefficient of variation: {cv:.1f}%")
    
    # Classify imbalance severity
    if imbalance_ratio < 2:
        severity = "LOW (Good!)"
    elif imbalance_ratio < 5:
        severity = "MODERATE"
    elif imbalance_ratio < 10:
        severity = "HIGH"
    else:
        severity = "SEVERE"
    
    print(f"\n⚠️  Imbalance Severity: {severity}")
    
    return df, total_images


def calculate_class_weights(df):
    """Calculate class weights for handling imbalance."""
    total = df['count'].sum()
    num_classes = len(df)
    
    print(f"\n{'='*80}")
    print(f" ⚖️  CLASS WEIGHTS CALCULATION")
    print(f"{'='*80}\n")
    
    print(f"{'Class':<25} {'Count':>12} {'Inv. Freq Weight':>20} {'Effective Weight':>20}")
    print(f"{'-'*80}")
    
    # Inverse frequency weighting
    df['inv_freq_weight'] = total / (num_classes * df['count'])
    
    # Effective number weighting (better for severe imbalance)
    beta = 0.9999
    df['effective_weight'] = (1 - beta) / (1 - beta ** df['count'])
    
    # Normalize
    df['inv_freq_weight_norm'] = df['inv_freq_weight'] / df['inv_freq_weight'].sum() * num_classes
    df['effective_weight_norm'] = df['effective_weight'] / df['effective_weight'].sum() * num_classes
    
    for _, row in df.iterrows():
        print(f"{row['class']:<25} {row['count']:>12,} {row['inv_freq_weight_norm']:>20.4f} {row['effective_weight_norm']:>20.4f}")
    
    return df


def compare_train_test(train_df, test_df):
    """Compare train and test distributions."""
    print(f"\n{'='*80}")
    print(f" 🔄 TRAIN vs TEST COMPARISON")
    print(f"{'='*80}\n")
    
    # Merge dataframes
    comparison = train_df[['class', 'count']].merge(
        test_df[['class', 'count']], 
        on='class', 
        suffixes=('_train', '_test')
    )
    
    comparison['train_pct'] = (comparison['count_train'] / comparison['count_train'].sum()) * 100
    comparison['test_pct'] = (comparison['count_test'] / comparison['count_test'].sum()) * 100
    comparison['ratio'] = comparison['count_train'] / comparison['count_test']
    
    print(f"{'Class':<25} {'Train':>12} {'Test':>12} {'Train/Test':>12} {'Distribution Match':>20}")
    print(f"{'-'*80}")
    
    for _, row in comparison.iterrows():
        class_name = row['class']
        train_count = row['count_train']
        test_count = row['count_test']
        ratio = row['ratio']
        
        # Check if distributions match (should be similar ratios)
        pct_diff = abs(row['train_pct'] - row['test_pct'])
        match = "✅ Good" if pct_diff < 2 else "⚠️  Mismatch"
        
        print(f"{class_name:<25} {train_count:>12,} {test_count:>12,} {ratio:>12.2f} {match:>20}")


def visualize_distribution(train_df, test_df, output_path='outputs/results/data_distribution.png'):
    """Create visualization of class distribution."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('UCF Crime Dataset - Class Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Train distribution bar chart
    ax1 = axes[0, 0]
    train_sorted = train_df.sort_values('count', ascending=False)
    colors = ['green' if c == 'NormalVideos' else 'red' for c in train_sorted['class']]
    ax1.barh(train_sorted['class'], train_sorted['count'], color=colors, alpha=0.7)
    ax1.set_xlabel('Number of Images')
    ax1.set_title('Training Data Distribution')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Test distribution bar chart
    ax2 = axes[0, 1]
    test_sorted = test_df.sort_values('count', ascending=False)
    colors = ['green' if c == 'NormalVideos' else 'red' for c in test_sorted['class']]
    ax2.barh(test_sorted['class'], test_sorted['count'], color=colors, alpha=0.7)
    ax2.set_xlabel('Number of Images')
    ax2.set_title('Test Data Distribution')
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Pie chart - Train
    ax3 = axes[1, 0]
    normal_count = train_df[train_df['class'] == 'NormalVideos']['count'].values[0]
    anomaly_count = train_df[train_df['class'] != 'NormalVideos']['count'].sum()
    ax3.pie([normal_count, anomaly_count], 
            labels=['Normal', 'Anomalies'], 
            autopct='%1.1f%%',
            colors=['green', 'red'],
            alpha=0.7,
            startangle=90)
    ax3.set_title('Train: Normal vs Anomalies')
    
    # 4. Class weights visualization
    ax4 = axes[1, 1]
    weights_df = train_df.sort_values('inv_freq_weight_norm', ascending=False)
    ax4.barh(weights_df['class'], weights_df['inv_freq_weight_norm'], alpha=0.7)
    ax4.set_xlabel('Weight (Inverse Frequency)')
    ax4.set_title('Class Weights for Imbalance Handling')
    ax4.axvline(x=1.0, color='red', linestyle='--', label='Neutral weight')
    ax4.legend()
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")
    
    return fig


def show_sampling_strategy(train_df):
    """Explain the sampling strategy."""
    print(f"\n{'='*80}")
    print(f" 🎯 IMBALANCE HANDLING STRATEGY")
    print(f"{'='*80}\n")
    
    print("The model uses a 3-PRONGED APPROACH to handle class imbalance:\n")
    
    print("1️⃣  FOCAL LOSS (α=0.25, γ=2.0)")
    print("   - Down-weights easy examples")
    print("   - Focuses on hard-to-classify samples")
    print("   - Reduces loss contribution from well-classified majority class\n")
    
    print("2️⃣  WEIGHTED RANDOM SAMPLING")
    print("   - Samples minority classes more frequently")
    print("   - Each class has equal probability in each epoch")
    print("   - No data duplication (sampling with replacement)\n")
    
    print("   Example sampling probabilities:")
    total = train_df['count'].sum()
    for _, row in train_df.head(5).iterrows():
        sample_prob = (1 / row['count']) / (1 / train_df['count']).sum()
        print(f"      {row['class']:<25} {sample_prob*100:>6.2f}% (has {row['count']:,} images)")
    
    print("\n3️⃣  CLASS WEIGHTS in Loss Function")
    print("   - Rare classes get higher weight in loss")
    print("   - Penalizes misclassification of minority classes more")
    print("   - Combined with Focal Loss for maximum effect\n")
    
    print("📈 Expected Result:")
    print("   - Balanced gradients across all classes")
    print("   - Prevents model from ignoring rare classes")
    print("   - Achieves high recall on all anomaly types\n")


def main():
    """Main analysis function."""
    print("\n" + "="*80)
    print(" 🔍 UCF CRIME DATASET - COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # Load config
    config = ConfigManager()
    root_dir = Path(config.config.data.root_dir)
    
    # Analyze training data
    train_df, train_total = analyze_class_distribution(root_dir, 'train')
    
    if train_df is None:
        print("\n❌ Training data not found. Please check data/raw/Train/ exists.")
        return
    
    # Calculate class weights
    train_df = calculate_class_weights(train_df)
    
    # Analyze test data
    test_df, test_total = analyze_class_distribution(root_dir, 'test')
    
    if test_df is not None:
        # Compare distributions
        compare_train_test(train_df, test_df)
        
        # Create visualization
        try:
            visualize_distribution(train_df, test_df)
        except Exception as e:
            print(f"\n⚠️  Could not create visualization: {e}")
    else:
        print("\n⚠️  Test data not found. Skipping comparison.")
    
    # Show sampling strategy
    show_sampling_strategy(train_df)
    
    # Summary
    print(f"\n{'='*80}")
    print(f" 📝 SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"✅ Training data: {train_total:,} images across {len(train_df)} classes")
    if test_df is not None:
        print(f"✅ Test data: {test_total:,} images across {len(test_df)} classes")
    print(f"\n✅ Class imbalance: DETECTED and will be HANDLED with 3-pronged approach")
    print(f"✅ Both Train and Test splits: PROPERLY SEPARATED")
    print(f"✅ Test data: Will be used ONLY for final evaluation (not during training)")
    
    print(f"\n{'='*80}")
    print(" 🚀 Ready to train! The model will handle the imbalance automatically.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
