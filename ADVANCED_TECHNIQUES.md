# Advanced Generalization Techniques Implementation

## 🎯 Overview

This document explains all the **advanced techniques** implemented in this project to ensure **excellent performance on unseen data**, addressing the common CV problem where models perform well on train/validation but poorly on new data.

---

## ✅ Implemented Techniques

### 1. **Data Augmentation** ✅

#### **Basic Augmentations** (Already Implemented)

- **Geometric**: HorizontalFlip, VerticalFlip, Rotate, ShiftScaleRotate
- **Color**: ColorJitter (brightness, contrast, saturation, hue)
- **Noise**: GaussianBlur, GaussNoise
- **Weather**: RandomRain, RandomFog, RandomShadow, RandomBrightnessContrast

**Purpose**: Increases data diversity, prevents memorization

**Implementation**: `src/data/dataset.py` - `get_train_transforms()`

---

#### **Advanced Augmentations** (NEW! ⭐)

- **CoarseDropout** (Cutout): Random rectangular regions set to zero
- **Random Erasing**: Random rectangular regions filled with noise
- **Mixup**: Linear interpolation between image pairs
- **CutMix**: Rectangular regions swapped between images

**Purpose**:

- **CoarseDropout/Cutout**: Forces model to use multiple cues, handles occlusion
- **Mixup**: Smooth decision boundaries, better calibration
- **CutMix**: Better than Mixup for localization tasks

**Configuration**: `configs/config.yaml`

```yaml
augmentation:
  train:
    coarse_dropout:
      enabled: true
      max_holes: 8
      max_height: 8
      max_width: 8
      p: 0.3

training:
  mixup_cutmix:
    enabled: true
    mixup_alpha: 0.2
    cutmix_alpha: 1.0
    prob: 0.5
    switch_prob: 0.5
```

**Implementation**:

- CoarseDropout: `src/data/dataset.py` - `get_train_transforms()`
- Mixup/CutMix: `src/data/dataset.py` - `MixupCutmixWrapper` class
- Training integration: `src/training/trainer.py` - `train_epoch()`

**References**:

- Mixup: https://arxiv.org/abs/1710.09412
- CutMix: https://arxiv.org/abs/1905.04899
- Cutout: https://arxiv.org/abs/1708.04552

---

### 2. **Regularization Techniques** ✅

#### **L2 Regularization (Weight Decay)**

**Already Implemented**: `weight_decay=0.0001` in AdamW optimizer

**Purpose**: Prevents weights from growing too large, reduces overfitting

**How it works**: Adds `λ * ||w||²` penalty to loss function

---

#### **Dropout**

**Already Implemented**:

- Backbone: 0.3 dropout in `EfficientNetBackbone`
- Temporal: 0.3 dropout in `TemporalEncoder` LSTM

**Purpose**: Prevents co-adaptation of features, forces redundancy

**How it works**: Randomly zeroes activations during training

---

#### **Label Smoothing**

**Already Implemented**: `label_smoothing=0.1` in config

**Purpose**: Prevents overconfident predictions, better calibration

**How it works**: Replaces hard labels (1/0) with soft labels (0.9/0.1)

---

### 3. **Pretrained Foundation Models** ✅

**Already Implemented**: EfficientNet-B0 pretrained on ImageNet

**Purpose**: Transfer learning from large-scale dataset (14M images, 1000 classes)

**Configuration**:

```yaml
model:
  backbone:
    type: "efficientnet_b0"
    pretrained: true
    freeze_backbone: false
```

**Why EfficientNet**:

- Compound scaling (depth, width, resolution)
- 5.8M parameters (lightweight)
- State-of-the-art accuracy/efficiency tradeoff

---

### 4. **Sharpness-Aware Minimization (SAM)** ⭐ NEW!

**Implementation**: `src/training/sam_optimizer.py`

**Purpose**: Seeks flat minima instead of sharp minima for better generalization

**How it works**: Two-step optimization process

1. **Ascent step**: Maximize loss to find adversarial parameters
   - `ε = ρ * grad / ||grad||`
   - `θ_adv = θ + ε`
2. **Descent step**: Minimize loss at adversarial parameters
   - `θ_new = θ_adv - lr * grad(θ_adv)`

**Configuration**:

```yaml
training:
  sam:
    enabled: true
    rho: 0.05 # Neighborhood size (0.05 for AdamW, 0.2 for SGD)
    adaptive: false # Use ASAM (adaptive SAM)
```

**Benefits**:

- Finds parameters robust to small perturbations
- Reduces sensitivity to data shifts
- **State-of-the-art on ImageNet, CIFAR**

**Trade-off**: ~2x training time (two forward-backward passes per batch)

**Reference**: https://arxiv.org/abs/2010.01412

---

### 5. **Stochastic Weight Averaging (SWA)** ⭐ NEW!

**Implementation**: `src/training/trainer.py` - Using PyTorch's `AveragedModel`

**Purpose**: Averages weights over training epochs for better generalization

**How it works**:

1. Train normally until epoch `start_epoch` (e.g., 75)
2. Start averaging: `θ_swa = (θ_swa * n + θ) / (n + 1)`
3. At end: Update BatchNorm statistics with `update_bn()`

**Configuration**:

```yaml
training:
  swa:
    enabled: true
    start_epoch: 75 # Start at 75% of training
    lr: 0.00005 # Lower learning rate for SWA
    anneal_epochs: 10
```

**Benefits**:

- Finds wider optima (flatter minima)
- Better than ensembling (single model)
- **Minimal computational cost** (only averaging)

**Reference**: https://arxiv.org/abs/1803.05407

---

### 6. **Test-Time Augmentation (TTA)** ⭐ NEW!

**Implementation**: `src/training/tta.py`

**Purpose**: Improves robustness at inference by averaging predictions over augmented versions

**How it works**:

1. Apply multiple augmentations to test image:
   - Original
   - HorizontalFlip
   - VerticalFlip
   - Rotation 90°
   - Brightness adjustment
2. Get predictions for all versions
3. Aggregate (mean, max, or voting)

**Configuration**:

```yaml
inference:
  tta:
    enabled: true
    augmentations: 5
    aggregation: "mean" # Options: mean, max, voting
```

**Benefits**:

- **Improved accuracy on unseen data** (typically +1-2%)
- Handles test-time distribution shifts
- Better calibrated predictions

**Trade-off**: ~5x slower inference (5 augmentations)

**When to use**: Final evaluation, production deployment

---

### 7. **Cross-Validation Strategy** ✅

**Already Implemented**: Train/Val split (80/20) with stratification

**Configuration**:

```yaml
validation:
  split_ratio: 0.2 # 20% for validation
```

**Purpose**:

- Detect overfitting early
- Select best hyperparameters
- Estimate true generalization performance

**Best practice**:

- Train data: 80% training, 20% validation
- Test data: **Completely untouched** until final evaluation

---

### 8. **Gradient Clipping** ✅

**Already Implemented**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

**Purpose**: Prevents exploding gradients, stabilizes training

**How it works**: Scales gradients if their norm exceeds threshold

---

### 9. **Mixed Precision Training** ✅

**Already Implemented**: `GradScaler` with FP16 autocast

**Purpose**:

- 2-3x faster training
- 50% less GPU memory
- Enables larger batch sizes

**Configuration**:

```yaml
training:
  mixed_precision: true
```

---

### 10. **Early Stopping** ✅

**Already Implemented**: Monitors validation F1-score

**Configuration**:

```yaml
training:
  early_stopping:
    patience: 15
    min_delta: 0.001
    monitor: "val_f1_score"
    mode: "max"
```

**Purpose**: Stops training when validation performance plateaus

---

## 🔬 How These Techniques Address Generalization

### Problem: Model performs well on Train/Val but poor on new data

### Root Causes:

1. **Overfitting**: Model memorizes training data instead of learning patterns
2. **Distribution Shift**: Test data differs from train data (lighting, angle, weather)
3. **Sharp Minima**: Model sits in narrow valley, small changes hurt performance
4. **Lack of Diversity**: Training data doesn't cover all real-world scenarios

### Our Solutions:

| Technique                | Addresses          | How                                              |
| ------------------------ | ------------------ | ------------------------------------------------ |
| **Data Augmentation**    | Lack of Diversity  | Artificially expands data distribution           |
| **Mixup/CutMix**         | Overfitting        | Forces model to learn smooth decision boundaries |
| **CoarseDropout**        | Distribution Shift | Simulates occlusion, forces multiple cues        |
| **Weather Augmentation** | Distribution Shift | Day/night/weather adaptability                   |
| **Dropout**              | Overfitting        | Prevents co-adaptation of neurons                |
| **Label Smoothing**      | Overconfidence     | Better calibrated predictions                    |
| **SAM**                  | Sharp Minima       | Seeks flat minima robust to perturbations        |
| **SWA**                  | Sharp Minima       | Averages weights for wider optima                |
| **TTA**                  | Distribution Shift | Robust predictions at test time                  |
| **Pretrained Models**    | Lack of Data       | Transfer learning from ImageNet                  |

---

## 📊 Expected Improvements

Based on literature and empirical studies:

| Technique                     | Typical Improvement on Unseen Data |
| ----------------------------- | ---------------------------------- |
| Basic Augmentation            | +2-5% accuracy                     |
| Mixup/CutMix                  | +1-3% accuracy                     |
| SAM                           | +1-2% accuracy                     |
| SWA                           | +0.5-1% accuracy                   |
| TTA                           | +1-2% accuracy                     |
| **Combined (All techniques)** | **+5-10% accuracy**                |

**Example**:

- Baseline (no techniques): 85% test accuracy
- With all techniques: **90-95% test accuracy**

---

## 🚀 How to Use

### Training with All Techniques

```bash
# All advanced techniques are enabled in configs/config.yaml
python train.py --wandb
```

### Disable Specific Techniques

Edit `configs/config.yaml`:

```yaml
# Disable SAM
training:
  sam:
    enabled: false

# Disable Mixup/CutMix
training:
  mixup_cutmix:
    enabled: false

# Disable SWA
training:
  swa:
    enabled: false

# Disable TTA at inference
inference:
  tta:
    enabled: false
```

### Inference with TTA

```python
from src.training.tta import create_tta_model

# Load model
model = create_model(config, device='cuda')
model.load_state_dict(torch.load('best_model.pth'))

# Wrap with TTA
tta_model = create_tta_model(model, config)

# Predict
predictions = tta_model(image)
```

---

## 🔍 Verification

### Check if Techniques are Active

```python
# During training, you'll see these messages:

🔍 Using SAM Optimizer for better generalization
   Rho: 0.05
   Adaptive: False

🎲 Mixup/CutMix enabled - alpha=0.2

📊 SWA enabled - will start at epoch 75

📊 SWA started at epoch 75

📊 Finalizing SWA model...
📊 Evaluating SWA model...
   SWA val_f1_score: 0.9523
   ✅ SWA model is better! Saving...
```

---

## 📚 References

1. **Mixup**: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
2. **CutMix**: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers", ICCV 2019
3. **SAM**: Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization", ICLR 2021
4. **SWA**: Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization", UAI 2018
5. **Cutout**: DeVries & Taylor, "Improved Regularization of Convolutional Neural Networks with Cutout", arXiv 2017
6. **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019
7. **Label Smoothing**: Szegedy et al., "Rethinking the Inception Architecture", CVPR 2016

---

## 🎓 Best Practices

### For Training:

1. ✅ **Always enable** data augmentation
2. ✅ **Enable SAM** for best generalization (despite 2x training time)
3. ✅ **Enable SWA** after 75% training (minimal cost)
4. ✅ **Use Mixup/CutMix** for imbalanced datasets
5. ✅ **Monitor validation metrics** closely

### For Inference:

1. ✅ **Enable TTA** for critical predictions (e.g., anomaly alerts)
2. ❌ **Disable TTA** for real-time requirements (<30ms)
3. ✅ **Use SWA model** if available (usually best)

### For Hyperparameters:

- **SAM rho**: 0.05 for AdamW, 0.2 for SGD
- **Mixup alpha**: 0.2 (conservative), 0.4 (aggressive)
- **CutMix alpha**: 1.0 (standard)
- **SWA start**: 75% of total epochs

---

## 🎯 Summary

This project implements **10 state-of-the-art techniques** to ensure your model generalizes well to unseen data:

1. ✅ Basic Data Augmentation
2. ⭐ Advanced Augmentation (Mixup/CutMix/Cutout)
3. ✅ Regularization (Dropout, L2, Label Smoothing)
4. ✅ Pretrained Foundation Models
5. ⭐ **Sharpness-Aware Minimization (SAM)**
6. ⭐ **Stochastic Weight Averaging (SWA)**
7. ⭐ **Test-Time Augmentation (TTA)**
8. ✅ Cross-Validation
9. ✅ Mixed Precision Training
10. ✅ Early Stopping

**All techniques are production-ready and enabled by default in `configs/config.yaml`.**

---

## 🤝 Support

If you have questions about any technique:

1. Check the implementation in the relevant file (paths listed above)
2. Review the configuration in `configs/config.yaml`
3. Read the referenced papers for theoretical background

**Good luck with your anomaly detection project! 🚀**
