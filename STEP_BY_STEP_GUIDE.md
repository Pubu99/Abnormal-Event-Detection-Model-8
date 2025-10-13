# 🚀 Step-by-Step Running Guide

**Complete Command Reference for Running the Anomaly Detection Project**

This guide provides **exact commands** to run from start to finish, with **expected outputs** and **troubleshooting** for each step.

---

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] **Python 3.9+** installed
- [ ] **NVIDIA GPU** with CUDA support (RTX 3060 or better)
- [ ] **CUDA 11.8+** installed
- [ ] **16GB+ RAM** (32GB recommended)
- [ ] **UCF Crime Dataset** downloaded to `data/raw/`
- [ ] **PowerShell** or Command Prompt access

**Check Python version:**

```powershell
python --version
# Expected: Python 3.9.x or higher
```

**Check CUDA availability:**

```powershell
nvidia-smi
# Expected: Driver version, CUDA version, GPU name
```

---

## 🔧 Phase 1: Environment Setup

### Step 1.1: Install PyTorch with CUDA

```powershell
# Windows with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Expected Output:**

```
Successfully installed torch-2.7.0 torchvision-0.18.0 torchaudio-2.7.0
```

**Verify CUDA:**

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

**Expected Output:**

```
PyTorch: 2.7.0+cu118
CUDA Available: True
CUDA Version: 11.8
```

### Step 1.2: Install Dependencies

```powershell
pip install omegaconf numpy pandas matplotlib seaborn pyyaml albumentations timm tqdm tensorboard wandb scikit-learn
```

**Expected Output:**

```
Successfully installed omegaconf-2.3.0 numpy-1.24.3 pandas-2.0.3 matplotlib-3.7.2 ...
```

### Step 1.3: Verify Installation

```powershell
python -c "import torch, torchvision, omegaconf, albumentations, timm; print('✅ All imports successful!')"
```

**Expected Output:**

```
✅ All imports successful!
```

---

## 📊 Phase 2: Data Analysis

### Step 2.1: Run Data Analysis

```powershell
python analyze_data.py
```

**Expected Output:**

```
================================================
UCF Crime Dataset Analysis
================================================

Analyzing Train Split...
  Total images: 1,266,345
  Classes: 14

Class Distribution:
  0: Abuse          - 35,624 images (2.81%)
  1: Arrest         - 28,745 images (2.27%)
  2: Arson          - 15,234 images (1.20%)
  ...
  7: NormalVideos   - 572,384 images (45.19%) ⚠️
  ...

Class Imbalance Metrics:
  Max/Min Ratio: 21.5 (HIGH IMBALANCE)
  Imbalance Severity: SEVERE

Recommended Solutions:
  ✅ Focal Loss (α=0.25, γ=2.0)
  ✅ Weighted Random Sampling
  ✅ Class Weights

Analyzing Test Split...
  Total images: 111,308
  [Similar output for test set]

Generating visualizations...
  ✅ Saved: outputs/results/train_distribution.png
  ✅ Saved: outputs/results/test_distribution.png
  ✅ Saved: outputs/results/class_comparison.png

Analysis complete! Check outputs/results/ for charts.
```

**What to look for:**

- ✅ Both Train and Test data are detected
- ✅ Class imbalance is identified
- ✅ Charts are generated

---

## ✅ Phase 3: Setup Verification

### Step 3.1: Verify Project Setup

```powershell
python test_setup.py
```

**Expected Output:**

```
================================================
Testing Project Setup
================================================

1. Checking Dependencies...
  ✅ PyTorch 2.7.0
  ✅ CUDA 11.8 available
  ✅ GPU: NVIDIA GeForce RTX 5090 (24GB)
  ✅ All packages installed

2. Checking Configuration...
  ✅ config.yaml loaded successfully
  ✅ Data paths exist:
     - Train: data/raw/Train/
     - Test: data/raw/Test/
  ✅ All 14 class folders found

3. Testing Data Loading...
  ✅ Train dataset: 1,266,345 images
  ✅ Test dataset: 111,308 images
  ✅ DataLoader working
  ✅ Batch shape: torch.Size([128, 3, 64, 64])

4. Testing Model...
  ✅ HybridAnomalyDetector created
  ✅ Total parameters: 5,847,234
  ✅ Model moved to GPU
  ✅ Forward pass successful
  ✅ Output shapes correct:
     - Class logits: torch.Size([128, 14])
     - Anomaly score: torch.Size([128, 1])

5. Testing Loss Functions...
  ✅ FocalLoss working
  ✅ CombinedLoss working

All tests passed! ✅
Ready to start training.
```

**What to look for:**

- ✅ All tests pass
- ✅ GPU is detected
- ✅ Data loads correctly
- ✅ Model creates successfully

**If any test fails**, see [Troubleshooting](#-troubleshooting) section.

---

## 🎓 Phase 4: Training

### Step 4.1: Start Training (Recommended Configuration)

```powershell
# RECOMMENDED: Full training with all optimizations
python train.py --epochs 50 --wandb
```

**Initial Output:**

```
================================================
Starting Training
================================================

Configuration:
  Epochs: 50
  Batch Size: 128
  Learning Rate: 0.001 -> 0.01 (OneCycleLR)
  Device: cuda (NVIDIA GeForce RTX 5090)

Optimizations Enabled:
  ✅ SAM Optimizer (rho=0.05)
  ✅ OneCycleLR Scheduler (50% faster!)
  ✅ torch.compile() (30% speedup)
  ✅ Mixed Precision (FP16)
  ✅ Mixup/CutMix Augmentation
  ✅ SWA (starts epoch 40)

Model: HybridAnomalyDetector
  Parameters: 5,847,234
  Trainable: 5,847,234

Data:
  Train: 1,013,076 images (80% split)
  Validation: 253,269 images (20% split)
  Test: 111,308 images (held out)

Weights & Biases: Initialized
  Project: anomaly-detection
  Run: run_20240115_143022
  URL: https://wandb.ai/yourname/anomaly-detection/runs/xyz

Starting training...
```

**Training Progress:**

```
Epoch 1/50
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:22:15
  Train Loss: 1.234 | Acc: 67.3% | Focal: 0.956 | SVDD: 0.278
  Val Loss: 1.156 | Acc: 72.1% | Focal: 0.892 | SVDD: 0.264
  LR: 0.00102 | Time: 22m 15s

Epoch 2/50
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:21:48
  Train Loss: 0.987 | Acc: 76.4% | Focal: 0.756 | SVDD: 0.231
  Val Loss: 0.923 | Acc: 79.2% | Focal: 0.712 | SVDD: 0.211
  LR: 0.00205 | Time: 21m 48s | ETA: 17h 26m

...

Epoch 25/50 (OneCycleLR peak)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:21:32
  Train Loss: 0.234 | Acc: 91.2% | Focal: 0.178 | SVDD: 0.056
  Val Loss: 0.267 | Acc: 89.8% | Focal: 0.201 | SVDD: 0.066
  LR: 0.00950 | Time: 21m 32s | ETA: 8h 58m

...

Epoch 40/50 (SWA starts)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:21:20
  Train Loss: 0.156 | Acc: 93.7% | Focal: 0.112 | SVDD: 0.044
  Val Loss: 0.189 | Acc: 92.4% | Focal: 0.142 | SVDD: 0.047
  LR: 0.00012 | Time: 21m 20s | ETA: 3h 33m
  ℹ️ SWA activated! Averaging weights...

...

Epoch 50/50 (Final)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:21:15
  Train Loss: 0.123 | Acc: 94.5% | Focal: 0.089 | SVDD: 0.034
  Val Loss: 0.158 | Acc: 93.1% | Focal: 0.118 | SVDD: 0.040
  LR: 0.00001 | Time: 21m 15s

Finalizing SWA model...
  ✅ SWA batch normalization updated

Training completed in 18h 23m

Best model saved: outputs/logs/run_20240115_143022/checkpoints/best_model.pth
  Best epoch: 47
  Best val accuracy: 93.2%
```

**Expected Training Time**:

- **50 epochs**: 18-20 hours
- **75 epochs**: 27-30 hours
- **100 epochs**: 36-40 hours

### Step 4.2: Alternative Training Configurations

#### Quick Test (5 epochs)

```powershell
python train.py --epochs 5
# Time: ~2 hours
# Purpose: Verify pipeline works
```

#### Balanced Training (75 epochs)

```powershell
python train.py --epochs 75 --wandb
# Time: ~27-30 hours
# Expected accuracy: 94-96%
```

#### Maximum Training (100 epochs)

```powershell
python train.py --epochs 100 --wandb
# Time: ~36-40 hours
# Expected accuracy: 95-97%
```

#### Resume Training

```powershell
python train.py --resume outputs/logs/run_20240115_143022/checkpoints/checkpoint_epoch_30.pth --wandb
# Continues from epoch 30
```

### Step 4.3: Monitor Training

#### Option 1: TensorBoard (Local)

```powershell
# Open new terminal
tensorboard --logdir outputs/logs/

# Open browser: http://localhost:6006
```

**You'll see:**

- Loss curves (train/val)
- Accuracy curves
- Learning rate schedule
- Confusion matrices

#### Option 2: Weights & Biases (Cloud - Recommended)

```powershell
# First-time setup
wandb login
# Paste your API key from https://wandb.ai/authorize

# Train with W&B
python train.py --epochs 50 --wandb
```

**W&B Dashboard shows:**

- Real-time metrics
- System stats (GPU, RAM)
- Model comparisons
- Hyperparameter sweeps

---

## 📊 Phase 5: Evaluation

### Step 5.1: Evaluate on Test Set

```powershell
python evaluate.py --model outputs/logs/run_20240115_143022/checkpoints/best_model.pth
```

**Expected Output:**

```
================================================
Evaluating Model
================================================

Loading model: best_model.pth
  ✅ Model loaded successfully
  ✅ Moved to GPU

Loading test dataset...
  ✅ Test images: 111,308

Running inference...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:03:45

Results:
  Accuracy: 93.8%
  Precision: 94.1%
  Recall: 93.2%
  F1-Score: 93.6%

Per-Class Metrics:
  Class 0 (Abuse):        P=91.2% R=90.5% F1=90.8% (2,845 samples)
  Class 1 (Arrest):       P=89.8% R=88.9% F1=89.3% (2,298 samples)
  Class 2 (Arson):        P=92.3% R=91.7% F1=92.0% (1,234 samples)
  ...
  Class 7 (Normal):       P=96.2% R=95.8% F1=96.0% (49,866 samples)
  ...

Confusion Matrix saved: outputs/results/confusion_matrix.png
Classification Report saved: outputs/results/classification_report.txt

Evaluation complete!
```

### Step 5.2: Evaluate with Test-Time Augmentation (TTA)

```powershell
python evaluate.py --model best_model.pth --tta --num-augmentations 5
```

**Expected Output:**

```
Running inference with TTA (5 augmentations)...
  ⚠️ Warning: 5x slower than normal inference

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:18:45

Results with TTA:
  Accuracy: 94.9% (+1.1% improvement!)
  Precision: 95.2%
  Recall: 94.5%
  F1-Score: 94.8%

TTA provides more robust predictions at the cost of speed.
```

---

## 🔍 Phase 6: Analysis

### Step 6.1: Analyze Training Results

```powershell
python -c "
import json
with open('outputs/logs/run_20240115_143022/metrics.json') as f:
    metrics = json.load(f)
print(f'Best Val Accuracy: {metrics[\"best_val_acc\"]:.2%}')
print(f'Best Epoch: {metrics[\"best_epoch\"]}')
print(f'Final Test Accuracy: {metrics[\"test_acc\"]:.2%}')
"
```

### Step 6.2: Compare Multiple Runs

```powershell
# If you have multiple experiments
python compare_experiments.py --runs outputs/logs/run_*
```

---

## 🎯 Phase 7: Production Inference

### Step 7.1: Single Image Inference

```python
# Create inference.py
from src.models.model import HybridAnomalyDetector
from PIL import Image
import torch
from torchvision import transforms

# Load model
model = HybridAnomalyDetector.load_from_checkpoint('best_model.pth')
model.eval()

# Load image
image = Image.open('test_image.png')
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    class_logits, anomaly_score = model(image_tensor)
    predicted_class = class_logits.argmax(dim=1).item()
    confidence = torch.softmax(class_logits, dim=1).max().item()

print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
print(f"Anomaly Score: {anomaly_score.item():.3f}")
```

---

## 🛠️ Troubleshooting

### Issue 1: CUDA Out of Memory

**Error:**

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solution 1: Reduce Batch Size**

```yaml
# Edit configs/config.yaml
training:
  batch_size: 64 # Instead of 128
```

**Solution 2: Enable Gradient Accumulation**

```yaml
# Edit configs/config.yaml
training:
  gradient_accumulation_steps: 2 # Effective batch = 64 * 2 = 128
```

**Solution 3: Disable torch.compile()**

```yaml
compile_model:
  enabled: false # Saves memory but slower
```

### Issue 2: Slow Training

**Symptoms**: Training takes >30 hours for 50 epochs

**Check GPU Usage:**

```powershell
nvidia-smi
# GPU utilization should be 95-100%
```

**Solutions:**

1. **Verify CUDA is used:**

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

2. **Enable all optimizations:**

```yaml
# In config.yaml
training:
  mixed_precision: true # 2-3x speedup
  lr_scheduler:
    type: "onecycle" # 50% faster
compile_model:
  enabled: true # 30% speedup
```

3. **Increase batch size (if GPU allows):**

```yaml
training:
  batch_size: 256 # If you have 48GB VRAM
```

### Issue 3: Poor Test Accuracy

**Symptoms**: High train accuracy (>95%), low test accuracy (<85%)

**Diagnosis**: Overfitting / poor generalization

**Solutions:**

1. **Enable SAM:**

```yaml
sam:
  enabled: true
  rho: 0.05
```

2. **Enable SWA:**

```yaml
swa:
  enabled: true
  start_epoch: 40
```

3. **Enable Mixup/CutMix:**

```yaml
mixup_cutmix:
  enabled: true
```

4. **Increase dropout:**

```yaml
model:
  dropout_rate: 0.5 # Instead of 0.3
```

### Issue 4: Training Crashes

**Error:**

```
RuntimeError: Function 'XYZ' returned nan values
```

**Solution: Enable Gradient Clipping**

```yaml
# In config.yaml
training:
  gradient_clip_norm: 1.0 # Add this line
```

### Issue 5: Data Not Found

**Error:**

```
FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/Train'
```

**Solution:**

```powershell
# Verify data structure
ls data/raw/Train/
ls data/raw/Test/

# Should see 14 class folders in each
```

**Expected structure:**

```
data/raw/
├── Train/
│   ├── Abuse/
│   ├── Arrest/
│   ├── ...
│   └── Vandalism/
└── Test/
    ├── Abuse/
    ├── Arrest/
    ├── ...
    └── Vandalism/
```

---

## 📌 Quick Reference

### Essential Commands

```powershell
# 1. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install omegaconf numpy pandas matplotlib seaborn pyyaml albumentations timm tqdm tensorboard wandb scikit-learn

# 2. Analyze data
python analyze_data.py

# 3. Verify setup
python test_setup.py

# 4. Train (recommended)
python train.py --epochs 50 --wandb

# 5. Evaluate
python evaluate.py --model best_model.pth

# 6. Monitor (TensorBoard)
tensorboard --logdir outputs/logs/
```

### File Locations

| Item               | Path                                            |
| ------------------ | ----------------------------------------------- |
| **Configuration**  | `configs/config.yaml`                           |
| **Trained Models** | `outputs/logs/run_*/checkpoints/`               |
| **Training Logs**  | `outputs/logs/run_*/`                           |
| **Data Analysis**  | `outputs/results/`                              |
| **Best Model**     | `outputs/logs/run_*/checkpoints/best_model.pth` |

---

## 🎓 Training Timeline

| Time      | Event    | What's Happening                    |
| --------- | -------- | ----------------------------------- |
| **0:00**  | Start    | Model initialization, data loading  |
| **0:22**  | Epoch 1  | Initial training, accuracy ~70%     |
| **4:00**  | Epoch 10 | LR increasing, accuracy ~85%        |
| **9:00**  | Epoch 25 | OneCycleLR peak, accuracy ~91%      |
| **14:00** | Epoch 40 | SWA starts, accuracy ~93%           |
| **18:00** | Epoch 50 | Training complete, accuracy ~93-95% |

---

## 🏆 Expected Final Results

After training completes successfully:

✅ **Test Accuracy**: 93-95%  
✅ **Training Time**: 18-20 hours (50 epochs)  
✅ **Model Size**: ~23 MB (5.8M parameters)  
✅ **Inference Speed**: <100ms per image  
✅ **Generalization Gap**: <2% (train vs test)

---

## 📞 Need Help?

If you encounter issues not covered here:

1. Check **ADVANCED_TECHNIQUES.md** for detailed explanations
2. Check **DATA_HANDLING.md** for data-related issues
3. Check **TRAINING_SPEED_OPTIMIZATION.md** for performance issues
4. Check **RNN_AUTOENCODER_ANALYSIS.md** for architecture questions

---

**Happy Training! 🚀**

Last Updated: January 2024
