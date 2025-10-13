# 🚀 Setup and Training Guide

## Quick Start (5 Steps to Training)

### 1️⃣ Install Dependencies

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install PyTorch with CUDA 12.8 (RTX 5090)
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install -r requirements.txt
```

### 2️⃣ Verify GPU Setup

```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'CUDA Version: {torch.version.cuda}')"
```

Expected output:

```
CUDA Available: True
GPU: NVIDIA GeForce RTX 5090
CUDA Version: 12.8
```

### 3️⃣ Verify Data Structure

Make sure your data is organized as:

```
data/raw/
├── Train/
│   ├── Abuse/
│   ├── Arrest/
│   ├── Arson/
│   ├── Assault/
│   ├── Burglary/
│   ├── Explosion/
│   ├── Fighting/
│   ├── NormalVideos/
│   ├── RoadAccidents/
│   ├── Robbery/
│   ├── Shooting/
│   ├── Shoplifting/
│   ├── Stealing/
│   └── Vandalism/
└── Test/
    └── (same structure)
```

### 4️⃣ Configure Training (Optional)

Edit `configs/config.yaml` to customize:

- Batch size (default: 128 for RTX 5090)
- Learning rate
- Epochs
- Model architecture
- Augmentation settings

### 5️⃣ Start Training!

```powershell
# Basic training
python train.py

# With custom config
python train.py --config configs/config.yaml

# Enable W&B logging
python train.py --wandb

# Custom batch size and epochs
python train.py --batch-size 256 --epochs 50
```

---

## 📊 Monitor Training

### TensorBoard

```powershell
tensorboard --logdir outputs/logs/
```

Open browser: http://localhost:6006

### Weights & Biases

If enabled, W&B dashboard opens automatically.

---

## 🧪 Evaluate Model

After training:

```powershell
# Evaluate best model
python evaluate.py --checkpoint outputs/logs/EXPERIMENT_NAME/checkpoints/best_model.pth

# Save predictions
python evaluate.py --checkpoint outputs/logs/EXPERIMENT_NAME/checkpoints/best_model.pth --save-predictions
```

---

## 📈 Expected Performance

With the provided configuration and RTX 5090:

| Metric         | Target                | Expected Time  |
| -------------- | --------------------- | -------------- |
| Training Speed | ~800-1200 samples/sec | -              |
| Epoch Time     | ~15-20 min            | Per epoch      |
| Total Training | 25-33 hours           | 100 epochs     |
| Val Accuracy   | >95%                  | By epoch 50-70 |
| Val F1-Score   | >0.93                 | By epoch 50-70 |

**Memory Usage:**

- Model Size: ~15 MB
- Training VRAM: ~8-12 GB (batch size 128)
- Can increase batch size to 256-512 on RTX 5090

---

## 🔧 Troubleshooting

### Out of Memory (OOM)

```powershell
# Reduce batch size in config.yaml
training:
  batch_size: 64  # Reduce from 128
```

### Slow Training

```powershell
# Increase num_workers in config.yaml
data:
  num_workers: 12  # Increase from 8
```

### Low Accuracy

1. Train longer (100+ epochs)
2. Adjust learning rate
3. Enable data augmentation
4. Use class weights (already enabled)

### CUDA Errors

```powershell
# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"
```

---

## 📁 Output Structure

After training:

```
outputs/
├── logs/
│   └── baseline_v1_TIMESTAMP/
│       ├── checkpoints/
│       │   ├── best_model.pth
│       │   └── checkpoint_epoch_*.pth
│       ├── tensorboard/
│       ├── train.log
│       └── wandb/
├── models/
└── results/
    ├── evaluation.txt
    └── predictions.csv
```

---

## 🎯 Next Steps After Training

1. **Evaluate on Test Set**

   ```powershell
   python evaluate.py --checkpoint outputs/logs/.../best_model.pth
   ```

2. **Analyze Results**

   - Check confusion matrix
   - Review per-class performance
   - Identify misclassifications

3. **Optimize Model**

   - Adjust hyperparameters
   - Try different architectures
   - Add more augmentation

4. **Deploy Model**
   - Export to ONNX for production
   - Integrate with multi-camera system
   - Build real-time inference pipeline

---

## 💡 Pro Tips

### Faster Training

1. **Mixed Precision**: Already enabled (FP16)
2. **Larger Batch Size**: Increase to 256-512 on RTX 5090
3. **More Workers**: Set `num_workers: 12`
4. **Persistent Workers**: Already enabled

### Better Accuracy

1. **Longer Training**: 100-150 epochs
2. **Learning Rate Warmup**: Already configured
3. **Label Smoothing**: Already enabled (0.1)
4. **Class Weights**: Already enabled for imbalance

### Memory Optimization

1. **Gradient Accumulation**: Set to 2-4 for effective larger batch
2. **Empty Cache**: Periodically clear CUDA cache
3. **Lower Precision**: Use FP16 (already enabled)

---

## 🔬 Advanced Configuration

### Custom Experiment

```yaml
# configs/config.yaml
experiment:
  name: "efficientnet_b3_experiment"
  notes: "Testing larger backbone"

model:
  backbone:
    type: "efficientnet_b3" # Try b1, b2, b3
```

### Different Loss Function

```yaml
training:
  loss:
    type: "focal"
    alpha: 0.25
    gamma: 2.5 # Increase for harder focus
```

### Custom Augmentation

```yaml
augmentation:
  train:
    random_rotation: 20
    random_fog: 0.3
    random_rain: 0.3
```

---

## 📞 Support

If you encounter issues:

1. Check the `train.log` file
2. Review TensorBoard graphs
3. Verify data loading with test script
4. Check GPU memory usage

---

## 🎓 Model Architecture Summary

```
Input (64x64x3)
    ↓
EfficientNet-B0 Backbone
    ↓
Bi-LSTM Temporal Encoder
    ↓
Feature Projection (512-dim)
    ↓
├─→ Multi-Class Classifier (14 classes)
├─→ Binary Anomaly Detector (Normal/Anomaly)
└─→ Deep SVDD Head (Anomaly scoring)
```

**Key Features:**

- ✅ Class imbalance handling (Focal Loss + Weighted Sampling)
- ✅ Mixed precision training (FP16)
- ✅ Multi-task learning
- ✅ Attention mechanism
- ✅ Real-time optimized (~100ms inference)

---

Happy Training! 🚀
