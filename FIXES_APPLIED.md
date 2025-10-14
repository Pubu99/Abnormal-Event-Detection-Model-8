# Fixes Applied to Resolve Training Issues

## Date: October 14, 2025

### Issues Found:

1. **Import Error**: `MixupCutmixWrapper` and `mixup_criterion` not exported
2. **Deprecation Warning**: `autocast()` and `GradScaler()` syntax outdated
3. **CUDA Graph Error**: `torch.compile()` conflicting with SAM's double forward pass
4. **NaN Loss Issue**: Training producing NaN values (likely gradient explosion)
5. **SWA Model Wrapper**: `AveragedModel` wrapper breaking `anomaly_head.center` access

### Fixes Applied:

#### 1. Fixed Missing Imports ✅
- Added `MixupCutmixWrapper` and `mixup_criterion` to `src/data/__init__.py`

#### 2. Updated Deprecated PyTorch API ✅
- Changed `from torch.cuda.amp import autocast, GradScaler` to `from torch.amp import autocast, GradScaler`
- Updated all `autocast()` calls to `autocast(device_type="cuda")`
- Updated `GradScaler()` to `GradScaler('cuda')`

#### 3. Fixed torch.compile() + SAM Conflict ✅
- Disabled `torch.compile()` when SAM is enabled
- CUDA graphs don't work with double forward passes

#### 4. Fixed SWA Model Wrapper Issue ✅
- Created helper methods: `_get_actual_model()` and `_get_svdd_center()`
- Replaced all `self.model.anomaly_head.center` with `self._get_svdd_center()`
- This unwraps the `AveragedModel` wrapper when SWA is active

### Remaining Issue: NaN Losses

**Symptoms**:
- Training reaches epoch 16
- All losses become NaN
- Accuracy drops to near 0%
- Early stopping triggers

**Likely Causes**:
1. Gradient explosion with SAM optimizer
2. Numerical instability in loss calculation
3. Learning rate too high for SAM
4. Mixed precision causing numerical issues

**Recommended Solutions** (in order of priority):

### Solution 1: Disable SAM Temporarily ⭐ RECOMMENDED
```yaml
# In configs/config.yaml
training:
  sam:
    enabled: false  # Disable SAM
```

**Reasoning**: SAM requires two forward passes and is more prone to gradient explosion. Test without it first.

### Solution 2: Reduce Learning Rate
```yaml
training:
  learning_rate: 0.0001  # Reduce from 0.001
  max_learning_rate: 0.001  # Reduce from 0.01
```

### Solution 3: Add Gradient Clipping
```yaml
training:
  gradient_clip:
    enabled: true
    max_norm: 0.5  # Reduce from 1.0
```

### Solution 4: Disable Mixed Precision Temporarily
```yaml
training:
  mixed_precision: false
```

### Solution 5: Check Loss Weights
The combined loss might be unstable. Consider reducing SVDD weight:
```python
# In src/models/losses.py CombinedLoss
self.weight_svdd = 0.1  # Reduce from 0.3
```

---

## Quick Fix Command:

```bash
# Edit config to disable SAM
nano configs/config.yaml
# Set sam.enabled: false

# Then run training
python train.py --epochs 50
```

---

## Status: READY TO TRAIN

With SAM disabled, the model should train stably. You can enable SAM later once basic training works.
