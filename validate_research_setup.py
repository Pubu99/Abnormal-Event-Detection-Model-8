"""
Pre-Training Validation Script.
Quick test to ensure all components work before full training.

Tests:
1. Config loads correctly
2. Sequence dataset creates properly
3. Model initializes and forward pass works
4. Trainer initializes
5. One training step completes
"""

import sys
# Avoid hardcoded absolute paths; use repository-relative imports when possible

import torch
from src.utils.config import ConfigManager
from src.models.research_model import create_research_model
from src.data.sequence_dataset import create_sequence_dataloaders
from src.training.research_trainer import ResearchTrainer

print("=" * 80)
print("🧪 Pre-Training Validation")
print("=" * 80)

# Test 1: Load config
print("\n1️⃣  Testing configuration...")
try:
    config = ConfigManager('configs/config_research_enhanced.yaml').config
    print("   ✅ Config loaded successfully")
except Exception as e:
    print(f"   ❌ Config failed: {e}")
    sys.exit(1)

# Test 2: Create dataloaders
print("\n2️⃣  Testing sequence dataset...")
try:
    train_loader, val_loader = create_sequence_dataloaders(config, use_weighted_sampling=False)
    print(f"   ✅ Dataloaders created")
    print(f"      Train: {len(train_loader.dataset):,} sequences")
    print(f"      Val: {len(val_loader.dataset):,} sequences")
except Exception as e:
    print(f"   ❌ Dataset failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Load one batch
print("\n3️⃣  Testing batch loading...")
try:
    frames, labels, future_frames = next(iter(train_loader))
    print(f"   ✅ Batch loaded")
    print(f"      Frames: {frames.shape}")
    print(f"      Labels: {labels.shape}")
    print(f"      Future frames: {future_frames.shape}")
except Exception as e:
    print(f"   ❌ Batch loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Create model
print("\n4️⃣  Testing model creation...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_research_model(config, device=device)
    print(f"   ✅ Model created on {device}")
except Exception as e:
    print(f"   ❌ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Forward pass
print("\n5️⃣  Testing forward pass...")
try:
    frames = frames.to(device)
    labels = labels.to(device)
    future_frames = future_frames.to(device)
    
    with torch.no_grad():
        outputs = model(frames, return_all=True)
    
    print(f"   ✅ Forward pass successful")
    print(f"      Regression: {outputs['regression'].shape}")
    print(f"      Classification: {outputs['class_logits'].shape}")
    if 'vae' in outputs:
        print(f"      VAE: {outputs['vae']['reconstructed'].shape}")
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Create trainer
print("\n6️⃣  Testing trainer initialization...")
try:
    trainer = ResearchTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        logger=None
    )
    print(f"   ✅ Trainer initialized")
except Exception as e:
    print(f"   ❌ Trainer initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: One training step
print("\n7️⃣  Testing one training step...")
try:
    model.train()
    optimizer = trainer.optimizer
    
    # Forward
    outputs = model(frames, return_all=True)
    
    # Compute loss
    loss, loss_dict = trainer.compute_losses(
        outputs, labels, future_frames
    )
    
    # Backward
    loss.backward()
    
    # Step
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"   ✅ Training step successful")
    print(f"      Total loss: {loss.item():.4f}")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"      {key}: {value.item():.4f}")
except Exception as e:
    print(f"   ❌ Training step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: One validation step
print("\n8️⃣  Testing one validation step...")
try:
    model.eval()
    
    with torch.no_grad():
        val_frames, val_labels, val_future = next(iter(val_loader))
        val_frames = val_frames.to(device)
        val_labels = val_labels.to(device)
        val_future = val_future.to(device)
        
        val_outputs = model(val_frames, return_all=True)
        val_loss, val_loss_dict = trainer.compute_losses(
            val_outputs, val_labels, val_future
        )
    
    print(f"   ✅ Validation step successful")
    print(f"      Total loss: {val_loss.item():.4f}")
except Exception as e:
    print(f"   ❌ Validation step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL PRE-TRAINING TESTS PASSED!")
print("=" * 80)
print("\n🚀 Ready to start full training:")
print(f"   Command: python train_research.py --config configs/config_research_enhanced.yaml")
print(f"\n   Expected training time: 6-8 hours on a single GPU")
print(f"   Target performance: 85-88% test accuracy")
print(f"   Current baseline: 54% test accuracy")
print(f"   Improvement goal: +31-34% absolute")
print("=" * 80)
