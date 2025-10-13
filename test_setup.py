"""
Quick test script to verify installation and data loading.
Run this before starting full training.

Usage:
    python test_setup.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all required packages are installed."""
    print("="*70)
    print(" 🧪 TESTING IMPORTS")
    print("="*70)
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('cv2', 'OpenCV'),
        ('albumentations', 'Albumentations'),
        ('timm', 'TIMM'),
        ('sklearn', 'Scikit-learn'),
        ('tqdm', 'tqdm'),
        ('yaml', 'PyYAML'),
        ('tensorboard', 'TensorBoard'),
        ('loguru', 'Loguru'),
    ]
    
    failed = []
    for module, name in required_packages:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - NOT INSTALLED")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All packages installed successfully!")
    return True


def test_cuda():
    """Test CUDA availability."""
    print("\n" + "="*70)
    print(" 🖥️  TESTING CUDA")
    print("="*70)
    
    import torch
    
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   GPU name: {torch.cuda.get_device_name(0)}")
        
        # Test GPU computation
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = x @ y
            print(f"   ✅ GPU computation test passed")
        except Exception as e:
            print(f"   ❌ GPU computation test failed: {e}")
            return False
    else:
        print("   ⚠️  CUDA not available - will use CPU (slower)")
    
    return True


def test_config():
    """Test configuration loading."""
    print("\n" + "="*70)
    print(" ⚙️  TESTING CONFIGURATION")
    print("="*70)
    
    try:
        from src.utils import ConfigManager
        
        config = ConfigManager()
        print(f"   ✅ Config loaded successfully")
        print(f"   Project: {config.config.project.name}")
        print(f"   Classes: {config.num_classes}")
        print(f"   Device: {config.device}")
        return True
    except Exception as e:
        print(f"   ❌ Config loading failed: {e}")
        return False


def test_data():
    """Test data loading."""
    print("\n" + "="*70)
    print(" 📦 TESTING DATA LOADING")
    print("="*70)
    
    try:
        from src.utils import ConfigManager
        from src.data import UCFCrimeDataset, get_train_transforms
        
        config = ConfigManager().config
        transform = get_train_transforms(config)
        
        # Test training data
        print("\n   Testing training data...")
        train_dataset = UCFCrimeDataset(
            root_dir=config.data.root_dir,
            split='train',
            transform=transform,
            normal_class_idx=config.data.normal_class_idx
        )
        
        # Load one sample
        image, label, is_anomaly = train_dataset[0]
        print(f"   ✅ Train dataset: {len(train_dataset):,} samples")
        print(f"   Sample shape: {image.shape}")
        print(f"   Sample label: {label} ({train_dataset.classes[label]})")
        print(f"   Is anomaly: {is_anomaly}")
        
        # Test test data
        print("\n   Testing test data...")
        test_dataset = UCFCrimeDataset(
            root_dir=config.data.root_dir,
            split='test',
            transform=transform,
            normal_class_idx=config.data.normal_class_idx
        )
        
        print(f"   ✅ Test dataset: {len(test_dataset):,} samples")
        
        return True
    except FileNotFoundError as e:
        print(f"   ❌ Data directory not found: {e}")
        print("   Make sure data is in the correct location:")
        print("   data/raw/Train/ and data/raw/Test/")
        return False
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test model creation."""
    print("\n" + "="*70)
    print(" 🤖 TESTING MODEL")
    print("="*70)
    
    try:
        import torch
        from src.utils import ConfigManager
        from src.models import create_model
        
        config = ConfigManager().config
        
        # Create model
        device = 'cpu'  # Use CPU for testing
        model = create_model(config, device=device)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 64, 64)
        outputs = model(dummy_input)
        
        print(f"   ✅ Model created successfully")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Class logits shape: {outputs['class_logits'].shape}")
        print(f"   Binary logits shape: {outputs['binary_logits'].shape}")
        print(f"   Anomaly scores shape: {outputs['anomaly_scores'].shape}")
        
        return True
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """Test dataloader with batching."""
    print("\n" + "="*70)
    print(" 🔄 TESTING DATALOADER")
    print("="*70)
    
    try:
        from src.utils import ConfigManager
        from src.data import create_dataloaders
        
        config = ConfigManager().config
        
        # Temporarily reduce batch size for testing
        original_batch_size = config.training.batch_size
        config.training.batch_size = 8
        
        print(f"   Creating dataloaders (batch_size={config.training.batch_size})...")
        train_loader, val_loader, _, _ = create_dataloaders(config)
        
        # Test one batch
        images, labels, is_anomaly = next(iter(train_loader))
        
        print(f"   ✅ Dataloaders created successfully")
        print(f"   Batch shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        
        # Restore original batch size
        config.training.batch_size = original_batch_size
        
        return True
    except Exception as e:
        print(f"   ❌ Dataloader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """Test one training step."""
    print("\n" + "="*70)
    print(" 🏃 TESTING TRAINING STEP")
    print("="*70)
    
    try:
        import torch
        from src.utils import ConfigManager
        from src.models import create_model, create_loss_function
        from src.data import create_dataloaders
        
        config = ConfigManager().config
        config.training.batch_size = 4  # Small batch for testing
        device = 'cpu'  # Use CPU for testing
        
        # Create model and loss
        model = create_model(config, device=device)
        criterion = create_loss_function(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create dataloader
        train_loader, _, _, _ = create_dataloaders(config)
        
        # One training step
        model.train()
        images, labels, is_anomaly = next(iter(train_loader))
        images = images.to(device)
        labels = labels.to(device)
        is_anomaly = is_anomaly.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, return_embeddings=True)
        losses = criterion(outputs, labels, is_anomaly, model.anomaly_head.center)
        loss = losses['total']
        loss.backward()
        optimizer.step()
        
        print(f"   ✅ Training step successful")
        print(f"   Total loss: {loss.item():.4f}")
        print(f"   Classification loss: {losses['classification'].item():.4f}")
        print(f"   Binary loss: {losses['binary'].item():.4f}")
        
        return True
    except Exception as e:
        print(f"   ❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" 🚀 RUNNING SETUP VERIFICATION TESTS")
    print("="*70)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("CUDA", test_cuda),
        ("Configuration", test_config),
        ("Data Loading", test_data),
        ("Model Creation", test_model),
        ("DataLoader", test_dataloader),
        ("Training Step", test_training_step),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} test crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print(" 📊 TEST SUMMARY")
    print("="*70)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} - {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print(" ✅ ALL TESTS PASSED - READY TO TRAIN!")
        print("="*70)
        print("\nNext steps:")
        print("   1. Review configs/config.yaml")
        print("   2. Start training: python train.py")
        print("   3. Monitor with: tensorboard --logdir outputs/logs/")
        return 0
    else:
        print(" ❌ SOME TESTS FAILED - FIX ISSUES BEFORE TRAINING")
        print("="*70)
        print("\nPlease fix the failed tests and run again.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
