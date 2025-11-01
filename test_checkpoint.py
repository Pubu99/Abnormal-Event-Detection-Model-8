import torch
print("Testing checkpoint load...")

try:
    ckpt = torch.load('models/best_model.pth', map_location='cpu')
    print("✅ SUCCESS with weights_only default")
except Exception as e1:
    print(f"❌ Failed with default: {type(e1).__name__}: {str(e1)[:100]}")
    
    try:
        ckpt = torch.load('models/best_model.pth', map_location='cpu', weights_only=False)
        print("✅ SUCCESS with weights_only=False")
        print(f"Checkpoint type: {type(ckpt)}")
        if isinstance(ckpt, dict):
            print(f"Keys: {list(ckpt.keys())[:10]}")
    except Exception as e2:
        print(f"❌ Failed with weights_only=False: {e2}")
