import sys
sys.path.insert(0, '.')

from pathlib import Path
from inference.engine import AnomalyDetector

print("=" * 70)
print("Testing AnomalyDetector initialization...")
print("=" * 70)

try:
    detector = AnomalyDetector(
        model_path="models/best_model.pth",
        config_path="configs/config_research_enhanced.yaml",
        yolo_model="yolov8n.pt",
        device="cpu",
        confidence_threshold=0.7
    )
    print("\n✅✅✅ SUCCESS! Detector initialized successfully!")
    print(f"Model ready on device: {detector.device}")
    print(f"YOLO detector: {detector.detector_name}")
except Exception as e:
    print(f"\n❌❌❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
