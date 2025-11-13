"""
Quick Integration Test for Professional Fusion System
Tests the fusion engine with sample data through the unified pipeline
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "backend"))

import numpy as np
import cv2

print("="*70)
print("🧪 PROFESSIONAL FUSION SYSTEM - INTEGRATION TEST")
print("="*70)
print()

# Test 1: Import fusion engine
print("📦 Test 1: Importing Fusion Engine...")
try:
    from backend.services.intelligent_fusion import IntelligentFusionEngine, AnomalyType
    engine = IntelligentFusionEngine()
    print("   ✅ Fusion Engine imported successfully")
    print(f"   ✅ Detection threshold: {engine.ANOMALY_THRESHOLD}")
    print(f"   ✅ Weights: ML={engine.WEIGHT_ML}, YOLO={engine.WEIGHT_OBJECTS}, Pose={engine.WEIGHT_POSE}, Motion={engine.WEIGHT_MOTION}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

print()

# Test 2: Test critical override (Weapon Detection)
print("🔫 Test 2: Critical Override (Weapon Detection)...")
try:
    detection = engine.fuse_detections(
        ml_result={'class': 'Normal', 'confidence': 0.95, 'probabilities': [0.95, 0.05]},
        yolo_detections=[
            {'class': 'knife', 'bbox': (100, 100, 50, 50), 'confidence': 0.95},
            {'class': 'person', 'bbox': (200, 200, 80, 120), 'confidence': 0.90}
        ],
        pose_result={'is_anomalous': False, 'anomaly_type': None, 'confidence': 0.1},
        motion_result={'is_unusual': False, 'anomaly_type': None, 'confidence': 0.1},
        frame_number=1
    )
    
    if detection:
        print(f"   ✅ Weapon detected: {detection.anomaly_type.value}")
        print(f"   ✅ Severity: {detection.severity.value}")
        print(f"   ✅ Fusion Score: {detection.fusion_score:.3f}")
    else:
        print("   ❌ Weapon not detected")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 3: Person Falling Detection (ML Override)
print("🚑 Test 3: Person Falling Detection (ML says Normal)...")
try:
    detection = engine.fuse_detections(
        ml_result={'class': 'Normal', 'confidence': 0.80, 'probabilities': [0.80, 0.20]},
        yolo_detections=[
            {'class': 'person', 'bbox': (150, 150, 70, 100), 'confidence': 0.92}
        ],
        pose_result={'is_anomalous': True, 'anomaly_type': 'PERSON_FALLING', 'confidence': 0.92},
        motion_result={'is_unusual': True, 'anomaly_type': 'RAPID_MOVEMENT', 'confidence': 0.85},
        frame_number=2
    )
    
    if detection and detection.anomaly_type == AnomalyType.PERSON_FALLING:
        print(f"   ✅ Person falling detected!")
        print(f"   ✅ ML said: Normal (ignored)")
        print(f"   ✅ Fusion Score: {detection.fusion_score:.3f}")
        print(f"   ✅ Reasoning: {detection.reasoning[:2]}")
    else:
        print(f"   ❌ Person falling not detected: {detection}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 4: Normal Scene (No Report)
print("✅ Test 4: Normal Scene (Should NOT report)...")
try:
    detection = engine.fuse_detections(
        ml_result={'class': 'Normal', 'confidence': 0.98, 'probabilities': [0.98, 0.02]},
        yolo_detections=[
            {'class': 'person', 'bbox': (100, 100, 60, 80), 'confidence': 0.90},
            {'class': 'car', 'bbox': (300, 200, 150, 100), 'confidence': 0.85}
        ],
        pose_result={'is_anomalous': False, 'anomaly_type': None, 'confidence': 0.1},
        motion_result={'is_unusual': False, 'anomaly_type': None, 'confidence': 0.1},
        frame_number=3
    )
    
    if detection is None:
        print("   ✅ Correctly ignored normal scene (no anomaly reported)")
    else:
        print(f"   ❌ False alarm! Reported: {detection.anomaly_type.value}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print()

# Test 5: Detection History
print("📊 Test 5: Detection History & Statistics...")
try:
    history = engine.detection_history
    
    print(f"   ✅ Total detections: {len(history)}")
    print(f"   ✅ Frames processed: {engine.frame_count}")
    
    if len(history) > 0:
        print(f"   ✅ Recent detections:")
        for det in history[:3]:
            print(f"      - {det.anomaly_type.value} (Score: {det.fusion_score:.3f})")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print()
print("="*70)
print("✨ INTEGRATION TEST COMPLETE!")
print("="*70)
print()
print("🚀 Next Steps:")
print("   1. Start backend: .\\START_BACKEND.ps1")
print("   2. Start frontend: .\\START_FRONTEND.ps1")
print("   3. Open http://localhost:3000 and test live camera")
print()
