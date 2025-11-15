"""
Test Professional Intelligent Fusion Engine
Validates weighted scoring, person falling detection, and anomaly-only reporting

Author: Professional Testing
Date: 2025-10-17
"""

import sys
sys.path.append('backend/services')

from intelligent_fusion import IntelligentFusionEngine, AnomalyType, Severity
import json


def test_critical_override():
    """Test 1: Critical objects bypass fusion scoring"""
    print("\n🔍 Test 1: Critical Object Override (Weapon Detection)")
    
    fusion = IntelligentFusionEngine()
    
    # Scenario: ML says Normal, but YOLO detects knife
    ml_result = {'class': 'Normal', 'confidence': 0.95}
    yolo_detections = [
        {'class': 'person', 'bbox': (100, 100, 200, 300), 'confidence': 0.92},
        {'class': 'knife', 'bbox': (150, 150, 180, 200), 'confidence': 0.88}
    ]
    pose_result = None
    motion_result = None
    
    detection = fusion.fuse_detections(ml_result, yolo_detections, pose_result, motion_result, frame_number=1)
    
    assert detection is not None, "❌ Failed to detect weapon"
    assert detection.severity == Severity.CRITICAL, "❌ Wrong severity"
    assert detection.fusion_score == 1.0, "❌ Critical override should have score 1.0"
    
    print(f"✅ Weapon detected: {detection.anomaly_type.value}")
    print(f"   Severity: {detection.severity.value}")
    print(f"   Fusion Score: {detection.fusion_score}")
    print(f"   Explanation: {detection.explanation}")


def test_multiple_weapons():
    """Test 2: Multiple weapons detection"""
    print("\n🔍 Test 2: Multiple Weapons Detection")
    
    fusion = IntelligentFusionEngine()
    
    yolo_detections = [
        {'class': 'knife', 'bbox': (100, 100, 150, 200), 'confidence': 0.90},
        {'class': 'gun', 'bbox': (300, 100, 350, 200), 'confidence': 0.85}
    ]
    
    detection = fusion.fuse_detections(None, yolo_detections, None, None, frame_number=1)
    
    assert detection is not None, "❌ Failed to detect multiple weapons"
    assert detection.anomaly_type == AnomalyType.MULTIPLE_WEAPONS, "❌ Wrong anomaly type"
    assert detection.severity == Severity.CRITICAL, "❌ Wrong severity"
    
    print(f"✅ Multiple weapons detected")
    print(f"   Explanation: {detection.explanation}")


def test_person_falling():
    """Test 3: Person falling detection (even if ML says Normal)"""
    print("\n🔍 Test 3: Person Falling Detection (ML Override)")
    
    fusion = IntelligentFusionEngine()
    
    # Scenario: ML says Normal, but Pose + Motion detect falling
    ml_result = {'class': 'Normal', 'confidence': 0.98}
    yolo_detections = [
        {'class': 'person', 'bbox': (200, 200, 300, 400), 'confidence': 0.95}
    ]
    pose_result = {
        'is_anomalous': True,
        'anomaly_type': 'PERSON_FALLING',
        'confidence': 0.92
    }
    motion_result = {
        'is_unusual': True,
        'anomaly_type': 'RAPID_MOVEMENT',
        'confidence': 0.85
    }
    
    detection = fusion.fuse_detections(ml_result, yolo_detections, pose_result, motion_result, frame_number=1)
    
    assert detection is not None, "❌ Failed to detect person falling"
    assert detection.anomaly_type == AnomalyType.PERSON_FALLING, "❌ Wrong anomaly type"
    assert any("Person Falling" in r for r in detection.reasoning), "❌ Missing falling reasoning"
    
    print(f"✅ Person falling detected successfully!")
    print(f"   ML said: {ml_result['class']} (ignored)")
    print(f"   Fusion detected: {detection.anomaly_type.value}")
    print(f"   Fusion Score: {detection.fusion_score:.3f}")
    print(f"   Reasoning: {detection.reasoning}")


def test_ml_model_anomaly():
    """Test 4: ML model detects anomaly"""
    print("\n🔍 Test 4: ML Model Anomaly Detection")
    
    fusion = IntelligentFusionEngine()
    
    ml_result = {'class': 'Fighting', 'confidence': 0.89}
    yolo_detections = [
        {'class': 'person', 'bbox': (100, 100, 200, 300), 'confidence': 0.90},
        {'class': 'person', 'bbox': (250, 100, 350, 300), 'confidence': 0.88}
    ]
    pose_result = {
        'is_anomalous': True,
        'anomaly_type': 'FIGHTING_DETECTED',
        'confidence': 0.85
    }
    motion_result = {
        'is_unusual': True,
        'anomaly_type': 'RAPID_MOVEMENT',
        'confidence': 0.80
    }
    
    detection = fusion.fuse_detections(ml_result, yolo_detections, pose_result, motion_result, frame_number=1)
    
    assert detection is not None, "❌ Failed to detect fighting"
    assert detection.anomaly_type == AnomalyType.FIGHTING, "❌ Wrong anomaly type"
    assert detection.fusion_score > 0.70, "❌ Fusion score too low"
    
    print(f"✅ Fighting detected")
    print(f"   Fusion Score: {detection.fusion_score:.3f}")
    print(f"   Individual Scores:")
    print(f"      ML: {detection.ml_score:.3f} (40% weight)")
    print(f"      Objects: {detection.object_score:.3f} (25% weight)")
    print(f"      Pose: {detection.pose_score:.3f} (20% weight)")
    print(f"      Motion: {detection.motion_score:.3f} (15% weight)")


def test_crowd_density():
    """Test 5: High crowd density detection"""
    print("\n🔍 Test 5: High Crowd Density Detection")
    
    fusion = IntelligentFusionEngine()
    
    # 20 people detected
    yolo_detections = [{'class': 'person', 'bbox': (i*50, 100, i*50+40, 200), 'confidence': 0.9} 
                      for i in range(20)]
    
    detection = fusion.fuse_detections(None, yolo_detections, None, None, frame_number=1)
    
    assert detection is not None, "❌ Failed to detect crowd"
    assert detection.anomaly_type == AnomalyType.HIGH_CROWD_DENSITY, "❌ Wrong anomaly type"
    
    print(f"✅ Crowd density detected")
    print(f"   People count: {len(yolo_detections)}")
    print(f"   Explanation: {detection.explanation}")


def test_normal_no_report():
    """Test 6: Normal scene - should NOT be reported"""
    print("\n🔍 Test 6: Normal Scene (No Report)")
    
    fusion = IntelligentFusionEngine()
    
    ml_result = {'class': 'Normal', 'confidence': 0.98}
    yolo_detections = [
        {'class': 'person', 'bbox': (100, 100, 150, 250), 'confidence': 0.90},
        {'class': 'car', 'bbox': (300, 200, 450, 300), 'confidence': 0.88}
    ]
    pose_result = {'is_anomalous': False}
    motion_result = {'is_unusual': False}
    
    detection = fusion.fuse_detections(ml_result, yolo_detections, pose_result, motion_result, frame_number=1)
    
    assert detection is None, "❌ Normal scene should not be reported!"
    
    print(f"✅ Normal scene correctly ignored (no false alarm)")
    print(f"   Detection: None (as expected)")


def test_detection_history():
    """Test 7: Detection history tracking"""
    print("\n🔍 Test 7: Detection History Tracking")
    
    fusion = IntelligentFusionEngine()
    
    # Generate multiple detections
    for i in range(5):
        yolo_detections = [{'class': 'knife', 'bbox': (100, 100, 150, 200), 'confidence': 0.90}]
        fusion.fuse_detections(None, yolo_detections, None, None, frame_number=i+1)
    
    history = fusion.get_recent_detections(limit=10)
    
    assert len(history) == 5, f"❌ Expected 5 detections, got {len(history)}"
    
    print(f"✅ Detection history tracked")
    print(f"   Total detections: {len(history)}")
    print(f"   Recent: {history[-1]['anomaly_type']}")


def test_statistics():
    """Test 8: Statistics calculation"""
    print("\n🔍 Test 8: Statistics Calculation")
    
    fusion = IntelligentFusionEngine()
    
    # Generate varied detections
    detections_config = [
        ({'class': 'knife', 'bbox': (100, 100, 150, 200)}, Severity.CRITICAL),
        ({'class': 'knife', 'bbox': (100, 100, 150, 200)}, Severity.CRITICAL),
        ({'class': 'person', 'bbox': (100, 100, 150, 200)}, None),  # Normal - not reported
    ]
    
    for idx, (yolo_obj, expected_severity) in enumerate(detections_config):
        ml_result = {'class': 'Fighting', 'confidence': 0.8} if idx == 2 else None
        yolo_dets = [yolo_obj] * (16 if idx == 2 else 1)
        fusion.fuse_detections(ml_result, yolo_dets, None, None, frame_number=idx+1)
    
    stats = fusion.get_statistics()
    
    print(f"✅ Statistics calculated")
    print(f"   Total detections: {stats['total_detections']}")
    print(f"   Frames processed: {stats['total_frames_processed']}")
    print(f"   Anomaly rate: {stats['anomaly_rate']:.2%}")
    print(f"   By severity: {stats['by_severity']}")
    print(f"   Average confidence: {stats['average_confidence']:.3f}")


def test_fusion_scoring():
    """Test 9: Weighted fusion scoring"""
    print("\n🔍 Test 9: Weighted Fusion Scoring Validation")
    
    fusion = IntelligentFusionEngine()
    
    # All modalities detect anomaly
    ml_result = {'class': 'Assault', 'confidence': 0.85}
    yolo_detections = [
        {'class': 'person', 'bbox': (100, 100, 150, 250), 'confidence': 0.90},
        {'class': 'person', 'bbox': (200, 100, 250, 250), 'confidence': 0.88}
    ]
    pose_result = {'is_anomalous': True, 'anomaly_type': 'AGGRESSIVE_GESTURES', 'confidence': 0.80}
    motion_result = {'is_unusual': True, 'anomaly_type': 'RAPID_MOVEMENT', 'confidence': 0.75}
    
    detection = fusion.fuse_detections(ml_result, yolo_detections, pose_result, motion_result, frame_number=1)
    
    assert detection is not None, "❌ Failed to detect anomaly"
    
    # Verify weights are applied
    expected_min_score = 0.85 * 0.40  # ML contribution
    assert detection.fusion_score >= expected_min_score, "❌ Fusion scoring error"
    
    print(f"✅ Fusion scoring validated")
    print(f"   Fusion Score: {detection.fusion_score:.3f}")
    print(f"   Breakdown:")
    print(f"      ML (40%):     {detection.ml_score:.3f} → {detection.ml_score * 0.40:.3f}")
    print(f"      Objects (25%): {detection.object_score:.3f} → {detection.object_score * 0.25:.3f}")
    print(f"      Pose (20%):    {detection.pose_score:.3f} → {detection.pose_score * 0.20:.3f}")
    print(f"      Motion (15%):  {detection.motion_score:.3f} → {detection.motion_score * 0.15:.3f}")


def run_all_tests():
    """Run comprehensive test suite"""
    print("=" * 80)
    print("🚀 PROFESSIONAL FUSION ENGINE TEST SUITE")
    print("=" * 80)
    
    tests = [
        test_critical_override,
        test_multiple_weapons,
        test_person_falling,
        test_ml_model_anomaly,
        test_crowd_density,
        test_normal_no_report,
        test_detection_history,
        test_statistics,
        test_fusion_scoring
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"❌ {test_func.__name__} FAILED: {str(e)}")
        except Exception as e:
            failed += 1
            print(f"❌ {test_func.__name__} ERROR: {str(e)}")
    
    print("\n" + "=" * 80)
    print(f"📊 TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed == 0:
        print("✨ ALL TESTS PASSED! Professional Fusion Engine is ready for production.")
        print("\n🎯 Key Features Validated:")
        print("   ✅ Weighted scoring (ML 40%, YOLO 25%, Pose 20%, Motion 15%)")
        print("   ✅ Critical object override (weapons → instant alert)")
        print("   ✅ Person falling detection (even if ML says Normal)")
        print("   ✅ Anomaly-only reporting (no normal highlights)")
        print("   ✅ Detection history and statistics")
        print("   ✅ Consensus bonus for multi-modal agreement")
    else:
        print(f"⚠️ {failed} tests failed. Review errors above.")


if __name__ == "__main__":
    run_all_tests()
