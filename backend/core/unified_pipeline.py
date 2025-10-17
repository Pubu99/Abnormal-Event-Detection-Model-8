"""
Enhanced Inference Core - Unified Detection Pipeline
Integrates all detection services for comprehensive anomaly detection

Author: AI Assistant  
Date: 2025-10-17
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from services.motion_analysis import MotionAnalyzer
from services.pose_estimation import PoseEstimator
from services.rule_engine import RuleEngine, AlertLevel
from services.object_tracking import SimpleTracker, SpeedAnalyzer
from services.intelligent_fusion import IntelligentFusionEngine, AnomalyLevel
import cv2
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import base64


class UnifiedDetectionPipeline:
    """
    Professional unified detection pipeline combining:
    - ML Model (Anomaly Classification)
    - YOLO (Object Detection)
    - Motion Analysis (Optical Flow, BG Subtraction)
    - Pose Estimation (MediaPipe)
    - Object Tracking (Centroid Tracking)
    - Rule Engine (Context-Aware Alerts)
    """
    
    def __init__(self, anomaly_detector):
        """
        Initialize unified pipeline
        
        Args:
            anomaly_detector: Existing AnomalyDetector instance
        """
        self.anomaly_detector = anomaly_detector
        
        # Initialize all services
        print("🔧 Initializing Enhanced Detection Services...")
        
        self.motion_analyzer = MotionAnalyzer(
            motion_threshold=5.0,
            static_threshold=2.0
        )
        print("   ✅ Motion Analyzer ready")
        
        self.pose_estimator = PoseEstimator(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("   ✅ Pose Estimator ready")
        
        self.tracker = SimpleTracker(
            max_disappeared=30,
            max_distance=100.0
        )
        print("   ✅ Object Tracker ready")
        
        self.speed_analyzer = SpeedAnalyzer()
        print("   ✅ Speed Analyzer ready")
        
        self.rule_engine = RuleEngine(
            crowd_density_threshold=15,
            loitering_time_seconds=30.0
        )
        print("   ✅ Rule Engine ready")
        
        self.fusion_engine = IntelligentFusionEngine()
        print("   ✅ Intelligent Fusion Engine ready")
        
        print("🚀 Unified Detection Pipeline initialized!\n")
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process single frame through complete pipeline
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Comprehensive detection result
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'frame_shape': frame.shape,
            'detections': {},
            'alerts': [],
            'visualization': None
        }
        
        try:
            # 1. YOLO Object Detection
            yolo_results = self.anomaly_detector.detect_objects(frame)
            
            # Convert YOLO format for other services
            yolo_detections = []
            for i, obj_class in enumerate(yolo_results['objects']):
                bbox = yolo_results['boxes'][i]
                x1, y1, x2, y2 = map(int, bbox)
                yolo_detections.append({
                    'class': obj_class,
                    'bbox': (x1, y1, x2 - x1, y2 - y1),  # x, y, w, h
                    'confidence': yolo_results['confidences'][i]
                })
            
            result['detections']['yolo'] = {
                'objects': yolo_results['objects'],
                'count': len(yolo_results['objects']),
                'dangerous_objects': yolo_results.get('dangerous', False)
            }
            
            # 2. ML Model Prediction (if frame buffer ready)
            self.anomaly_detector.frame_buffer.append(frame)
            
            ml_prediction = None
            if len(self.anomaly_detector.frame_buffer) == self.anomaly_detector.sequence_length:
                # Convert frames to tensor sequence
                sequence_tensor = self.anomaly_detector.create_sequence(
                    list(self.anomaly_detector.frame_buffer)
                )
                pred_result = self.anomaly_detector.predict_sequence(sequence_tensor)
                ml_prediction = pred_result
                
                result['detections']['ml_model'] = {
                    'predicted_class': pred_result['predicted_class'],
                    'confidence': pred_result['confidence'],
                    'is_anomaly': pred_result['is_anomaly'],
                    'anomaly_score': pred_result.get('anomaly_score', 0.0),
                    'top_3': pred_result.get('top3_predictions', [])  # Note: engine uses 'top3_predictions'
                }
            
            # 3. Motion Analysis
            motion_result = self.motion_analyzer.analyze(frame)
            
            result['detections']['motion'] = {
                'magnitude': motion_result.motion_magnitude,
                'direction': motion_result.motion_direction,
                'regions_count': len(motion_result.motion_regions),
                'is_unusual': motion_result.is_unusual,
                'anomaly_type': motion_result.anomaly_type,
                'confidence': motion_result.confidence
            }
            
            # 4. Pose Estimation
            pose_result = self.pose_estimator.analyze(frame)
            
            result['detections']['pose'] = {
                'persons_detected': pose_result.persons_detected,
                'is_anomalous': pose_result.is_anomalous,
                'anomaly_type': pose_result.anomaly_type,
                'confidence': pose_result.confidence
            }
            
            # 5. Object Tracking
            tracking_result = self.tracker.update(yolo_detections)
            
            tracked_objects_data = []
            for obj in tracking_result.tracked_objects:
                tracked_objects_data.append({
                    'track_id': obj.track_id,
                    'class': obj.class_name,
                    'bbox': obj.bbox,
                    'speed': obj.speed,
                    'direction': obj.direction,
                    'age': obj.age
                })
            
            result['detections']['tracking'] = {
                'tracked_objects': tracked_objects_data,
                'total_tracks': tracking_result.total_tracks,
                'new_tracks': tracking_result.new_tracks,
                'lost_tracks': tracking_result.lost_tracks
            }
            
            # 6. Speed Analysis
            speed_analysis = self.speed_analyzer.analyze(tracking_result.tracked_objects)
            result['detections']['speed'] = speed_analysis
            
            # 7. Rule Engine - Generate Context-Aware Alerts
            rule_result = self.rule_engine.evaluate(
                yolo_detections=yolo_detections,
                motion_result=result['detections']['motion'],
                pose_result=result['detections']['pose'],
                anomaly_class=ml_prediction['predicted_class'] if ml_prediction else None,
                anomaly_confidence=ml_prediction['confidence'] if ml_prediction else None
            )
            
            # Format alerts
            alerts = []
            for alert in rule_result.alerts:
                alerts.append({
                    'level': alert.level.value,
                    'title': alert.title,
                    'message': alert.message,
                    'confidence': alert.confidence,
                    'timestamp': alert.timestamp,
                    'location': alert.location,
                    'objects_involved': alert.objects_involved,
                    'metadata': alert.metadata
                })
            
            result['alerts'] = alerts
            result['threat_level'] = rule_result.threat_level.value
            result['is_dangerous'] = rule_result.is_dangerous
            result['summary'] = rule_result.summary
            
            # 8. Intelligent Fusion - Align all detections professionally
            fusion_result = self.fusion_engine.fuse_detections(
                ml_result=ml_prediction,
                yolo_objects=yolo_results['objects'],
                pose_result=result['detections']['pose'],
                motion_result=result['detections']['motion'],
                frame_number=len(self.anomaly_detector.frame_buffer),
                timestamp=result['timestamp']
            )
            
            # Add fusion results to response
            result['fusion'] = {
                'final_decision': fusion_result.final_decision.name,
                'confidence': fusion_result.confidence,
                'reasoning': fusion_result.reasoning,
                'ml_weight': fusion_result.ml_weight,
                'context_weight': fusion_result.context_weight,
                'override_applied': fusion_result.override_applied,
                'score_breakdown': {
                    'ml_score': fusion_result.ml_score,
                    'object_score': fusion_result.object_score,
                    'pose_score': fusion_result.pose_score,
                    'motion_score': fusion_result.motion_score
                }
            }
            
            # Override final decision if fusion is more confident
            if fusion_result.override_applied or fusion_result.confidence > 0.7:
                result['threat_level'] = fusion_result.final_decision.name
                result['is_dangerous'] = fusion_result.final_decision.value >= AnomalyLevel.ABNORMAL.value
            
            # 9. Create Visualization
            vis_frame = self._create_visualization(
                frame,
                yolo_detections,
                tracking_result.tracked_objects,
                pose_result,
                motion_result,
                rule_result
            )
            
            # Encode frame as JPEG base64
            _, buffer = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            result['frame_base64'] = base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            result['error'] = str(e)
            print(f"❌ Error in pipeline: {e}")
        
        return result
    
    def _create_visualization(self,
                            frame: np.ndarray,
                            yolo_detections: List[Dict],
                            tracked_objects: List,
                            pose_result,
                            motion_result,
                            rule_result) -> np.ndarray:
        """Create annotated frame with all detections"""
        
        vis_frame = frame.copy()
        
        # Draw pose skeletons
        if pose_result.is_anomalous:
            vis_frame = self.pose_estimator.draw_pose(vis_frame, pose_result)
        
        # Draw tracked objects with bounding boxes
        for obj in tracked_objects:
            x, y, w, h = obj.bbox
            
            # Color based on class
            if obj.class_name == 'person':
                color = (0, 255, 0)  # Green
            elif obj.class_name in ['knife', 'gun', 'weapon']:
                color = (0, 0, 255)  # Red
            else:
                color = (255, 144, 30)  # Blue
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with track ID and speed
            label = f"ID:{obj.track_id} {obj.class_name}"
            if obj.speed > 5:
                label += f" {obj.speed:.1f}px/f"
            
            # Background for text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_frame, (x, y - th - 4), (x + tw, y), color, -1)
            cv2.putText(vis_frame, label, (x, y - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw trajectory
            if len(obj.trajectory) > 2:
                points = np.array(obj.trajectory, dtype=np.int32)
                cv2.polylines(vis_frame, [points], False, color, 2)
        
        # Draw motion regions
        for region in motion_result.motion_regions:
            x, y, w, h = region
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        
        # Draw alert indicators
        alert_y = 30
        for alert in rule_result.alerts:
            color = self.rule_engine.get_alert_color(alert.level)
            
            # Alert box
            text = f"{alert.title} ({alert.confidence*100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_frame, (10, alert_y), (20 + tw, alert_y + th + 10), color, -1)
            cv2.putText(vis_frame, text, (15, alert_y + th + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            alert_y += th + 20
        
        return vis_frame
    
    def reset(self):
        """Reset all service states"""
        self.motion_analyzer.reset()
        self.pose_estimator.reset()
        self.tracker.reset()
        self.anomaly_detector.frame_buffer.clear()
