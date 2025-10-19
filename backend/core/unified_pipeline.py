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
from services.intelligent_fusion import IntelligentFusionEngine
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
            # 1. YOLO Object Detection with Tracking
            yolo_results = self.anomaly_detector.detect_objects(frame)
            
            # Convert YOLO format for other services
            yolo_detections = []
            for i, obj_class in enumerate(yolo_results['objects']):
                bbox = yolo_results['boxes'][i]
                x1, y1, x2, y2 = map(int, bbox)
                track_id = yolo_results['track_ids'][i] if 'track_ids' in yolo_results else None
                yolo_detections.append({
                    'class': obj_class,
                    'bbox': (x1, y1, x2 - x1, y2 - y1),  # x, y, w, h
                    'confidence': yolo_results['confidences'][i],
                    'track_id': track_id  # NEW: Tracking ID for smooth movement
                })
            
            result['detections']['yolo'] = {
                'objects': yolo_detections,  # ⭐ FIXED: Send full objects with bbox, not just class names
                'count': len(yolo_results['objects']),
                'dangerous_objects': yolo_results.get('dangerous', False),
                'class_names': yolo_results['objects']  # Keep original for backwards compatibility
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
                
                # Transform for fusion engine (expects 'class' not 'predicted_class')
                ml_prediction = {
                    'class': pred_result['predicted_class'],
                    'confidence': pred_result['confidence'],
                    'probabilities': pred_result.get('probabilities', [])
                }
                
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
            
            # 7. PROFESSIONAL INTELLIGENT FUSION ENGINE
            # Weighted scoring: ML (40%), YOLO (25%), Pose (20%), Motion (15%)
            # Anomaly-only reporting: No "Normal" highlights (threshold 0.70)
            fusion_detection = self.fusion_engine.fuse_detections(
                ml_result=ml_prediction,
                yolo_detections=yolo_detections,
                pose_result=result['detections']['pose'],
                motion_result=result['detections']['motion'],
                frame_number=len(self.anomaly_detector.frame_buffer)
            )
            
            # ONLY REPORT ANOMALIES (fusion_score >= 0.70)
            if fusion_detection is None:
                # Normal scene - no anomaly detected
                result['fusion'] = None
                result['anomaly_detected'] = False
                result['alerts'] = []
                result['threat_level'] = 'NORMAL'
                result['is_dangerous'] = False
                result['summary'] = 'Normal activity - No anomalies detected'
            else:
                # ANOMALY DETECTED - Professional fusion result
                result['fusion'] = {
                    'anomaly_type': fusion_detection.anomaly_type.value,
                    'severity': fusion_detection.severity.value,
                    # Provide a final_decision field for UI compatibility
                    'final_decision': fusion_detection.severity.value,
                    'fusion_score': round(fusion_detection.fusion_score, 3),
                    'confidence': round(fusion_detection.confidence, 3),
                    'reasoning': fusion_detection.reasoning,
                    'explanation': fusion_detection.explanation,
                    
                    # Weighted score breakdown (Professional Display)
                    'score_breakdown': {
                        'ml_model': {
                            'score': round(fusion_detection.ml_score, 3),
                            'weight': '40%',
                            'weighted_contribution': round(fusion_detection.ml_score * 0.40, 3)
                        },
                        'yolo_objects': {
                            'score': round(fusion_detection.object_score, 3),
                            'weight': '25%',
                            'weighted_contribution': round(fusion_detection.object_score * 0.25, 3)
                        },
                        'pose_estimation': {
                            'score': round(fusion_detection.pose_score, 3),
                            'weight': '20%',
                            'weighted_contribution': round(fusion_detection.pose_score * 0.20, 3)
                        },
                        'motion_analysis': {
                            'score': round(fusion_detection.motion_score, 3),
                            'weight': '15%',
                            'weighted_contribution': round(fusion_detection.motion_score * 0.15, 3)
                        }
                    },
                    
                    # Consensus information
                    'consensus': {
                        'agreement_count': fusion_detection.consensus_count,
                        'consensus_bonus': 0.15 if fusion_detection.consensus_count >= 2 else 0.0
                    },
                    
                    # Critical override flag
                    'critical_override': fusion_detection.critical_override
                }
                
                # Set threat level based on severity
                severity_to_threat = {
                    'CRITICAL': 'CRITICAL',
                    'HIGH': 'HIGH',
                    'MEDIUM': 'MEDIUM',
                    'LOW': 'LOW'
                }
                
                result['anomaly_detected'] = True
                result['threat_level'] = severity_to_threat.get(fusion_detection.severity.value, 'MEDIUM')
                result['is_dangerous'] = fusion_detection.severity.value in ['CRITICAL', 'HIGH']
                result['summary'] = f"{fusion_detection.anomaly_type.value}: {fusion_detection.explanation}"
                
                # Format as professional alert
                result['alerts'] = [{
                    'level': fusion_detection.severity.value,
                    'title': fusion_detection.anomaly_type.value,
                    'message': fusion_detection.explanation,
                    'confidence': round(fusion_detection.confidence, 3),
                    'timestamp': fusion_detection.timestamp,
                    'fusion_score': round(fusion_detection.fusion_score, 3),
                    'reasoning': fusion_detection.reasoning,
                    'metadata': {
                        'frame_number': fusion_detection.frame_number,
                        'detection_id': fusion_detection.detection_id,
                        'modalities_agreed': fusion_detection.consensus_count
                    }
                }]

            # 8. Rule Engine evaluation (context-aware alerts)
            rule_result = self.rule_engine.evaluate(
                yolo_detections=yolo_detections,
                motion_result=result['detections']['motion'],
                pose_result=result['detections']['pose'],
                anomaly_class=(ml_prediction.get('class') if ml_prediction else None),
                anomaly_confidence=(ml_prediction.get('confidence') if ml_prediction else None)
            )

            # Merge alerts: keep fusion alert(s) and add rule alerts
            result['alerts'] = (result.get('alerts', []) or []) + [
                {
                    'level': a.level.value,
                    'title': a.title,
                    'message': a.message,
                    'confidence': a.confidence,
                    'timestamp': a.timestamp,
                    'location': a.location,
                    'objects_involved': a.objects_involved,
                    'metadata': a.metadata,
                }
                for a in rule_result.alerts
            ]

            # Elevate threat level to worst between fusion and rules
            severity_rank = { 'INFO': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4 }
            current_level = result.get('threat_level', 'INFO')
            rule_level = rule_result.threat_level.value
            result['threat_level'] = max(current_level, rule_level, key=lambda k: severity_rank.get(k, 0))
            result['is_dangerous'] = result['threat_level'] in ['HIGH', 'CRITICAL']
            if rule_result.summary and rule_result.alerts:
                result['summary'] = rule_result.summary
            
            # 9. Create Professional Visualization
            vis_frame = self._create_visualization(
                frame,
                yolo_detections,
                tracking_result.tracked_objects,
                pose_result,
                motion_result,
                result.get('alerts', []),
                fusion_detection
            )
            
            # ⭐ OPTIMIZED ENCODING FOR SMOOTH REAL-TIME STREAMING ⭐
            # Quality 70 = Optimal balance for real-time (faster encoding/decoding)
            # JPEG_OPTIMIZE = 1 enables Huffman optimization for smaller files
            _, buffer = cv2.imencode('.jpg', vis_frame, [
                cv2.IMWRITE_JPEG_QUALITY, 70,  # Reduced to 70 for speed
                cv2.IMWRITE_JPEG_OPTIMIZE, 1   # Enable optimization
            ])
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
                            alerts: List[Dict],
                            fusion_detection) -> np.ndarray:
        """Create professional annotated frame with fusion-based detections"""
        
        vis_frame = frame.copy()
        
        # ALWAYS draw YOLO bounding boxes (fast, real-time)
        for obj in tracked_objects:
            x, y, w, h = obj.bbox
            
            # Color based on class and fusion detection
            if obj.class_name in ['knife', 'gun', 'weapon']:
                color = (0, 0, 255)  # Red for weapons (CRITICAL)
            elif obj.class_name == 'person':
                color = (0, 255, 0)  # Green for people
            else:
                color = (255, 144, 30)  # Blue for other objects
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with track ID
            label = f"ID:{obj.track_id} {obj.class_name}"
            if obj.speed > 5:
                label += f" {obj.speed:.1f}px/f"
            
            # Background for text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_frame, (x, y - th - 4), (x + tw, y), color, -1)
            cv2.putText(vis_frame, label, (x, y - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # ONLY ADD HEAVY PROCESSING IF ANOMALY DETECTED
        if fusion_detection is not None:
            # Draw pose skeletons (only for anomalies)
            if pose_result.is_anomalous:
                vis_frame = self.pose_estimator.draw_pose(vis_frame, pose_result)
            alert_y = 30
            for alert in alerts:
                # Severity color mapping
                severity_colors = {
                    'CRITICAL': (0, 0, 255),      # Red
                    'HIGH': (0, 128, 255),         # Orange
                    'MEDIUM': (0, 255, 255),       # Yellow
                    'LOW': (255, 255, 0)           # Cyan
                }
                color = severity_colors.get(alert['level'], (255, 255, 255))
                
                # Alert box with fusion score
                text = f"🚨 {alert['title']} (Score: {alert.get('fusion_score', 0):.2f})"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis_frame, (10, alert_y), (20 + tw, alert_y + th + 10), color, -1)
                cv2.putText(vis_frame, text, (15, alert_y + th + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                alert_y += th + 20
            
            # Draw fusion score indicator in bottom-right
            h, w = vis_frame.shape[:2]
            score_text = f"Fusion: {fusion_detection.fusion_score:.3f}"
            (tw, th), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_frame, (w - tw - 20, h - th - 20), (w - 10, h - 10), (0, 0, 0), -1)
            cv2.putText(vis_frame, score_text, (w - tw - 15, h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis_frame
    
    def reset(self):
        """Reset all service states"""
        self.motion_analyzer.reset()
        self.pose_estimator.reset()
        self.tracker.reset()
        self.anomaly_detector.frame_buffer.clear()
