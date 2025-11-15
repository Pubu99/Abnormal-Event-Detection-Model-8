"""
Enhanced Inference Core - Unified Detection Pipeline
Integrates all detection services for comprehensive anomaly detection

Author: AI Assistant  
Date: 2025-10-17
"""

import sys
from pathlib import Path

# Add paths for imports
backend_path = str(Path(__file__).parent.parent / "backend")
src_path = str(Path(__file__).parent.parent.parent / "src")
sys.path.append(backend_path)
sys.path.append(src_path)

from services.motion_analysis import MotionAnalyzer
from services.pose_estimation import PoseEstimator
from services.rule_engine import RuleEngine, AlertLevel
from services.object_tracking import SimpleTracker, SpeedAnalyzer
from services.intelligent_fusion import IntelligentFusionEngine

# Advanced tracking and classification
try:
    from models.object_tracker import get_object_tracker
    from models.body_part_detector import get_body_part_detector
    from models.contextual_classifier import get_contextual_classifier
    ADVANCED_TRACKING_AVAILABLE = True
except ImportError:
    ADVANCED_TRACKING_AVAILABLE = False
    print("⚠️  Advanced tracking modules not available - using basic tracking")

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
        
        # 📦 OBJECT ANOMALY DETECTOR: Comprehensive object-based anomaly detection
        try:
            from services.object_anomaly_detector import get_object_anomaly_detector
            self.object_anomaly_detector = get_object_anomaly_detector(
                abandoned_threshold=15.0,
                stationary_threshold=2.0,
                crowd_density_threshold=15,
                fps=30.0
            )
            print("   📦 Object Anomaly Detector ready")
        except ImportError:
            self.object_anomaly_detector = None
            print("   ⚠️ Object Anomaly Detector unavailable")
        
        self.fusion_engine = IntelligentFusionEngine()
        print("   ✅ Intelligent Fusion Engine ready")
        
        # Advanced tracking and contextual classification (if available)
        if ADVANCED_TRACKING_AVAILABLE:
            self.object_tracker = get_object_tracker(max_distance=50.0, fps=30)
            self.body_part_detector = get_body_part_detector()
            self.contextual_classifier = get_contextual_classifier()
            print("   ✅ Object Tracker ready")
            print("   ✅ Body Part Detector ready")
            print("   ✅ Contextual Classifier ready")
        else:
            self.object_tracker = None
            self.body_part_detector = None
            self.contextual_classifier = None

        # RL policy integration (disabled by default)
        # To enable, set attributes on the `anomaly_detector` instance:
        #   anomaly_detector.use_rl_policy = True
        #   anomaly_detector.rl_threshold = 0.5  # probability threshold to accept alerts
        self.use_rl_policy = getattr(anomaly_detector, 'use_rl_policy', False)
        self.rl_threshold = float(getattr(anomaly_detector, 'rl_threshold', 0.5))

        # Multi-rate processing and caches for performance
        self.frame_count = 0
        # By default use multi-rate processing to save CPU/GPU cycles.
        # If `force_all_modalities` is set to True the pipeline will run
        # ML and Pose every frame (or attempt a lightweight fallback)
        # to guarantee that all modalities contribute to the fusion.
        self.force_all_modalities = getattr(anomaly_detector, 'force_all_modalities', False)

        # Adjust intervals for fast_mode to prioritize YOLO/motion and reduce heavy ML runs
        if getattr(anomaly_detector, 'fast_mode', False):
            ml_interval = 3 if not self.force_all_modalities else 1    # Run ML every 3 frames in fast mode (was 2)
            pose_interval = 4 if not self.force_all_modalities else 2   # Run pose every 4 frames (was 3)
        else:
            ml_interval = 1 if self.force_all_modalities else 10
            pose_interval = 1 if self.force_all_modalities else 5

        self.processing_intervals = {
            'yolo': 1,      # every frame
            'ml_model': ml_interval, # every N frames
            'pose': pose_interval,      # every N frames
            'motion': 2,    # every 2 frames (was every frame - optical flow is expensive)
            'fusion': 1     # every frame (uses cached data)
        }
        self._cache = {
            'ml_result': None,
            'pose_result': None,
            'motion_result': None
        }

        # Performance monitoring
        self.perf_times = {k: [] for k in ['yolo', 'ml', 'pose', 'motion', 'fusion', 'total']}

        print("🚀 Unified Detection Pipeline initialized!\n")
    
    def process_frame(self, frame: np.ndarray, camera_id: Optional[str] = None) -> Dict:
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
        
        import time
        start_total = time.time()

        try:
            # 1. YOLO Object Detection with Tracking
            t0 = time.time()
            yolo_results = self.anomaly_detector.detect_objects(frame, camera_id=camera_id)
            self.perf_times['yolo'].append(time.time() - t0)
            
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
            
            # 2. ML Model Prediction (multi-rate with caching)
            # Always append current frame to the temporal buffer
            self.anomaly_detector.frame_buffer.append(frame)
            ml_prediction = None
            if (self.frame_count % self.processing_intervals['ml_model']) == 0:
                t_ml = time.time()
                # If we have a full buffer, run the full sequence model.
                if len(self.anomaly_detector.frame_buffer) == self.anomaly_detector.sequence_length:
                    # Use TTA if enabled on the detector to stabilize predictions
                    if getattr(self.anomaly_detector, 'enable_tta', False):
                        pred_result = self.anomaly_detector.predict_sequence_tta(
                            list(self.anomaly_detector.frame_buffer)
                        )
                    else:
                        sequence_tensor = self.anomaly_detector.create_sequence(
                            list(self.anomaly_detector.frame_buffer)
                        )
                        pred_result = self.anomaly_detector.predict_sequence(sequence_tensor)

                    ml_prediction = {
                        'class': pred_result['predicted_class'],
                        'confidence': pred_result['confidence'],
                        'probabilities': pred_result.get('all_confidences', {})
                    }
                    result['detections']['ml_model'] = {
                        'predicted_class': pred_result['predicted_class'],
                        'confidence': pred_result['confidence'],
                        'is_anomaly': pred_result['is_anomaly'],
                        'anomaly_score': pred_result.get('anomaly_score', 0.0),
                        'top_3': pred_result.get('top3_predictions', [])
                    }
                else:
                    # If the buffer is not yet full but the user requested forcing all
                    # modalities, produce a lightweight fallback prediction by
                    # duplicating the current frame to form a sequence. This ensures
                    # the fusion engine still receives an ML signal even before the
                    # full temporal buffer is available.
                    if self.force_all_modalities:
                        try:
                            frames_needed = self.anomaly_detector.sequence_length
                            cur = list(self.anomaly_detector.frame_buffer)
                            # pad/duplicate last frame to meet length
                            while len(cur) < frames_needed:
                                cur.append(frame)
                            if getattr(self.anomaly_detector, 'enable_tta', False):
                                pred_result = self.anomaly_detector.predict_sequence_tta(
                                    cur[-frames_needed:]
                                )
                            else:
                                seq = self.anomaly_detector.create_sequence(cur[-frames_needed:])
                                pred_result = self.anomaly_detector.predict_sequence(seq)
                            ml_prediction = {
                                'class': pred_result['predicted_class'],
                                'confidence': pred_result['confidence'],
                                'probabilities': pred_result.get('all_confidences', {})
                            }
                            result['detections']['ml_model'] = {
                                'predicted_class': pred_result['predicted_class'],
                                'confidence': pred_result['confidence'],
                                'is_anomaly': pred_result['is_anomaly'],
                                'anomaly_score': pred_result.get('anomaly_score', 0.0),
                                'top_3': pred_result.get('top3_predictions', [])
                            }
                        except Exception:
                            ml_prediction = None
                    else:
                        ml_prediction = None
                self._cache['ml_result'] = ml_prediction
                self.perf_times['ml'].append(time.time() - t_ml)
            else:
                ml_prediction = self._cache.get('ml_result')
            
            # 3. Motion Analysis (heavily downscaled for speed)
            if (self.frame_count % self.processing_intervals['motion']) == 0:
                t_motion = time.time()
                # Downscale to 33% for much faster optical flow (was 50%)
                h, w = frame.shape[:2]
                motion_frame = cv2.resize(frame, (max(1, w//3), max(1, h//3)))
                motion_result = self.motion_analyzer.analyze(motion_frame)
                self._cache['motion_result'] = motion_result
                self.perf_times['motion'].append(time.time() - t_motion)
            else:
                motion_result = self._cache.get('motion_result')

            result['detections']['motion'] = {
                'magnitude': getattr(motion_result, 'motion_magnitude', 0.0),
                'direction': getattr(motion_result, 'motion_direction', 0.0),
                'regions_count': len(getattr(motion_result, 'motion_regions', []) or []),
                'is_unusual': getattr(motion_result, 'is_unusual', False),
                'anomaly_type': getattr(motion_result, 'anomaly_type', None),
                'confidence': getattr(motion_result, 'confidence', 0.0)
            }

            # 4. Pose Estimation (conditional + multi-rate)
            # Only run pose occasionally and only if people detected
            people_count = len([obj for obj in yolo_detections if obj['class'] == 'person'])
            if (self.frame_count % self.processing_intervals['pose']) == 0 and people_count > 0:
                t_pose = time.time()
                # Downscale to 33% for much faster pose detection (was 50%)
                h, w = frame.shape[:2]
                pose_frame = cv2.resize(frame, (max(1, w//3), max(1, h//3)))
                # ⚡ ENHANCED: Pass YOLO detections for synergistic pose-object analysis
                pose_result = self.pose_estimator.analyze(
                    pose_frame, 
                    camera_id=camera_id,
                    yolo_detections=yolo_detections  # Enable synergistic analysis
                )
                self._cache['pose_result'] = pose_result
                self.perf_times['pose'].append(time.time() - t_pose)
            else:
                pose_result = self._cache.get('pose_result') or type('empty', (), {
                    'persons_detected': 0,
                    'is_anomalous': False,
                    'anomaly_type': None,
                    'confidence': 0.0
                })()

            result['detections']['pose'] = {
                'persons_detected': getattr(pose_result, 'persons_detected', 0),
                'is_anomalous': getattr(pose_result, 'is_anomalous', False),
                'anomaly_type': getattr(pose_result, 'anomaly_type', None),
                'confidence': getattr(pose_result, 'confidence', 0.0)
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
            
            # 5.1 📦 OBJECT ANOMALY DETECTION (abandoned objects, crowds, weapons, etc.)
            object_anomalies = []
            if self.object_anomaly_detector is not None:
                try:
                    # Extract pose data for weapon detection
                    pose_keypoints = getattr(pose_result, 'keypoints', []) if pose_result else []
                    pose_bboxes = []
                    if pose_keypoints:
                        # Extract bounding boxes for persons from YOLO
                        for obj in yolo_detections:
                            if obj.get('class') == 'person':
                                pose_bboxes.append(obj.get('bbox', (0, 0, 0, 0)))
                    
                    object_anomalies = self.object_anomaly_detector.detect_anomalies(
                        frame=frame,
                        yolo_detections=yolo_detections,
                        tracked_objects=tracked_objects_data,
                        frame_number=self.frame_count,
                        pose_keypoints=pose_keypoints,  # Enable weapon detection
                        pose_bboxes=pose_bboxes  # Person bounding boxes
                    )
                    
                    # Add to result
                    result['detections']['object_anomalies'] = [
                        {
                            'type': anomaly.anomaly_type.value,
                            'confidence': anomaly.confidence,
                            'object_class': anomaly.object_class,
                            'bbox': anomaly.bbox,
                            'duration': anomaly.duration,
                            'details': anomaly.details,
                            'severity': anomaly.severity
                        }
                        for anomaly in object_anomalies
                    ]
                except Exception as e:
                    print(f"⚠️ Object anomaly detection error: {e}")
                    result['detections']['object_anomalies'] = []
            else:
                result['detections']['object_anomalies'] = []
            
            # 5.5 ADVANCED OBJECT TRACKING & CONTEXTUAL ANALYSIS (if available)
            advanced_tracks = []
            contextual_predictions = []
            if ADVANCED_TRACKING_AVAILABLE and self.object_tracker is not None:
                try:
                    # Prepare detections for advanced tracker (needs centroid + keypoints)
                    advanced_detections = []
                    for obj in yolo_detections:
                        x, y, w, h = obj['bbox']
                        centroid = (x + w//2, y + h//2)
                        
                        detection_dict = {
                            'class': obj['class'],
                            'confidence': obj['confidence'],
                            'bbox': obj['bbox'],
                            'centroid': centroid,
                        }
                        
                        # Extract pose keypoints if available for this person
                        if obj['class'] == 'person' and pose_result and hasattr(pose_result, 'persons_detected'):
                            if pose_result.persons_detected > 0:
                                # Use pose keypoints if available (simplified - attach empty for now)
                                detection_dict['keypoints'] = []
                        
                        advanced_detections.append(detection_dict)
                    
                    # Update advanced tracker
                    advanced_tracks = self.object_tracker.update(advanced_detections)
                    
                    # For each track, perform contextual analysis
                    for track_id, track in advanced_tracks.items():
                        try:
                            # Create pose context from body part detector
                            pose_context = None
                            if self.body_part_detector and track.class_name == 'person':
                                # Extract body part context (simplified - would use pose keypoints)
                                pose_context = self.body_part_detector.analyze_pose(
                                    keypoints=[],  # would come from pose estimator
                                    detected_objects=yolo_detections,
                                    frame_height=frame.shape[0],
                                    frame_width=frame.shape[1]
                                )
                            
                            # Classify anomaly with context
                            if self.contextual_classifier:
                                prediction = self.contextual_classifier.classify_track(
                                    track=track,
                                    pose_context=pose_context,
                                    restricted_zones=None,  # would be from config
                                    known_persons=None
                                )
                                
                                # Calculate track duration and other contextual metrics
                                track_duration = track.age_frames / 30.0  # Convert frames to seconds (assuming 30 FPS)
                                
                                # Calculate loitering score (higher if standing still)
                                loitering_score = 0.0
                                if track.movement_speed < 0.5 and track_duration > 10.0:
                                    loitering_score = min(track_duration / 20.0, 1.0)
                                
                                # Calculate temporal consistency (how stable the track is)
                                temporal_consistency = min(track.confidence, 1.0)
                                
                                contextual_predictions.append({
                                    'track_id': track_id,
                                    'anomaly_type': prediction.anomaly_type.value,
                                    'confidence': prediction.confidence,
                                    'threat_level': prediction.threat_level,
                                    'reasoning': prediction.reasoning,
                                    # Additional contextual features for RL training
                                    'track_duration': track_duration,
                                    'movement_speed': prediction.movement_speed,
                                    'loitering_score': loitering_score,
                                    'track_confidence': track.confidence,
                                    'gesture_score': prediction.pose_threat,
                                    'held_object_count': len(track.held_items),
                                    'body_pose_score': prediction.pose_threat,
                                    'temporal_consistency': temporal_consistency
                                })
                        except Exception as e:
                            print(f"⚠️  Error in contextual analysis for track {track_id}: {e}")
                    
                    result['detections']['advanced_tracking'] = {
                        'tracks': len(advanced_tracks),
                        'contextual_predictions': contextual_predictions
                    }
                    
                except Exception as e:
                    print(f"⚠️  Error in advanced tracking: {e}")
            
            # Speed Analysis
            speed_analysis = self.speed_analyzer.analyze(tracking_result.tracked_objects)
            result['detections']['speed'] = speed_analysis
            
            # 6. PROFESSIONAL INTELLIGENT FUSION ENGINE
            # Weighted scoring: ML (40%), YOLO (25%), Pose (20%), Motion (15%)
            # Anomaly-only reporting: No "Normal" highlights (threshold 0.70)
            # NOW WITH CONTEXTUAL PREDICTIONS (Priority 0)
            t_f = time.time()
            fusion_detection = self.fusion_engine.fuse_detections(
                ml_result=ml_prediction,
                yolo_detections=yolo_detections,
                pose_result=result['detections']['pose'],
                motion_result=result['detections']['motion'],
                frame_number=len(self.anomaly_detector.frame_buffer),
                contextual_predictions=contextual_predictions if contextual_predictions else None
            )
            # If the fusion engine has a global decline-all suppression active,
            # or this specific detection id has been suppressed, then suppress
            # the fusion_detection so no alerts will be produced or sent to the UI.
            try:
                # Only suppress detections when their specific id is in the
                # suppressed_ids set. Remove reliance on the old
                # decline_all_active global flag which could block all future
                # detections unintentionally.
                if fusion_detection is not None and getattr(self.fusion_engine, 'suppressed_ids', None) is not None:
                    if fusion_detection.detection_id in self.fusion_engine.suppressed_ids:
                        fusion_detection = None
            except Exception:
                pass
            self.perf_times['fusion'].append(time.time() - t_f)

            # RL policy consult: allow an RL agent to accept/suppress detections
            rl_decision = None
            try:
                if fusion_detection is not None and self.use_rl_policy:
                    from services.rl_agent import get_agent
                    agent = get_agent()
                    if agent is not None:
                        severity_map = {'CRITICAL': 3, 'HIGH': 2, 'MEDIUM': 1, 'LOW': 0}
                        features = {
                            'fusion_score': float(fusion_detection.fusion_score or 0.0),
                            'ml_score': float(fusion_detection.ml_score or 0.0),
                            'object_score': float(fusion_detection.object_score or 0.0),
                            'pose_score': float(fusion_detection.pose_score or 0.0),
                            'motion_score': float(fusion_detection.motion_score or 0.0),
                            'consensus_count': float(fusion_detection.consensus_count or 0),
                            'confidence': float(fusion_detection.confidence or 0.0),
                            'severity': float(severity_map.get(fusion_detection.severity.value if fusion_detection.severity else 'LOW', 0))
                        }
                        prob = agent.predict_proba(features)
                        rl_decision = {'probability': prob, 'accepted': prob >= self.rl_threshold}
                        # expose RL decision to frontend via result['fusion'] later
                        # If agent rejects, suppress the fusion_detection (no alerts)
                        if not rl_decision['accepted']:
                            # mark suppression
                            result['fusion'] = {
                                'anomaly_type': fusion_detection.anomaly_type.value,
                                'severity': fusion_detection.severity.value,
                                'fusion_score': round(fusion_detection.fusion_score, 3),
                                'confidence': round(fusion_detection.confidence, 3),
                                'reasoning': fusion_detection.reasoning,
                                'explanation': fusion_detection.explanation,
                                'rl_policy': {
                                    'accepted': False,
                                    'probability': prob,
                                    'threshold': self.rl_threshold
                                }
                            }
                            # suppress alerts and mark as normal for downstream
                            result['anomaly_detected'] = False
                            result['alerts'] = []
                            result['threat_level'] = 'NORMAL'
                            result['is_dangerous'] = False
                            result['summary'] = f"Detection suppressed by RL policy (p={prob:.3f})"
                            fusion_detection = None
            except Exception:
                # non-fatal; if RL agent not available just continue
                rl_decision = None

            # Always provide a top-level score breakdown for the UI so the
            # "DETECTION METHODS USED" panel shows all modalities even when
            # fusion does not flag an anomaly. These are best-effort scores
            # computed from the most recent modality outputs / cache.
            try:
                ml_score = ml_prediction.get('confidence', 0.0) if ml_prediction else 0.0
            except Exception:
                ml_score = 0.0

            # Simple object score: prioritize dangerous objects, otherwise use
            # normalized object count (cap at 5 objects).
            try:
                dangerous_present = any(o['class'] in ['gun', 'knife', 'weapon', 'pistol', 'rifle', 'fire'] for o in yolo_detections)
                if dangerous_present:
                    object_score = 1.0
                else:
                    object_score = min(len(yolo_detections) / 5.0, 1.0)
            except Exception:
                object_score = 0.0

            pose_score = float(result['detections']['pose'].get('confidence', 0.0))
            motion_score = float(result['detections']['motion'].get('confidence', 0.0))

            result['score_breakdown'] = {
                'ml_model': {'score': round(ml_score, 3), 'weight': '40%'},
                'yolo_objects': {'score': round(object_score, 3), 'weight': '25%'},
                'pose_estimation': {'score': round(pose_score, 3), 'weight': '20%'},
                'motion_analysis': {'score': round(motion_score, 3), 'weight': '15%'}
            }
            
            # ONLY REPORT ANOMALIES (fusion_score >= 0.70)
            if fusion_detection is None:
                # Start with neutral defaults; we'll update after rule evaluation below
                result['fusion'] = None
                result['anomaly_detected'] = False
                result['alerts'] = []
                result['threat_level'] = 'INFO'
                result['is_dangerous'] = False
                # Don't show any "normal activity" banner; this system is anomaly-only
                result['summary'] = ''
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
            # If no fusion anomaly, but rules raised alerts above INFO, treat as anomaly-only output
            if fusion_detection is None:
                if rule_result.alerts and result['threat_level'] != 'INFO':
                    result['anomaly_detected'] = True
                    result['summary'] = rule_result.summary or ''
                else:
                    # Keep anomaly_detected False and leave summary empty (no normal banner)
                    result['anomaly_detected'] = False
                    result['summary'] = ''
            else:
                # With fusion anomaly, prefer rule summary when present
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
            
            # ⚡ ULTRA-OPTIMIZED ENCODING FOR MAXIMUM FPS ⚡
            # Quality 40-45 = Aggressive compression for real-time streaming
            # JPEG_OPTIMIZE = 0 = Skip Huffman optimization (faster encoding)
            # Prioritize FPS over slight quality loss
            _, buffer = cv2.imencode('.jpg', vis_frame, [
                cv2.IMWRITE_JPEG_QUALITY, 40,  # Aggressive compression for speed
                cv2.IMWRITE_JPEG_OPTIMIZE, 0,  # Skip optimization for speed
                cv2.IMWRITE_JPEG_PROGRESSIVE, 0  # Disable progressive (faster)
            ])
            result['frame_base64'] = base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            result['error'] = str(e)
            print(f"❌ Error in pipeline: {e}")

        # Performance logging
        total_elapsed = time.time() - start_total
        self.perf_times['total'].append(total_elapsed)
        self.frame_count += 1
        # log every 30 frames
        if self.frame_count % 30 == 0:
            try:
                import numpy as _np
                print('\n' + '='*60)
                print('📊 PERFORMANCE (Last 30 frames)')
                print('='*60)
                for key, times in self.perf_times.items():
                    if times:
                        avg_ms = _np.mean(times[-30:]) * 1000
                        print(f"   {key.upper():12s}: {avg_ms:6.2f} ms")
                if self.perf_times['total']:
                    avg_total = _np.mean(self.perf_times['total'][-30:])
                    fps = 1.0 / avg_total if avg_total > 0 else 0
                    print(f"\n   TARGET: >20 FPS | CURRENT: {fps:.1f} FPS")
                print('='*60 + '\n')
            except Exception:
                pass

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
