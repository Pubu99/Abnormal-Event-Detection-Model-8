"""
Pose Estimation Service - ENHANCED WITH ADVANCED ALGORITHMS
Detects human poses and identifies anomalous behaviors using:
- Synergistic pose & object detection
- Temporal-spatial graph modeling
- Advanced motion feature extraction
- Multi-modal fusion

Implements state-of-the-art research techniques for professional-grade detection.

Author: Professional AI/ML Implementation
Date: 2025-11-13
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import math

# Import advanced analyzer
try:
    from .advanced_pose_motion import get_advanced_pose_motion_analyzer
    _ADVANCED_ANALYZER_AVAILABLE = True
except ImportError:
    _ADVANCED_ANALYZER_AVAILABLE = False

# Prefer OpenPose if available; fallback to MediaPipe
_OPENPOSE_AVAILABLE = False
_MEDIAPIPE_AVAILABLE = False
try:
    # Try the official module name used by CMU OpenPose builds
    from openpose import pyopenpose as op  # type: ignore
    _OPENPOSE_AVAILABLE = True
    print("🕺 OpenPose detected (openpose.pyopenpose)")
except Exception:
    try:
        # Some community builds expose pyopenpose directly
        import pyopenpose as op  # type: ignore
        _OPENPOSE_AVAILABLE = True
        print("🕺 OpenPose detected (pyopenpose)")
    except Exception:
        op = None
        _OPENPOSE_AVAILABLE = False
        # proceed to MediaPipe fallback
    try:
        import mediapipe as mp
        _MEDIAPIPE_AVAILABLE = True
        print("🧍 MediaPipe detected: using MediaPipe for pose estimation")
    except Exception:
        mp = None
        print("⚠️ No pose backend installed. Pose estimation disabled.")


@dataclass
class PoseResult:
    """Pose detection result with weapon detection"""
    persons_detected: int
    poses: List[Dict]  # List of detected poses with keypoints
    is_anomalous: bool
    anomaly_type: Optional[str]
    confidence: float
    timestamp: str
    keypoints: List[List[Tuple[float, float, float]]] = field(default_factory=list)
    weapon_detections: List = field(default_factory=list)  # List of WeaponDetection objects



class PoseEstimator:
    """
    ENHANCED Human Pose Estimation with Advanced Algorithms
    
    Features:
    - Synergistic pose & object detection
    - Temporal-spatial graph modeling
    - Advanced motion feature extraction
    - Multi-modal fusion for robust anomaly detection
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 enable_advanced_analysis: bool = True):
        """
        Initialize pose estimator
        
        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            enable_advanced_analysis: Enable advanced pose-motion analyzer
        """
        self.enabled = _OPENPOSE_AVAILABLE or _MEDIAPIPE_AVAILABLE
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # ⚡ ADVANCED: Initialize advanced pose-motion analyzer
        self.enable_advanced = enable_advanced_analysis and _ADVANCED_ANALYZER_AVAILABLE
        if self.enable_advanced:
            self.advanced_analyzer = get_advanced_pose_motion_analyzer(
                history_length=30,
                fps=30.0
            )
            print("   ⚡ Advanced Pose-Motion Analyzer: ENABLED")
        else:
            self.advanced_analyzer = None
            if enable_advanced_analysis:
                print("   ⚠️  Advanced analyzer unavailable - using basic pose estimation")
        
        # Lazy initialization - only create when first needed
        self.mp_pose = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self.pose = None
        self.openpose_wrapper = None
        self._initialized = False
        self.backend = None  # 'openpose' or 'mediapipe'
        
        # Pose history for temporal analysis
        self.pose_history = []
        self.history_size = 30  # 1 second at 30fps
        # Previous raw keypoints for simple exponential smoothing
        self._prev_keypoints = None
        self._smoothing_alpha = 0.45
        
        # Frame counter for temporal tracking
        self.frame_number = 0

        # Counters for persistence-based heuristics
        self._anomaly_counters = {}
        self._persistence_threshold = 3  # frames
        # Directory for saving sample frames for auditing
        self._sample_dir = Path(__file__).parent.parent / 'data' / 'fall_samples'
        try:
            self._sample_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    
    def _lazy_init(self):
        """Initialize MediaPipe only when first needed"""
        if self._initialized or not self.enabled:
            return
        # Initialize OpenPose if available, else MediaPipe
        if _OPENPOSE_AVAILABLE:
            try:
                params = dict()
                params["model_folder"] = "models/openpose"  # allow user to place models here
                params["hand"] = False
                params["face"] = False
                params["net_resolution"] = "-1x256"  # speed/quality trade-off
                self.openpose_wrapper = op.WrapperPython()
                self.openpose_wrapper.configure(params)
                self.openpose_wrapper.start()
                self.backend = 'openpose'
                self._initialized = True
                return
            except Exception as e:
                print(f"⚠️ OpenPose init failed, falling back to MediaPipe: {e}")
        if _MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                model_complexity=1
            )
            self.backend = 'mediapipe'
            self._initialized = True
        
        
    def analyze(self, frame: np.ndarray, 
                camera_id: Optional[str] = None,
                yolo_detections: Optional[List[Dict]] = None) -> PoseResult:
        """
        ENHANCED: Analyze poses with advanced temporal-spatial modeling
        
        Args:
            frame: Input BGR frame
            camera_id: Optional camera identifier
            yolo_detections: Optional YOLO object detections for synergistic analysis
            
        Returns:
            PoseResult with detected anomalies
        """
        self.frame_number += 1
        
        if not self.enabled:
            return PoseResult(
                persons_detected=0,
                poses=[],
                is_anomalous=False,
                anomaly_type=None,
                confidence=0.0,
                timestamp=datetime.now().isoformat()
            )
        
        # Initialize MediaPipe on first use
        self._lazy_init()
        
        poses = []
        keypoints_list = []
        bounding_boxes = []
        
        # Extract bounding boxes from YOLO detections (for synergistic analysis)
        if yolo_detections:
            for det in yolo_detections:
                if det.get('class') == 'person':
                    bbox = det.get('bbox')
                    if bbox:
                        bounding_boxes.append(bbox)
        
        # Run pose detection
        if _OPENPOSE_AVAILABLE and self.openpose_wrapper is not None:
            # OpenPose inference
            try:
                datum = op.Datum()
                imageToProcess = frame
                datum.cvInputData = imageToProcess
                self.openpose_wrapper.emplaceAndPop([datum])
                if datum.poseKeypoints is not None and len(datum.poseKeypoints.shape) >= 2:
                    # datum.poseKeypoints shape: (numPeople, 25, 3)
                    for person_idx in range(datum.poseKeypoints.shape[0]):
                        person = datum.poseKeypoints[person_idx]
                        # Normalize to width/height
                        h, w = frame.shape[:2]
                        keypoints = [(kp[0] / max(w,1e-6), kp[1] / max(h,1e-6), kp[2]) for kp in person]
                        keypoints_np = np.array([[kp[0]*w, kp[1]*h, kp[2]] for kp in keypoints])
                        keypoints_list.append(keypoints_np)
                        try:
                            sk = self._smooth_keypoints(keypoints)
                            pose_data = self._extract_pose_features(sk, frame.shape)
                            poses.append(pose_data)
                        except Exception as e:
                            print(f"⚠️ Pose feature extraction failed (OpenPose): {e}")
            except Exception as e:
                print(f"⚠️ OpenPose inference failed this frame: {e}")
        elif _MEDIAPIPE_AVAILABLE and self.pose is not None:
            # MediaPipe inference
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                h, w = frame.shape[:2]
                keypoints = [
                    (lm.x, lm.y, lm.visibility)
                    for lm in landmarks
                ]
                # Convert to pixel coordinates for advanced analyzer
                keypoints_np = np.array([[lm.x * w, lm.y * h, lm.visibility] for lm in landmarks])
                keypoints_list.append(keypoints_np)
                try:
                    sk = self._smooth_keypoints(keypoints)
                    pose_data = self._extract_pose_features(sk, frame.shape)
                    poses.append(pose_data)
                except Exception as e:
                    print(f"⚠️ Pose feature extraction failed (MediaPipe): {e}")
        
        # ⚡ ADVANCED ANALYSIS: Use temporal-spatial graph modeling
        advanced_result = None
        if self.enable_advanced and self.advanced_analyzer and keypoints_list:
            try:
                # Fill bounding boxes if missing (estimate from keypoints)
                if len(bounding_boxes) < len(keypoints_list):
                    for kpts in keypoints_list[len(bounding_boxes):]:
                        if len(kpts) > 0:
                            valid_kpts = kpts[kpts[:, 2] > 0.3][:, :2]
                            if len(valid_kpts) > 0:
                                x_min, y_min = valid_kpts.min(axis=0)
                                x_max, y_max = valid_kpts.max(axis=0)
                                w_bbox = int(x_max - x_min)
                                h_bbox = int(y_max - y_min)
                                bounding_boxes.append((int(x_min), int(y_min), w_bbox, h_bbox))
                
                # Run advanced analysis
                advanced_result = self.advanced_analyzer.analyze_frame(
                    frame=frame,
                    pose_keypoints=keypoints_list,
                    bounding_boxes=bounding_boxes,
                    frame_number=self.frame_number
                )
            except Exception as e:
                print(f"⚠️ Advanced pose-motion analysis failed: {e}")
                advanced_result = None
        
        # Update history (store flattened per-frame pose features)
        frame_pose = poses[0] if poses else None
        self.pose_history.append(frame_pose)
        if len(self.pose_history) > self.history_size:
            self.pose_history.pop(0)
        self._last_frame_for_sample = frame.copy()
        
        # Detect anomalies using BOTH basic and advanced methods
        is_anomalous, anomaly_type, confidence = self._detect_pose_anomaly(
            poses, camera_id=camera_id, advanced_result=advanced_result
        )
        
        return PoseResult(
            persons_detected=len(poses),
            poses=poses,
            is_anomalous=is_anomalous,
            anomaly_type=anomaly_type,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            keypoints=keypoints_list,
            weapon_detections=[]  # Weapon detection now handled by object_anomaly_detector
        )

    def _smooth_keypoints(self, keypoints: List[Tuple[float,float,float]]) -> List[Tuple[float,float,float]]:
        """Apply a simple exponential smoothing to keypoints to reduce jitter."""
        if self._prev_keypoints is None:
            self._prev_keypoints = keypoints
            return keypoints

        alpha = self._smoothing_alpha
        sk = []
        for prev, cur in zip(self._prev_keypoints, keypoints):
            sx = alpha * cur[0] + (1 - alpha) * prev[0]
            sy = alpha * cur[1] + (1 - alpha) * prev[1]
            sc = alpha * cur[2] + (1 - alpha) * prev[2]
            sk.append((sx, sy, sc))
        self._prev_keypoints = sk
        return sk
    
    def _extract_pose_features(self, keypoints: List, frame_shape: Tuple) -> Dict:
        """Extract meaningful features from pose keypoints"""
        h, w = frame_shape[:2]
        
        # Key body parts indices depending on backend
        if self.backend == 'openpose':
            # OpenPose BODY_25 mapping
            NOSE = 0
            NECK = 1
            RIGHT_SHOULDER = 2
            RIGHT_ELBOW = 3
            RIGHT_WRIST = 4
            LEFT_SHOULDER = 5
            LEFT_ELBOW = 6
            LEFT_WRIST = 7
            MID_HIP = 8
            RIGHT_HIP = 9
            RIGHT_KNEE = 10
            RIGHT_ANKLE = 11
            LEFT_HIP = 12
            LEFT_KNEE = 13
            LEFT_ANKLE = 14
        else:
            # MediaPipe (33 landmarks)
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_ELBOW = 13
            RIGHT_ELBOW = 14
            LEFT_WRIST = 15
            RIGHT_WRIST = 16
            LEFT_HIP = 23
            RIGHT_HIP = 24
            LEFT_KNEE = 25
            RIGHT_KNEE = 26
            LEFT_ANKLE = 27
            RIGHT_ANKLE = 28
            NOSE = 0
        
        # Extract coordinates
        def get_point(idx):
            return (keypoints[idx][0] * w, keypoints[idx][1] * h)
        
        # Calculate angles
        left_elbow_angle = self._calculate_angle(
            get_point(LEFT_SHOULDER),
            get_point(LEFT_ELBOW),
            get_point(LEFT_WRIST)
        )
        
        right_elbow_angle = self._calculate_angle(
            get_point(RIGHT_SHOULDER),
            get_point(RIGHT_ELBOW),
            get_point(RIGHT_WRIST)
        )
        
        # Body posture
        shoulder_center = (
            (get_point(LEFT_SHOULDER)[0] + get_point(RIGHT_SHOULDER)[0]) / 2,
            (get_point(LEFT_SHOULDER)[1] + get_point(RIGHT_SHOULDER)[1]) / 2
        )
        
        hip_center = (
            (get_point(LEFT_HIP)[0] + get_point(RIGHT_HIP)[0]) / 2,
            (get_point(LEFT_HIP)[1] + get_point(RIGHT_HIP)[1]) / 2
        )
        
        # Body tilt angle
        body_angle = math.degrees(math.atan2(
            hip_center[1] - shoulder_center[1],
            hip_center[0] - shoulder_center[0]
        ))
        
        # Arms raised detection
        arms_raised = (
            get_point(LEFT_WRIST)[1] < get_point(LEFT_SHOULDER)[1] and
            get_point(RIGHT_WRIST)[1] < get_point(RIGHT_SHOULDER)[1]
        )
        
        # Hands near head (surrender, distress)
        hands_near_head = (
            abs(get_point(LEFT_WRIST)[1] - get_point(NOSE)[1]) < 50 or
            abs(get_point(RIGHT_WRIST)[1] - get_point(NOSE)[1]) < 50
        )
        
        return {
            'left_elbow_angle': left_elbow_angle,
            'right_elbow_angle': right_elbow_angle,
            'body_angle': body_angle,
            'arms_raised': arms_raised,
            'hands_near_head': hands_near_head,
            'shoulder_center': shoulder_center,
            'hip_center': hip_center
        }
    
    def _calculate_angle(self, point1: Tuple, point2: Tuple, point3: Tuple) -> float:
        """Calculate angle between three points"""
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        
        return angle
    
    def _detect_pose_anomaly(self, poses: List[Dict], 
                            camera_id: Optional[str] = None,
                            advanced_result: Optional[Dict] = None) -> Tuple[bool, Optional[str], float]:
        """
        ENHANCED: Detect anomalous poses using multi-modal fusion
        
        Combines basic pose analysis with advanced temporal-spatial modeling
        
        Args:
            poses: Basic pose features
            camera_id: Optional camera identifier
            advanced_result: Results from advanced pose-motion analyzer
            
        Returns:
            (is_anomalous, anomaly_type, confidence)
        """
        # ⚡ PRIORITY 1: Advanced analysis (if available)
        if advanced_result and advanced_result.get('is_anomalous'):
            return (
                True,
                advanced_result['anomaly_type'],
                advanced_result['confidence']
            )
        
        # PRIORITY 2: Basic pose analysis (fallback)
        if not poses:
            return False, None, 0.0

        # Load per-camera thresholds if camera_id provided
        body_angle_thresh = 45.0
        vel_thresh = 0.02
        accel_thresh = 0.02
        persistence = self._persistence_threshold
        try:
            if camera_id:
                from services.camera_manager import get_camera_manager
                cm = get_camera_manager()
                cam = cm.get_camera(camera_id)
                if cam:
                    body_angle_thresh = float(getattr(cam, 'pose_body_angle_thresh', body_angle_thresh))
                    vel_thresh = float(getattr(cam, 'pose_velocity_thresh', vel_thresh))
                    accel_thresh = float(getattr(cam, 'pose_acceleration_thresh', accel_thresh))
                    persistence = int(getattr(cam, 'pose_persistence_frames', persistence))
        except Exception:
            # best-effort: fall back to defaults if camera manager unavailable
            pass

        for pose in poses:
            # 1. Falling detection (extreme body tilt)
            if abs(pose.get('body_angle', 0.0)) > body_angle_thresh:
                key = f"PERSON_FALLING::{camera_id or 'global'}"
                self._anomaly_counters[key] = self._anomaly_counters.get(key, 0) + 1
                if self._anomaly_counters.get(key, 0) >= max(1, persistence):
                    # reset other counters for this camera to avoid duplicate alerts
                    keys_to_clear = [k for k in list(self._anomaly_counters.keys()) if k.endswith(f"::{camera_id}") and k != key]
                    for k in keys_to_clear:
                        self._anomaly_counters.pop(k, None)
                    return True, "PERSON_FALLING", 0.85
                else:
                    # persistence not yet reached
                    return False, None, 0.0
            
            # 2. Fighting detection (aggressive arm movements)
            if pose['left_elbow_angle'] < 90 and pose['right_elbow_angle'] < 90:
                # Check temporal pattern
                if len(self.pose_history) > 10:
                    rapid_arm_movement = self._check_rapid_arm_movement()
                    if rapid_arm_movement:
                        return True, "FIGHTING_DETECTED", 0.80
            
            # 3. Surrender/Distress pose (hands raised near head)
            if pose.get('arms_raised') and pose.get('hands_near_head'):
                key = f"DISTRESS_POSE::{camera_id or 'global'}"
                self._anomaly_counters[key] = self._anomaly_counters.get(key, 0) + 1
                if self._anomaly_counters.get(key, 0) >= max(1, persistence):
                    return True, "DISTRESS_POSE", 0.75
                else:
                    return False, None, 0.0
            
            # 4. Weapon handling pose (one arm extended, rigid posture)
            if (pose.get('left_elbow_angle', 0.0) > 160 or pose.get('right_elbow_angle', 0.0) > 160):
                # Straight arm could indicate weapon
                key = f"SUSPICIOUS_POSE::{camera_id or 'global'}"
                self._anomaly_counters[key] = self._anomaly_counters.get(key, 0) + 1
                if self._anomaly_counters.get(key, 0) >= max(1, persistence):
                    return True, "SUSPICIOUS_POSE", 0.65
                else:
                    return False, None, 0.0
        
        # 5. Multiple people with aggressive poses (group fighting)
        if len(poses) >= 2:
            aggressive_count = sum(
                1 for p in poses 
                if p.get('left_elbow_angle', 180) < 90 or p.get('right_elbow_angle', 180) < 90
            )
            if aggressive_count >= 2:
                key = f"GROUP_ALTERCATION::{camera_id or 'global'}"
                self._anomaly_counters[key] = self._anomaly_counters.get(key, 0) + 1
                if self._anomaly_counters.get(key, 0) >= max(1, persistence):
                    return True, "GROUP_ALTERCATION", 0.78
                else:
                    return False, None, 0.0
        
        # If we got here, no persistent anomaly detected: decay counters for this camera
        try:
            if camera_id:
                for k in list(self._anomaly_counters.keys()):
                    if k.endswith(f"::{camera_id}"):
                        # decay by 1 per non-event frame to avoid permanent lock
                        self._anomaly_counters[k] = max(0, self._anomaly_counters[k] - 1)
        except Exception:
            pass

        return False, None, 0.0
    
    def _check_rapid_arm_movement(self) -> bool:
        """Check for rapid arm movements in pose history"""
        if len(self.pose_history) < 10:
            return False
        
        # Calculate arm angle variance over time
        arm_angles = []
        for frame_poses in self.pose_history[-10:]:
            if frame_poses:
                pose = frame_poses[0]
                avg_angle = (pose['left_elbow_angle'] + pose['right_elbow_angle']) / 2
                arm_angles.append(avg_angle)
        
        if len(arm_angles) > 5:
            variance = np.var(arm_angles)
            return variance > 500  # High variance indicates rapid movement
        
        return False
    
    def draw_pose(self, frame: np.ndarray, pose_result: PoseResult) -> np.ndarray:
        """
        Draw pose landmarks on frame
        
        Args:
            frame: Input BGR frame
            pose_result: PoseResult from analyze()
            
        Returns:
            Frame with pose overlay
        """
        if not self.enabled or not pose_result.keypoints:
            return frame
        annotated_frame = frame.copy()
        # Choose color based on anomaly
        if pose_result.is_anomalous:
            if pose_result.anomaly_type in ["FIGHTING_DETECTED", "GROUP_ALTERCATION"]:
                landmark_color = (0, 0, 255)  # Red
            elif pose_result.anomaly_type in ["PERSON_FALLING", "DISTRESS_POSE"]:
                landmark_color = (0, 165, 255)  # Orange
            else:
                landmark_color = (0, 255, 255)  # Yellow
        else:
            landmark_color = (0, 255, 0)  # Green

        try:
            if _OPENPOSE_AVAILABLE and self.openpose_wrapper is not None and pose_result.keypoints:
                # Draw simple circles for keypoints (OpenPose has many connections; keep it light)
                h, w = frame.shape[:2]
                for (x_norm, y_norm, conf) in pose_result.keypoints[0]:
                    x = int(x_norm * w)
                    y = int(y_norm * h)
                    if conf > 0.05:
                        cv2.circle(annotated_frame, (x, y), 3, landmark_color, -1)
                return annotated_frame
            elif _MEDIAPIPE_AVAILABLE and self.pose is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=landmark_color,
                            thickness=2,
                            circle_radius=2
                        ),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=landmark_color,
                            thickness=2,
                            circle_radius=2
                        )
                    )
                    return annotated_frame
        except Exception:
            pass
        return frame
    
    def reset(self):
        """Reset estimator state"""
        self.pose_history.clear()
