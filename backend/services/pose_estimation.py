"""
Pose Estimation Service
Detects human poses and identifies anomalous behaviors:
- Fighting detection
- Falling detection
- Weapon handling poses
- Abnormal gestures

Author: AI Assistant
Date: 2025-10-17
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import math

try:
    import mediapipe as mp
except ImportError:
    mp = None
    print("⚠️ MediaPipe not installed. Pose estimation disabled.")


@dataclass
class PoseResult:
    """Pose detection result"""
    persons_detected: int
    poses: List[Dict]  # List of detected poses with keypoints
    is_anomalous: bool
    anomaly_type: Optional[str]
    confidence: float
    timestamp: str
    keypoints: List[List[Tuple[float, float, float]]] = field(default_factory=list)


class PoseEstimator:
    """Human pose estimation for anomaly detection"""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize pose estimator
        
        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.enabled = mp is not None
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Lazy initialization - only create when first needed
        self.mp_pose = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self.pose = None
        self._initialized = False
        
        # Pose history for temporal analysis
        self.pose_history = []
        self.history_size = 30  # 1 second at 30fps
    
    def _lazy_init(self):
        """Initialize MediaPipe only when first needed"""
        if self._initialized or not self.enabled:
            return
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            model_complexity=1  # 0=lite, 1=full, 2=heavy
        )
        self._initialized = True
        
    def analyze(self, frame: np.ndarray) -> PoseResult:
        """
        Analyze poses in the frame
        
        Args:
            frame: Input BGR frame
            
        Returns:
            PoseResult with detected anomalies
        """
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
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(rgb_frame)
        
        poses = []
        keypoints_list = []
        
        if results.pose_landmarks:
            # Extract keypoints
            landmarks = results.pose_landmarks.landmark
            keypoints = [
                (lm.x, lm.y, lm.visibility)
                for lm in landmarks
            ]
            keypoints_list.append(keypoints)
            
            # Get pose information
            pose_data = self._extract_pose_features(keypoints, frame.shape)
            poses.append(pose_data)
        
        # Update history
        self.pose_history.append(poses)
        if len(self.pose_history) > self.history_size:
            self.pose_history.pop(0)
        
        # Detect anomalies
        is_anomalous, anomaly_type, confidence = self._detect_pose_anomaly(poses)
        
        return PoseResult(
            persons_detected=len(poses),
            poses=poses,
            is_anomalous=is_anomalous,
            anomaly_type=anomaly_type,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            keypoints=keypoints_list
        )
    
    def _extract_pose_features(self, keypoints: List, frame_shape: Tuple) -> Dict:
        """Extract meaningful features from pose keypoints"""
        h, w = frame_shape[:2]
        
        # Key body parts (MediaPipe indices)
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
    
    def _detect_pose_anomaly(self, poses: List[Dict]) -> Tuple[bool, Optional[str], float]:
        """
        Detect anomalous poses
        
        Returns:
            (is_anomalous, anomaly_type, confidence)
        """
        if not poses:
            return False, None, 0.0
        
        for pose in poses:
            # 1. Falling detection (extreme body tilt)
            if abs(pose['body_angle']) > 45:
                return True, "PERSON_FALLING", 0.85
            
            # 2. Fighting detection (aggressive arm movements)
            if pose['left_elbow_angle'] < 90 and pose['right_elbow_angle'] < 90:
                # Check temporal pattern
                if len(self.pose_history) > 10:
                    rapid_arm_movement = self._check_rapid_arm_movement()
                    if rapid_arm_movement:
                        return True, "FIGHTING_DETECTED", 0.80
            
            # 3. Surrender/Distress pose (hands raised near head)
            if pose['arms_raised'] and pose['hands_near_head']:
                return True, "DISTRESS_POSE", 0.75
            
            # 4. Weapon handling pose (one arm extended, rigid posture)
            if (pose['left_elbow_angle'] > 160 or pose['right_elbow_angle'] > 160):
                # Straight arm could indicate weapon
                return True, "SUSPICIOUS_POSE", 0.65
        
        # 5. Multiple people with aggressive poses (group fighting)
        if len(poses) >= 2:
            aggressive_count = sum(
                1 for p in poses 
                if p['left_elbow_angle'] < 90 or p['right_elbow_angle'] < 90
            )
            if aggressive_count >= 2:
                return True, "GROUP_ALTERCATION", 0.78
        
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
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process again to get landmarks for drawing
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Draw landmarks
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
            
            # Draw skeleton
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
        
        return frame
    
    def reset(self):
        """Reset estimator state"""
        self.pose_history.clear()
