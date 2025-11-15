"""
Advanced Pose Estimation and Motion Detection Enhancement
Implements state-of-the-art techniques for abnormal event detection:

1. Synergistic Pose & Object Detection
2. Temporal-Spatial Graph Modeling
3. Multi-Modal Motion Feature Extraction
4. Real-Time Efficient Processing
5. Context-Aware Fusion

Author: Professional AI/ML Implementation
Date: 2025-11-13
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
from datetime import datetime
import math


@dataclass
class EnhancedPoseFrame:
    """Enhanced pose data with temporal and spatial context"""
    keypoints: np.ndarray  # (N, 17, 3) - N persons, 17 keypoints, (x, y, confidence)
    timestamp: float
    frame_number: int
    bounding_boxes: List[Tuple[int, int, int, int]]
    velocities: Optional[np.ndarray] = None  # Keypoint velocities
    accelerations: Optional[np.ndarray] = None  # Keypoint accelerations
    pose_angles: Optional[Dict[str, float]] = None  # Joint angles
    skeleton_features: Optional[np.ndarray] = None  # Geometric features


@dataclass
class MotionFeatures:
    """Advanced motion features beyond basic optical flow"""
    optical_flow_magnitude: float
    optical_flow_direction: float
    motion_entropy: float  # Chaos/randomness in motion
    motion_uniformity: float  # How uniform the motion is
    dominant_motion_direction: Tuple[float, float]  # (x, y) unit vector
    motion_acceleration: float  # Change in motion magnitude
    motion_regions: int  # Number of distinct motion regions
    is_crowd_panic: bool
    is_rapid_movement: bool
    is_erratic: bool


class AdvancedPoseMotionAnalyzer:
    """
    Professional pose and motion analyzer with:
    - Temporal-spatial graph modeling
    - Multi-modal feature extraction
    - Real-time optimization
    - Context-aware anomaly detection
    """
    
    def __init__(self, history_length: int = 30, fps: float = 30.0):
        """
        Initialize advanced analyzer
        
        Args:
            history_length: Number of frames to keep in temporal buffer
            fps: Expected frames per second for velocity calculations
        """
        self.history_length = history_length
        self.fps = fps
        self.dt = 1.0 / fps  # Time delta between frames
        
        # Temporal buffers for spatial-temporal modeling
        self.pose_history: deque = deque(maxlen=history_length)
        self.motion_history: deque = deque(maxlen=history_length)
        self.optical_flow_history: deque = deque(maxlen=10)
        
        # Previous frame for optical flow
        self.prev_gray = None
        
        # COCO keypoint indices for skeletal analysis
        self.KEYPOINT_INDICES = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2,
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }
        
        # Skeletal connections for graph structure
        self.SKELETON_CONNECTIONS = [
            # Head
            (0, 1), (0, 2), (1, 3), (2, 4),
            # Arms
            (5, 7), (7, 9), (6, 8), (8, 10),
            # Body
            (5, 6), (5, 11), (6, 12), (11, 12),
            # Legs
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        # Anomaly detection thresholds (HIGHLY SENSITIVE)
        # These values are optimized for MAXIMUM sensitivity and fast detection
        
        # POSE THRESHOLDS
        self.FALL_ANGLE_THRESHOLD = 30  # degrees from vertical (MORE SENSITIVE)
        self.LYING_DURATION_THRESHOLD = 1.5  # seconds (detect lying poses FASTER)
        self.COLLAPSE_VELOCITY_THRESHOLD = 70  # pixels/sec for rapid fall (MORE SENSITIVE)
        
        # MOTION THRESHOLDS
        self.RAPID_MOVEMENT_THRESHOLD = 40  # pixels per frame (MORE SENSITIVE)
        self.PANIC_FLOW_MAGNITUDE = 20  # optical flow magnitude for crowd panic (MORE SENSITIVE)
        self.ERRATIC_MOTION_ENTROPY = 0.60  # motion entropy threshold (MORE SENSITIVE)
        self.FREEZE_MOTION_THRESHOLD = 2.0  # pixels/frame (detect immobility)
        
        # GAIT ANALYSIS
        self.ABNORMAL_GAIT_ASYMMETRY = 0.35  # leg movement asymmetry ratio (MORE SENSITIVE)
        self.LIMPING_THRESHOLD = 0.35  # limb velocity difference
        
        # VIOLENT POSE DETECTION
        self.AGGRESSIVE_ARM_ANGLE = 110  # degrees (arms extended aggressively)
        self.GRABBING_PROXIMITY = 30  # pixels (hands near another person)
        
        # LOITERING DETECTION
        self.LOITERING_TIME_THRESHOLD = 6.0  # seconds of stationary behavior (MORE SENSITIVE)
        self.PACING_PATTERN_CYCLES = 3  # number of direction reversals for suspicious pacing (MORE SENSITIVE)
        
        # TEMPORAL CONSISTENCY
        self.ANOMALY_PERSISTENCE_FRAMES = 3  # frames to confirm anomaly (reduce flicker)
        
        # PERFORMANCE OPTIMIZATION
        self.frame_counter = 0
        self.last_skeletal_features = None
        self.cached_motion_features = None
    
    def reset_cache(self):
        """
        Reset all in-memory caches
        Used when starting fresh analysis or clearing stale data
        """
        self.pose_history.clear()
        self.motion_history.clear()
        self.optical_flow_history.clear()
        self.prev_gray = None
        self.frame_counter = 0
        self.last_skeletal_features = None
        self.cached_motion_features = None
        
    def analyze_frame(self, 
                     frame: np.ndarray,
                     pose_keypoints: List[np.ndarray],
                     bounding_boxes: List[Tuple[int, int, int, int]],
                     frame_number: int) -> Dict[str, Any]:
        """
        Comprehensive pose and motion analysis with performance optimization
        
        Args:
            frame: Current BGR frame
            pose_keypoints: List of keypoint arrays for detected persons
            bounding_boxes: Bounding boxes for each person
            frame_number: Current frame number
            
        Returns:
            Comprehensive analysis results
        """
        timestamp = frame_number / self.fps
        self.frame_counter += 1
        
        # FAST PATH: Skip heavy processing if no persons detected
        if not pose_keypoints or len(pose_keypoints) == 0:
            return self._create_empty_result()
        
        # 1. SYNERGISTIC POSE & OBJECT DETECTION (lightweight)
        enhanced_poses = self._enhance_poses_with_context(
            pose_keypoints, bounding_boxes, frame
        )
        
        # 2. TEMPORAL-SPATIAL GRAPH MODELING (every frame for temporal consistency)
        pose_velocities, pose_accelerations = self._compute_temporal_dynamics(
            enhanced_poses
        )
        
        # 3. SKELETAL FEATURE EXTRACTION (cached - only compute every 2 frames)
        if self.frame_counter % 2 == 0:
            skeletal_features = self._extract_skeletal_features(enhanced_poses)
            self.last_skeletal_features = skeletal_features
        else:
            skeletal_features = self.last_skeletal_features or self._extract_skeletal_features(enhanced_poses)
        
        # 4. ADVANCED MOTION FEATURE EXTRACTION (cached - only compute every 2 frames)
        if self.frame_counter % 2 == 0:
            motion_features = self._extract_motion_features(frame)
            self.cached_motion_features = motion_features
        else:
            motion_features = self.cached_motion_features or self._extract_motion_features(frame)
        
        # 5. POSE-BASED ANOMALY DETECTION
        pose_anomalies = self._detect_pose_anomalies(
            enhanced_poses, skeletal_features, pose_velocities
        )
        
        # 6. MOTION-BASED ANOMALY DETECTION
        motion_anomalies = self._detect_motion_anomalies(motion_features)
        
        # 7. MULTI-MODAL FUSION
        fused_result = self._fuse_pose_motion_anomalies(
            pose_anomalies, motion_anomalies, motion_features
        )
        
        # Store in temporal buffer
        pose_frame = EnhancedPoseFrame(
            keypoints=np.array(enhanced_poses) if enhanced_poses else np.array([]),
            timestamp=timestamp,
            frame_number=frame_number,
            bounding_boxes=bounding_boxes,
            velocities=pose_velocities,
            accelerations=pose_accelerations,
            pose_angles=skeletal_features.get('joint_angles', {}),
            skeleton_features=skeletal_features.get('geometric_features')
        )
        self.pose_history.append(pose_frame)
        self.motion_history.append(motion_features)
        
        return fused_result
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Return empty result when no persons detected for fast processing"""
        return {
            'is_anomalous': False,
            'anomaly_type': None,
            'confidence': 0.0,
            'pose_anomalies': [],
            'motion_anomalies': [],
            'details': []
        }
    
    def _enhance_poses_with_context(self,
                                    pose_keypoints: List[np.ndarray],
                                    bounding_boxes: List[Tuple[int, int, int, int]],
                                    frame: np.ndarray) -> List[np.ndarray]:
        """
        Enhance pose keypoints with object and scene context
        
        Strategy: Use bounding box constraints to refine keypoint locations
        and filter out spurious detections outside person regions
        """
        enhanced_poses = []
        
        for kpts, bbox in zip(pose_keypoints, bounding_boxes):
            if len(kpts) == 0:
                continue
                
            x, y, w, h = bbox
            
            # Refine keypoints: filter those outside bounding box
            refined_kpts = kpts.copy()
            for i in range(len(refined_kpts)):
                kp_x, kp_y = refined_kpts[i, 0], refined_kpts[i, 1]
                
                # Check if keypoint is within reasonable distance of bbox
                margin = 0.2  # Allow 20% margin outside bbox
                if not (x - w*margin < kp_x < x + w*(1+margin) and 
                       y - h*margin < kp_y < y + h*(1+margin)):
                    # Keypoint outside reasonable region - reduce confidence
                    refined_kpts[i, 2] *= 0.3
            
            enhanced_poses.append(refined_kpts)
        
        return enhanced_poses
    
    def _compute_temporal_dynamics(self, 
                                   current_poses: List[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute velocities and accelerations using temporal differencing
        
        Returns:
            (velocities, accelerations) for each person's keypoints
        """
        if len(self.pose_history) < 2:
            return None, None
        
        prev_frame = self.pose_history[-1]
        
        if len(prev_frame.keypoints) == 0 or len(current_poses) == 0:
            return None, None
        
        # Match poses between frames (simple nearest neighbor)
        velocities = []
        accelerations = []
        
        for curr_pose in current_poses:
            # Find closest previous pose (by center of mass)
            curr_com = np.mean(curr_pose[:, :2], axis=0)
            
            best_match_idx = -1
            best_dist = float('inf')
            
            for i, prev_pose in enumerate(prev_frame.keypoints):
                if len(prev_pose) == 0:
                    continue
                prev_com = np.mean(prev_pose[:, :2], axis=0)
                dist = np.linalg.norm(curr_com - prev_com)
                
                if dist < best_dist:
                    best_dist = dist
                    best_match_idx = i
            
            if best_match_idx >= 0 and best_dist < 100:  # Max 100 pixels movement
                prev_pose = prev_frame.keypoints[best_match_idx]
                
                # Compute velocity
                vel = (curr_pose[:, :2] - prev_pose[:, :2]) / self.dt
                velocities.append(vel)
                
                # Compute acceleration if we have previous velocity
                if prev_frame.velocities is not None and best_match_idx < len(prev_frame.velocities):
                    prev_vel = prev_frame.velocities[best_match_idx]
                    accel = (vel - prev_vel) / self.dt
                    accelerations.append(accel)
                else:
                    accelerations.append(np.zeros_like(vel))
            else:
                # No match - new person
                velocities.append(np.zeros((len(curr_pose), 2)))
                accelerations.append(np.zeros((len(curr_pose), 2)))
        
        return (np.array(velocities) if velocities else None,
                np.array(accelerations) if accelerations else None)
    
    def _extract_skeletal_features(self, poses: List[np.ndarray]) -> Dict[str, Any]:
        """
        Extract geometric skeletal features for graph-based modeling
        
        Features include:
        - Joint angles
        - Limb lengths
        - Body orientation
        - Symmetry measures
        """
        if not poses:
            return {'joint_angles': {}, 'geometric_features': None}
        
        all_angles = []
        all_features = []
        
        for pose in poses:
            if len(pose) < 17:
                continue
            
            # Calculate key joint angles
            angles = {}
            
            # Torso angle (vertical alignment)
            if pose[11, 2] > 0.3 and pose[5, 2] > 0.3:  # Left hip and shoulder
                torso_vec = pose[5, :2] - pose[11, :2]
                vertical = np.array([0, -1])
                torso_angle = self._angle_between_vectors(torso_vec, vertical)
                angles['torso_vertical'] = torso_angle
            
            # Arm angles (for gesture detection)
            # Left arm angle
            if all(pose[idx, 2] > 0.3 for idx in [5, 7, 9]):  # shoulder, elbow, wrist
                angles['left_arm'] = self._calculate_joint_angle(
                    pose[5, :2], pose[7, :2], pose[9, :2]
                )
            
            # Right arm angle
            if all(pose[idx, 2] > 0.3 for idx in [6, 8, 10]):
                angles['right_arm'] = self._calculate_joint_angle(
                    pose[6, :2], pose[8, :2], pose[10, :2]
                )
            
            # Leg angles (for fall detection)
            if all(pose[idx, 2] > 0.3 for idx in [11, 13, 15]):  # Left leg
                angles['left_leg'] = self._calculate_joint_angle(
                    pose[11, :2], pose[13, :2], pose[15, :2]
                )
            
            if all(pose[idx, 2] > 0.3 for idx in [12, 14, 16]):  # Right leg
                angles['right_leg'] = self._calculate_joint_angle(
                    pose[12, :2], pose[14, :2], pose[16, :2]
                )
            
            all_angles.append(angles)
            
            # Geometric features vector
            features = []
            # Limb lengths (normalized)
            limb_lengths = self._calculate_limb_lengths(pose)
            features.extend(limb_lengths)
            
            # Body aspect ratio
            if pose[11, 2] > 0.3 and pose[5, 2] > 0.3:
                height = abs(pose[5, 1] - pose[15, 1])  # shoulder to ankle
                width = abs(pose[5, 0] - pose[6, 0])  # shoulder width
                aspect_ratio = height / (width + 1e-6)
                features.append(aspect_ratio)
            else:
                features.append(0.0)
            
            all_features.append(features)
        
        return {
            'joint_angles': all_angles[0] if all_angles else {},
            'geometric_features': np.array(all_features) if all_features else None
        }
    
    def _extract_motion_features(self, frame: np.ndarray) -> MotionFeatures:
        """
        Extract advanced motion features using optical flow and statistical analysis
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize defaults
        motion_features = MotionFeatures(
            optical_flow_magnitude=0.0,
            optical_flow_direction=0.0,
            motion_entropy=0.0,
            motion_uniformity=1.0,
            dominant_motion_direction=(0.0, 0.0),
            motion_acceleration=0.0,
            motion_regions=0,
            is_crowd_panic=False,
            is_rapid_movement=False,
            is_erratic=False
        )
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return motion_features
        
        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Extract flow magnitude and direction
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Average flow magnitude
        avg_magnitude = np.mean(magnitude)
        
        # Dominant motion direction
        flow_x = np.mean(flow[..., 0])
        flow_y = np.mean(flow[..., 1])
        norm = math.sqrt(flow_x**2 + flow_y**2) + 1e-6
        dominant_dir = (flow_x / norm, flow_y / norm)
        
        # Motion entropy (measure of chaos)
        hist, _ = np.histogram(angle, bins=36, range=(0, 2*np.pi), density=True)
        hist = hist + 1e-10  # Avoid log(0)
        entropy = -np.sum(hist * np.log2(hist))
        normalized_entropy = entropy / np.log2(36)  # Normalize to [0, 1]
        
        # Motion uniformity (how similar motion vectors are)
        direction_variance = np.std(angle)
        uniformity = 1.0 - min(direction_variance / np.pi, 1.0)
        
        # Motion acceleration (change in magnitude over time)
        if len(self.optical_flow_history) > 0:
            prev_magnitude = self.optical_flow_history[-1].optical_flow_magnitude
            acceleration = abs(avg_magnitude - prev_magnitude) / self.dt
        else:
            acceleration = 0.0
        
        # Count distinct motion regions using connected components
        motion_mask = (magnitude > 2.0).astype(np.uint8)
        num_labels, _ = cv2.connectedComponents(motion_mask)
        motion_regions = num_labels - 1  # Subtract background
        
        # Anomaly detection flags
        is_rapid = avg_magnitude > self.RAPID_MOVEMENT_THRESHOLD
        is_panic = (avg_magnitude > self.PANIC_FLOW_MAGNITUDE and 
                   motion_regions > 5 and normalized_entropy > 0.6)
        is_erratic = normalized_entropy > self.ERRATIC_MOTION_ENTROPY
        
        motion_features = MotionFeatures(
            optical_flow_magnitude=float(avg_magnitude),
            optical_flow_direction=float(np.mean(angle)),
            motion_entropy=float(normalized_entropy),
            motion_uniformity=float(uniformity),
            dominant_motion_direction=dominant_dir,
            motion_acceleration=float(acceleration),
            motion_regions=int(motion_regions),
            is_crowd_panic=bool(is_panic),
            is_rapid_movement=bool(is_rapid),
            is_erratic=bool(is_erratic)
        )
        
        self.prev_gray = gray
        return motion_features
    
    def _detect_pose_anomalies(self,
                               poses: List[np.ndarray],
                               skeletal_features: Dict[str, Any],
                               velocities: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        COMPREHENSIVE POSE ANOMALY DETECTION
        
        Detects:
        1. Falls and tumbling
        2. Lying down (collapsed/motionless)
        3. Violent/aggressive stances
        4. Unnatural body postures
        5. Abnormal gait/limping
        """
        anomalies = {
            'is_anomalous': False,
            'anomaly_type': None,
            'confidence': 0.0,
            'details': [],
            'sub_anomalies': []  # Multiple concurrent anomalies
        }
        
        if not poses or 'joint_angles' not in skeletal_features:
            return anomalies
        
        angles = skeletal_features['joint_angles']
        max_confidence = 0.0
        primary_anomaly = None
        
        # ==================== FALL DETECTION ====================
        if 'torso_vertical' in angles:
            torso_angle = angles['torso_vertical']
            
            # CRITICAL: Rapid fall (high priority)
            if velocities is not None:
                avg_velocity = np.mean(np.linalg.norm(velocities, axis=2))
                if torso_angle > self.FALL_ANGLE_THRESHOLD and avg_velocity > self.COLLAPSE_VELOCITY_THRESHOLD:
                    confidence = min(0.95, (torso_angle / 90.0) * (avg_velocity / 100.0))
                    anomalies['sub_anomalies'].append({
                        'type': 'RAPID_FALL',
                        'confidence': confidence,
                        'detail': f"Rapid fall detected (angle: {torso_angle:.1f}°, velocity: {avg_velocity:.1f})"
                    })
                    if confidence > max_confidence:
                        max_confidence = confidence
                        primary_anomaly = 'PERSON_FALLING'
                        anomalies['details'].append(f"🚨 CRITICAL: Rapid fall (angle: {torso_angle:.1f}°, velocity: {avg_velocity:.1f})")
            
            # FALL: Normal speed but extreme angle
            if torso_angle > self.FALL_ANGLE_THRESHOLD:
                confidence = min(0.88, (torso_angle - self.FALL_ANGLE_THRESHOLD) / 55.0)
                if confidence > max_confidence:
                    max_confidence = confidence
                    primary_anomaly = 'PERSON_FALLING'
                    anomalies['details'].append(f"Person falling (torso angle: {torso_angle:.1f}°)")
        
        # ==================== LYING DOWN DETECTION ====================
        # Check if person is horizontal for extended period
        if len(self.pose_history) >= int(self.LYING_DURATION_THRESHOLD * self.fps):
            recent_angles = []
            for hist_frame in list(self.pose_history)[-int(self.LYING_DURATION_THRESHOLD * self.fps):]:
                if hist_frame and hasattr(hist_frame, 'pose_angles') and hist_frame.pose_angles:
                    recent_angles.append(hist_frame.pose_angles.get('torso_vertical', 0))
            
            if recent_angles and len(recent_angles) >= self.LYING_DURATION_THRESHOLD * self.fps * 0.7:
                avg_angle = np.mean(recent_angles)
                if avg_angle > 60:  # Horizontal for extended period
                    confidence = 0.82
                    anomalies['sub_anomalies'].append({
                        'type': 'LYING_MOTIONLESS',
                        'confidence': confidence,
                        'detail': f"Person lying motionless for {self.LYING_DURATION_THRESHOLD}s"
                    })
                    if confidence > max_confidence:
                        max_confidence = confidence
                        primary_anomaly = 'PERSON_LYING'
                        anomalies['details'].append(f"⚠️ Person lying motionless ({self.LYING_DURATION_THRESHOLD}s)")
        
        # ==================== VIOLENT/AGGRESSIVE POSES ====================
        left_arm = angles.get('left_arm', 180)
        right_arm = angles.get('right_arm', 180)
        
        # FIGHTING STANCE: Both arms bent aggressively
        if left_arm < 90 and right_arm < 90:
            # Check for rapid arm movements (fighting pattern)
            if velocities is not None and len(self.pose_history) > 5:
                # Extract arm velocities from recent frames
                arm_velocities = []
                for hist_frame in list(self.pose_history)[-5:]:
                    if hist_frame and hist_frame.velocities is not None:
                        # Velocities for wrist keypoints (indices 9, 10 in COCO)
                        arm_vels = hist_frame.velocities[0, [9, 10], :]
                        arm_velocities.append(np.linalg.norm(arm_vels))
                
                if arm_velocities and np.mean(arm_velocities) > 30:
                    confidence = 0.85
                    anomalies['sub_anomalies'].append({
                        'type': 'FIGHTING',
                        'confidence': confidence,
                        'detail': "Fighting stance with rapid arm movements"
                    })
                    if confidence > max_confidence:
                        max_confidence = confidence
                        primary_anomaly = 'FIGHTING'
                        anomalies['details'].append("⚔️ Fighting detected (aggressive arm movements)")
        
        # THREATENING GESTURE: Arms extended forward (pushing/grabbing)
        if left_arm > self.AGGRESSIVE_ARM_ANGLE or right_arm > self.AGGRESSIVE_ARM_ANGLE:
            if velocities is not None:
                arm_vel = np.mean(np.linalg.norm(velocities[0, [9, 10], :], axis=1))
                if arm_vel > 20:  # Moving arms aggressively
                    confidence = 0.78
                    anomalies['sub_anomalies'].append({
                        'type': 'THREATENING_GESTURE',
                        'confidence': confidence,
                        'detail': "Threatening gesture (extended arms with movement)"
                    })
                    if confidence > max_confidence:
                        max_confidence = confidence
                        primary_anomaly = 'AGGRESSIVE_POSTURE'
                        anomalies['details'].append("🤜 Threatening gesture detected")
        
        # ==================== UNNATURAL POSTURES ====================
        # Check for extreme joint angles that are biomechanically unusual
        left_leg = angles.get('left_leg', 180)
        right_leg = angles.get('right_leg', 180)
        
        # AWKWARD POSE: Both legs at extreme angles
        if (left_leg < 30 or left_leg > 170) and (right_leg < 30 or right_leg > 170):
            confidence = 0.68
            anomalies['sub_anomalies'].append({
                'type': 'AWKWARD_POSTURE',
                'confidence': confidence,
                'detail': f"Unnatural leg positions (left: {left_leg:.1f}°, right: {right_leg:.1f}°)"
            })
            if confidence > max_confidence and primary_anomaly is None:
                max_confidence = confidence
                primary_anomaly = 'UNUSUAL_POSE'
                anomalies['details'].append("⚠️ Awkward body posture detected")
        
        # ==================== ABNORMAL GAIT/LIMPING ====================
        if velocities is not None and len(self.pose_history) > 10:
            # Analyze leg movement symmetry over time
            left_ankle_vels = []
            right_ankle_vels = []
            
            for hist_frame in list(self.pose_history)[-10:]:
                if hist_frame and hist_frame.velocities is not None:
                    # Ankle keypoints (indices 15, 16 in COCO)
                    left_ankle_vels.append(np.linalg.norm(hist_frame.velocities[0, 15, :]))
                    right_ankle_vels.append(np.linalg.norm(hist_frame.velocities[0, 16, :]))
            
            if left_ankle_vels and right_ankle_vels:
                avg_left = np.mean(left_ankle_vels)
                avg_right = np.mean(right_ankle_vels)
                
                # LIMPING: Significant asymmetry in leg movement
                if avg_left > 0.1 or avg_right > 0.1:  # Person is moving
                    asymmetry = abs(avg_left - avg_right) / (max(avg_left, avg_right) + 1e-6)
                    
                    if asymmetry > self.ABNORMAL_GAIT_ASYMMETRY:
                        confidence = min(0.75, asymmetry / 0.6)
                        anomalies['sub_anomalies'].append({
                            'type': 'ABNORMAL_GAIT',
                            'confidence': confidence,
                            'detail': f"Limping/abnormal gait (asymmetry: {asymmetry:.2f})"
                        })
                        if confidence > max_confidence and primary_anomaly is None:
                            max_confidence = confidence
                            primary_anomaly = 'ABNORMAL_GAIT'
                            anomalies['details'].append(f"🚶 Abnormal gait/limping detected")
        
        # ==================== RAPID POSE CHANGES ====================
        if velocities is not None:
            avg_velocity = np.mean(np.linalg.norm(velocities, axis=2))
            if avg_velocity > self.RAPID_MOVEMENT_THRESHOLD:
                confidence = min(0.82, avg_velocity / 100.0)
                anomalies['sub_anomalies'].append({
                    'type': 'RAPID_MOVEMENT',
                    'confidence': confidence,
                    'detail': f"Rapid pose change (velocity: {avg_velocity:.1f})"
                })
                if confidence > max_confidence and primary_anomaly is None:
                    max_confidence = confidence
                    primary_anomaly = 'RAPID_MOVEMENT'
                    anomalies['details'].append(f"⚡ Rapid movement detected")
        
        # Set final results with LOWERED threshold for faster detection
        if max_confidence >= 0.40:  # Lower threshold (was 0.50) for faster detection
            anomalies['is_anomalous'] = True
            anomalies['anomaly_type'] = primary_anomaly
            anomalies['confidence'] = max_confidence
        
        return anomalies
    
    def _detect_motion_anomalies(self, motion_features: MotionFeatures) -> Dict[str, Any]:
        """
        COMPREHENSIVE MOTION ANOMALY DETECTION
        
        Detects:
        1. Sudden abrupt movements/jerks
        2. Loitering and strange pacing
        3. Freezing/immobility when motion expected
        4. Erratic motion patterns
        5. Crowd panic
        """
        anomalies = {
            'is_anomalous': False,
            'anomaly_type': None,
            'confidence': 0.0,
            'details': [],
            'sub_anomalies': []
        }
        
        max_confidence = 0.0
        primary_anomaly = None
        
        # ==================== CROWD PANIC ====================
        # High magnitude + high entropy + multiple regions
        if motion_features.is_crowd_panic:
            confidence = min(0.92, motion_features.optical_flow_magnitude / 50.0)
            anomalies['sub_anomalies'].append({
                'type': 'CROWD_PANIC',
                'confidence': confidence,
                'detail': f"Crowd panic (flow: {motion_features.optical_flow_magnitude:.1f}, regions: {motion_features.motion_regions})"
            })
            if confidence > max_confidence:
                max_confidence = confidence
                primary_anomaly = 'CROWD_PANIC'
                anomalies['details'].append(f"🚨 CRITICAL: Crowd panic detected")
        
        # ==================== SUDDEN ABRUPT MOVEMENTS ====================
        # High acceleration indicates sudden jerks/jolts
        if motion_features.motion_acceleration > 15:
            confidence = min(0.86, motion_features.motion_acceleration / 30.0)
            anomalies['sub_anomalies'].append({
                'type': 'ABRUPT_MOVEMENT',
                'confidence': confidence,
                'detail': f"Sudden jerk (acceleration: {motion_features.motion_acceleration:.1f})"
            })
            if confidence > max_confidence:
                max_confidence = confidence
                primary_anomaly = 'ABRUPT_MOVEMENT'
                anomalies['details'].append(f"⚡ Sudden abrupt movement detected")
        
        # ==================== ERRATIC MOTION ====================
        # High entropy indicates chaotic, unpredictable motion
        if motion_features.is_erratic:
            confidence = min(0.80, motion_features.motion_entropy / 0.8)
            anomalies['sub_anomalies'].append({
                'type': 'ERRATIC_MOTION',
                'confidence': confidence,
                'detail': f"Erratic motion (entropy: {motion_features.motion_entropy:.2f})"
            })
            if confidence > max_confidence and primary_anomaly != 'CROWD_PANIC':
                max_confidence = confidence
                primary_anomaly = 'ERRATIC_MOTION'
                anomalies['details'].append(f"🔀 Erratic motion pattern detected")
        
        # ==================== LOITERING DETECTION ====================
        # Very low motion for extended period
        if len(self.motion_history) >= int(self.LOITERING_TIME_THRESHOLD * self.fps):
            recent_magnitudes = [
                m.optical_flow_magnitude 
                for m in list(self.motion_history)[-int(self.LOITERING_TIME_THRESHOLD * self.fps):]
            ]
            
            if recent_magnitudes:
                avg_magnitude = np.mean(recent_magnitudes)
                # LOITERING: Person present but very little motion
                if avg_magnitude < self.FREEZE_MOTION_THRESHOLD and motion_features.motion_regions >= 1:
                    confidence = 0.75
                    anomalies['sub_anomalies'].append({
                        'type': 'LOITERING',
                        'confidence': confidence,
                        'detail': f"Loitering detected ({self.LOITERING_TIME_THRESHOLD}s stationary)"
                    })
                    if confidence > max_confidence and primary_anomaly is None:
                        max_confidence = confidence
                        primary_anomaly = 'LOITERING'
                        anomalies['details'].append(f"⏱️ Loitering detected ({self.LOITERING_TIME_THRESHOLD}s)")
        
        # ==================== STRANGE PACING ====================
        # Detect repetitive back-and-forth motion patterns
        if len(self.motion_history) >= 20:
            # Analyze direction changes
            recent_directions = [
                m.optical_flow_direction 
                for m in list(self.motion_history)[-20:]
            ]
            
            if recent_directions and len(recent_directions) >= 15:
                # Count direction reversals
                direction_changes = 0
                for i in range(1, len(recent_directions)):
                    angle_diff = abs(recent_directions[i] - recent_directions[i-1])
                    # Normalize to [0, pi]
                    angle_diff = min(angle_diff, 2*np.pi - angle_diff)
                    if angle_diff > np.pi * 0.7:  # ~126 degrees - significant reversal
                        direction_changes += 1
                
                # PACING: Multiple direction reversals (back-and-forth)
                if direction_changes >= self.PACING_PATTERN_CYCLES:
                    confidence = min(0.72, direction_changes / 8.0)
                    anomalies['sub_anomalies'].append({
                        'type': 'STRANGE_PACING',
                        'confidence': confidence,
                        'detail': f"Strange pacing pattern ({direction_changes} reversals)"
                    })
                    if confidence > max_confidence and primary_anomaly is None:
                        max_confidence = confidence
                        primary_anomaly = 'SUSPICIOUS_PACING'
                        anomalies['details'].append(f"🔄 Strange pacing pattern detected")
        
        # ==================== FREEZING (SUDDEN STOP) ====================
        # Sudden drop in motion when previous motion was significant
        if len(self.motion_history) >= 5:
            prev_magnitude = np.mean([
                m.optical_flow_magnitude 
                for m in list(self.motion_history)[-5:-1]
            ])
            current_magnitude = motion_features.optical_flow_magnitude
            
            # FREEZE: Was moving, now suddenly stopped
            if prev_magnitude > 15 and current_magnitude < self.FREEZE_MOTION_THRESHOLD:
                confidence = 0.70
                anomalies['sub_anomalies'].append({
                    'type': 'SUDDEN_FREEZE',
                    'confidence': confidence,
                    'detail': f"Sudden freeze (was moving at {prev_magnitude:.1f}, now {current_magnitude:.1f})"
                })
                if confidence > max_confidence and primary_anomaly is None:
                    max_confidence = confidence
                    primary_anomaly = 'SUDDEN_STOP'
                    anomalies['details'].append(f"🛑 Sudden freeze detected")
        
        # ==================== RAPID MOVEMENT ====================
        if motion_features.is_rapid_movement:
            confidence = min(0.84, motion_features.optical_flow_magnitude / 100.0)
            anomalies['sub_anomalies'].append({
                'type': 'RAPID_MOVEMENT',
                'confidence': confidence,
                'detail': f"Rapid movement (magnitude: {motion_features.optical_flow_magnitude:.1f})"
            })
            if confidence > max_confidence and primary_anomaly is None:
                max_confidence = confidence
                primary_anomaly = 'RAPID_MOVEMENT'
                anomalies['details'].append(f"⚡ Rapid movement detected")
        
        # Set final results with LOWERED threshold for faster detection
        if max_confidence >= 0.40:  # Lower threshold (was 0.50) for faster detection
            anomalies['is_anomalous'] = True
            anomalies['anomaly_type'] = primary_anomaly
            anomalies['confidence'] = max_confidence
        
        return anomalies
    
    def _fuse_pose_motion_anomalies(self,
                                    pose_anomalies: Dict[str, Any],
                                    motion_anomalies: Dict[str, Any],
                                    motion_features: MotionFeatures) -> Dict[str, Any]:
        """
        Multi-modal fusion of pose and motion anomalies
        
        Strategy: Weighted voting with context-aware confidence boosting
        """
        fused = {
            'is_anomalous': False,
            'anomaly_type': None,
            'confidence': 0.0,
            'pose_score': 0.0,
            'motion_score': 0.0,
            'fusion_details': [],
            'motion_features': {
                'optical_flow_magnitude': motion_features.optical_flow_magnitude,
                'motion_entropy': motion_features.motion_entropy,
                'motion_regions': motion_features.motion_regions
            }
        }
        
        pose_score = pose_anomalies['confidence'] if pose_anomalies['is_anomalous'] else 0.0
        motion_score = motion_anomalies['confidence'] if motion_anomalies['is_anomalous'] else 0.0
        
        # Weighted fusion: Pose 60%, Motion 40%
        fusion_score = pose_score * 0.6 + motion_score * 0.4
        
        # Consensus boost: if both detect anomaly, boost confidence
        if pose_anomalies['is_anomalous'] and motion_anomalies['is_anomalous']:
            fusion_score = min(fusion_score * 1.3, 1.0)
            fused['fusion_details'].append("⚡ Pose and motion consensus boost")
        
        fused['pose_score'] = pose_score
        fused['motion_score'] = motion_score
        fused['confidence'] = fusion_score
        
        # Determine primary anomaly type
        if fusion_score >= 0.5:
            fused['is_anomalous'] = True
            
            # Prioritize pose anomalies (more reliable)
            if pose_anomalies['is_anomalous']:
                fused['anomaly_type'] = pose_anomalies['anomaly_type']
                fused['fusion_details'].extend(pose_anomalies['details'])
            else:
                fused['anomaly_type'] = motion_anomalies['anomaly_type']
                fused['fusion_details'].extend(motion_anomalies['details'])
            
            # Add motion details if available
            if motion_anomalies['is_anomalous'] and pose_anomalies['is_anomalous']:
                fused['fusion_details'].extend(motion_anomalies['details'])
        
        return fused
    
    # Helper methods
    
    def _angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in degrees"""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return math.degrees(angle_rad)
    
    def _calculate_joint_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at joint p2 formed by points p1-p2-p3"""
        v1 = p1 - p2
        v2 = p3 - p2
        return self._angle_between_vectors(v1, v2)
    
    def _calculate_limb_lengths(self, pose: np.ndarray) -> List[float]:
        """Calculate normalized limb lengths"""
        lengths = []
        
        for connection in self.SKELETON_CONNECTIONS:
            i, j = connection
            if pose[i, 2] > 0.3 and pose[j, 2] > 0.3:
                length = np.linalg.norm(pose[i, :2] - pose[j, :2])
                lengths.append(float(length))
            else:
                lengths.append(0.0)
        
        return lengths


def get_advanced_pose_motion_analyzer(history_length: int = 30, fps: float = 30.0) -> AdvancedPoseMotionAnalyzer:
    """Factory function to get analyzer instance"""
    return AdvancedPoseMotionAnalyzer(history_length=history_length, fps=fps)
