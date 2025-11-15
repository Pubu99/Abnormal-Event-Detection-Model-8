"""
Advanced Weapon Detection Service
Combines multiple signals for weapon detection WITHOUT specialized weapon-detection models

DETECTION STRATEGIES:
1. Pose-based weapon holding detection (arm angles, hand positions)
2. Object shape analysis (elongated objects in hands)
3. Suspicious object carrying patterns
4. Context-aware heuristics (person + suspicious object)

Author: Professional Implementation
Date: 2025-11-13
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class WeaponType(Enum):
    """Detected weapon types"""
    HANDGUN = "handgun"
    RIFLE = "rifle"
    KNIFE = "knife"
    BLUNT_OBJECT = "blunt_object"
    UNKNOWN_WEAPON = "unknown_weapon"


@dataclass
class WeaponDetection:
    """Weapon detection result"""
    weapon_type: WeaponType
    confidence: float
    person_bbox: Tuple[int, int, int, int]
    weapon_bbox: Optional[Tuple[int, int, int, int]]
    detection_method: str
    details: str


class WeaponDetector:
    """
    Professional weapon detection using pose and context analysis
    NO specialized models needed - uses existing YOLO + Pose data
    """
    
    def __init__(self):
        """Initialize weapon detector"""
        # COCO keypoint indices
        self.KEYPOINT_INDICES = {
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12
        }
        
        # Suspicious object classes that could be weapons
        self.SUSPICIOUS_OBJECTS = [
            'bottle',  # Could be weapon
            'umbrella',  # Could conceal weapon
            'baseball bat', 'sports ball',
            'scissors', 'knife',  # If YOLO detects these
            'cell phone'  # Could be gun-shaped
        ]
        
    def detect_weapons(self,
                      frame: np.ndarray,
                      yolo_detections: List[Dict],
                      pose_keypoints: List[np.ndarray],
                      pose_bboxes: List[Tuple[int, int, int, int]]) -> List[WeaponDetection]:
        """
        Detect potential weapons using multi-signal fusion
        
        Args:
            frame: Current frame
            yolo_detections: YOLO object detections
            pose_keypoints: Pose keypoints for detected persons
            pose_bboxes: Bounding boxes for persons
            
        Returns:
            List of weapon detections
        """
        detections = []
        
        # METHOD 1: Pose-based weapon holding detection
        pose_weapons = self._detect_weapon_holding_pose(
            pose_keypoints, pose_bboxes
        )
        detections.extend(pose_weapons)
        
        # METHOD 2: Suspicious object + person correlation
        object_weapons = self._detect_suspicious_objects(
            frame, yolo_detections, pose_bboxes
        )
        detections.extend(object_weapons)
        
        # METHOD 3: Hand region analysis for elongated objects
        hand_weapons = self._analyze_hand_regions(
            frame, pose_keypoints, pose_bboxes
        )
        detections.extend(hand_weapons)
        
        return detections
    
    def _detect_weapon_holding_pose(self,
                                    keypoints_list: List[np.ndarray],
                                    bboxes: List[Tuple[int, int, int, int]]) -> List[WeaponDetection]:
        """
        CRITICAL: Detect weapon-holding poses
        
        Weapons patterns:
        - Handgun: Arm extended forward, wrist aligned with elbow/shoulder
        - Rifle: Both arms forward, parallel configuration
        - Knife: Arm raised or extended with bent elbow
        """
        detections = []
        
        for keypoints, bbox in zip(keypoints_list, bboxes):
            if keypoints is None or len(keypoints) < 17:
                continue
            
            # Extract key points
            try:
                left_shoulder = keypoints[self.KEYPOINT_INDICES['left_shoulder']]
                right_shoulder = keypoints[self.KEYPOINT_INDICES['right_shoulder']]
                left_elbow = keypoints[self.KEYPOINT_INDICES['left_elbow']]
                right_elbow = keypoints[self.KEYPOINT_INDICES['right_elbow']]
                left_wrist = keypoints[self.KEYPOINT_INDICES['left_wrist']]
                right_wrist = keypoints[self.KEYPOINT_INDICES['right_wrist']]
                
                # Check confidence scores (keypoints are [x, y, confidence])
                if left_shoulder[2] < 0.3 or right_shoulder[2] < 0.3:
                    continue
                
                # HANDGUN DETECTION: Extended arm with alignment
                left_arm_extended = self._check_handgun_pose(
                    left_shoulder, left_elbow, left_wrist
                )
                right_arm_extended = self._check_handgun_pose(
                    right_shoulder, right_elbow, right_wrist
                )
                
                if left_arm_extended or right_arm_extended:
                    confidence = 0.75 if (left_arm_extended and right_arm_extended) else 0.65
                    detections.append(WeaponDetection(
                        weapon_type=WeaponType.HANDGUN,
                        confidence=confidence,
                        person_bbox=bbox,
                        weapon_bbox=None,
                        detection_method="pose_analysis",
                        details="Handgun holding pose detected (extended arm alignment)"
                    ))
                
                # RIFLE DETECTION: Both arms forward, parallel
                rifle_pose = self._check_rifle_pose(
                    left_shoulder, right_shoulder,
                    left_elbow, right_elbow,
                    left_wrist, right_wrist
                )
                
                if rifle_pose:
                    detections.append(WeaponDetection(
                        weapon_type=WeaponType.RIFLE,
                        confidence=0.80,
                        person_bbox=bbox,
                        weapon_bbox=None,
                        detection_method="pose_analysis",
                        details="Rifle holding pose detected (parallel arm configuration)"
                    ))
                
                # KNIFE/BLUNT OBJECT: Raised arm with bent elbow
                knife_pose_left = self._check_knife_pose(left_shoulder, left_elbow, left_wrist)
                knife_pose_right = self._check_knife_pose(right_shoulder, right_elbow, right_wrist)
                
                if knife_pose_left or knife_pose_right:
                    detections.append(WeaponDetection(
                        weapon_type=WeaponType.KNIFE,
                        confidence=0.60,
                        person_bbox=bbox,
                        weapon_bbox=None,
                        detection_method="pose_analysis",
                        details="Knife/weapon holding pose detected (raised arm)"
                    ))
                    
            except (IndexError, KeyError):
                continue
        
        return detections
    
    def _check_handgun_pose(self, shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> bool:
        """Check if arm configuration matches handgun holding"""
        if shoulder[2] < 0.3 or elbow[2] < 0.3 or wrist[2] < 0.3:
            return False
        
        # Calculate arm extension angle
        shoulder_to_elbow = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
        elbow_to_wrist = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
        
        # Normalize vectors
        se_norm = np.linalg.norm(shoulder_to_elbow)
        ew_norm = np.linalg.norm(elbow_to_wrist)
        
        if se_norm < 1e-6 or ew_norm < 1e-6:
            return False
        
        # Check alignment (should be nearly straight for gun pose)
        dot_product = np.dot(shoulder_to_elbow, elbow_to_wrist)
        cos_angle = dot_product / (se_norm * ew_norm)
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        # Handgun: Arm extended forward (angle > 150 degrees = nearly straight)
        # Wrist should be forward of shoulder
        is_extended = angle_deg > 145
        is_forward = wrist[0] > shoulder[0] - 20  # Allow some tolerance
        
        return is_extended and is_forward
    
    def _check_rifle_pose(self,
                         left_shoulder: np.ndarray, right_shoulder: np.ndarray,
                         left_elbow: np.ndarray, right_elbow: np.ndarray,
                         left_wrist: np.ndarray, right_wrist: np.ndarray) -> bool:
        """Check if pose matches rifle holding (both arms forward, parallel)"""
        # Need good confidence on all points
        if any(kp[2] < 0.3 for kp in [left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist]):
            return False
        
        # Both wrists should be forward of shoulders
        left_forward = left_wrist[0] > left_shoulder[0]
        right_forward = right_wrist[0] > right_shoulder[0]
        
        # Wrists should be relatively close together (holding same object)
        wrist_distance = np.linalg.norm([left_wrist[0] - right_wrist[0], left_wrist[1] - right_wrist[1]])
        shoulders_distance = np.linalg.norm([left_shoulder[0] - right_shoulder[0], left_shoulder[1] - right_shoulder[1]])
        
        wrists_close = wrist_distance < shoulders_distance * 1.2
        
        # Both elbows should be bent (not fully extended like handgun)
        left_arm_bent = self._calculate_elbow_angle(left_shoulder, left_elbow, left_wrist) < 150
        right_arm_bent = self._calculate_elbow_angle(right_shoulder, right_elbow, right_wrist) < 150
        
        return left_forward and right_forward and wrists_close and left_arm_bent and right_arm_bent
    
    def _check_knife_pose(self, shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> bool:
        """Check if pose matches knife/blunt weapon holding (raised or extended arm)"""
        if shoulder[2] < 0.3 or elbow[2] < 0.3 or wrist[2] < 0.3:
            return False
        
        # Wrist above shoulder (raised arm)
        wrist_raised = wrist[1] < shoulder[1] - 30
        
        # OR wrist extended forward with bent elbow (stabbing pose)
        elbow_angle = self._calculate_elbow_angle(shoulder, elbow, wrist)
        bent_forward = (elbow_angle < 120) and (wrist[0] > elbow[0])
        
        return wrist_raised or bent_forward
    
    def _calculate_elbow_angle(self, shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
        """Calculate elbow joint angle"""
        v1 = np.array([shoulder[0] - elbow[0], shoulder[1] - elbow[1]])
        v2 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 180.0
        
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        return angle
    
    def _detect_suspicious_objects(self,
                                   frame: np.ndarray,
                                   yolo_detections: List[Dict],
                                   person_bboxes: List[Tuple[int, int, int, int]]) -> List[WeaponDetection]:
        """
        Detect suspicious objects near persons
        """
        detections = []
        
        # Check for suspicious objects
        suspicious_objs = [
            obj for obj in yolo_detections 
            if obj.get('class', '') in self.SUSPICIOUS_OBJECTS
        ]
        
        for obj in suspicious_objs:
            obj_bbox = obj.get('bbox')
            if not obj_bbox:
                continue
            
            ox, oy, ow, oh = obj_bbox
            obj_center = (ox + ow/2, oy + oh/2)
            
            # Find closest person
            for person_bbox in person_bboxes:
                px, py, pw, ph = person_bbox
                
                # Check if object is within person bbox
                if (ox >= px and ox + ow <= px + pw and 
                    oy >= py and oy + oh <= py + ph):
                    
                    # Object is being held/carried
                    detections.append(WeaponDetection(
                        weapon_type=WeaponType.UNKNOWN_WEAPON,
                        confidence=0.55,
                        person_bbox=person_bbox,
                        weapon_bbox=obj_bbox,
                        detection_method="object_correlation",
                        details=f"Suspicious object: {obj.get('class')} held by person"
                    ))
                    break
        
        return detections
    
    def _analyze_hand_regions(self,
                             frame: np.ndarray,
                             keypoints_list: List[np.ndarray],
                             bboxes: List[Tuple[int, int, int, int]]) -> List[WeaponDetection]:
        """
        Analyze hand regions for elongated objects (guns, knives)
        """
        detections = []
        
        for keypoints, bbox in zip(keypoints_list, bboxes):
            if keypoints is None or len(keypoints) < 17:
                continue
            
            try:
                left_wrist = keypoints[self.KEYPOINT_INDICES['left_wrist']]
                right_wrist = keypoints[self.KEYPOINT_INDICES['right_wrist']]
                
                # Analyze regions around wrists for elongated objects
                for wrist, side in [(left_wrist, 'left'), (right_wrist, 'right')]:
                    if wrist[2] < 0.3:
                        continue
                    
                    # Extract hand region (30x30 pixels around wrist)
                    wx, wy = int(wrist[0]), int(wrist[1])
                    h, w = frame.shape[:2]
                    
                    x1 = max(0, wx - 15)
                    y1 = max(0, wy - 15)
                    x2 = min(w, wx + 15)
                    y2 = min(h, wy + 15)
                    
                    if x2 - x1 < 10 or y2 - y1 < 10:
                        continue
                    
                    hand_region = frame[y1:y2, x1:x2]
                    
                    # Check for elongated dark objects (potential weapons)
                    has_elongated = self._detect_elongated_object(hand_region)
                    
                    if has_elongated:
                        detections.append(WeaponDetection(
                            weapon_type=WeaponType.UNKNOWN_WEAPON,
                            confidence=0.50,
                            person_bbox=bbox,
                            weapon_bbox=(x1, y1, x2-x1, y2-y1),
                            detection_method="hand_region_analysis",
                            details=f"Elongated object detected in {side} hand region"
                        ))
                        
            except (IndexError, KeyError):
                continue
        
        return detections
    
    def _detect_elongated_object(self, region: np.ndarray) -> bool:
        """
        Detect elongated objects in hand region using edge detection
        """
        if region.size == 0:
            return False
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check for elongated contours
            for contour in contours:
                if len(contour) < 5:
                    continue
                
                # Fit ellipse
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (x, y), (MA, ma), angle = ellipse
                    
                    # Elongated if major axis > 2x minor axis
                    if MA > 0 and ma > 0 and MA / ma > 2.0:
                        return True
                except:
                    continue
            
            return False
            
        except Exception:
            return False


# Singleton instance
_weapon_detector_instance = None

def get_weapon_detector() -> WeaponDetector:
    """Get singleton weapon detector instance"""
    global _weapon_detector_instance
    if _weapon_detector_instance is None:
        _weapon_detector_instance = WeaponDetector()
    return _weapon_detector_instance
