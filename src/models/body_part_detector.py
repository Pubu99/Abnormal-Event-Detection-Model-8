"""
Body Part & Context Detector
Detects body parts (hands, head, limbs) and contextual objects to understand:
- What body parts are visible and their positions
- What objects are held in hands
- Pose/gesture information
- Threat context (e.g., hand near face, weapon in hand)

Uses YOLOv8-Pose for keypoint detection (already in your system).

Author: Ravishan
Date: 2025-11-09
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class BodyPart(Enum):
    """COCO pose keypoints (YOLOv8 standard 17 points)"""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


class GestureType(Enum):
    """Common gestures that indicate threats or normal activity"""
    NEUTRAL = "neutral"
    HANDS_RAISED = "hands_raised"  # threat indicator
    HAND_NEAR_FACE = "hand_near_face"  # normal (scratching, etc)
    HAND_TO_POCKET = "hand_to_pocket"  # suspicious
    WEAPON_IN_HAND = "weapon_in_hand"  # high threat
    BAG_HELD = "bag_held"  # normal
    RUNNING = "running"  # context-dependent
    CROUCHING = "crouching"  # suspicious
    UNKNOWN = "unknown"


class ThreatContext(Enum):
    """Threat level based on pose + object context"""
    LOW = 1  # normal pose, no threats
    MEDIUM = 2  # suspicious pose or unfamiliar object
    HIGH = 3  # weapon detected or aggressive pose
    CRITICAL = 4  # multiple threat indicators


@dataclass
class BodyPartDetection:
    """Detected body part with position and confidence"""
    part: BodyPart
    position: Tuple[float, float]  # (x, y) in image
    confidence: float
    visibility: float = 1.0  # 0-1, how visible this part is


@dataclass
class HeldObject:
    """Object detected in person's hand/grasp"""
    object_class: str  # "gun", "knife", "phone", "bag", "beer_bottle", etc.
    confidence: float
    position: Tuple[float, float]  # center of object
    hand: str  # "left" or "right"
    is_dangerous: bool = False  # weapon classification


@dataclass
class PoseContext:
    """Analysis of detected pose and gesture"""
    keypoints: Dict[BodyPart, BodyPartDetection]
    gesture: GestureType
    threat_level: ThreatContext
    held_objects: List[HeldObject] = field(default_factory=list)
    
    # Pose metrics
    body_angle: float = 0.0  # angle of torso
    hand_height_ratio: float = 0.0  # hand height / body height
    is_upright: bool = True
    
    # Confidence scores
    pose_confidence: float = 0.0
    gesture_confidence: float = 0.0
    threat_confidence: float = 0.0


class BodyPartDetector:
    """
    Analyzes pose keypoints and detects:
    - Body part locations
    - Gestures and poses
    - Held objects
    - Threat context
    """
    
    def __init__(self):
        # Gesture thresholds (in pixels for distance-based checks)
        self.hand_raise_threshold = 0.7  # hand above shoulder ratio
        self.hand_face_distance_threshold = 80  # pixels
        self.hand_pocket_distance_threshold = 100  # pixels
        
        # Dangerous object classes that increase threat level
        self.dangerous_objects = {
            'gun', 'rifle', 'pistol', 'knife', 'blade', 'axe', 'sword',
            'bomb', 'explosive', 'grenade', 'weapon'
        }
        
        # Normal objects that don't increase threat
        self.normal_objects = {
            'phone', 'bag', 'backpack', 'beer_bottle', 'water_bottle',
            'cigarette', 'cup', 'keys', 'wallet'
        }
    
    def analyze_pose(self, 
                    keypoints: List[Tuple[float, float, float]],
                    detected_objects: List[dict],
                    frame_height: int,
                    frame_width: int) -> PoseContext:
        """
        Analyze pose keypoints and objects to determine threat context
        
        Args:
            keypoints: List of (x, y, confidence) tuples (17 points for COCO)
            detected_objects: List of detected objects from YOLO
            frame_height, frame_width: Frame dimensions
        
        Returns:
            PoseContext with pose analysis and threat assessment
        """
        
        # Parse keypoints
        keypoints_dict = self._parse_keypoints(keypoints)
        
        # Detect gesture
        gesture = self._detect_gesture(keypoints_dict)
        
        # Detect held objects in hands
        held_objects = self._detect_held_objects(detected_objects, keypoints_dict)
        
        # Calculate pose metrics
        body_angle = self._calculate_body_angle(keypoints_dict)
        hand_height_ratio = self._calculate_hand_height(keypoints_dict, frame_height)
        is_upright = body_angle < 30  # less than 30 degrees from vertical
        
        # Calculate pose confidence
        pose_confidence = self._calculate_keypoint_confidence(keypoints_dict)
        
        # Determine threat level based on gesture + objects
        threat_level, threat_confidence = self._assess_threat(
            gesture, held_objects, is_upright, pose_confidence
        )
        
        return PoseContext(
            keypoints=keypoints_dict,
            gesture=gesture,
            threat_level=threat_level,
            held_objects=held_objects,
            body_angle=body_angle,
            hand_height_ratio=hand_height_ratio,
            is_upright=is_upright,
            pose_confidence=pose_confidence,
            threat_confidence=threat_confidence
        )
    
    def _parse_keypoints(self, keypoints: List[Tuple[float, float, float]]) -> Dict[BodyPart, BodyPartDetection]:
        """Convert raw keypoints to BodyPartDetection objects"""
        kp_dict = {}
        for i, (x, y, conf) in enumerate(keypoints):
            if i < len(BodyPart):
                part = BodyPart(i)
                kp_dict[part] = BodyPartDetection(
                    part=part,
                    position=(x, y),
                    confidence=conf,
                    visibility=1.0 if conf > 0.5 else 0.0
                )
        return kp_dict
    
    def _detect_gesture(self, keypoints: Dict[BodyPart, BodyPartDetection]) -> GestureType:
        """Detect gesture/pose type from keypoints"""
        
        if not self._has_keypoint(keypoints, BodyPart.LEFT_SHOULDER):
            return GestureType.UNKNOWN
        
        left_shoulder = keypoints[BodyPart.LEFT_SHOULDER].position
        right_shoulder = keypoints[BodyPart.RIGHT_SHOULDER].position
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        
        # Check hand positions relative to shoulders
        if self._has_keypoint(keypoints, BodyPart.LEFT_WRIST):
            left_wrist = keypoints[BodyPart.LEFT_WRIST].position
            if left_wrist[1] < shoulder_y - 50:  # hand above shoulder
                if self._has_keypoint(keypoints, BodyPart.RIGHT_WRIST):
                    right_wrist = keypoints[BodyPart.RIGHT_WRIST].position
                    if right_wrist[1] < shoulder_y - 50:
                        return GestureType.HANDS_RAISED
        
        # Check hand near face (normal behavior)
        if self._has_keypoint(keypoints, BodyPart.NOSE):
            nose = keypoints[BodyPart.NOSE].position
            if self._has_keypoint(keypoints, BodyPart.LEFT_WRIST):
                left_wrist = keypoints[BodyPart.LEFT_WRIST].position
                if self._distance(left_wrist, nose) < self.hand_face_distance_threshold:
                    return GestureType.HAND_NEAR_FACE
            
            if self._has_keypoint(keypoints, BodyPart.RIGHT_WRIST):
                right_wrist = keypoints[BodyPart.RIGHT_WRIST].position
                if self._distance(right_wrist, nose) < self.hand_face_distance_threshold:
                    return GestureType.HAND_NEAR_FACE
        
        # Check crouching pose
        if self._has_keypoint(keypoints, BodyPart.LEFT_KNEE):
            left_knee = keypoints[BodyPart.LEFT_KNEE].position
            if left_knee[1] < shoulder_y + 100:  # knees high (crouching)
                return GestureType.CROUCHING
        
        return GestureType.NEUTRAL
    
    def _detect_held_objects(self, 
                            detected_objects: List[dict],
                            keypoints: Dict[BodyPart, BodyPartDetection]) -> List[HeldObject]:
        """Detect objects held in hands based on spatial proximity"""
        held_items = []
        
        if not self._has_keypoint(keypoints, BodyPart.LEFT_WRIST):
            return held_items
        
        left_wrist = keypoints[BodyPart.LEFT_WRIST].position
        right_wrist = keypoints[BodyPart.RIGHT_WRIST].position if self._has_keypoint(keypoints, BodyPart.RIGHT_WRIST) else None
        
        # Check each detected object's proximity to hands
        for obj in detected_objects:
            if 'centroid' not in obj or 'class' not in obj:
                continue
            
            obj_center = obj['centroid']
            obj_class = obj['class'].lower()
            
            # Check if object is near left hand
            if self._distance(obj_center, left_wrist) < 150:  # 150 pixel threshold
                is_dangerous = obj_class in self.dangerous_objects
                held_items.append(HeldObject(
                    object_class=obj_class,
                    confidence=obj.get('confidence', 0.8),
                    position=obj_center,
                    hand='left',
                    is_dangerous=is_dangerous
                ))
            
            # Check if object is near right hand
            elif right_wrist and self._distance(obj_center, right_wrist) < 150:
                is_dangerous = obj_class in self.dangerous_objects
                held_items.append(HeldObject(
                    object_class=obj_class,
                    confidence=obj.get('confidence', 0.8),
                    position=obj_center,
                    hand='right',
                    is_dangerous=is_dangerous
                ))
        
        return held_items
    
    def _calculate_body_angle(self, keypoints: Dict[BodyPart, BodyPartDetection]) -> float:
        """Calculate body angle (tilt) from shoulders and hips"""
        if not (self._has_keypoint(keypoints, BodyPart.LEFT_SHOULDER) and
                self._has_keypoint(keypoints, BodyPart.RIGHT_SHOULDER)):
            return 0.0
        
        left_shoulder = keypoints[BodyPart.LEFT_SHOULDER].position
        right_shoulder = keypoints[BodyPart.RIGHT_SHOULDER].position
        
        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        return abs(angle)  # Return absolute angle
    
    def _calculate_hand_height(self, keypoints: Dict[BodyPart, BodyPartDetection], frame_height: int) -> float:
        """Calculate average hand height as ratio of frame height"""
        hand_positions = []
        
        if self._has_keypoint(keypoints, BodyPart.LEFT_WRIST):
            hand_positions.append(keypoints[BodyPart.LEFT_WRIST].position[1])
        if self._has_keypoint(keypoints, BodyPart.RIGHT_WRIST):
            hand_positions.append(keypoints[BodyPart.RIGHT_WRIST].position[1])
        
        if not hand_positions:
            return 0.0
        
        avg_hand_y = np.mean(hand_positions)
        return avg_hand_y / frame_height  # 0 = top, 1 = bottom
    
    def _calculate_keypoint_confidence(self, keypoints: Dict[BodyPart, BodyPartDetection]) -> float:
        """Calculate average confidence of visible keypoints"""
        if not keypoints:
            return 0.0
        
        confidences = [kp.confidence for kp in keypoints.values() if kp.visibility > 0]
        return np.mean(confidences) if confidences else 0.0
    
    def _assess_threat(self, 
                      gesture: GestureType,
                      held_objects: List[HeldObject],
                      is_upright: bool,
                      pose_confidence: float) -> Tuple[ThreatContext, float]:
        """Assess threat level based on gesture and objects"""
        
        threat_score = 0.0
        max_score = 10.0
        
        # Gesture-based threat
        if gesture == GestureType.HANDS_RAISED:
            threat_score += 2.0  # could indicate surrender or threat
        elif gesture == GestureType.CROUCHING:
            threat_score += 1.5  # suspicious
        elif gesture == GestureType.HAND_NEAR_FACE:
            threat_score += 0.0  # normal
        
        # Object-based threat
        for obj in held_objects:
            if obj.is_dangerous:
                threat_score += 4.0  # weapon detected
            else:
                threat_score += 0.5  # normal object
        
        # Posture
        if not is_upright:
            threat_score += 0.5
        
        # Confidence multiplier
        if pose_confidence < 0.5:
            threat_score *= 0.7  # lower confidence = less reliable assessment
        
        # Map score to threat level
        threat_confidence = min(threat_score / max_score, 1.0)
        
        if threat_score >= 7.0:
            return ThreatContext.CRITICAL, threat_confidence
        elif threat_score >= 4.5:
            return ThreatContext.HIGH, threat_confidence
        elif threat_score >= 2.5:
            return ThreatContext.MEDIUM, threat_confidence
        else:
            return ThreatContext.LOW, threat_confidence
    
    def _has_keypoint(self, keypoints: Dict[BodyPart, BodyPartDetection], part: BodyPart) -> bool:
        """Check if keypoint is present and visible"""
        return part in keypoints and keypoints[part].visibility > 0
    
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# Singleton instance
_body_detector = None

def get_body_part_detector() -> BodyPartDetector:
    """Get or create singleton detector instance"""
    global _body_detector
    if _body_detector is None:
        _body_detector = BodyPartDetector()
    return _body_detector
