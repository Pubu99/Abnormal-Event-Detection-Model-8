"""
Contextual Anomaly Classifier
Uses object tracking context + body part detection + movement analysis to classify
anomalies with reduced false positives and increased accuracy.

Considers:
- Object type (person, animal, etc.)
- Held objects (weapon vs normal)
- Body pose and gestures
- Movement patterns (stationary, running, etc.)
- Temporal context (how long in anomalous state)
- Zone context (restricted area vs public area)

Author: Ravishan
Date: 2025-11-09
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

from .object_tracker import ObjectTrack
from .body_part_detector import PoseContext, ThreatContext, GestureType, HeldObject


class AnomalyType(Enum):
    """Types of detected anomalies"""
    WEAPON_DETECTED = "weapon_detected"
    UNAUTHORIZED_PERSON = "unauthorized_person"
    LOITERING = "loitering"  # person stays in area too long
    RUNNING = "running"  # person running in restricted area
    FORCED_ENTRY = "forced_entry"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"
    THEFT = "theft"  # object moved without authority
    CROWD_GATHERING = "crowd_gathering"
    VEHICLE_VIOLATION = "vehicle_violation"
    NORMAL = "normal"


class ConfidenceLevel(Enum):
    """Confidence/severity of anomaly detection"""
    LOW = 0.1  # 10% - very uncertain
    MEDIUM = 0.5  # 50% - somewhat likely
    HIGH = 0.8  # 80% - very likely
    CRITICAL = 0.95  # 95% - almost certain


@dataclass
class AnomalyPrediction:
    """Prediction of anomaly with confidence and context"""
    anomaly_type: AnomalyType
    confidence: float
    reasoning: str  # explanation for the classification
    threat_level: str  # "low", "medium", "high", "critical"
    
    # Context info
    track_id: int
    object_class: str
    age_frames: int
    movement_speed: float
    
    # Components
    pose_threat: float = 0.0
    object_threat: float = 0.0
    behavior_threat: float = 0.0
    temporal_threat: float = 0.0


class ContextualClassifier:
    """
    Classify anomalies using track context and pose information
    with sophisticated logic to reduce false positives
    """
    
    def __init__(self):
        # Thresholds (configurable)
        self.weapon_confidence_threshold = 0.7
        self.running_speed_threshold = 200  # pixels per frame
        self.loitering_time_threshold = 300  # frames (~10 seconds at 30fps)
        self.loitering_distance_threshold = 100  # pixels (movement area)
        
        # Temporal context
        self.threat_history_frames = 30  # remember threat state for this many frames
        self.threat_hysteresis = 0.1  # prevent flickering between states
        
        # Track threat history for hysteresis
        self._threat_history: Dict[int, List[float]] = {}
    
    def classify_track(self,
                      track: ObjectTrack,
                      pose_context: Optional[PoseContext] = None,
                      restricted_zones: Optional[List[dict]] = None,
                      known_persons: Optional[set] = None) -> AnomalyPrediction:
        """
        Classify a tracked object as normal or anomalous
        
        Args:
            track: ObjectTrack with position, movement, age info
            pose_context: PoseContext with body parts and threat assessment
            restricted_zones: List of zone dicts with boundaries
            known_persons: Set of known person identifiers
        
        Returns:
            AnomalyPrediction with type, confidence, and explanation
        """
        
        # Component threat scores (0-1)
        pose_threat = self._calculate_pose_threat(pose_context) if pose_context is not None else 0.0
        object_threat = self._calculate_object_threat(track, pose_context) if pose_context is not None else 0.0
        behavior_threat = self._calculate_behavior_threat(track)
        temporal_threat = self._calculate_temporal_threat(track, behavior_threat)
        zone_threat = self._calculate_zone_threat(track, restricted_zones)
        
        # Overall anomaly score
        anomaly_score = (
            pose_threat * 0.25 +
            object_threat * 0.35 +
            behavior_threat * 0.20 +
            temporal_threat * 0.15 +
            zone_threat * 0.05
        )
        
        # Apply hysteresis to prevent flickering
        anomaly_score = self._apply_hysteresis(track.track_id, anomaly_score)
        
        # Determine anomaly type and confidence
        anomaly_type, confidence = self._determine_anomaly_type(
            track, pose_context, pose_threat, object_threat,
            behavior_threat, temporal_threat, zone_threat
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            track, pose_context, pose_threat, object_threat,
            behavior_threat, temporal_threat, anomaly_type
        )
        
        # Determine threat level string
        threat_level_str = self._score_to_threat_level(anomaly_score)
        
        return AnomalyPrediction(
            anomaly_type=anomaly_type,
            confidence=confidence,
            reasoning=reasoning,
            threat_level=threat_level_str,
            track_id=track.track_id,
            object_class=track.class_name,
            age_frames=track.age_frames,
            movement_speed=track.movement_speed,
            pose_threat=pose_threat,
            object_threat=object_threat,
            behavior_threat=behavior_threat,
            temporal_threat=temporal_threat
        )
    
    def _calculate_pose_threat(self, pose_context: PoseContext) -> float:
        """Calculate threat from body pose and gestures"""
        if pose_context is None:
            return 0.0
        
        threat = 0.0
        
        # Threat level from pose analysis
        if pose_context.threat_level == ThreatContext.CRITICAL:
            threat = 0.9
        elif pose_context.threat_level == ThreatContext.HIGH:
            threat = 0.7
        elif pose_context.threat_level == ThreatContext.MEDIUM:
            threat = 0.4
        else:  # LOW
            threat = 0.1
        
        # Gesture-specific adjustments
        if pose_context.gesture == GestureType.HANDS_RAISED:
            threat += 0.15  # could be threat
        elif pose_context.gesture == GestureType.CROUCHING:
            threat += 0.1  # suspicious
        elif pose_context.gesture == GestureType.HAND_NEAR_FACE:
            threat -= 0.1  # reduce threat (normal behavior)
        
        # Confidence adjustment
        threat *= pose_context.threat_confidence
        
        return min(threat, 1.0)
    
    def _calculate_object_threat(self, track: ObjectTrack, pose_context: PoseContext) -> float:
        """Calculate threat from held objects"""
        if pose_context is None or not pose_context.held_objects:
            return 0.0
        
        threat = 0.0
        
        for obj in pose_context.held_objects:
            if obj.is_dangerous:
                # Weapon detected - high threat
                threat += obj.confidence * 0.9
            else:
                # Normal object - low threat
                threat += obj.confidence * 0.1
        
        # Cap at 1.0
        return min(threat, 1.0)
    
    def _calculate_behavior_threat(self, track: ObjectTrack) -> float:
        """Calculate threat from movement behavior"""
        threat = 0.0
        
        # Running detection
        speed = track.movement_speed
        if speed > self.running_speed_threshold:
            threat += 0.5  # running
            
            # Direction changes
            if len(track.centroid_history) > 10:
                recent_positions = list(track.centroid_history)[-10:]
                direction_changes = self._count_direction_changes(recent_positions)
                if direction_changes > 3:
                    threat += 0.2  # erratic movement
        
        # Stationary for long time (not always bad)
        duration_seconds = track.age_frames / 30.0  # Assuming 30 FPS
        if track.movement_speed < 1.0 and duration_seconds > 3:
            threat += 0.1  # slight threat (loitering)
        
        return min(threat, 1.0)
    
    def _calculate_temporal_threat(self, track: ObjectTrack, current_behavior_threat: float) -> float:
        """Calculate threat based on how long object has been in anomalous state"""
        if current_behavior_threat < 0.3:
            return 0.0  # not anomalous currently
        
        # Longer time in anomalous state = higher threat
        # But cap at reasonable limits to avoid permanent threats
        temporal_factor = min(track.age / 300.0, 1.0)  # 300 frames = ~10 seconds max
        
        return current_behavior_threat * temporal_factor
    
    def _calculate_zone_threat(self, track: ObjectTrack, restricted_zones: Optional[List[dict]]) -> float:
        """Calculate threat from presence in restricted zones"""
        if not restricted_zones or not track.centroid:
            return 0.0
        
        threat = 0.0
        
        for zone in restricted_zones:
            if self._point_in_zone(track.centroid, zone):
                # In restricted zone - increase threat based on object type
                if track.class_name in ['person', 'animal']:
                    threat = 0.4  # person/animal in restricted area
                else:
                    threat = 0.2
        
        return min(threat, 1.0)
    
    def _apply_hysteresis(self, track_id: int, new_score: float) -> float:
        """Apply hysteresis to prevent flickering between threat states"""
        if track_id not in self._threat_history:
            self._threat_history[track_id] = []
        
        history = self._threat_history[track_id]
        history.append(new_score)
        
        # Keep only recent history
        if len(history) > self.threat_history_frames:
            history.pop(0)
        
        # Average with hysteresis
        if len(history) > 1:
            avg_score = np.mean(history)
            # Smooth transition
            smoothed = avg_score * 0.7 + new_score * 0.3
            return smoothed
        
        return new_score
    
    def _determine_anomaly_type(self,
                               track: ObjectTrack,
                               pose_context: Optional[PoseContext],
                               pose_threat: float,
                               object_threat: float,
                               behavior_threat: float,
                               temporal_threat: float,
                               zone_threat: float) -> Tuple[AnomalyType, float]:
        """Determine which specific anomaly is detected"""
        
        # Weapon detection - highest priority
        if object_threat > 0.7 and pose_context:
            for obj in pose_context.held_objects:
                if obj.is_dangerous and obj.confidence > self.weapon_confidence_threshold:
                    return AnomalyType.WEAPON_DETECTED, min(object_threat, 1.0)
        
        # Running in restricted area
        if behavior_threat > 0.5 and zone_threat > 0.2:
            confidence = (behavior_threat + zone_threat) / 2
            return AnomalyType.RUNNING, confidence
        
        # Loitering (staying too long)
        if temporal_threat > 0.6 and track.is_stationary():
            return AnomalyType.LOITERING, min(temporal_threat, 1.0)
        
        # Suspicious behavior
        if pose_threat > 0.6 or behavior_threat > 0.6:
            confidence = max(pose_threat, behavior_threat)
            return AnomalyType.SUSPICIOUS_BEHAVIOR, confidence
        
        # Zone violation
        if zone_threat > 0.5:
            return AnomalyType.UNAUTHORIZED_PERSON, zone_threat
        
        # Normal
        overall_threat = max(pose_threat, object_threat, behavior_threat, zone_threat)
        return AnomalyType.NORMAL, 1.0 - overall_threat
    
    def _generate_reasoning(self,
                           track: ObjectTrack,
                           pose_context: Optional[PoseContext],
                           pose_threat: float,
                           object_threat: float,
                           behavior_threat: float,
                           temporal_threat: float,
                           anomaly_type: AnomalyType) -> str:
        """Generate human-readable explanation for classification"""
        
        reasons = []
        
        if anomaly_type == AnomalyType.WEAPON_DETECTED:
            if pose_context and pose_context.held_objects:
                for obj in pose_context.held_objects:
                    if obj.is_dangerous:
                        reasons.append(f"Weapon detected: {obj.object_class} in {obj.hand} hand")
        
        elif anomaly_type == AnomalyType.RUNNING:
            reasons.append(f"Running in restricted area (speed: {track.movement_speed:.0f}px/frame)")
        
        elif anomaly_type == AnomalyType.LOITERING:
            reasons.append(f"Loitering: {track.age} frames in area (stationary)")
        
        elif anomaly_type == AnomalyType.SUSPICIOUS_BEHAVIOR:
            if pose_context:
                reasons.append(f"Suspicious pose: {pose_context.gesture.value}")
            if behavior_threat > 0.5:
                reasons.append(f"Erratic movement detected")
        
        elif anomaly_type == AnomalyType.NORMAL:
            # ⭐ FIX: Don't add "Normal activity detected" to reasoning
            # This is an anomaly detection system - we only report threats, not normal activity
            # If truly normal, this prediction will be filtered by fusion engine
            pass
        
        # Add component details
        if pose_threat > 0.5 and pose_context:
            reasons.append(f"Threat level: {pose_context.threat_level.name}")
        
        if object_threat > 0.3 and pose_context:
            held = [f"{obj.object_class}" for obj in pose_context.held_objects]
            if held:
                reasons.append(f"Holding: {', '.join(held)}")
        
        if not reasons:
            reasons.append("Unknown activity")
        
        return " | ".join(reasons)
    
    def _score_to_threat_level(self, score: float) -> str:
        """Convert numerical score to threat level"""
        if score >= 0.8:
            return "critical"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _count_direction_changes(self, positions: List[Tuple[float, float]]) -> int:
        """Count direction changes in movement path"""
        if len(positions) < 3:
            return 0
        
        changes = 0
        for i in range(1, len(positions) - 1):
            # Vector from i-1 to i
            v1 = (positions[i][0] - positions[i-1][0], positions[i][1] - positions[i-1][1])
            # Vector from i to i+1
            v2 = (positions[i+1][0] - positions[i][0], positions[i+1][1] - positions[i][1])
            
            # Dot product to detect direction change
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            if dot < 0:  # vectors pointing in different directions
                changes += 1
        
        return changes
    
    def _point_in_zone(self, point: Tuple[float, float], zone: dict) -> bool:
        """Check if point is inside zone polygon"""
        # Simple bounding box check for now
        if 'bbox' in zone:
            x1, y1, x2, y2 = zone['bbox']
            return x1 <= point[0] <= x2 and y1 <= point[1] <= y2
        
        # Support polygon zones in future
        if 'polygon' in zone:
            # TODO: Implement point-in-polygon test
            pass
        
        return False


# Singleton instance
_contextual_classifier = None

def get_contextual_classifier() -> ContextualClassifier:
    """Get or create singleton classifier instance"""
    global _contextual_classifier
    if _contextual_classifier is None:
        _contextual_classifier = ContextualClassifier()
    return _contextual_classifier
