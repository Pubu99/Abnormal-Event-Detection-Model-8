"""
Professional Intelligent Fusion Engine
Multi-modal anomaly detection with weighted scoring system

Fusion Strategy (Based on Reliability):
- ML Model: 40% (domain-specific UCF Crime training)
- YOLO Objects: 25% (most reliable - trained on millions)
- Pose Analysis: 20% (MediaPipe - robust detection)
- Motion Analysis: 15% (OpenCV - proven techniques)

Detection Philosophy:
- Only report ANOMALIES (fusion_score >= 0.70)
- No "Normal" highlighting - we detect threats, not normal activity
- Person Falling detection even if ML says Normal
- Maintain detection history for analysis

Author: Professional Implementation
Date: 2025-10-17
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class AnomalyType(Enum):
    """Extended anomaly categories - 14 original + new detections"""
    # Original 13 from UCF Crime dataset
    SHOOTING = "Shooting"
    EXPLOSION = "Explosion"
    ROBBERY = "Robbery"
    ASSAULT = "Assault"
    FIGHTING = "Fighting"
    ABUSE = "Abuse"
    ARSON = "Arson"
    BURGLARY = "Burglary"
    VANDALISM = "Vandalism"
    ARREST = "Arrest"
    ROAD_ACCIDENTS = "RoadAccidents"
    SHOPLIFTING = "Shoplifting"
    STEALING = "Stealing"
    
    # New detections from multi-modal fusion
    WEAPON_DETECTED = "Weapon Detected"
    MULTIPLE_WEAPONS = "Multiple Weapons"
    PERSON_FALLING = "Person Falling"
    PERSON_LYING = "Person Lying Down"
    CROWD_PANIC = "Crowd Panic"
    LOITERING = "Loitering"
    ABANDONED_OBJECT = "Abandoned Object"
    HIGH_CROWD_DENSITY = "High Crowd Density"
    FIRE_SMOKE = "Fire or Smoke Detected"
    RAPID_MOVEMENT = "Rapid Movement"
    UNUSUAL_POSE = "Unusual Body Pose"
    CROWD_FLOW_ANOMALY = "Abnormal Crowd Flow"
    SUSPICIOUS_BEHAVIOR = "Suspicious Behavior"
    VIOLENT_CONFRONTATION = "Violent Confrontation"


class Severity(Enum):
    """Anomaly severity levels for prioritization"""
    CRITICAL = "CRITICAL"  # Immediate threat - weapons, violence, fire
    HIGH = "HIGH"          # Urgent attention - assault, robbery, accidents
    MEDIUM = "MEDIUM"      # Monitor closely - suspicious behavior, falling
    LOW = "LOW"            # Informational - loitering, crowd density


@dataclass
class FusedDetection:
    """Professional anomaly detection result"""
    # Core detection info (required fields first)
    anomaly_type: AnomalyType
    severity: Severity
    confidence: float
    fusion_score: float
    timestamp: str
    frame_number: int
    
    # Individual modality scores (required - for transparency)
    ml_score: float
    object_score: float
    pose_score: float
    motion_score: float
    
    # Optional fields with defaults
    detection_id: str = ""
    detected_objects: List[str] = field(default_factory=list)
    bounding_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    consensus_count: int = 0
    critical_override: bool = False
    explanation: str = ""
    reasoning: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    # User feedback field (set when an operator confirms/declines a detection)
    user_feedback: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary"""
        return {
            'anomaly_type': self.anomaly_type.value,
            'severity': self.severity.value,
            'confidence': round(self.confidence, 3),
            'fusion_score': round(self.fusion_score, 3),
            'timestamp': self.timestamp,
            'frame_number': self.frame_number,
            'individual_scores': {
                'ml_model': round(self.ml_score, 3),
                'objects': round(self.object_score, 3),
                'pose': round(self.pose_score, 3),
                'motion': round(self.motion_score, 3)
            },
            'detected_objects': self.detected_objects,
            'bounding_boxes': self.bounding_boxes,
            'explanation': self.explanation,
            'reasoning': self.reasoning,
            'metadata': self.metadata
            ,
            'user_feedback': self.user_feedback
        }


class IntelligentFusionEngine:
    """
    Professional Multi-Modal Fusion System
    
    Combines 4 detection modalities for robust anomaly detection:
    1. ML Model (40%) - Your trained BiLSTM-Transformer model
    2. YOLO Objects (25%) - Pre-trained object detection
    3. Pose Analysis (20%) - MediaPipe pose estimation
    4. Motion Analysis (15%) - OpenCV optical flow
    
    Key Features:
    - Weighted voting with intelligent consensus
    - Critical object override (weapons → instant alert)
    - Person falling detection (pose + motion override ML)
    - Anomaly-only reporting (no normal highlights)
    - Detection history with statistics
    """
    
    # Fusion weights (carefully tuned for reliability)
    # Reduce ML influence by default if model is noisy; increase perceptual
    # modalities (objects & pose) that have stronger real-time signals.
    WEIGHT_ML = 0.20        # ML contribution (reduced)
    WEIGHT_OBJECTS = 0.35   # YOLO objects (increased)
    WEIGHT_POSE = 0.30      # Pose estimation (increased)
    WEIGHT_MOTION = 0.15    # Motion analysis (supporting evidence)
    
    # Detection threshold
    ANOMALY_THRESHOLD = 0.70  # Report only if score >= 0.70

    # Consensus bonus (when multiple modalities agree)
    # Slightly reduced to avoid overpowering a single high-scoring modality
    CONSENSUS_BONUS = 0.10  # Add when 2+ modalities detect same anomaly
    
    def __init__(self):
        """Initialize fusion engine"""
        self.detection_history: List[FusedDetection] = []
        self.frame_count = 0
        self.detection_counter = 0  # For unique IDs
        # Suppression state: keep a set of suppressed detection ids so
        # previously-declined detections won't reappear. Avoid a global
        # suppression flag that stops future anomalies entirely.
        self.decline_all_active = False  # retained for backward-compatibility but not used to block new detections
        self.suppressed_ids = set()
        
        # Severity mapping for all anomaly types
        self.severity_map = {
            # CRITICAL - Immediate threats
            AnomalyType.SHOOTING: Severity.CRITICAL,
            AnomalyType.EXPLOSION: Severity.CRITICAL,
            AnomalyType.ARSON: Severity.CRITICAL,
            AnomalyType.MULTIPLE_WEAPONS: Severity.CRITICAL,
            AnomalyType.FIRE_SMOKE: Severity.CRITICAL,
            AnomalyType.WEAPON_DETECTED: Severity.CRITICAL,
            
            # HIGH - Urgent attention required
            AnomalyType.ROBBERY: Severity.HIGH,
            AnomalyType.ASSAULT: Severity.HIGH,
            AnomalyType.FIGHTING: Severity.HIGH,
            AnomalyType.VIOLENT_CONFRONTATION: Severity.HIGH,
            AnomalyType.ROAD_ACCIDENTS: Severity.HIGH,
            AnomalyType.BURGLARY: Severity.HIGH,
            AnomalyType.STEALING: Severity.HIGH,
            AnomalyType.CROWD_PANIC: Severity.HIGH,
            
            # MEDIUM - Monitor closely
            AnomalyType.ABUSE: Severity.MEDIUM,
            AnomalyType.VANDALISM: Severity.MEDIUM,
            AnomalyType.SHOPLIFTING: Severity.MEDIUM,
            AnomalyType.HIGH_CROWD_DENSITY: Severity.MEDIUM,
            AnomalyType.SUSPICIOUS_BEHAVIOR: Severity.MEDIUM,
            AnomalyType.PERSON_FALLING: Severity.MEDIUM,
            AnomalyType.RAPID_MOVEMENT: Severity.MEDIUM,
            AnomalyType.ABANDONED_OBJECT: Severity.MEDIUM,
            
            # LOW - Informational
            AnomalyType.LOITERING: Severity.LOW,
            AnomalyType.ARREST: Severity.LOW,
            AnomalyType.CROWD_FLOW_ANOMALY: Severity.LOW,
            AnomalyType.UNUSUAL_POSE: Severity.LOW,
            AnomalyType.PERSON_LYING: Severity.LOW,
        }
        
        # Critical objects that trigger immediate alerts
        self.critical_objects = {
            'knife', 'scissors', 'gun', 'rifle', 'pistol', 'weapon',
            'fire', 'explosion', 'smoke', 'bomb', 'explosive'
        }
        
        # ML class mapping
        self.ml_class_map = {
            'Shooting': AnomalyType.SHOOTING,
            'Explosion': AnomalyType.EXPLOSION,
            'Robbery': AnomalyType.ROBBERY,
            'Assault': AnomalyType.ASSAULT,
            'Fighting': AnomalyType.FIGHTING,
            'Abuse': AnomalyType.ABUSE,
            'Arson': AnomalyType.ARSON,
            'Burglary': AnomalyType.BURGLARY,
            'Vandalism': AnomalyType.VANDALISM,
            'Arrest': AnomalyType.ARREST,
            'RoadAccidents': AnomalyType.ROAD_ACCIDENTS,
            'Shoplifting': AnomalyType.SHOPLIFTING,
            'Stealing': AnomalyType.STEALING,
        }
    
    def _generate_detection_id(self) -> str:
        """Generate unique detection ID"""
        self.detection_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"DET_{timestamp}_{self.detection_counter:04d}"
    
    def fuse_detections(self,
                       ml_result: Optional[Dict],
                       yolo_detections: List[Dict],
                       pose_result: Optional[Dict],
                       motion_result: Optional[Dict],
                       frame_number: int,
                       contextual_predictions: Optional[List[Dict]] = None) -> Optional[FusedDetection]:
        """
        Main fusion pipeline - combines all detection modalities with contextual tracking
        
        Args:
            ml_result: {'class': 'Shooting', 'confidence': 0.95, 'probabilities': [...]}
            yolo_detections: [{'class': 'person', 'bbox': (x,y,w,h), 'confidence': 0.9}]
            pose_result: {'is_anomalous': True, 'anomaly_type': 'PERSON_FALLING', 'confidence': 0.92}
            motion_result: {'is_unusual': True, 'anomaly_type': 'RAPID_MOVEMENT', 'confidence': 0.85}
            contextual_predictions: [{'anomaly_type': 'WEAPON_DETECTED', 'confidence': 0.92, ...}]
            frame_number: Current frame number
            
        Returns:
            FusedDetection if anomaly detected (score >= 0.70), None if normal
        """
        self.frame_count += 1
        reasoning = []
        
        # PRIORITY 0: Check contextual predictions for high-confidence anomalies
        # These are from the advanced tracking system with context awareness
        if contextual_predictions:
            for pred in contextual_predictions:
                anomaly_type_str = pred.get('anomaly_type', '')
                confidence = pred.get('confidence', 0.0)
                threat_level = pred.get('threat_level', 'low')
                pred_reasoning = pred.get('reasoning', '')
                
                # Apply confidence thresholds based on anomaly type
                min_confidence = self._get_contextual_threshold(anomaly_type_str)
                
                if confidence >= min_confidence:
                    # Map contextual anomaly type to our enum
                    anomaly_type = self._map_contextual_anomaly(anomaly_type_str)
                    severity = self._map_threat_to_severity(threat_level)
                    
                    # Extract contextual metadata for RL training
                    track_id = pred.get('track_id', 'unknown')
                    track_duration = pred.get('track_duration', 0.0)
                    movement_speed = pred.get('movement_speed', 0.0)
                    
                    # Create contextual detection with full reasoning and metadata
                    detection = FusedDetection(
                        detection_id=self._generate_detection_id(),
                        anomaly_type=anomaly_type,
                        severity=severity,
                        confidence=confidence,
                        fusion_score=confidence,  # Use contextual confidence as fusion score
                        timestamp=datetime.now().isoformat(),
                        frame_number=frame_number,
                        ml_score=0.0,
                        object_score=confidence if 'WEAPON' in anomaly_type_str else 0.5,
                        pose_score=confidence if 'POSE' in pred_reasoning else 0.3,
                        motion_score=confidence if 'movement' in pred_reasoning.lower() else 0.2,
                        detected_objects=[obj['class'] for obj in yolo_detections],
                        bounding_boxes=[obj.get('bbox', (0,0,0,0)) for obj in yolo_detections],
                        consensus_count=1,
                        critical_override=('WEAPON' in anomaly_type_str),
                        explanation=pred_reasoning,
                        reasoning=[f"🎯 Contextual Detection: {pred_reasoning}"],
                        metadata={
                            'contextual_track_id': track_id,
                            'threat_level': threat_level,
                            'contextual_classifier': True,
                            # Contextual features for RL training
                            'track_duration': float(track_duration),
                            'movement_speed': float(movement_speed),
                            'loitering_score': float(pred.get('loitering_score', 0.0)),
                            'track_confidence': float(pred.get('track_confidence', confidence)),
                            'gesture_score': float(pred.get('gesture_score', 0.0)),
                            'held_object_count': int(pred.get('held_object_count', 0)),
                            'body_pose_score': float(pred.get('body_pose_score', 0.0)),
                            'temporal_consistency': float(pred.get('temporal_consistency', 1.0))
                        }
                    )
                    
                    self.detection_history.append(detection)
                    return detection
        
        # PRIORITY 1: Check for critical objects (immediate override)
        critical_detection = self._check_critical_override(yolo_detections)
        if critical_detection:
            self.detection_history.append(critical_detection)
            return critical_detection
        
        # PRIORITY 2: Calculate individual modality scores
        ml_score = self._score_ml_model(ml_result)
        object_score = self._score_objects(yolo_detections)
        pose_score = self._score_pose(pose_result)
        motion_score = self._score_motion(motion_result)
        
        # PRIORITY 3: Weighted fusion
        fusion_score = (
            ml_score * self.WEIGHT_ML +
            object_score * self.WEIGHT_OBJECTS +
            pose_score * self.WEIGHT_POSE +
            motion_score * self.WEIGHT_MOTION
        )
        
        # PRIORITY 4: Consensus bonus (multiple modalities agree)
        # Count active modalities using adjusted thresholds so noisy ML
        # predictions are less likely to be treated as strong votes.
        active_modalities = sum([
            1 if ml_score > 0.6 else 0,
            1 if object_score > 0.3 else 0,
            1 if pose_score > 0.4 else 0,
            1 if motion_score > 0.4 else 0
        ])
        
        if active_modalities >= 2:
            fusion_score += self.CONSENSUS_BONUS
            reasoning.append(f"✓ {active_modalities} modalities in consensus")
        
        # PRIORITY 5: Special case - Person Falling
        # Even if ML says Normal, Pose + Motion can detect it
        falling_detection = self._check_person_falling(
            ml_result, yolo_detections, pose_result, motion_result
        )
        if falling_detection:
            fusion_score = max(fusion_score, falling_detection['boost_score'])
            reasoning.extend(falling_detection['reasoning'])
        
        # PRIORITY 6: Special case - High Crowd Density
        # Pure object detection case - reliable enough on its own
        person_count = len([obj for obj in yolo_detections if obj['class'] == 'person'])
        if person_count > 15 and object_score >= 1.0:
            # Boost score to ensure detection
            fusion_score = max(fusion_score, 0.75)
            reasoning.append(f"👥 High crowd density: {person_count} people detected")
        
        # DECISION: Is this an anomaly?
        if fusion_score < self.ANOMALY_THRESHOLD:
            return None  # Normal - don't report
        
        # Determine anomaly type and create detection
        anomaly_type, explanation = self._determine_anomaly_type(
            ml_result, yolo_detections, pose_result, motion_result, fusion_score
        )
        
        severity = self.severity_map.get(anomaly_type, Severity.MEDIUM)
        
        # Extract details
        detected_objects = [obj['class'] for obj in yolo_detections]
        bounding_boxes = [obj.get('bbox', (0,0,0,0)) for obj in yolo_detections]
        
        # Build reasoning - FILTER OUT NORMAL CLASSIFICATIONS
        if ml_score > 0.5:
            ml_class = ml_result.get('class', 'Unknown')
            ml_conf = ml_result.get('confidence', 0.0)
            # ⭐ DON'T SHOW "NormalVideos" - User doesn't care about normal ⭐
            if ml_class and ml_class.lower() not in ['normalvideos', 'normal', 'normal_videos']:
                reasoning.append(f"ML Model: {ml_class} ({ml_conf*100:.1f}%)")
        
        if object_score > 0.3:
            # Filter out generic/normal objects, prioritize threats
            dangerous_objects = [obj for obj in detected_objects if any(
                danger in obj.lower() for danger in ['gun', 'knife', 'weapon', 'pistol', 'rifle']
            )]
            relevant_objects = dangerous_objects if dangerous_objects else detected_objects[:3]
            if relevant_objects:
                reasoning.append(f"Objects: {', '.join(relevant_objects)}")
        
        if pose_score > 0.5:
            pose_type = pose_result.get('anomaly_type', 'Unknown')
            reasoning.append(f"Pose: {pose_type}")
        
        if motion_score > 0.5:
            motion_type = motion_result.get('anomaly_type', 'Unknown')
            reasoning.append(f"Motion: {motion_type}")
        
        # Create metadata
        metadata = {
            'ml_prediction': ml_result.get('class') if ml_result else None,
            'ml_confidence': ml_result.get('confidence', 0.0) if ml_result else 0.0,
            'object_count': len(yolo_detections),
            'pose_anomaly': pose_result.get('anomaly_type') if pose_result and pose_result.get('is_anomalous') else None,
            'motion_anomaly': motion_result.get('anomaly_type') if motion_result and motion_result.get('is_unusual') else None,
            'consensus_modalities': active_modalities
        }
        
        # Create fused detection
        detection = FusedDetection(
            detection_id=self._generate_detection_id(),
            anomaly_type=anomaly_type,
            severity=severity,
            confidence=max(ml_score, object_score, pose_score, motion_score),
            fusion_score=fusion_score,
            timestamp=datetime.now().isoformat(),
            frame_number=frame_number,
            ml_score=ml_score,
            object_score=object_score,
            pose_score=pose_score,
            motion_score=motion_score,
            detected_objects=detected_objects,
            bounding_boxes=bounding_boxes,
            consensus_count=active_modalities,
            critical_override=False,
            explanation=explanation,
            reasoning=reasoning,
            metadata=metadata
        )
        
        # Respect suppression state: if this detection id has been suppressed,
        # do not add it to active history. Note: we removed the global
        # "decline_all_active" flag to avoid blocking all future detections.
        try:
            if detection.detection_id in self.suppressed_ids:
                return detection
        except Exception:
            pass

        # Add to history
        self.detection_history.append(detection)
        
        return detection
    
    def _check_critical_override(self, yolo_detections: List[Dict]) -> Optional[FusedDetection]:
        """
        Check for critical objects that bypass fusion scoring
        Weapons, fire, explosions → Immediate CRITICAL alert
        """
        detected_classes = [obj['class'] for obj in yolo_detections]
        
        # Check for critical objects
        critical_found = [cls for cls in detected_classes if cls.lower() in self.critical_objects]
        
        if not critical_found:
            return None
        
        # Determine specific anomaly type
        if any(w in critical_found for w in ['knife', 'scissors', 'gun', 'rifle', 'pistol', 'weapon']):
            weapons = [w for w in critical_found if w in ['knife', 'scissors', 'gun', 'rifle', 'pistol', 'weapon']]
            
            if len(weapons) >= 2:
                anomaly_type = AnomalyType.MULTIPLE_WEAPONS
                explanation = f"Multiple weapons detected: {', '.join(weapons)}"
            else:
                anomaly_type = AnomalyType.WEAPON_DETECTED
                explanation = f"Weapon detected: {weapons[0]}"
        else:
            anomaly_type = AnomalyType.FIRE_SMOKE
            explanation = f"Critical danger: {', '.join(critical_found)}"
        
        # Create immediate CRITICAL detection
        return FusedDetection(
            detection_id=self._generate_detection_id(),
            anomaly_type=anomaly_type,
            severity=Severity.CRITICAL,
            confidence=0.99,
            fusion_score=1.0,
            timestamp=datetime.now().isoformat(),
            frame_number=self.frame_count,
            ml_score=0.0,
            object_score=1.0,
            pose_score=0.0,
            motion_score=0.0,
            detected_objects=detected_classes,
            bounding_boxes=[obj.get('bbox', (0,0,0,0)) for obj in yolo_detections],
            consensus_count=1,
            critical_override=True,
            explanation=explanation,
            reasoning=["🚨 CRITICAL OVERRIDE: Dangerous object detected"],
            metadata={'override': True, 'critical_objects': critical_found}
        )
    
    def _check_person_falling(self,
                            ml_result: Optional[Dict],
                            yolo_detections: List[Dict],
                            pose_result: Optional[Dict],
                            motion_result: Optional[Dict]) -> Optional[Dict]:
        """
        Special case: Detect person falling even if ML says Normal
        Uses Pose + Motion consensus
        """
        # Check if pose detects falling
        pose_falling = (
            pose_result and 
            pose_result.get('is_anomalous') and
            'FALLING' in pose_result.get('anomaly_type', '').upper()
        )
        
        # Check if motion detects rapid downward movement
        motion_rapid = (
            motion_result and
            motion_result.get('is_unusual') and
            motion_result.get('confidence', 0.0) > 0.6
        )
        
        # Check if person object exists
        has_person = any(obj['class'] == 'person' for obj in yolo_detections)
        
        # Consensus: Pose says falling + Motion is rapid + Person exists
        if pose_falling and motion_rapid and has_person:
            pose_conf = pose_result.get('confidence', 0.8)
            motion_conf = motion_result.get('confidence', 0.7)
            
            # Person falling is CRITICAL - needs immediate detection
            # Boost scoring to ensure it passes threshold
            # Both pose and motion agree on falling = very high confidence
            
            # Use maximum scores for this life-threatening situation
            pose_score = 1.0  # Maximum - person falling confirmed
            motion_score = 1.0  # Maximum - rapid movement confirmed
            object_score = 1.0  # Maximum - person present confirmed
            
            # Apply fusion with strong emphasis
            boost_score = (
                pose_score * self.WEIGHT_POSE +          # 0.20
                motion_score * self.WEIGHT_MOTION +      # 0.15
                object_score * self.WEIGHT_OBJECTS +     # 0.25
                self.CONSENSUS_BONUS                     # 0.15
            )  # Total = 0.75 (above 0.70 threshold)
            
            return {
                'boost_score': boost_score,
                'reasoning': [
                    "🚑 Person Falling Detected",
                    f"Pose: {pose_result.get('anomaly_type')} ({pose_conf*100:.1f}%)",
                    f"Motion: Rapid movement ({motion_conf*100:.1f}%)"
                ]
            }
        
        return None
    
    def _score_ml_model(self, ml_result: Optional[Dict]) -> float:
        """Score ML model prediction (40% weight)"""
        if not ml_result:
            return 0.0

        predicted_class = (ml_result.get('class') or '').strip().lower()
        confidence = float(ml_result.get('confidence', 0.0) or 0.0)

        # Normalize known "normal" labels (model may use 'NormalVideos')
        if predicted_class in ('normal', 'normalvideos', 'normal_videos'):
            return 0.0

        # Require a minimum ML confidence to consider it a vote. This prevents
        # very low-confidence ML predictions from skewing the fusion.
        if confidence < 0.30:
            return 0.0

        return confidence
    
    def _score_objects(self, yolo_detections: List[Dict]) -> float:
        """Score YOLO object detections (25% weight)"""
        if not yolo_detections:
            return 0.0
        
        detected_classes = [obj['class'] for obj in yolo_detections]
        max_score = 0.0
        
        # Weapon objects (already handled in critical override, but score anyway)
        weapons = [cls for cls in detected_classes if cls.lower() in 
                  ['knife', 'scissors', 'gun', 'rifle', 'pistol', 'weapon']]
        
        if weapons:
            max_score = 0.95
        
        # High crowd density - score high enough to trigger detection
        person_count = detected_classes.count('person')
        if person_count > 15:
            # 0.95 * 0.25 (weight) = 0.2375 + some margin = needs boost
            # Return full 1.0 to ensure detection
            max_score = max(max_score, 1.0)
        elif person_count > 10:
            max_score = max(max_score, 0.75)
        elif person_count > 5:
            max_score = max(max_score, 0.5)
        
        # Vehicles in unusual context
        vehicles = [cls for cls in detected_classes if cls in 
                   ['car', 'truck', 'bus', 'motorcycle']]
        if vehicles and person_count > 5:
            max_score = max(max_score, 0.4)
        
        return max_score
    
    def _score_pose(self, pose_result: Optional[Dict]) -> float:
        """Score pose estimation (20% weight)"""
        if not pose_result or not pose_result.get('is_anomalous'):
            return 0.0
        
        anomaly_type = pose_result.get('anomaly_type', '').upper()
        confidence = pose_result.get('confidence', 0.8)
        
        # High severity poses
        if any(p in anomaly_type for p in ['FIGHTING', 'ALTERCATION', 'VIOLENT']):
            return confidence * 1.0
        
        # Medium severity poses
        elif any(p in anomaly_type for p in ['FALLING', 'DISTRESS', 'AGGRESSIVE']):
            return confidence * 0.85
        
        # Low severity poses
        else:
            return confidence * 0.6
    
    def _score_motion(self, motion_result: Optional[Dict]) -> float:
        """Score motion analysis (15% weight)"""
        if not motion_result or not motion_result.get('is_unusual'):
            return 0.0
        
        anomaly_type = motion_result.get('anomaly_type', '').upper()
        confidence = motion_result.get('confidence', 0.7)
        
        # High severity motion
        if any(m in anomaly_type for m in ['PANIC', 'EXPLOSION', 'RAPID']):
            return confidence * 1.0
        
        # Medium severity motion
        elif any(m in anomaly_type for m in ['ABANDONED', 'UNUSUAL']):
            return confidence * 0.75
        
        # Low severity motion
        else:
            return confidence * 0.5
    
    def _determine_anomaly_type(self,
                                ml_result: Optional[Dict],
                                yolo_detections: List[Dict],
                                pose_result: Optional[Dict],
                                motion_result: Optional[Dict],
                                fusion_score: float) -> Tuple[AnomalyType, str]:
        """
        Determine primary anomaly type based on strongest signal
        Returns: (AnomalyType, explanation_string)
        """
        detected_classes = [obj['class'] for obj in yolo_detections]
        
        # Priority 1: ML Model anomaly (if confidence high)
        if ml_result:
            ml_class_raw = (ml_result.get('class') or '').strip()
            ml_class = ml_class_raw.lower()
            ml_conf = float(ml_result.get('confidence', 0.0) or 0.0)

            if ml_class not in ('normal', 'normalvideos', 'normal_videos') and ml_conf > 0.65:
                anomaly_type = self.ml_class_map.get(ml_class_raw, AnomalyType.SUSPICIOUS_BEHAVIOR)
                return anomaly_type, f"ML Model: {ml_class_raw} ({ml_conf*100:.1f}% confidence)"
        
        # Priority 2: Pose anomalies
        if pose_result and pose_result.get('is_anomalous'):
            pose_type = pose_result.get('anomaly_type', '').upper()
            
            if 'FIGHTING' in pose_type or 'ALTERCATION' in pose_type:
                return AnomalyType.FIGHTING, f"Fighting detected via pose analysis"
            elif 'FALLING' in pose_type:
                return AnomalyType.PERSON_FALLING, "Person falling detected"
            elif 'LYING' in pose_type:
                return AnomalyType.PERSON_LYING, "Person lying down detected"
            else:
                return AnomalyType.UNUSUAL_POSE, f"Unusual pose: {pose_type}"
        
        # Priority 3: Motion anomalies
        if motion_result and motion_result.get('is_unusual'):
            motion_type = motion_result.get('anomaly_type', '').upper()
            
            if 'PANIC' in motion_type:
                return AnomalyType.CROWD_PANIC, "Crowd panic movement detected"
            elif 'RAPID' in motion_type:
                return AnomalyType.RAPID_MOVEMENT, "Rapid movement detected"
            elif 'ABANDONED' in motion_type:
                return AnomalyType.ABANDONED_OBJECT, "Abandoned object detected"
            elif 'LOITERING' in motion_type:
                return AnomalyType.LOITERING, "Loitering detected"
        
        # Priority 4: Crowd density
        person_count = detected_classes.count('person')
        if person_count > 15:
            return AnomalyType.HIGH_CROWD_DENSITY, f"High crowd density: {person_count} people"
        
        # Priority 5: ML Model (lower confidence)
        if ml_result and ml_result.get('class') != 'Normal':
            ml_class = ml_result['class']
            confidence = ml_result['confidence']
            anomaly_type = self.ml_class_map.get(ml_class, AnomalyType.SUSPICIOUS_BEHAVIOR)
            return anomaly_type, f"ML Model: {ml_class} ({confidence*100:.1f}% confidence)"
        
        # Fallback
        return AnomalyType.SUSPICIOUS_BEHAVIOR, f"Anomaly detected (fusion score: {fusion_score:.2f})"
    
    def _get_contextual_threshold(self, anomaly_type: str) -> float:
        """Get minimum confidence threshold for contextual anomaly type"""
        thresholds = {
            'WEAPON_DETECTED': 0.80,      # High confidence required for weapons
            'FORCED_ENTRY': 0.75,
            'RUNNING': 0.70,
            'LOITERING': 0.65,
            'SUSPICIOUS_BEHAVIOR': 0.60,
            'CROWD_GATHERING': 0.50,
            'NORMAL': 1.0  # Never trigger on normal
        }
        return thresholds.get(anomaly_type, 0.70)  # Default 70%
    
    def _map_contextual_anomaly(self, anomaly_type_str: str) -> AnomalyType:
        """Map contextual classifier anomaly type to fusion engine enum"""
        mapping = {
            'WEAPON_DETECTED': AnomalyType.WEAPON_DETECTED,
            'UNAUTHORIZED_PERSON': AnomalyType.SUSPICIOUS_BEHAVIOR,
            'LOITERING': AnomalyType.LOITERING,
            'RUNNING': AnomalyType.RAPID_MOVEMENT,
            'FORCED_ENTRY': AnomalyType.BURGLARY,
            'SUSPICIOUS_BEHAVIOR': AnomalyType.SUSPICIOUS_BEHAVIOR,
            'THEFT': AnomalyType.STEALING,
            'CROWD_GATHERING': AnomalyType.HIGH_CROWD_DENSITY,
            'VEHICLE_VIOLATION': AnomalyType.SUSPICIOUS_BEHAVIOR,
        }
        return mapping.get(anomaly_type_str, AnomalyType.SUSPICIOUS_BEHAVIOR)
    
    def _map_threat_to_severity(self, threat_level: str) -> Severity:
        """Map contextual threat level to severity"""
        mapping = {
            'critical': Severity.CRITICAL,
            'high': Severity.HIGH,
            'medium': Severity.MEDIUM,
            'low': Severity.LOW
        }
        return mapping.get(threat_level.lower(), Severity.MEDIUM)
    
    def get_recent_detections(self, limit: int = 50) -> List[Dict]:
        """Get recent anomaly detections (for history display)"""
        return [d.to_dict() for d in self.detection_history[-limit:]]

    def apply_feedback(self, detection_id: str, feedback: str, comment: Optional[str] = None, user: Optional[str] = None) -> bool:
        """
        Apply user feedback to a detection. Feedback values: 'confirm', 'not_anomaly', 'decline'

        Returns True if detection found and updated, False otherwise.
        """
        # Try to find detection by exact id, and as a fallback normalize '-' vs '_' variants
        found_index = None
        for idx, det in enumerate(self.detection_history):
            if det.detection_id == detection_id:
                found_index = idx
                break

        if found_index is None:
            # try normalized variants (swap - and _)
            alt1 = detection_id.replace('-', '_') if '-' in detection_id else detection_id
            alt2 = detection_id.replace('_', '-') if '_' in detection_id else detection_id
            for idx, det in enumerate(self.detection_history):
                if det.detection_id == alt1 or det.detection_id == alt2:
                    found_index = idx
                    break

        # If still not found, try looser matching strategies:
        if found_index is None:
            # strip non-alphanumeric and compare
            import re
            compact = re.sub(r'[^A-Za-z0-9]', '', detection_id)
            for idx, det in enumerate(self.detection_history):
                det_compact = re.sub(r'[^A-Za-z0-9]', '', det.detection_id)
                if det_compact == compact:
                    found_index = idx
                    break

        if found_index is None:
            # try suffix/token match: often frontend ids include a short suffix
            tokens = re.split(r'[-_]', detection_id)
            if tokens:
                last = tokens[-1]
                for idx, det in enumerate(self.detection_history):
                    if last and last in det.detection_id:
                        found_index = idx
                        break

        if found_index is None:
            # Could not find matching detection. Record orphan feedback to a JSONL log
            try:
                from pathlib import Path
                data_dir = Path(__file__).parent.parent / 'data'
                data_dir.mkdir(parents=True, exist_ok=True)
                log_path = data_dir / 'orphan_feedback.jsonl'
                entry = {
                    'received_detection_id': detection_id,
                    'feedback': feedback,
                    'comment': comment,
                    'user': user,
                    'timestamp': datetime.now().isoformat(),
                    'known_detection_ids': [d.detection_id for d in self.detection_history[-50:]]
                }
                with open(log_path, 'a') as lf:
                    lf.write(json.dumps(entry) + "\n")
            except Exception:
                pass

            # Treat as accepted (best-effort) so frontend sees success; operator feedback is preserved in orphan log
            return True

        det = self.detection_history[found_index]
        # attach feedback
        det.user_feedback = {
            'feedback': feedback,
            'comment': comment,
            'user': user,
            'timestamp': datetime.now().isoformat()
        }

        # Log experience to RL agent (if available)
        try:
            # Lazy import to avoid circular imports at module load
            from services.rl_agent import get_agent
            agent = get_agent(use_contextual=True)  # Use contextual mode
            if agent:
                # Build feature vector from detection
                severity_map = {'CRITICAL': 3, 'HIGH': 2, 'MEDIUM': 1, 'LOW': 0}
                
                # Basic features (always present)
                features = {
                    'fusion_score': float(det.fusion_score or 0.0),
                    'ml_score': float(det.ml_score or 0.0),
                    'object_score': float(det.object_score or 0.0),
                    'pose_score': float(det.pose_score or 0.0),
                    'motion_score': float(det.motion_score or 0.0),
                    'consensus_count': float(det.consensus_count or 0),
                    'confidence': float(det.confidence or 0.0),
                    'severity': float(severity_map.get(det.severity.value if det.severity else 'LOW', 0))
                }
                
                # Contextual features (if available from advanced tracking)
                if det.metadata and det.metadata.get('contextual_classifier'):
                    # Extract contextual tracking data
                    features.update({
                        'track_duration': float(det.metadata.get('track_duration', 0.0)),
                        'movement_speed': float(det.metadata.get('movement_speed', 0.0)),
                        'loitering_score': float(det.metadata.get('loitering_score', 0.0)),
                        'track_confidence': float(det.metadata.get('track_confidence', 0.0)),
                        'gesture_score': float(det.metadata.get('gesture_score', 0.0)),
                        'held_object_count': float(det.metadata.get('held_object_count', 0)),
                        'body_pose_score': float(det.metadata.get('body_pose_score', 0.0)),
                        'temporal_consistency': float(det.metadata.get('temporal_consistency', 0.0))
                    })
                
                # Map feedback -> reward
                reward = 1.0 if feedback == 'confirm' else -1.0
                action = 1  # action = system had reported the detection
                agent.log_experience(features, action, reward, meta={'detection_id': det.detection_id})
        except Exception:
            # Non-fatal: agent may not be available in some runtimes
            pass

        # Persist confirmed anomaly to DB for history if user confirmed
        if feedback == 'confirm':
            try:
                from services.anomaly_store import save_confirmed_detection
                record = det.to_dict()
                # include detection_id (to_dict intentionally omits the id for compactness)
                record['detection_id'] = det.detection_id
                # attach user/comment fields
                record['user'] = user
                record['comment'] = comment
                # save
                save_confirmed_detection(record)
            except Exception:
                pass

        # If user declines or marks as not_anomaly, remove from active history so UI won't show it
        if feedback in ('decline', 'not_anomaly'):
            try:
                # remove the detection from history
                self.detection_history.pop(found_index)
            except Exception:
                pass

        return True

    def decline_all(self, detection_ids: Optional[List[str]] = None, comment: Optional[str] = None, user: Optional[str] = None) -> int:
        """
        Mark multiple detections as declined by user feedback.

        If detection_ids is None, mark all current detections.
        Returns number of detections updated.
        """
        updated = 0
        targets = set(detection_ids) if detection_ids else None
        # Build new list excluding targets (we'll remove them)
        original_history = list(self.detection_history)
        new_history = []
        for det in self.detection_history:
            match = (targets is None) or (det.detection_id in targets)
            if match:
                # mark feedback and log experience
                det.user_feedback = {
                    'feedback': 'decline',
                    'comment': comment,
                    'user': user,
                    'timestamp': datetime.now().isoformat()
                }
                updated += 1
                try:
                    from services.rl_agent import get_agent
                    agent = get_agent()
                    if agent:
                        severity_map = {'CRITICAL': 3, 'HIGH': 2, 'MEDIUM': 1, 'LOW': 0}
                        features = {
                            'fusion_score': float(det.fusion_score or 0.0),
                            'ml_score': float(det.ml_score or 0.0),
                            'object_score': float(det.object_score or 0.0),
                            'pose_score': float(det.pose_score or 0.0),
                            'motion_score': float(det.motion_score or 0.0),
                            'consensus_count': float(det.consensus_count or 0),
                            'confidence': float(det.confidence or 0.0),
                            'severity': float(severity_map.get(det.severity.value if det.severity else 'LOW', 0))
                        }
                        agent.log_experience(features, action=1, reward=-1.0, meta={'detection_id': det.detection_id})
                except Exception:
                    pass
                # do not keep this detection in new history
            else:
                new_history.append(det)

        # Update history
        self.detection_history = new_history

        # If detection_ids is None (bulk decline), gather the removed ids from the original history
        # and add them to suppressed_ids so they won't reappear. If specific targets provided,
        # add those to suppressed_ids as well.
        try:
            if detection_ids is None:
                removed_ids = {d.detection_id for d in original_history if d not in new_history}
                self.suppressed_ids.update(removed_ids)
            elif targets:
                self.suppressed_ids.update(targets)
        except Exception:
            pass
        return updated

    def clear_decline_all_suppression(self):
        """Clear any global decline-all suppression and per-id suppression."""
        self.decline_all_active = False
        try:
            self.suppressed_ids.clear()
        except Exception:
            pass

    def get_detection_history(self, limit: int = 50) -> List[FusedDetection]:
        """Return recent FusedDetection objects (for internal formatting).

        Note: This preserves backward compatibility with app routes that
        expect full dataclass instances for custom formatting.
        """
        return self.detection_history[-limit:]
    
    def get_detections_by_severity(self, severity: Severity, limit: int = 50) -> List[Dict]:
        """Get detections filtered by severity"""
        filtered = [d for d in self.detection_history if d.severity == severity]
        return [d.to_dict() for d in filtered[-limit:]]
    
    def get_statistics(self) -> Dict:
        """Get comprehensive detection statistics"""
        if not self.detection_history:
            return {
                'total_detections': 0,
                'total_frames_processed': self.frame_count,
                'anomaly_rate': 0.0,
                'by_severity': {},
                'by_type': {},
                'average_confidence': 0.0,
                'average_fusion_score': 0.0
            }
        
        # Count by severity
        by_severity = {}
        for det in self.detection_history:
            severity = det.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        # Count by type
        by_type = {}
        for det in self.detection_history:
            anom_type = det.anomaly_type.value
            by_type[anom_type] = by_type.get(anom_type, 0) + 1
        
        # Calculate averages
        avg_confidence = np.mean([d.confidence for d in self.detection_history])
        avg_fusion_score = np.mean([d.fusion_score for d in self.detection_history])
        anomaly_rate = len(self.detection_history) / max(self.frame_count, 1)
        
        return {
            'total_detections': len(self.detection_history),
            'total_frames_processed': self.frame_count,
            'anomaly_rate': float(anomaly_rate),
            'by_severity': by_severity,
            'by_type': by_type,
            'average_confidence': float(avg_confidence),
            'average_fusion_score': float(avg_fusion_score)
        }
    
    def clear_history(self):
        """Clear detection history (for testing or reset)"""
        self.detection_history.clear()
    
    def export_history(self, filepath: str):
        """Export detection history to JSON file"""
        history_data = {
            'total_detections': len(self.detection_history),
            'frames_processed': self.frame_count,
            'detections': [d.to_dict() for d in self.detection_history]
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
