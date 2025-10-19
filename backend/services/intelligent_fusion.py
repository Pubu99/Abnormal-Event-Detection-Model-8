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
    WEIGHT_ML = 0.40        # Domain-specific training
    WEIGHT_OBJECTS = 0.25   # Most reliable pre-training
    WEIGHT_POSE = 0.20      # Robust human detection
    WEIGHT_MOTION = 0.15    # Supporting evidence
    
    # Detection threshold
    ANOMALY_THRESHOLD = 0.70  # Report only if score >= 0.70
    
    # Consensus bonus (when multiple modalities agree)
    CONSENSUS_BONUS = 0.15  # Add when 2+ modalities detect same anomaly
    
    def __init__(self):
        """Initialize fusion engine"""
        self.detection_history: List[FusedDetection] = []
        self.frame_count = 0
        self.detection_counter = 0  # For unique IDs
        
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
                       frame_number: int) -> Optional[FusedDetection]:
        """
        Main fusion pipeline - combines all detection modalities
        
        Args:
            ml_result: {'class': 'Shooting', 'confidence': 0.95, 'probabilities': [...]}
            yolo_detections: [{'class': 'person', 'bbox': (x,y,w,h), 'confidence': 0.9}]
            pose_result: {'is_anomalous': True, 'anomaly_type': 'PERSON_FALLING', 'confidence': 0.92}
            motion_result: {'is_unusual': True, 'anomaly_type': 'RAPID_MOVEMENT', 'confidence': 0.85}
            frame_number: Current frame number
            
        Returns:
            FusedDetection if anomaly detected (score >= 0.70), None if normal
        """
        self.frame_count += 1
        reasoning = []
        
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
        active_modalities = sum([
            1 if ml_score > 0.5 else 0,
            1 if object_score > 0.3 else 0,
            1 if pose_score > 0.5 else 0,
            1 if motion_score > 0.5 else 0
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
        
        predicted_class = ml_result.get('class', 'Normal')
        confidence = ml_result.get('confidence', 0.0)
        
        # Normal = no anomaly
        if predicted_class == 'Normal':
            return 0.0
        
        # Anomaly detected by ML model
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
        if ml_result and ml_result.get('class') != 'Normal':
            ml_class = ml_result['class']
            confidence = ml_result['confidence']
            
            if confidence > 0.7:
                anomaly_type = self.ml_class_map.get(ml_class, AnomalyType.SUSPICIOUS_BEHAVIOR)
                return anomaly_type, f"ML Model: {ml_class} ({confidence*100:.1f}% confidence)"
        
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
    
    def get_recent_detections(self, limit: int = 50) -> List[Dict]:
        """Get recent anomaly detections (for history display)"""
        return [d.to_dict() for d in self.detection_history[-limit:]]

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
