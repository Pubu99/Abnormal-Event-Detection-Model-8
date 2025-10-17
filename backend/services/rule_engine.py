"""
Rule-Based Alert Engine
Context-aware alert generation based on multiple signals:
- Person + Weapon = RED ALERT
- Crowd Density threshold = YELLOW ALERT
- Loitering > threshold = BLUE ALERT
- Restricted zone violation = ORANGE ALERT

Author: AI Assistant
Date: 2025-10-17
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels"""
    CRITICAL = "CRITICAL"  # Red - Immediate threat
    HIGH = "HIGH"  # Orange - High risk
    MEDIUM = "MEDIUM"  # Yellow - Attention needed
    LOW = "LOW"  # Blue - Monitor
    INFO = "INFO"  # Green - Normal


@dataclass
class Alert:
    """Alert information"""
    level: AlertLevel
    title: str
    message: str
    timestamp: str
    confidence: float
    location: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    objects_involved: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class RuleResult:
    """Rule engine evaluation result"""
    alerts: List[Alert]
    threat_level: AlertLevel
    is_dangerous: bool
    summary: str
    timestamp: str


class RuleEngine:
    """Smart rule-based anomaly detection engine"""
    
    # Dangerous objects (from YOLO COCO classes)
    WEAPON_CLASSES = ['knife', 'scissors', 'gun', 'rifle']
    DANGEROUS_OBJECTS = ['fire', 'explosion', 'smoke']
    VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']
    
    def __init__(self,
                 crowd_density_threshold: int = 15,
                 loitering_time_seconds: float = 30.0,
                 restricted_zones: List[Tuple[int, int, int, int]] = None):
        """
        Initialize rule engine
        
        Args:
            crowd_density_threshold: Max people count before alert
            loitering_time_seconds: Seconds before loitering alert
            restricted_zones: List of (x, y, w, h) restricted areas
        """
        self.crowd_density_threshold = crowd_density_threshold
        self.loitering_time_seconds = loitering_time_seconds
        self.restricted_zones = restricted_zones or []
        
        # Tracking data
        self.person_positions = {}  # {person_id: (position, start_time)}
        self.alert_history = []
        
    def evaluate(self,
                 yolo_detections: List[Dict],
                 motion_result: Optional[Dict] = None,
                 pose_result: Optional[Dict] = None,
                 anomaly_class: Optional[str] = None,
                 anomaly_confidence: Optional[float] = None) -> RuleResult:
        """
        Evaluate all rules and generate alerts
        
        Args:
            yolo_detections: List of YOLO detected objects
            motion_result: Motion analysis result
            pose_result: Pose estimation result
            anomaly_class: Predicted anomaly class from ML model
            anomaly_confidence: ML model confidence
            
        Returns:
            RuleResult with all alerts
        """
        alerts = []
        
        # Extract detected objects
        detected_classes = [obj['class'] for obj in yolo_detections]
        people_count = detected_classes.count('person')
        
        # RULE 1: Person + Weapon = CRITICAL ALERT
        weapons_detected = [cls for cls in detected_classes if cls in self.WEAPON_CLASSES]
        if people_count > 0 and weapons_detected:
            for weapon in weapons_detected:
                weapon_obj = next(obj for obj in yolo_detections if obj['class'] == weapon)
                alerts.append(Alert(
                    level=AlertLevel.CRITICAL,
                    title="🚨 WEAPON DETECTED",
                    message=f"Person with {weapon.upper()} detected! Immediate action required.",
                    timestamp=datetime.now().isoformat(),
                    confidence=weapon_obj['confidence'],
                    location=weapon_obj['bbox'],
                    objects_involved=['person', weapon],
                    metadata={'weapon_type': weapon, 'people_nearby': people_count}
                ))
        
        # RULE 2: Crowd Density Alert
        if people_count > self.crowd_density_threshold:
            alerts.append(Alert(
                level=AlertLevel.MEDIUM,
                title="⚠️ HIGH CROWD DENSITY",
                message=f"{people_count} people detected. Monitoring for crowd control.",
                timestamp=datetime.now().isoformat(),
                confidence=0.95,
                objects_involved=['person'] * people_count,
                metadata={'count': people_count, 'threshold': self.crowd_density_threshold}
            ))
        
        # RULE 3: Dangerous Objects
        dangerous_objects = [cls for cls in detected_classes if cls in self.DANGEROUS_OBJECTS]
        if dangerous_objects:
            for obj_class in dangerous_objects:
                obj = next(o for o in yolo_detections if o['class'] == obj_class)
                alerts.append(Alert(
                    level=AlertLevel.CRITICAL,
                    title=f"🔥 {obj_class.upper()} DETECTED",
                    message=f"Dangerous condition: {obj_class} detected!",
                    timestamp=datetime.now().isoformat(),
                    confidence=obj['confidence'],
                    location=obj['bbox'],
                    objects_involved=[obj_class]
                ))
        
        # RULE 4: ML Model Anomaly + Objects = Context Alert
        if anomaly_class and anomaly_class != 'Normal' and anomaly_confidence > 0.7:
            severity_map = {
                'Shooting': AlertLevel.CRITICAL,
                'Explosion': AlertLevel.CRITICAL,
                'Robbery': AlertLevel.CRITICAL,
                'Assault': AlertLevel.HIGH,
                'Fighting': AlertLevel.HIGH,
                'Abuse': AlertLevel.HIGH,
                'Arson': AlertLevel.CRITICAL,
                'Burglary': AlertLevel.HIGH,
                'Vandalism': AlertLevel.MEDIUM,
                'Arrest': AlertLevel.INFO,
                'RoadAccidents': AlertLevel.HIGH,
                'Shoplifting': AlertLevel.MEDIUM,
                'Stealing': AlertLevel.HIGH
            }
            
            level = severity_map.get(anomaly_class, AlertLevel.MEDIUM)
            
            # Add context from YOLO
            context_objects = [obj['class'] for obj in yolo_detections[:5]]
            
            alerts.append(Alert(
                level=level,
                title=f"🎯 {anomaly_class.upper()} DETECTED",
                message=f"ML Model detected {anomaly_class} with {anomaly_confidence*100:.1f}% confidence. Objects: {', '.join(context_objects)}",
                timestamp=datetime.now().isoformat(),
                confidence=anomaly_confidence,
                objects_involved=context_objects,
                metadata={
                    'anomaly_type': anomaly_class,
                    'ml_confidence': anomaly_confidence,
                    'object_context': context_objects
                }
            ))
        
        # RULE 5: Motion Anomalies
        if motion_result and motion_result.get('is_unusual'):
            anomaly_type = motion_result.get('anomaly_type')
            confidence = motion_result.get('confidence', 0.0)
            
            severity_map = {
                'CROWD_PANIC': AlertLevel.HIGH,
                'RAPID_MOVEMENT': AlertLevel.MEDIUM,
                'ABANDONED_OBJECT': AlertLevel.HIGH,
                'LOITERING': AlertLevel.LOW,
                'UNUSUAL_MOTION_PATTERN': AlertLevel.MEDIUM
            }
            
            level = severity_map.get(anomaly_type, AlertLevel.LOW)
            
            alerts.append(Alert(
                level=level,
                title=f"🏃 {anomaly_type.replace('_', ' ')}",
                message=f"Unusual motion pattern detected: {anomaly_type.replace('_', ' ').lower()}",
                timestamp=datetime.now().isoformat(),
                confidence=confidence,
                metadata={'motion_type': anomaly_type}
            ))
        
        # RULE 6: Pose Anomalies
        if pose_result and pose_result.get('is_anomalous'):
            anomaly_type = pose_result.get('anomaly_type')
            confidence = pose_result.get('confidence', 0.0)
            
            severity_map = {
                'FIGHTING_DETECTED': AlertLevel.CRITICAL,
                'GROUP_ALTERCATION': AlertLevel.CRITICAL,
                'PERSON_FALLING': AlertLevel.HIGH,
                'DISTRESS_POSE': AlertLevel.HIGH,
                'SUSPICIOUS_POSE': AlertLevel.MEDIUM
            }
            
            level = severity_map.get(anomaly_type, AlertLevel.MEDIUM)
            
            alerts.append(Alert(
                level=level,
                title=f"🤺 {anomaly_type.replace('_', ' ')}",
                message=f"Abnormal pose detected: {anomaly_type.replace('_', ' ').lower()}",
                timestamp=datetime.now().isoformat(),
                confidence=confidence,
                metadata={'pose_type': anomaly_type}
            ))
        
        # RULE 7: Restricted Zone Violation
        for person_obj in [obj for obj in yolo_detections if obj['class'] == 'person']:
            for zone_idx, zone in enumerate(self.restricted_zones):
                if self._is_inside_zone(person_obj['bbox'], zone):
                    alerts.append(Alert(
                        level=AlertLevel.HIGH,
                        title="🚷 RESTRICTED ZONE VIOLATION",
                        message=f"Person detected in restricted zone {zone_idx + 1}",
                        timestamp=datetime.now().isoformat(),
                        confidence=person_obj['confidence'],
                        location=person_obj['bbox'],
                        objects_involved=['person'],
                        metadata={'zone_id': zone_idx, 'zone': zone}
                    ))
        
        # RULE 8: Vehicle in Pedestrian Area
        vehicles = [obj for obj in yolo_detections if obj['class'] in self.VEHICLE_CLASSES]
        if vehicles and people_count > 3:
            alerts.append(Alert(
                level=AlertLevel.MEDIUM,
                title="🚗 VEHICLE IN CROWD",
                message=f"Vehicle detected near {people_count} people. Monitor for safety.",
                timestamp=datetime.now().isoformat(),
                confidence=vehicles[0]['confidence'],
                location=vehicles[0]['bbox'],
                objects_involved=['vehicle', 'person'],
                metadata={'vehicle_type': vehicles[0]['class'], 'people_count': people_count}
            ))
        
        # Determine overall threat level
        if any(alert.level == AlertLevel.CRITICAL for alert in alerts):
            threat_level = AlertLevel.CRITICAL
            is_dangerous = True
        elif any(alert.level == AlertLevel.HIGH for alert in alerts):
            threat_level = AlertLevel.HIGH
            is_dangerous = True
        elif any(alert.level == AlertLevel.MEDIUM for alert in alerts):
            threat_level = AlertLevel.MEDIUM
            is_dangerous = False
        elif any(alert.level == AlertLevel.LOW for alert in alerts):
            threat_level = AlertLevel.LOW
            is_dangerous = False
        else:
            threat_level = AlertLevel.INFO
            is_dangerous = False
        
        # Generate summary
        if alerts:
            summary = f"{len(alerts)} alert(s): " + ", ".join([a.title for a in alerts[:3]])
        else:
            summary = "No threats detected. All systems normal."
        
        return RuleResult(
            alerts=alerts,
            threat_level=threat_level,
            is_dangerous=is_dangerous,
            summary=summary,
            timestamp=datetime.now().isoformat()
        )
    
    def _is_inside_zone(self, bbox: Tuple[int, int, int, int], 
                       zone: Tuple[int, int, int, int]) -> bool:
        """Check if bounding box center is inside zone"""
        x, y, w, h = bbox
        zx, zy, zw, zh = zone
        
        # Center of bbox
        cx, cy = x + w // 2, y + h // 2
        
        # Check if inside zone
        return (zx <= cx <= zx + zw) and (zy <= cy <= zy + zh)
    
    def get_alert_color(self, level: AlertLevel) -> Tuple[int, int, int]:
        """Get BGR color for alert level"""
        color_map = {
            AlertLevel.CRITICAL: (0, 0, 255),      # Red
            AlertLevel.HIGH: (0, 165, 255),        # Orange
            AlertLevel.MEDIUM: (0, 255, 255),      # Yellow
            AlertLevel.LOW: (255, 144, 30),        # Blue
            AlertLevel.INFO: (0, 255, 0)           # Green
        }
        return color_map.get(level, (255, 255, 255))
