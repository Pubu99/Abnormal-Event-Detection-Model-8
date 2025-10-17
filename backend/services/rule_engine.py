"""
Enhanced Rule-Based Alert Engine
Context-aware alert generation based on multiple signals

ACTIVE RULES (No Configuration Required):
1. Person + Weapon = RED ALERT
2. Multiple weapons detection = CRITICAL
3. Crowd Density threshold = YELLOW ALERT
4. Dangerous objects (fire, explosion) = CRITICAL
5. ML Model anomaly context = Severity-based
6. Motion anomalies = Context-based
7. Pose anomalies = Context-based
8. Loitering > threshold = BLUE ALERT
9. Abnormal crowd flow direction = MEDIUM ALERT
10. Sudden appearance/disappearance = LOW ALERT

FUTURE ENHANCEMENTS (Requires Zone Configuration):
- Running in restricted areas
- Large bags in restricted areas
- Vehicle speeding in speed zones
- Tripwire boundary crossing
- Restricted zone violations

To enable zone-based rules:
1. Configure zones via zone_manager.add_zone()
2. Uncomment zone-based rule calls in evaluate() method

Author: AI Assistant
Date: 2025-10-17
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from zone_manager import RestrictedZoneManager, ZoneType


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
    """Smart rule-based anomaly detection engine with spatial zones"""
    
    # Dangerous objects (from YOLO COCO classes)
    WEAPON_CLASSES = ['knife', 'scissors', 'gun', 'rifle', 'pistol', 'weapon']
    DANGEROUS_OBJECTS = ['fire', 'explosion', 'smoke', 'bomb']
    VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
    LARGE_OBJECT_CLASSES = ['backpack', 'suitcase', 'handbag', 'bag', 'luggage']
    
    def __init__(self,
                 crowd_density_threshold: int = 15,
                 loitering_time_seconds: float = 30.0,
                 running_speed_threshold: float = 15.0,  # px/frame
                 vehicle_speed_limit: float = 20.0,  # px/frame
                 large_object_size_threshold: int = 5000,  # pixels²
                 zone_manager: Optional[RestrictedZoneManager] = None):
        """
        Initialize enhanced rule engine
        
        Args:
            crowd_density_threshold: Max people count before alert
            loitering_time_seconds: Seconds before loitering alert
            running_speed_threshold: Speed threshold for running detection
            vehicle_speed_limit: Speed limit for vehicles
            large_object_size_threshold: Minimum size for large object alert
            zone_manager: Zone manager for spatial rules (optional)
        """
        self.crowd_density_threshold = crowd_density_threshold
        self.loitering_time_seconds = loitering_time_seconds
        self.running_speed_threshold = running_speed_threshold
        self.vehicle_speed_limit = vehicle_speed_limit
        self.large_object_size_threshold = large_object_size_threshold
        
        # Zone manager for spatial rules
        self.zone_manager = zone_manager or RestrictedZoneManager()
        
        # Tracking data
        self.person_positions = {}  # {person_id: (position, start_time)}
        self.object_history = {}  # {object_id: {'last_seen': time, 'positions': []}}
        self.tripwire_crossings = {}  # {tripwire_id: set of crossed object IDs}
        self.alert_history = []
        
        # Normal crowd flow direction (can be configured per camera)
        self.normal_flow_angle = 90  # degrees (0 = right, 90 = down)
        self.flow_tolerance = 45  # degrees tolerance
        
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
            for zone_id, zone in self.zone_manager.zones.items():
                if zone.zone_type in [ZoneType.RESTRICTED, ZoneType.HIGH_SECURITY]:
                    if self.zone_manager.is_point_in_zone(
                        self._get_bbox_center(person_obj['bbox']), zone_id):
                        alerts.append(Alert(
                            level=AlertLevel.HIGH,
                            title="🚷 RESTRICTED ZONE VIOLATION",
                            message=f"Person detected in restricted zone '{zone_id}'",
                            timestamp=datetime.now().isoformat(),
                            confidence=person_obj['confidence'],
                            location=person_obj['bbox'],
                            objects_involved=['person'],
                            metadata={'zone_id': zone_id, 'zone_type': zone.zone_type.value}
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
        
        # ========================================================================
        # NEW RULES - Enhanced Detection
        # ========================================================================
        
        # RULE 9: Multiple Weapons Detection
        multi_weapon_alert = self.check_multiple_weapons(detected_classes)
        if multi_weapon_alert:
            alerts.append(multi_weapon_alert)
        
        # For the following rules, we need tracked objects with speed/direction
        # Calculate speed from velocity if available
        tracked_objects = []
        for obj in yolo_detections:
            obj_copy = obj.copy()
            if 'velocity' in obj and not 'speed' in obj:
                vx, vy = obj['velocity']
                obj_copy['speed'] = np.sqrt(vx**2 + vy**2)
            tracked_objects.append(obj_copy)
        
        # ========================================================================
        # ZONE-BASED RULES (Future Enhancement - Requires Zone Configuration)
        # ========================================================================
        # These rules are disabled by default. To enable, configure zones via
        # zone_manager and uncomment the rule calls below.
        #
        # # RULE 10: Running in Restricted Zone
        # running_alert = self.check_running_in_restricted_zone(tracked_objects)
        # if running_alert:
        #     alerts.append(running_alert)
        #
        # # RULE 11: Large Objects in Restricted Zones
        # large_object_alert = self.check_large_object_in_zone(tracked_objects)
        # if large_object_alert:
        #     alerts.append(large_object_alert)
        #
        # # RULE 12: Vehicle Speeding (Zone-based)
        # speeding_alert = self.check_vehicle_speeding(tracked_objects)
        # if speeding_alert:
        #     alerts.append(speeding_alert)
        #
        # # RULE 13: Tripwire Crossing
        # tripwire_alert = self.check_tripwire_crossing(tracked_objects)
        # if tripwire_alert:
        #     alerts.append(tripwire_alert)
        
        # ========================================================================
        # ACTIVE RULES (No Configuration Required)
        # ========================================================================
        
        # RULE 10: Abnormal Crowd Flow Direction
        crowd_flow_alert = self.check_abnormal_crowd_flow(tracked_objects)
        if crowd_flow_alert:
            alerts.append(crowd_flow_alert)
        
        # RULE 11: Sudden Appearance/Disappearance
        appearance_alert = self.check_sudden_appearance_disappearance(tracked_objects)
        if appearance_alert:
            alerts.append(appearance_alert)
        
        # ========================================================================
        # END NEW RULES
        # ========================================================================
        
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
    
    # ============================================================================
    # NEW RULES - Spatial and Behavioral Detection
    # ============================================================================
    
    def check_multiple_weapons(self, objects: List[str]) -> Optional[Alert]:
        """
        RULE: Multiple Weapons Detected → CRITICAL
        Detect multiple dangerous weapons in close proximity
        """
        weapons = [obj for obj in objects if obj in self.WEAPON_CLASSES]
        
        if len(weapons) >= 2:
            return Alert(
                level=AlertLevel.CRITICAL,
                title="🚨 MULTIPLE WEAPONS DETECTED",
                message=f"{len(weapons)} weapons detected: {', '.join(set(weapons))}",
                timestamp=datetime.now().isoformat(),
                confidence=0.95,
                objects_involved=weapons,
                metadata={'weapon_count': len(weapons)}
            )
        return None
    
    def check_running_in_restricted_zone(self, 
                                        tracked_objects: List[Dict]) -> Optional[Alert]:
        """
        RULE: Person Running in Restricted Area → HIGH
        Detect rapid movement in sensitive zones (potential escape/threat)
        """
        if not self.zone_manager or len(self.zone_manager) == 0:
            return None
        
        for obj in tracked_objects:
            if obj.get('class') == 'person' and obj.get('speed', 0) > self.running_speed_threshold:
                # Check if in restricted zone
                bbox = obj.get('bbox')
                if bbox:
                    is_in_zone, zone_id = self.zone_manager.is_bbox_in_zone(
                        bbox, 
                        zone_type=ZoneType.RESTRICTED
                    )
                    
                    if is_in_zone:
                        return Alert(
                            level=AlertLevel.HIGH,
                            title="🏃 RUNNING IN RESTRICTED AREA",
                            message=f"Person running at {obj['speed']:.1f} px/f in {zone_id}",
                            timestamp=datetime.now().isoformat(),
                            confidence=0.85,
                            location=bbox,
                            metadata={'zone_id': zone_id, 'speed': obj['speed']}
                        )
        return None
    
    def check_large_object_in_zone(self, 
                                   yolo_detections: List[Dict]) -> Optional[Alert]:
        """
        RULE: Large Bags/Objects in Restricted Areas → MEDIUM
        Flag unauthorized large object transportation in sensitive zones
        """
        if not self.zone_manager or len(self.zone_manager) == 0:
            return None
        
        for obj in yolo_detections:
            obj_class = obj.get('class', '')
            
            if obj_class in self.LARGE_OBJECT_CLASSES:
                bbox = obj.get('bbox')
                if bbox:
                    # Check object size
                    x1, y1, x2, y2 = bbox
                    area = (x2 - x1) * (y2 - y1)
                    
                    if area > self.large_object_size_threshold:
                        # Check if in restricted zone
                        is_in_zone, zone_id = self.zone_manager.is_bbox_in_zone(
                            bbox,
                            zone_type=ZoneType.RESTRICTED
                        )
                        
                        if is_in_zone:
                            return Alert(
                                level=AlertLevel.MEDIUM,
                                title="📦 LARGE OBJECT IN RESTRICTED AREA",
                                message=f"{obj_class} ({area}px²) detected in {zone_id}",
                                timestamp=datetime.now().isoformat(),
                                confidence=obj.get('confidence', 0.8),
                                location=bbox,
                                metadata={'zone_id': zone_id, 'object_type': obj_class, 'size': area}
                            )
        return None
    
    def check_vehicle_speeding(self, 
                              tracked_objects: List[Dict]) -> Optional[Alert]:
        """
        RULE: Vehicle Speed Limit Violation → HIGH
        Detect fast vehicle movement in restricted or pedestrian areas
        """
        for obj in tracked_objects:
            obj_class = obj.get('class', '')
            
            if obj_class in self.VEHICLE_CLASSES:
                speed = obj.get('speed', 0)
                
                if speed > self.vehicle_speed_limit:
                    bbox = obj.get('bbox')
                    
                    # Extra HIGH alert if in restricted zone
                    if bbox and self.zone_manager and len(self.zone_manager) > 0:
                        is_in_zone, zone_id = self.zone_manager.is_bbox_in_zone(
                            bbox,
                            zone_type=ZoneType.SPEED_LIMIT
                        )
                        
                        if is_in_zone:
                            return Alert(
                                level=AlertLevel.HIGH,
                                title="🚗 VEHICLE SPEEDING IN RESTRICTED ZONE",
                                message=f"{obj_class} at {speed:.1f} px/f (limit: {self.vehicle_speed_limit}) in {zone_id}",
                                timestamp=datetime.now().isoformat(),
                                confidence=0.9,
                                location=bbox,
                                metadata={'zone_id': zone_id, 'speed': speed, 'limit': self.vehicle_speed_limit}
                            )
                    
                    return Alert(
                        level=AlertLevel.MEDIUM,
                        title="🚗 VEHICLE SPEEDING",
                        message=f"{obj_class} moving at {speed:.1f} px/f (limit: {self.vehicle_speed_limit})",
                        timestamp=datetime.now().isoformat(),
                        confidence=0.85,
                        location=bbox,
                        metadata={'speed': speed, 'limit': self.vehicle_speed_limit}
                    )
        return None
    
    def check_tripwire_crossing(self, 
                               tracked_objects: List[Dict]) -> Optional[Alert]:
        """
        RULE: Crossing Virtual Tripwire → HIGH
        Detect objects crossing predefined virtual boundaries
        """
        if not self.zone_manager or len(self.zone_manager) == 0:
            return None
        
        tripwires = self.zone_manager.get_zones_by_type(ZoneType.TRIPWIRE)
        if not tripwires:
            return None
        
        for obj in tracked_objects:
            track_id = obj.get('track_id')
            if not track_id:
                continue
            
            current_pos = obj.get('centroid')
            previous_pos = obj.get('previous_position')
            
            if not current_pos or not previous_pos:
                continue
            
            # Check each tripwire
            for zone_id in tripwires:
                # Check if trajectory crosses tripwire
                if self.zone_manager.check_trajectory_crosses_zone(
                    previous_pos, current_pos, zone_id
                ):
                    # Check if already recorded
                    if zone_id not in self.tripwire_crossings:
                        self.tripwire_crossings[zone_id] = set()
                    
                    if track_id not in self.tripwire_crossings[zone_id]:
                        self.tripwire_crossings[zone_id].add(track_id)
                        
                        return Alert(
                            level=AlertLevel.HIGH,
                            title="🚷 TRIPWIRE CROSSED",
                            message=f"{obj.get('class', 'object')} crossed virtual boundary: {zone_id}",
                            timestamp=datetime.now().isoformat(),
                            confidence=0.9,
                            location=obj.get('bbox'),
                            metadata={'zone_id': zone_id, 'track_id': track_id}
                        )
        return None
    
    def check_abnormal_crowd_flow(self, 
                                  tracked_objects: List[Dict]) -> Optional[Alert]:
        """
        RULE: Abnormal Crowd Movement Direction → MEDIUM
        Detect crowd moving against usual flow patterns (panic, conflict)
        """
        # Need at least 5 people for crowd analysis
        people = [obj for obj in tracked_objects if obj.get('class') == 'person']
        
        if len(people) < 5:
            return None
        
        # Calculate average movement direction
        angles = []
        for person in people:
            direction = person.get('direction')
            if direction is not None:
                angles.append(direction)
        
        if not angles:
            return None
        
        avg_angle = np.mean(angles)
        angle_std = np.std(angles)
        
        # Calculate difference from normal flow
        angle_diff = abs(avg_angle - self.normal_flow_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # Alert if crowd moving opposite direction (dispersal/panic)
        if angle_diff > (180 - self.flow_tolerance):
            return Alert(
                level=AlertLevel.MEDIUM,
                title="⚠️ ABNORMAL CROWD FLOW",
                message=f"Crowd moving at {avg_angle:.0f}° (normal: {self.normal_flow_angle}°) - possible dispersal or panic",
                timestamp=datetime.now().isoformat(),
                confidence=0.75,
                metadata={
                    'crowd_size': len(people),
                    'avg_direction': avg_angle,
                    'normal_direction': self.normal_flow_angle,
                    'angle_deviation': angle_diff
                }
            )
        
        # Alert if high variance (chaotic movement)
        if angle_std > 60:  # High variance in directions
            return Alert(
                level=AlertLevel.MEDIUM,
                title="⚠️ CHAOTIC CROWD MOVEMENT",
                message=f"Crowd movement is disorganized (std: {angle_std:.1f}°) - possible conflict or confusion",
                timestamp=datetime.now().isoformat(),
                confidence=0.7,
                metadata={
                    'crowd_size': len(people),
                    'direction_variance': angle_std
                }
            )
        
        return None
    
    def check_sudden_appearance_disappearance(self,
                                             tracked_objects: List[Dict]) -> Optional[Alert]:
        """
        RULE: Sudden Appearance/Disappearance → VARIABLE
        Detect abrupt changes in scene composition (potential suspicious activity)
        """
        current_time = datetime.now()
        current_ids = {obj.get('track_id') for obj in tracked_objects if obj.get('track_id')}
        
        # Update object history
        new_objects = []
        for obj in tracked_objects:
            track_id = obj.get('track_id')
            if track_id:
                if track_id not in self.object_history:
                    # New object appeared
                    self.object_history[track_id] = {
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'class': obj.get('class'),
                        'bbox': obj.get('bbox')
                    }
                    new_objects.append(obj)
                else:
                    # Update last seen
                    self.object_history[track_id]['last_seen'] = current_time
        
        # Check for sudden appearance of multiple objects (except people)
        non_person_new = [obj for obj in new_objects 
                         if obj.get('class') not in ['person']]
        
        if len(non_person_new) >= 2:
            return Alert(
                level=AlertLevel.LOW,
                title="👀 SUDDEN OBJECT APPEARANCE",
                message=f"{len(non_person_new)} objects suddenly appeared in frame",
                timestamp=current_time.isoformat(),
                confidence=0.6,
                objects_involved=[obj.get('class') for obj in non_person_new],
                metadata={'new_objects': len(non_person_new)}
            )
        
        # Clean up old entries (not seen for >10 seconds)
        cutoff_time = current_time - timedelta(seconds=10)
        disappeared = []
        
        for track_id in list(self.object_history.keys()):
            if self.object_history[track_id]['last_seen'] < cutoff_time:
                disappeared.append(self.object_history[track_id]['class'])
                del self.object_history[track_id]
        
        return None
    
    # ============================================================================
    # END NEW RULES
    # ============================================================================
    
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
    
    def _get_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get center point of bounding box"""
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)
