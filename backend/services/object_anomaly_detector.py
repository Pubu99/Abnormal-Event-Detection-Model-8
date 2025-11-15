"""
Comprehensive Object Anomaly Detection System
Detects suspicious objects and scenarios WITHOUT specialized models

DETECTION CATEGORIES:
1. Weapons (guns, knives via pose analysis - NO specialized model)
2. Abandoned objects (unattended luggage, bags, packages)
3. Unauthorized vehicles in restricted zones
4. Suspicious packages and containers
5. Crowd anomalies (clustering, dispersal, stampedes)
6. Hazardous materials (misplaced fire extinguishers, chemicals)
7. Missing objects (shoplifting, theft)
8. Unusual personal belongings (oversized bags, hidden objects)
9. Stationary suspicious objects in dynamic areas

Uses: YOLO detections + pose analysis + temporal tracking + zone analytics

Author: Professional Implementation
Date: 2025-11-13
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum


class ObjectAnomalyType(Enum):
    """Types of object-based anomalies"""
    # Weapons (pose-based detection)
    WEAPON_HANDGUN = "weapon_handgun"
    WEAPON_RIFLE = "weapon_rifle"
    WEAPON_KNIFE = "weapon_knife"
    WEAPON_UNKNOWN = "weapon_unknown"
    # Abandoned objects
    ABANDONED_LUGGAGE = "abandoned_luggage"
    ABANDONED_BAG = "abandoned_bag"
    ABANDONED_PACKAGE = "abandoned_package"
    # Vehicles
    UNAUTHORIZED_VEHICLE = "unauthorized_vehicle"
    # Suspicious items
    SUSPICIOUS_PACKAGE = "suspicious_package"
    # Crowds
    CROWD_CLUSTERING = "crowd_clustering"
    CROWD_DISPERSAL = "crowd_dispersal"
    CROWD_STAMPEDE = "crowd_stampede"
    # Hazards
    HAZARDOUS_MATERIAL = "hazardous_material"
    # Theft
    MISSING_OBJECT = "missing_object"
    # Belongings
    UNUSUAL_BELONGING = "unusual_belonging"
    # Stationary
    STATIONARY_SUSPICIOUS = "stationary_suspicious"


@dataclass
class TrackedObjectInfo:
    """Extended tracking info for anomaly detection"""
    object_id: int
    class_name: str
    first_seen: datetime
    last_seen: datetime
    positions: deque  # Recent positions
    stationary_duration: float
    associated_person_id: Optional[int] = None
    is_abandoned: bool = False
    is_suspicious: bool = False


@dataclass
class ObjectAnomalyDetection:
    """Object anomaly detection result"""
    anomaly_type: ObjectAnomalyType
    confidence: float
    object_class: str
    bbox: Tuple[int, int, int, int]
    duration: float  # seconds
    details: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW


class ObjectAnomalyDetector:
    """
    Professional unified object + weapon anomaly detection
    Combines temporal tracking, pose analysis, and context awareness
    """
    
    # Object categories
    LUGGAGE_CLASSES = ['suitcase', 'backpack', 'handbag', 'luggage', 'bag']
    VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'motorbike']
    CONTAINER_CLASSES = ['suitcase', 'backpack', 'handbag', 'box', 'bottle', 'cup', 'bowl']
    HAZARD_CLASSES = ['fire extinguisher', 'bottle', 'cup']  # Context-dependent
    VALUABLE_CLASSES = ['laptop', 'cell phone', 'handbag', 'backpack', 'suitcase']
    
    # COCO keypoint indices for weapon detection
    KEYPOINT_INDICES = {
        'left_shoulder': 5, 'right_shoulder': 6,
        'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10,
        'left_hip': 11, 'right_hip': 12
    }
    
    def __init__(self,
                 abandoned_threshold: float = 15.0,  # seconds
                 stationary_threshold: float = 2.0,  # pixels/sec movement
                 crowd_density_threshold: int = 15,  # people count
                 crowd_cluster_distance: float = 100.0,  # pixels
                 fps: float = 30.0,
                 enable_weapon_detection: bool = True):
        """
        Initialize unified object + weapon anomaly detector
        
        Args:
            abandoned_threshold: Seconds before object considered abandoned
            stationary_threshold: Movement threshold for stationary detection
            crowd_density_threshold: People count for crowd anomaly
            crowd_cluster_distance: Distance for crowd clustering
            fps: Frames per second
            enable_weapon_detection: Enable pose-based weapon detection
        """
        self.abandoned_threshold = abandoned_threshold
        self.stationary_threshold = stationary_threshold
        self.crowd_density_threshold = crowd_density_threshold
        self.crowd_cluster_distance = crowd_cluster_distance
        self.fps = fps
        self.enable_weapon_detection = enable_weapon_detection
        
        # Tracking storage
        self.tracked_objects: Dict[int, TrackedObjectInfo] = {}
        self.person_object_associations: Dict[int, Set[int]] = defaultdict(set)
        
        # Historical data for missing object detection
        self.expected_objects: Dict[str, List] = defaultdict(list)  # zone -> objects
        self.previous_frame_objects: Set[str] = set()
        
        # Crowd history for anomaly detection
        self.crowd_history: deque = deque(maxlen=60)  # Last 2 seconds at 30fps
        
        # Zone definitions (can be configured externally)
        self.restricted_zones: List[Tuple[int, int, int, int]] = []  # [(x, y, w, h), ...]
        self.dynamic_zones: List[Tuple[int, int, int, int]] = []  # Areas with expected movement
        
    def detect_anomalies(self,
                        frame: np.ndarray,
                        yolo_detections: List[Dict],
                        tracked_objects: List[Dict],
                        frame_number: int,
                        pose_keypoints: Optional[List[np.ndarray]] = None,
                        pose_bboxes: Optional[List[Tuple[int, int, int, int]]] = None) -> List[ObjectAnomalyDetection]:
        """
        Detect all object-based anomalies including weapons
        
        Args:
            frame: Current frame
            yolo_detections: YOLO detections with class, bbox, confidence
            tracked_objects: Tracked objects with IDs
            frame_number: Current frame number
            pose_keypoints: Optional pose keypoints for weapon detection
            pose_bboxes: Optional person bounding boxes for weapon detection
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        current_time = datetime.now()
        
        # Update tracking information
        self._update_tracking(yolo_detections, tracked_objects, current_time)
        
        # 0. 🔫 WEAPON DETECTION (HIGHEST PRIORITY - pose-based, no specialized model)
        if self.enable_weapon_detection and pose_keypoints and pose_bboxes:
            weapons = self._detect_weapons_pose_based(pose_keypoints, pose_bboxes, yolo_detections, frame)
            anomalies.extend(weapons)
        
        # 1. ABANDONED OBJECT DETECTION
        abandoned = self._detect_abandoned_objects(current_time)
        anomalies.extend(abandoned)
        
        # 2. UNAUTHORIZED VEHICLE DETECTION
        vehicles = self._detect_unauthorized_vehicles(yolo_detections)
        anomalies.extend(vehicles)
        
        # 3. SUSPICIOUS PACKAGE DETECTION
        packages = self._detect_suspicious_packages(yolo_detections, current_time)
        anomalies.extend(packages)
        
        # 4. CROWD ANOMALIES
        crowd = self._detect_crowd_anomalies(yolo_detections, frame.shape)
        anomalies.extend(crowd)
        
        # 5. HAZARDOUS MATERIAL DETECTION
        hazards = self._detect_hazardous_materials(yolo_detections)
        anomalies.extend(hazards)
        
        # 6. MISSING OBJECT DETECTION
        missing = self._detect_missing_objects(yolo_detections)
        anomalies.extend(missing)
        
        # 7. UNUSUAL PERSONAL BELONGINGS
        unusual = self._detect_unusual_belongings(yolo_detections, frame.shape)
        anomalies.extend(unusual)
        
        # 8. STATIONARY SUSPICIOUS OBJECTS
        stationary = self._detect_stationary_suspicious(frame.shape)
        anomalies.extend(stationary)
        
        # Update history
        self._update_history(yolo_detections)
        
        return anomalies
    
    def _update_tracking(self,
                        yolo_detections: List[Dict],
                        tracked_objects: List[Dict],
                        current_time: datetime):
        """Update object tracking with temporal information"""
        current_ids = set()
        
        for obj in tracked_objects:
            obj_id = obj.get('track_id')
            if obj_id is None:
                continue
            
            current_ids.add(obj_id)
            class_name = obj.get('class_name', obj.get('class', ''))
            bbox = obj.get('bbox', (0, 0, 0, 0))
            
            if obj_id in self.tracked_objects:
                # Update existing
                tracked = self.tracked_objects[obj_id]
                tracked.last_seen = current_time
                tracked.positions.append(bbox)
                
                # Calculate movement
                if len(tracked.positions) >= 2:
                    prev_bbox = tracked.positions[-2]
                    curr_bbox = tracked.positions[-1]
                    
                    # Center movement
                    prev_center = (prev_bbox[0] + prev_bbox[2]/2, prev_bbox[1] + prev_bbox[3]/2)
                    curr_center = (curr_bbox[0] + curr_bbox[2]/2, curr_bbox[1] + curr_bbox[3]/2)
                    movement = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                     (curr_center[1] - prev_center[1])**2)
                    
                    # Update stationary duration
                    if movement < self.stationary_threshold:
                        duration = (current_time - tracked.first_seen).total_seconds()
                        tracked.stationary_duration = duration
                    else:
                        tracked.stationary_duration = 0.0
            else:
                # New object
                self.tracked_objects[obj_id] = TrackedObjectInfo(
                    object_id=obj_id,
                    class_name=class_name,
                    first_seen=current_time,
                    last_seen=current_time,
                    positions=deque([bbox], maxlen=30),
                    stationary_duration=0.0
                )
        
        # Remove objects not seen recently (disappeared)
        disappeared_ids = [oid for oid, obj in self.tracked_objects.items()
                          if (current_time - obj.last_seen).total_seconds() > 2.0]
        for oid in disappeared_ids:
            del self.tracked_objects[oid]
    
    def _detect_abandoned_objects(self, current_time: datetime) -> List[ObjectAnomalyDetection]:
        """Detect abandoned luggage, bags, packages"""
        anomalies = []
        
        for obj_id, tracked in self.tracked_objects.items():
            # Check if object is luggage/bag type
            if tracked.class_name not in self.LUGGAGE_CLASSES + self.CONTAINER_CLASSES:
                continue
            
            # Check if stationary for threshold duration
            duration = (current_time - tracked.first_seen).total_seconds()
            if duration < self.abandoned_threshold:
                continue
            
            # Check if person nearby (not abandoned if owner present)
            has_nearby_person = self._has_nearby_person(tracked, current_time)
            if has_nearby_person:
                continue
            
            # ABANDONED OBJECT DETECTED
            if tracked.class_name in ['suitcase', 'luggage']:
                anomaly_type = ObjectAnomalyType.ABANDONED_LUGGAGE
                severity = "HIGH"
            elif tracked.class_name in ['backpack', 'handbag', 'bag']:
                anomaly_type = ObjectAnomalyType.ABANDONED_BAG
                severity = "MEDIUM"
            else:
                anomaly_type = ObjectAnomalyType.ABANDONED_PACKAGE
                severity = "MEDIUM"
            
            bbox = tracked.positions[-1] if tracked.positions else (0, 0, 0, 0)
            anomalies.append(ObjectAnomalyDetection(
                anomaly_type=anomaly_type,
                confidence=min(0.75 + (duration / self.abandoned_threshold) * 0.2, 0.95),
                object_class=tracked.class_name,
                bbox=bbox,
                duration=duration,
                details=f"Abandoned {tracked.class_name} detected ({duration:.1f}s unattended)",
                severity=severity
            ))
            
            tracked.is_abandoned = True
        
        return anomalies
    
    def _has_nearby_person(self, obj: TrackedObjectInfo, current_time: datetime, radius: float = 150.0) -> bool:
        """Check if person is near the object"""
        if not obj.positions:
            return False
        
        obj_bbox = obj.positions[-1]
        obj_center = (obj_bbox[0] + obj_bbox[2]/2, obj_bbox[1] + obj_bbox[3]/2)
        
        # Check all tracked persons
        for person_id, person in self.tracked_objects.items():
            if person.class_name != 'person':
                continue
            
            if (current_time - person.last_seen).total_seconds() > 1.0:
                continue
            
            if not person.positions:
                continue
            
            person_bbox = person.positions[-1]
            person_center = (person_bbox[0] + person_bbox[2]/2, person_bbox[1] + person_bbox[3]/2)
            
            distance = np.sqrt((obj_center[0] - person_center[0])**2 + 
                             (obj_center[1] - person_center[1])**2)
            
            if distance < radius:
                return True
        
        return False
    
    def _detect_unauthorized_vehicles(self, yolo_detections: List[Dict]) -> List[ObjectAnomalyDetection]:
        """Detect vehicles in restricted zones"""
        anomalies = []
        
        if not self.restricted_zones:
            return anomalies
        
        for obj in yolo_detections:
            if obj.get('class', '') not in self.VEHICLE_CLASSES:
                continue
            
            bbox = obj.get('bbox', (0, 0, 0, 0))
            obj_center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
            
            # Check if in restricted zone
            for zone in self.restricted_zones:
                zx, zy, zw, zh = zone
                if (zx <= obj_center[0] <= zx + zw and 
                    zy <= obj_center[1] <= zy + zh):
                    
                    anomalies.append(ObjectAnomalyDetection(
                        anomaly_type=ObjectAnomalyType.UNAUTHORIZED_VEHICLE,
                        confidence=0.85,
                        object_class=obj.get('class', 'vehicle'),
                        bbox=bbox,
                        duration=0.0,
                        details=f"Unauthorized {obj.get('class')} in restricted zone",
                        severity="HIGH"
                    ))
                    break
        
        return anomalies
    
    def _detect_suspicious_packages(self, yolo_detections: List[Dict], current_time: datetime) -> List[ObjectAnomalyDetection]:
        """Detect suspicious packages based on context"""
        anomalies = []
        
        for obj in yolo_detections:
            class_name = obj.get('class', '')
            if class_name not in self.CONTAINER_CLASSES:
                continue
            
            bbox = obj.get('bbox', (0, 0, 0, 0))
            confidence = obj.get('confidence', 0.0)
            
            # Suspicious if:
            # 1. Box/container in unusual location
            # 2. Not being carried by person
            # 3. Unusual size/shape
            
            is_suspicious = False
            details = ""
            
            # Check if carried by person
            has_person_nearby = False
            obj_center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
            
            for tracked in self.tracked_objects.values():
                if tracked.class_name != 'person':
                    continue
                if not tracked.positions:
                    continue
                
                person_bbox = tracked.positions[-1]
                person_center = (person_bbox[0] + person_bbox[2]/2, person_bbox[1] + person_bbox[3]/2)
                distance = np.sqrt((obj_center[0] - person_center[0])**2 + 
                                 (obj_center[1] - person_center[1])**2)
                
                if distance < 80:  # Close proximity
                    has_person_nearby = True
                    break
            
            # Unattended container is suspicious
            if not has_person_nearby and class_name in ['box', 'suitcase']:
                is_suspicious = True
                details = f"Unattended {class_name} detected"
            
            # Unusual size (very large backpack/bag)
            area = bbox[2] * bbox[3]
            if class_name in ['backpack', 'handbag'] and area > 20000:  # Large for the class
                is_suspicious = True
                details = f"Unusually large {class_name} detected"
            
            if is_suspicious:
                anomalies.append(ObjectAnomalyDetection(
                    anomaly_type=ObjectAnomalyType.SUSPICIOUS_PACKAGE,
                    confidence=0.70,
                    object_class=class_name,
                    bbox=bbox,
                    duration=0.0,
                    details=details,
                    severity="MEDIUM"
                ))
        
        return anomalies
    
    def _detect_crowd_anomalies(self, yolo_detections: List[Dict], frame_shape: Tuple) -> List[ObjectAnomalyDetection]:
        """Detect crowd clustering, dispersal, stampedes"""
        anomalies = []
        
        # Count people
        people = [obj for obj in yolo_detections if obj.get('class') == 'person']
        person_count = len(people)
        
        # Store in history
        self.crowd_history.append({
            'count': person_count,
            'positions': [obj.get('bbox', (0, 0, 0, 0)) for obj in people]
        })
        
        # Need history for temporal analysis
        if len(self.crowd_history) < 30:  # 1 second
            return anomalies
        
        # CROWD CLUSTERING: Sudden increase in people count
        prev_count = self.crowd_history[-30]['count']
        if person_count > self.crowd_density_threshold and person_count > prev_count * 1.5:
            # Check if people are clustered (not spread out)
            if self._is_crowd_clustered(people):
                anomalies.append(ObjectAnomalyDetection(
                    anomaly_type=ObjectAnomalyType.CROWD_CLUSTERING,
                    confidence=0.80,
                    object_class='crowd',
                    bbox=(0, 0, frame_shape[1], frame_shape[0]),
                    duration=0.0,
                    details=f"Crowd clustering detected ({person_count} people)",
                    severity="MEDIUM"
                ))
        
        # CROWD DISPERSAL: Sudden decrease in people count
        if prev_count > self.crowd_density_threshold and person_count < prev_count * 0.6:
            anomalies.append(ObjectAnomalyDetection(
                anomaly_type=ObjectAnomalyType.CROWD_DISPERSAL,
                confidence=0.75,
                object_class='crowd',
                bbox=(0, 0, frame_shape[1], frame_shape[0]),
                duration=0.0,
                details=f"Sudden crowd dispersal ({prev_count} -> {person_count} people)",
                severity="HIGH"
            ))
        
        # STAMPEDE: High density + rapid movement
        if person_count > self.crowd_density_threshold:
            avg_movement = self._calculate_crowd_movement()
            if avg_movement > 20:  # pixels/frame
                anomalies.append(ObjectAnomalyDetection(
                    anomaly_type=ObjectAnomalyType.CROWD_STAMPEDE,
                    confidence=0.85,
                    object_class='crowd',
                    bbox=(0, 0, frame_shape[1], frame_shape[0]),
                    duration=0.0,
                    details=f"Crowd stampede detected (movement: {avg_movement:.1f}px/f)",
                    severity="CRITICAL"
                ))
        
        return anomalies
    
    def _is_crowd_clustered(self, people: List[Dict]) -> bool:
        """Check if people are clustered together"""
        if len(people) < 3:
            return False
        
        # Calculate pairwise distances
        centers = []
        for p in people:
            bbox = p.get('bbox', (0, 0, 0, 0))
            centers.append((bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2))
        
        # Check if most people are within cluster distance
        close_pairs = 0
        total_pairs = 0
        
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                distance = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                                 (centers[i][1] - centers[j][1])**2)
                total_pairs += 1
                if distance < self.crowd_cluster_distance:
                    close_pairs += 1
        
        # Clustered if >60% of pairs are close
        return (close_pairs / max(total_pairs, 1)) > 0.6
    
    def _calculate_crowd_movement(self) -> float:
        """Calculate average crowd movement speed"""
        if len(self.crowd_history) < 2:
            return 0.0
        
        prev_positions = self.crowd_history[-2]['positions']
        curr_positions = self.crowd_history[-1]['positions']
        
        # Match positions (simple nearest neighbor)
        movements = []
        for curr_bbox in curr_positions:
            curr_center = (curr_bbox[0] + curr_bbox[2]/2, curr_bbox[1] + curr_bbox[3]/2)
            
            min_distance = float('inf')
            for prev_bbox in prev_positions:
                prev_center = (prev_bbox[0] + prev_bbox[2]/2, prev_bbox[1] + prev_bbox[3]/2)
                distance = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                 (curr_center[1] - prev_center[1])**2)
                if distance < min_distance:
                    min_distance = distance
            
            if min_distance < 200:  # Reasonable matching threshold
                movements.append(min_distance)
        
        return np.mean(movements) if movements else 0.0
    
    def _detect_hazardous_materials(self, yolo_detections: List[Dict]) -> List[ObjectAnomalyDetection]:
        """Detect misplaced hazardous materials"""
        anomalies = []
        
        for obj in yolo_detections:
            class_name = obj.get('class', '')
            
            # Fire extinguisher in unusual location
            if 'fire' in class_name.lower() or 'extinguisher' in class_name.lower():
                bbox = obj.get('bbox', (0, 0, 0, 0))
                
                # If not mounted (detected on floor/ground level)
                frame_height_estimate = 720  # Approximate
                if bbox[1] + bbox[3] > frame_height_estimate * 0.7:  # Lower half of frame
                    anomalies.append(ObjectAnomalyDetection(
                        anomaly_type=ObjectAnomalyType.HAZARDOUS_MATERIAL,
                        confidence=0.70,
                        object_class=class_name,
                        bbox=bbox,
                        duration=0.0,
                        details="Fire extinguisher in unusual location",
                        severity="MEDIUM"
                    ))
        
        return anomalies
    
    def _detect_missing_objects(self, yolo_detections: List[Dict]) -> List[ObjectAnomalyDetection]:
        """Detect suddenly missing valuable objects (shoplifting/theft)"""
        anomalies = []
        
        # Build current frame object set
        current_objects = set()
        for obj in yolo_detections:
            class_name = obj.get('class', '')
            if class_name in self.VALUABLE_CLASSES:
                bbox = obj.get('bbox', (0, 0, 0, 0))
                # Use position as identifier (rough)
                obj_id = f"{class_name}_{int(bbox[0]/50)}{int(bbox[1]/50)}"
                current_objects.add(obj_id)
        
        # Check for missing objects
        missing = self.previous_frame_objects - current_objects
        
        for missing_id in missing:
            class_name = missing_id.split('_')[0]
            anomalies.append(ObjectAnomalyDetection(
                anomaly_type=ObjectAnomalyType.MISSING_OBJECT,
                confidence=0.60,
                object_class=class_name,
                bbox=(0, 0, 0, 0),
                duration=0.0,
                details=f"Potential theft: {class_name} disappeared",
                severity="LOW"
            ))
        
        self.previous_frame_objects = current_objects
        return anomalies
    
    def _detect_unusual_belongings(self, yolo_detections: List[Dict], frame_shape: Tuple) -> List[ObjectAnomalyDetection]:
        """Detect unusually large or contextually inappropriate belongings"""
        anomalies = []
        
        frame_area = frame_shape[0] * frame_shape[1]
        
        for obj in yolo_detections:
            class_name = obj.get('class', '')
            if class_name not in ['backpack', 'suitcase', 'handbag']:
                continue
            
            bbox = obj.get('bbox', (0, 0, 0, 0))
            obj_area = bbox[2] * bbox[3]
            
            # Unusually large (>5% of frame)
            if obj_area > frame_area * 0.05:
                anomalies.append(ObjectAnomalyDetection(
                    anomaly_type=ObjectAnomalyType.UNUSUAL_BELONGING,
                    confidence=0.65,
                    object_class=class_name,
                    bbox=bbox,
                    duration=0.0,
                    details=f"Unusually large {class_name} detected",
                    severity="LOW"
                ))
        
        return anomalies
    
    def _detect_stationary_suspicious(self, frame_shape: Tuple) -> List[ObjectAnomalyDetection]:
        """Detect stationary objects in dynamic zones"""
        anomalies = []
        
        if not self.dynamic_zones:
            return anomalies
        
        for obj_id, tracked in self.tracked_objects.items():
            if tracked.class_name == 'person':
                continue
            
            # Check if in dynamic zone
            if not tracked.positions:
                continue
            
            bbox = tracked.positions[-1]
            obj_center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
            
            in_dynamic_zone = False
            for zone in self.dynamic_zones:
                zx, zy, zw, zh = zone
                if (zx <= obj_center[0] <= zx + zw and 
                    zy <= obj_center[1] <= zy + zh):
                    in_dynamic_zone = True
                    break
            
            if in_dynamic_zone and tracked.stationary_duration > 10.0:
                anomalies.append(ObjectAnomalyDetection(
                    anomaly_type=ObjectAnomalyType.STATIONARY_SUSPICIOUS,
                    confidence=0.70,
                    object_class=tracked.class_name,
                    bbox=bbox,
                    duration=tracked.stationary_duration,
                    details=f"Stationary {tracked.class_name} in dynamic area ({tracked.stationary_duration:.1f}s)",
                    severity="MEDIUM"
                ))
        
        return anomalies
    
    def _detect_weapons_pose_based(self,
                                   keypoints_list: List[np.ndarray],
                                   bboxes: List[Tuple[int, int, int, int]],
                                   yolo_detections: List[Dict],
                                   frame: np.ndarray) -> List[ObjectAnomalyDetection]:
        """
        🔫 WEAPON DETECTION using pose analysis (NO specialized model needed)
        Detects: handguns, rifles, knives via biomechanical pose analysis
        """
        anomalies = []
        
        for keypoints, bbox in zip(keypoints_list, bboxes):
            if keypoints is None or len(keypoints) < 17:
                continue
            
            try:
                # Extract key points
                left_shoulder = keypoints[self.KEYPOINT_INDICES['left_shoulder']]
                right_shoulder = keypoints[self.KEYPOINT_INDICES['right_shoulder']]
                left_elbow = keypoints[self.KEYPOINT_INDICES['left_elbow']]
                right_elbow = keypoints[self.KEYPOINT_INDICES['right_elbow']]
                left_wrist = keypoints[self.KEYPOINT_INDICES['left_wrist']]
                right_wrist = keypoints[self.KEYPOINT_INDICES['right_wrist']]
                
                # Check confidence scores
                if left_shoulder[2] < 0.3 or right_shoulder[2] < 0.3:
                    continue
                
                # HANDGUN DETECTION: Extended arm with alignment
                if self._check_handgun_pose(left_shoulder, left_elbow, left_wrist) or \
                   self._check_handgun_pose(right_shoulder, right_elbow, right_wrist):
                    anomalies.append(ObjectAnomalyDetection(
                        anomaly_type=ObjectAnomalyType.WEAPON_HANDGUN,
                        confidence=0.75,
                        object_class='person_with_handgun',
                        bbox=bbox,
                        duration=0.0,
                        details="Handgun holding pose detected (extended arm alignment)",
                        severity="CRITICAL"
                    ))
                    continue  # One weapon type per person
                
                # RIFLE DETECTION: Both arms forward, parallel
                if self._check_rifle_pose(left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist):
                    anomalies.append(ObjectAnomalyDetection(
                        anomaly_type=ObjectAnomalyType.WEAPON_RIFLE,
                        confidence=0.80,
                        object_class='person_with_rifle',
                        bbox=bbox,
                        duration=0.0,
                        details="Rifle holding pose detected (parallel arm configuration)",
                        severity="CRITICAL"
                    ))
                    continue
                
                # KNIFE/BLUNT WEAPON: Raised arm or stabbing pose
                if self._check_knife_pose(left_shoulder, left_elbow, left_wrist) or \
                   self._check_knife_pose(right_shoulder, right_elbow, right_wrist):
                    anomalies.append(ObjectAnomalyDetection(
                        anomaly_type=ObjectAnomalyType.WEAPON_KNIFE,
                        confidence=0.65,
                        object_class='person_with_knife',
                        bbox=bbox,
                        duration=0.0,
                        details="Knife/weapon holding pose detected (raised arm)",
                        severity="CRITICAL"
                    ))
                    
            except (IndexError, KeyError):
                continue
        
        return anomalies
    
    def _check_handgun_pose(self, shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> bool:
        """Check if arm configuration matches handgun holding"""
        if shoulder[2] < 0.3 or elbow[2] < 0.3 or wrist[2] < 0.3:
            return False
        
        # Calculate arm extension angle
        shoulder_to_elbow = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
        elbow_to_wrist = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
        
        se_norm = np.linalg.norm(shoulder_to_elbow)
        ew_norm = np.linalg.norm(elbow_to_wrist)
        
        if se_norm < 1e-6 or ew_norm < 1e-6:
            return False
        
        # Check alignment (nearly straight for gun pose)
        dot_product = np.dot(shoulder_to_elbow, elbow_to_wrist)
        cos_angle = dot_product / (se_norm * ew_norm)
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        # Handgun: Arm extended (angle > 145°), wrist forward
        is_extended = angle_deg > 145
        is_forward = wrist[0] > shoulder[0] - 20
        
        return is_extended and is_forward
    
    def _check_rifle_pose(self, left_shoulder: np.ndarray, right_shoulder: np.ndarray,
                         left_elbow: np.ndarray, right_elbow: np.ndarray,
                         left_wrist: np.ndarray, right_wrist: np.ndarray) -> bool:
        """Check if pose matches rifle holding (both arms forward, parallel)"""
        if any(kp[2] < 0.3 for kp in [left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist]):
            return False
        
        # Both wrists forward
        left_forward = left_wrist[0] > left_shoulder[0]
        right_forward = right_wrist[0] > right_shoulder[0]
        
        # Wrists close together (holding same object)
        wrist_distance = np.linalg.norm([left_wrist[0] - right_wrist[0], left_wrist[1] - right_wrist[1]])
        shoulders_distance = np.linalg.norm([left_shoulder[0] - right_shoulder[0], left_shoulder[1] - right_shoulder[1]])
        wrists_close = wrist_distance < shoulders_distance * 1.2
        
        return left_forward and right_forward and wrists_close
    
    def _check_knife_pose(self, shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> bool:
        """Check if pose matches knife/blunt weapon (raised or extended arm)"""
        if shoulder[2] < 0.3 or elbow[2] < 0.3 or wrist[2] < 0.3:
            return False
        
        # Wrist above shoulder (raised arm) OR wrist extended forward with bent elbow
        wrist_raised = wrist[1] < shoulder[1] - 30
        
        # Calculate elbow angle
        v1 = np.array([shoulder[0] - elbow[0], shoulder[1] - elbow[1]])
        v2 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return False
        
        elbow_angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1)))
        bent_forward = (elbow_angle < 120) and (wrist[0] > elbow[0])
        
        return wrist_raised or bent_forward
    
    def _update_history(self, yolo_detections: List[Dict]):
        """Update historical data"""
        pass  # Already updated in other methods
    
    def add_restricted_zone(self, x: int, y: int, w: int, h: int):
        """Add restricted zone for vehicle detection"""
        self.restricted_zones.append((x, y, w, h))
    
    def add_dynamic_zone(self, x: int, y: int, w: int, h: int):
        """Add dynamic zone where objects should not be stationary"""
        self.dynamic_zones.append((x, y, w, h))


# Singleton instance
_object_anomaly_detector_instance = None

def get_object_anomaly_detector(**kwargs) -> ObjectAnomalyDetector:
    """Get singleton object anomaly detector instance"""
    global _object_anomaly_detector_instance
    if _object_anomaly_detector_instance is None:
        _object_anomaly_detector_instance = ObjectAnomalyDetector(**kwargs)
    return _object_anomaly_detector_instance
