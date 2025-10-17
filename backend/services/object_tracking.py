"""
Object Tracking Service
Track objects across frames using SORT/DeepSORT algorithms:
- Multi-object tracking
- Speed calculation
- Direction analysis
- Trajectory prediction

Author: AI Assistant
Date: 2025-10-17
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import math


@dataclass
class TrackedObject:
    """Tracked object information"""
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    centroid: Tuple[float, float]
    speed: float  # pixels per frame
    direction: float  # angle in degrees
    trajectory: List[Tuple[float, float]]
    age: int  # frames since first detection
    
    
@dataclass
class TrackingResult:
    """Tracking analysis result"""
    tracked_objects: List[TrackedObject]
    total_tracks: int
    new_tracks: int
    lost_tracks: int
    timestamp: str


class SimpleTracker:
    """
    Simple centroid-based object tracker
    Good for real-time performance without deep features
    """
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100.0):
        """
        Initialize tracker
        
        Args:
            max_disappeared: Max frames object can be missing before removal
            max_distance: Max distance to associate detection with existing track
        """
        self.next_object_id = 0
        self.objects = {}  # {id: TrackedObject}
        self.disappeared = {}  # {id: frames_missing}
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Trajectory history
        self.trajectory_length = 30  # Keep last 30 positions
        
    def update(self, detections: List[Dict]) -> TrackingResult:
        """
        Update tracker with new detections
        
        Args:
            detections: List of YOLO detections with bbox, class, confidence
            
        Returns:
            TrackingResult with tracked objects
        """
        new_tracks = 0
        lost_tracks = 0
        
        # If no detections, mark all as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.objects[object_id]
                    del self.disappeared[object_id]
                    lost_tracks += 1
            
            return TrackingResult(
                tracked_objects=list(self.objects.values()),
                total_tracks=len(self.objects),
                new_tracks=0,
                lost_tracks=lost_tracks,
                timestamp=datetime.now().isoformat()
            )
        
        # Extract centroids from detections
        input_centroids = []
        for det in detections:
            x, y, w, h = det['bbox']
            cx, cy = x + w // 2, y + h // 2
            input_centroids.append((cx, cy))
        
        input_centroids = np.array(input_centroids)
        
        # If no existing objects, register all as new
        if len(self.objects) == 0:
            for i, det in enumerate(detections):
                self._register(det, input_centroids[i])
                new_tracks += 1
        else:
            # Get existing object centroids
            object_ids = list(self.objects.keys())
            object_centroids = np.array([
                self.objects[oid].centroid for oid in object_ids
            ])
            
            # Compute distance between existing and new centroids
            D = self._compute_distances(object_centroids, input_centroids)
            
            # Match existing objects to new detections
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self._update_object(object_id, detections[col], input_centroids[col])
                
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Mark unused objects as disappeared
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.objects[object_id]
                    del self.disappeared[object_id]
                    lost_tracks += 1
            
            # Register new objects
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                self._register(detections[col], input_centroids[col])
                new_tracks += 1
        
        return TrackingResult(
            tracked_objects=list(self.objects.values()),
            total_tracks=len(self.objects),
            new_tracks=new_tracks,
            lost_tracks=lost_tracks,
            timestamp=datetime.now().isoformat()
        )
    
    def _register(self, detection: Dict, centroid: Tuple[float, float]):
        """Register new object"""
        self.objects[self.next_object_id] = TrackedObject(
            track_id=self.next_object_id,
            class_name=detection['class'],
            bbox=detection['bbox'],
            confidence=detection['confidence'],
            centroid=centroid,
            speed=0.0,
            direction=0.0,
            trajectory=[centroid],
            age=1
        )
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def _update_object(self, object_id: int, detection: Dict, centroid: Tuple[float, float]):
        """Update existing object"""
        obj = self.objects[object_id]
        
        # Update trajectory
        obj.trajectory.append(centroid)
        if len(obj.trajectory) > self.trajectory_length:
            obj.trajectory.pop(0)
        
        # Calculate speed and direction
        if len(obj.trajectory) >= 2:
            prev_pos = obj.trajectory[-2]
            curr_pos = obj.trajectory[-1]
            
            # Speed (pixels per frame)
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            obj.speed = math.sqrt(dx**2 + dy**2)
            
            # Direction (angle in degrees)
            obj.direction = math.degrees(math.atan2(dy, dx))
        
        # Update other attributes
        obj.bbox = detection['bbox']
        obj.confidence = detection['confidence']
        obj.centroid = centroid
        obj.age += 1
        
        self.objects[object_id] = obj
    
    def _compute_distances(self, centroids_a: np.ndarray, centroids_b: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance matrix"""
        # Expand dimensions for broadcasting
        a = centroids_a[:, np.newaxis, :]
        b = centroids_b[np.newaxis, :, :]
        
        # Compute distances
        distances = np.linalg.norm(a - b, axis=2)
        return distances
    
    def reset(self):
        """Reset tracker state"""
        self.objects.clear()
        self.disappeared.clear()
        self.next_object_id = 0


class SpeedAnalyzer:
    """Analyze object speeds for anomaly detection"""
    
    # Speed thresholds (pixels per frame at 1080p, 30fps)
    PERSON_RUNNING_THRESHOLD = 15.0  # ~15 pixels/frame
    VEHICLE_SPEEDING_THRESHOLD = 30.0  # ~30 pixels/frame
    
    def __init__(self):
        self.speed_history = {}  # {track_id: [speeds]}
        self.history_size = 30
    
    def analyze(self, tracked_objects: List[TrackedObject]) -> Dict:
        """
        Analyze speeds of tracked objects
        
        Returns:
            Dict with speed anomalies
        """
        anomalies = []
        
        for obj in tracked_objects:
            # Update speed history
            if obj.track_id not in self.speed_history:
                self.speed_history[obj.track_id] = deque(maxlen=self.history_size)
            
            self.speed_history[obj.track_id].append(obj.speed)
            
            # Check for speed anomalies
            if obj.class_name == 'person' and obj.speed > self.PERSON_RUNNING_THRESHOLD:
                anomalies.append({
                    'track_id': obj.track_id,
                    'type': 'PERSON_RUNNING',
                    'speed': obj.speed,
                    'confidence': 0.80
                })
            
            elif obj.class_name in ['car', 'truck', 'bus', 'motorcycle']:
                if obj.speed > self.VEHICLE_SPEEDING_THRESHOLD:
                    anomalies.append({
                        'track_id': obj.track_id,
                        'type': 'VEHICLE_SPEEDING',
                        'speed': obj.speed,
                        'confidence': 0.75
                    })
        
        return {
            'has_anomalies': len(anomalies) > 0,
            'anomalies': anomalies,
            'timestamp': datetime.now().isoformat()
        }
