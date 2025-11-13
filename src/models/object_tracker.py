"""
Object Tracker Service
Tracks living objects (persons, animals) across video frames and maintains:
- Unique object IDs
- Centroid positions and movement history
- Object associations with held items
- Temporal metadata (duration, entry/exit times)

Uses Hungarian algorithm (scipy) for optimal centroid assignment across frames.

Author: Ravishan
Date: 2025-11-09
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from scipy.optimize import linear_sum_assignment
import logging

logger = logging.getLogger(__name__)


@dataclass
class ObjectTrack:
    """Represents a tracked object (person, animal, etc.)"""
    track_id: str
    class_name: str  # "person", "dog", "cat", etc.
    centroid: np.ndarray  # (x, y) position
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    
    # Temporal info
    created_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    age_frames: int = 0
    
    # Movement history
    centroid_history: deque = field(default_factory=lambda: deque(maxlen=30))
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=30))
    
    # Associated objects (e.g., hands, held items)
    associated_objects: Dict[str, List[dict]] = field(default_factory=dict)  # {body_part: [...]}
    held_items: List[dict] = field(default_factory=list)  # weapons, bags, etc.
    
    # Anomaly context
    is_moving: bool = False
    movement_speed: float = 0.0  # pixels per frame
    movement_direction: float = 0.0  # angle in degrees
    anomaly_score: float = 0.0
    
    def update(self, centroid: np.ndarray, bbox: Tuple[int, int, int, int], confidence: float):
        """Update track with new detection"""
        self.centroid_history.append(self.centroid.copy())
        self.bbox_history.append(self.bbox)
        
        self.centroid = centroid
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = datetime.now()
        self.age_frames += 1
        
        # Calculate movement
        if len(self.centroid_history) > 0:
            prev_centroid = self.centroid_history[-1]
            dx = centroid[0] - prev_centroid[0]
            dy = centroid[1] - prev_centroid[1]
            self.movement_speed = np.sqrt(dx**2 + dy**2)
            self.movement_direction = np.degrees(np.arctan2(dy, dx))
            self.is_moving = self.movement_speed > 5  # threshold: 5 pixels/frame
    
    def get_position_history(self) -> List[np.ndarray]:
        """Get centroid history as list"""
        return list(self.centroid_history) + [self.centroid]
    
    def is_stationary(self, duration_secs: float = 5.0) -> bool:
        """Check if object has been stationary for N seconds"""
        if self.movement_speed > 5:
            return False
        time_diff = (datetime.now() - self.last_seen).total_seconds()
        return time_diff <= duration_secs
    
    def is_active(self, max_age_secs: float = 2.0) -> bool:
        """Check if track is still active (recently seen)"""
        time_diff = (datetime.now() - self.last_seen).total_seconds()
        return time_diff <= max_age_secs


class ObjectTracker:
    """
    Multi-object tracker using centroid tracking with Hungarian algorithm
    
    Maintains unique IDs for detected objects across frames and tracks:
    - Position and movement
    - Associated body parts (hands, head)
    - Held items (weapons, bags)
    - Temporal information (age, duration in frame)
    """
    
    def __init__(self, 
                 max_distance: float = 50.0,
                 max_frames_inactive: int = 30,
                 fps: int = 30):
        """
        Initialize tracker
        
        Args:
            max_distance: Max centroid distance to associate detections (pixels)
            max_frames_inactive: Max frames to keep inactive track before removal
            fps: Frames per second for temporal calculations
        """
        self.max_distance = max_distance
        self.max_frames_inactive = max_frames_inactive
        self.fps = fps
        
        self.tracks: Dict[str, ObjectTrack] = {}
        self.next_track_id = 0
        self.frame_count = 0
    
    def update(self, detections: List[dict]) -> Dict[str, ObjectTrack]:
        """
        Update tracker with new frame detections
        
        Args:
            detections: List of detection dicts:
                {
                    'class': 'person',
                    'confidence': 0.95,
                    'bbox': (x1, y1, x2, y2),
                    'centroid': (x, y),
                    'keypoints': [...],  # pose keypoints if available
                    'held_items': [...]  # detected items in hands
                }
        
        Returns:
            Dictionary of active tracks
        """
        self.frame_count += 1
        
        # Extract centroids from detections
        detection_centroids = np.array([d['centroid'] for d in detections])
        
        # Match detections to existing tracks
        if len(self.tracks) == 0:
            # No existing tracks - create new ones
            for det in detections:
                self._create_track(det)
        else:
            # Use Hungarian algorithm to match detections to tracks
            self._match_detections(detections, detection_centroids)
        
        # Remove inactive tracks
        self._cleanup_inactive_tracks()
        
        return self.tracks
    
    def _match_detections(self, detections: List[dict], centroids: np.ndarray):
        """Match detections to existing tracks using Hungarian algorithm"""
        
        # Get current track centroids
        track_ids = list(self.tracks.keys())
        track_centroids = np.array([self.tracks[tid].centroid for tid in track_ids])
        
        if len(track_centroids) == 0:
            for det in detections:
                self._create_track(det)
            return
        
        # Compute distance matrix
        distances = self._compute_distances(track_centroids, centroids)
        
        # Solve assignment problem
        track_indices, det_indices = linear_sum_assignment(distances)
        
        # Track matched detections
        matched_det_indices = set()
        
        for track_idx, det_idx in zip(track_indices, det_indices):
            distance = distances[track_idx, det_idx]
            
            if distance <= self.max_distance:
                # Match found - update track
                track_id = track_ids[track_idx]
                det = detections[det_idx]
                self.tracks[track_id].update(
                    centroid=np.array(det['centroid']),
                    bbox=det['bbox'],
                    confidence=det['confidence']
                )
                
                # Update associated objects (body parts, held items)
                if 'keypoints' in det:
                    self._update_associated_objects(track_id, det)
                
                matched_det_indices.add(det_idx)
        
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_det_indices:
                self._create_track(det)
    
    def _create_track(self, detection: dict):
        """Create a new track for an unmatched detection"""
        track_id = f"TRK-{self.frame_count:06d}-{self.next_track_id:04d}"
        self.next_track_id += 1
        
        track = ObjectTrack(
            track_id=track_id,
            class_name=detection['class'],
            centroid=np.array(detection['centroid']),
            bbox=detection['bbox'],
            confidence=detection['confidence']
        )
        
        # Add track to dictionary BEFORE updating associated objects
        self.tracks[track_id] = track
        
        if 'keypoints' in detection:
            self._update_associated_objects(track_id, detection)
        
        logger.debug(f"Created track {track_id} for {detection['class']}")
    
    def _update_associated_objects(self, track_id: str, detection: dict):
        """Update body parts and held items for a track"""
        track = self.tracks[track_id]
        
        # Store keypoints/body parts
        if 'keypoints' in detection:
            keypoints = detection['keypoints']
            track.associated_objects['keypoints'] = keypoints
            
            # Extract hand positions
            if len(keypoints) > 10:  # COCO format has 17 keypoints
                left_hand = keypoints[9]  # left wrist
                right_hand = keypoints[10]  # right wrist
                track.associated_objects['hands'] = [left_hand, right_hand]
        
        # Store held items
        if 'held_items' in detection:
            track.held_items = detection['held_items']
    
    def _compute_distances(self, 
                          track_centroids: np.ndarray,
                          det_centroids: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances between track and detection centroids"""
        if len(track_centroids) == 0 or len(det_centroids) == 0:
            return np.empty((len(track_centroids), len(det_centroids)))
        
        # Compute pairwise distances
        distances = np.zeros((len(track_centroids), len(det_centroids)))
        for i, track_c in enumerate(track_centroids):
            for j, det_c in enumerate(det_centroids):
                distances[i, j] = np.linalg.norm(track_c - det_c)
        
        return distances
    
    def _cleanup_inactive_tracks(self):
        """Remove tracks that haven't been seen recently"""
        inactive_ids = [
            track_id for track_id, track in self.tracks.items()
            if self.frame_count - track.age_frames > self.max_frames_inactive
        ]
        
        for track_id in inactive_ids:
            logger.debug(f"Removed inactive track {track_id}")
            del self.tracks[track_id]
    
    def get_active_tracks(self) -> Dict[str, ObjectTrack]:
        """Get all currently active tracks"""
        return {tid: track for tid, track in self.tracks.items() if track.is_active()}
    
    def get_track_by_id(self, track_id: str) -> Optional[ObjectTrack]:
        """Get specific track by ID"""
        return self.tracks.get(track_id)
    
    def get_tracks_by_class(self, class_name: str) -> List[ObjectTrack]:
        """Get all tracks of specific class (person, dog, etc.)"""
        return [track for track in self.tracks.values() if track.class_name == class_name]
    
    def get_statistics(self) -> dict:
        """Get tracker statistics"""
        active_tracks = self.get_active_tracks()
        return {
            'total_tracks': len(self.tracks),
            'active_tracks': len(active_tracks),
            'frame_count': self.frame_count,
            'tracked_persons': len(self.get_tracks_by_class('person')),
            'tracked_animals': len([t for t in self.tracks.values() 
                                   if t.class_name in ['dog', 'cat', 'bird']]),
        }


# Singleton instance
_object_tracker = None

def get_object_tracker(max_distance: float = 50.0,
                      max_frames_inactive: int = 30,
                      fps: int = 30) -> ObjectTracker:
    """Get or create singleton tracker instance"""
    global _object_tracker
    if _object_tracker is None:
        _object_tracker = ObjectTracker(max_distance, max_frames_inactive, fps)
    return _object_tracker
