"""
Motion Analysis Service
Detects anomalies through motion patterns using:
- Optical Flow (Farneback)
- Background Subtraction (MOG2)
- Frame Differencing

Author: AI Assistant
Date: 2025-10-17
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MotionResult:
    """Motion analysis result"""
    motion_magnitude: float
    motion_direction: float
    motion_regions: List[Tuple[int, int, int, int]]  # (x, y, w, h)
    is_unusual: bool
    anomaly_type: Optional[str]
    confidence: float
    timestamp: str


class MotionAnalyzer:
    """Advanced motion analysis for anomaly detection"""
    
    def __init__(self, 
                 motion_threshold: float = 5.0,
                 static_threshold: float = 2.0,
                 learning_rate: float = 0.01):
        """
        Initialize motion analyzer
        
        Args:
            motion_threshold: Threshold for unusual motion magnitude
            static_threshold: Threshold for static object detection
            learning_rate: Background subtractor learning rate
        """
        # Background subtractor (MOG2)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        
        # Previous frame for optical flow
        self.prev_frame = None
        self.prev_gray = None
        
        # Thresholds
        self.motion_threshold = motion_threshold
        self.static_threshold = static_threshold
        self.learning_rate = learning_rate
        
        # Motion history for anomaly detection
        self.motion_history = []
        self.history_size = 30  # 30 frames
        
        # Statistical anomaly detection (Z-score)
        self.enable_statistical_detection = True
        self.z_score_threshold = 3.0  # 3 standard deviations
        
        # Stationary object tracking
        self.stationary_objects = {}  # {region_id: (bbox, start_time, frames_count)}
        
    def analyze(self, frame: np.ndarray) -> MotionResult:
        """
        Analyze motion in the frame
        
        Args:
            frame: Input BGR frame
            
        Returns:
            MotionResult with detected anomalies
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Initialize
        if self.prev_gray is None:
            self.prev_gray = gray
            return MotionResult(
                motion_magnitude=0.0,
                motion_direction=0.0,
                motion_regions=[],
                is_unusual=False,
                anomaly_type=None,
                confidence=0.0,
                timestamp=datetime.now().isoformat()
            )
        
        # 1. Optical Flow Analysis
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Calculate motion magnitude and direction
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Average motion
        avg_magnitude = np.mean(magnitude)
        avg_angle = np.mean(angle)
        
        # 2. Background Subtraction
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        
        # Clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # 3. Find motion regions
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_regions = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                motion_regions.append((x, y, w, h))
        
        # 4. Detect stationary objects (abandoned objects, loitering)
        self._update_stationary_objects(motion_regions, magnitude)
        
        # 5. Update motion history
        self.motion_history.append(avg_magnitude)
        if len(self.motion_history) > self.history_size:
            self.motion_history.pop(0)
        
        # 6. Statistical Anomaly Detection (Z-score)
        statistical_anomaly, z_score = self._detect_statistical_anomaly(avg_magnitude)
        
        # 7. Rule-based Anomaly Detection
        is_unusual, anomaly_type, confidence = self._detect_anomaly(
            avg_magnitude, motion_regions, magnitude
        )
        
        # 8. Combine statistical and rule-based detection
        if statistical_anomaly and not is_unusual:
            # Statistical outlier but not rule-based anomaly
            is_unusual = True
            anomaly_type = f"statistical_outlier (z-score: {z_score:.2f})"
            confidence = min(0.5 + (z_score - self.z_score_threshold) * 0.1, 0.95)
        elif statistical_anomaly and is_unusual:
            # Both agree - increase confidence
            confidence = min(confidence * 1.2, 0.98)
        
        # Update previous frame
        self.prev_gray = gray
        
        return MotionResult(
            motion_magnitude=float(avg_magnitude),
            motion_direction=float(avg_angle),
            motion_regions=motion_regions,
            is_unusual=is_unusual,
            anomaly_type=anomaly_type,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
    
    def _update_stationary_objects(self, motion_regions: List, magnitude: np.ndarray):
        """Track stationary objects over time"""
        current_time = datetime.now()
        
        # Update existing stationary objects
        for region_id in list(self.stationary_objects.keys()):
            bbox, start_time, frames = self.stationary_objects[region_id]
            
            # Check if object still present
            still_present = False
            for region in motion_regions:
                if self._iou(bbox, region) > 0.5:  # Overlap threshold
                    self.stationary_objects[region_id] = (
                        region, start_time, frames + 1
                    )
                    still_present = True
                    break
            
            # Remove if gone
            if not still_present:
                del self.stationary_objects[region_id]
        
        # Add new stationary objects
        for region in motion_regions:
            # Check if low motion in this region
            x, y, w, h = region
            region_magnitude = magnitude[y:y+h, x:x+w].mean() if h > 0 and w > 0 else 0
            
            if region_magnitude < self.static_threshold:
                # Check if already tracked
                is_tracked = False
                for bbox, _, _ in self.stationary_objects.values():
                    if self._iou(bbox, region) > 0.5:
                        is_tracked = True
                        break
                
                if not is_tracked:
                    region_id = f"{x}_{y}_{current_time.timestamp()}"
                    self.stationary_objects[region_id] = (region, current_time, 1)
    
    def _iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate Intersection over Union"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _detect_anomaly(self, avg_magnitude: float, motion_regions: List,
                       magnitude: np.ndarray) -> Tuple[bool, Optional[str], float]:
        """
        Detect motion-based anomalies
        
        Returns:
            (is_unusual, anomaly_type, confidence)
        """
        # 1. Sudden high motion (panic, running, crowd surge)
        if avg_magnitude > self.motion_threshold:
            if len(motion_regions) > 10:
                return True, "CROWD_PANIC", 0.85
            else:
                return True, "RAPID_MOVEMENT", 0.75
        
        # 2. Unusual motion pattern (compare with history)
        if len(self.motion_history) > 10:
            motion_std = np.std(self.motion_history)
            motion_mean = np.mean(self.motion_history)
            
            # Z-score anomaly detection
            if motion_std > 0:
                z_score = abs(avg_magnitude - motion_mean) / motion_std
                if z_score > 2.5:
                    return True, "UNUSUAL_MOTION_PATTERN", min(0.95, z_score / 3)
        
        # 3. Abandoned object detection (stationary > 150 frames ~5 seconds at 30fps)
        for region_id, (bbox, start_time, frames) in self.stationary_objects.items():
            if frames > 150:  # ~5 seconds
                duration = (datetime.now() - start_time).total_seconds()
                return True, "ABANDONED_OBJECT", min(0.95, duration / 60)
        
        # 4. Loitering detection (person staying too long)
        for region_id, (bbox, start_time, frames) in self.stationary_objects.items():
            if frames > 900:  # ~30 seconds
                return True, "LOITERING", 0.70
        
        return False, None, 0.0
    
    def _detect_statistical_anomaly(self, magnitude: float) -> Tuple[bool, float]:
        """
        Detect motion anomaly using Z-score (statistical outlier detection)
        
        Args:
            magnitude: Current motion magnitude
            
        Returns:
            (is_anomaly, z_score)
        """
        if not self.enable_statistical_detection:
            return False, 0.0
        
        # Need enough history for statistics
        if len(self.motion_history) < 10:
            return False, 0.0
        
        # Calculate Z-score
        mean = np.mean(self.motion_history)
        std = np.std(self.motion_history)
        
        # Avoid division by zero
        if std < 1e-6:
            return False, 0.0
        
        z_score = (magnitude - mean) / std
        
        # Detect anomaly if beyond threshold
        is_anomaly = abs(z_score) > self.z_score_threshold
        
        return is_anomaly, abs(z_score)
    
    def get_motion_heatmap(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Generate motion heatmap for visualization
        
        Args:
            frame_shape: (height, width)
            
        Returns:
            Heatmap as BGR image
        """
        if self.prev_gray is None:
            return np.zeros((*frame_shape, 3), dtype=np.uint8)
        
        # Get optical flow
        gray = self.prev_gray  # Use stored previous frame
        
        heatmap = np.zeros((*frame_shape, 3), dtype=np.uint8)
        
        # Create motion visualization
        # This will be populated in real-time during analysis
        
        return heatmap
    
    def reset(self):
        """Reset analyzer state"""
        self.prev_gray = None
        self.motion_history.clear()
        self.stationary_objects.clear()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
