"""
Zone Manager for Restricted Area Detection
Manages polygonal zones for spatial rule-based alerts

Author: AI Assistant
Date: 2025-10-17
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ZoneType(Enum):
    """Types of monitored zones"""
    RESTRICTED = "restricted"
    HIGH_SECURITY = "high_security"
    NO_LOITERING = "no_loitering"
    TRIPWIRE = "tripwire"
    SPEED_LIMIT = "speed_limit"
    PUBLIC = "public"


@dataclass
class Zone:
    """Zone definition"""
    zone_id: str
    polygon: np.ndarray
    zone_type: ZoneType
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RestrictedZoneManager:
    """
    Manage restricted zones and spatial rules
    
    Features:
    - Polygon-based zone definition
    - Point-in-polygon testing
    - BBox overlap detection
    - Zone visualization
    - Multiple zone types
    """
    
    def __init__(self):
        self.zones: Dict[str, Zone] = {}
        
    def add_zone(self, 
                 zone_id: str, 
                 polygon: List[Tuple[int, int]], 
                 zone_type: ZoneType = ZoneType.RESTRICTED,
                 **metadata):
        """
        Add a spatial zone
        
        Args:
            zone_id: Unique identifier (e.g., "entrance_1", "parking_lot")
            polygon: List of (x, y) points defining the zone boundary
            zone_type: Type of zone (RESTRICTED, HIGH_SECURITY, etc.)
            **metadata: Additional zone properties (speed_limit, allowed_objects, etc.)
            
        Example:
            zone_manager.add_zone(
                'entrance',
                [(100, 200), (300, 200), (300, 400), (100, 400)],
                zone_type=ZoneType.RESTRICTED
            )
        """
        polygon_array = np.array(polygon, dtype=np.int32)
        
        self.zones[zone_id] = Zone(
            zone_id=zone_id,
            polygon=polygon_array,
            zone_type=zone_type,
            metadata=metadata
        )
    
    def add_tripwire(self, 
                     zone_id: str,
                     line: Tuple[Tuple[int, int], Tuple[int, int]],
                     direction: str = 'both'):
        """
        Add a tripwire (virtual line)
        
        Args:
            zone_id: Unique identifier
            line: ((x1, y1), (x2, y2)) - line endpoints
            direction: 'forward', 'backward', or 'both'
        """
        # Create thin polygon around line
        (x1, y1), (x2, y2) = line
        
        # Calculate perpendicular offset (5 pixels)
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length > 0:
            nx = -dy / length * 5  # Normal vector
            ny = dx / length * 5
        else:
            nx, ny = 0, 5
        
        # Create polygon
        polygon = [
            (int(x1 + nx), int(y1 + ny)),
            (int(x2 + nx), int(y2 + ny)),
            (int(x2 - nx), int(y2 - ny)),
            (int(x1 - nx), int(y1 - ny))
        ]
        
        self.add_zone(
            zone_id,
            polygon,
            zone_type=ZoneType.TRIPWIRE,
            direction=direction,
            line=line
        )
    
    def is_point_in_zone(self, 
                        point: Tuple[int, int], 
                        zone_id: Optional[str] = None,
                        zone_type: Optional[ZoneType] = None) -> Tuple[bool, Optional[str]]:
        """
        Check if point is inside any zone
        
        Args:
            point: (x, y) coordinates
            zone_id: Check specific zone (optional)
            zone_type: Check zones of specific type (optional)
            
        Returns:
            (is_inside, zone_id) - True if inside any zone
        """
        x, y = point
        
        # Filter zones to check
        if zone_id:
            zones_to_check = {zone_id: self.zones[zone_id]} if zone_id in self.zones else {}
        elif zone_type:
            zones_to_check = {zid: z for zid, z in self.zones.items() if z.zone_type == zone_type}
        else:
            zones_to_check = self.zones
        
        # Check each zone
        for zid, zone in zones_to_check.items():
            result = cv2.pointPolygonTest(zone.polygon, (float(x), float(y)), False)
            if result >= 0:  # Inside or on boundary
                return True, zid
        
        return False, None
    
    def is_bbox_in_zone(self, 
                       bbox: Tuple[int, int, int, int],
                       zone_id: Optional[str] = None,
                       zone_type: Optional[ZoneType] = None,
                       threshold: float = 0.5) -> Tuple[bool, Optional[str]]:
        """
        Check if bounding box overlaps with zone
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box
            zone_id: Check specific zone (optional)
            zone_type: Check zones of specific type (optional)
            threshold: Overlap threshold (0.5 = center point must be in zone)
            
        Returns:
            (is_inside, zone_id) - True if overlaps with any zone
        """
        x1, y1, x2, y2 = bbox
        
        # Calculate center and corners
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # For strict checking (center must be in zone)
        if threshold >= 0.5:
            return self.is_point_in_zone(center, zone_id, zone_type)
        
        # For loose checking (any corner in zone)
        corners = [
            (x1, y1), (x2, y1), (x2, y2), (x1, y2), center
        ]
        
        for corner in corners:
            is_inside, zid = self.is_point_in_zone(corner, zone_id, zone_type)
            if is_inside:
                return True, zid
        
        return False, None
    
    def check_trajectory_crosses_zone(self,
                                     prev_pos: Tuple[int, int],
                                     curr_pos: Tuple[int, int],
                                     zone_id: str) -> bool:
        """
        Check if trajectory crosses a zone (useful for tripwires)
        
        Args:
            prev_pos: Previous position (x, y)
            curr_pos: Current position (x, y)
            zone_id: Zone to check
            
        Returns:
            True if trajectory crosses zone boundary
        """
        if zone_id not in self.zones:
            return False
        
        zone = self.zones[zone_id]
        
        # Check if line segment crosses polygon
        # Simple approach: check if prev and curr are on opposite sides
        prev_inside = cv2.pointPolygonTest(zone.polygon, 
                                          (float(prev_pos[0]), float(prev_pos[1])), False) >= 0
        curr_inside = cv2.pointPolygonTest(zone.polygon, 
                                          (float(curr_pos[0]), float(curr_pos[1])), False) >= 0
        
        # Crossing detected if states differ
        return prev_inside != curr_inside
    
    def get_zone_metadata(self, zone_id: str, key: str, default=None):
        """Get metadata for a zone"""
        if zone_id in self.zones:
            return self.zones[zone_id].metadata.get(key, default)
        return default
    
    def get_zones_by_type(self, zone_type: ZoneType) -> Dict[str, Zone]:
        """Get all zones of specific type"""
        return {zid: z for zid, z in self.zones.items() if z.zone_type == zone_type}
    
    def draw_zones(self, 
                   frame: np.ndarray,
                   zone_colors: Dict[ZoneType, Tuple[int, int, int]] = None,
                   show_labels: bool = True,
                   thickness: int = 2) -> np.ndarray:
        """
        Draw all zones on frame for visualization
        
        Args:
            frame: Input frame
            zone_colors: Custom colors per zone type
            show_labels: Show zone IDs
            thickness: Line thickness
            
        Returns:
            Frame with zones drawn
        """
        if zone_colors is None:
            zone_colors = {
                ZoneType.RESTRICTED: (0, 0, 255),      # Red
                ZoneType.HIGH_SECURITY: (0, 0, 139),   # Dark Red
                ZoneType.NO_LOITERING: (0, 165, 255),  # Orange
                ZoneType.TRIPWIRE: (255, 0, 255),      # Magenta
                ZoneType.SPEED_LIMIT: (0, 255, 255),   # Yellow
                ZoneType.PUBLIC: (0, 255, 0)           # Green
            }
        
        frame_copy = frame.copy()
        
        for zone_id, zone in self.zones.items():
            color = zone_colors.get(zone.zone_type, (128, 128, 128))
            
            # Draw polygon
            cv2.polylines(frame_copy, [zone.polygon], 
                         isClosed=True, color=color, thickness=thickness)
            
            # Fill with semi-transparent color
            overlay = frame_copy.copy()
            cv2.fillPoly(overlay, [zone.polygon], color)
            cv2.addWeighted(overlay, 0.2, frame_copy, 0.8, 0, frame_copy)
            
            if show_labels:
                # Add zone label at first point
                x, y = zone.polygon[0]
                label = f"{zone_id} ({zone.zone_type.value})"
                
                # Add background for text
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame_copy, (x, y - h - 10), (x + w, y - 5), color, -1)
                cv2.putText(frame_copy, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame_copy
    
    def clear_zones(self):
        """Remove all zones"""
        self.zones.clear()
    
    def remove_zone(self, zone_id: str):
        """Remove specific zone"""
        if zone_id in self.zones:
            del self.zones[zone_id]
    
    def __len__(self):
        return len(self.zones)
    
    def __contains__(self, zone_id):
        return zone_id in self.zones
