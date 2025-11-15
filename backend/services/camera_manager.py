"""
Camera Configuration Manager
Manages multiple cameras (webcam and IP cameras) for the surveillance system

Supports:
- Webcam cameras (browser getUserMedia)
- IP cameras via RTSP (even across different networks)
- Camera CRUD operations
- Persistent storage in JSON

Author: AI Assistant
Date: 2025-11-09
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum


class CameraType(Enum):
    """Camera types supported"""
    WEBCAM = "webcam"  # Browser webcam via getUserMedia
    IP_CAMERA = "ip_camera"  # IP camera via RTSP


class CameraStatus(Enum):
    """Camera connection status"""
    ONLINE = "online"
    OFFLINE = "offline"
    CONNECTING = "connecting"
    ERROR = "error"


@dataclass
class Camera:
    """Camera configuration"""
    id: str
    name: str
    type: CameraType
    location: str
    
    # IP camera specific
    rtsp_url: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    
    # Network access configuration (for cameras on different networks)
    network_access_method: str = "direct"  # direct, ssh_tunnel, rtsp_proxy, vpn
    ssh_host: Optional[str] = None  # SSH server for tunneling
    ssh_port: int = 22
    ssh_user: Optional[str] = None
    ssh_key_path: Optional[str] = None
    tunnel_local_port: Optional[int] = None  # Local port for SSH tunnel
    proxy_host: Optional[str] = None  # RTSP proxy host
    proxy_port: Optional[int] = None  # RTSP proxy port
    
    # Status and metadata
    status: CameraStatus = CameraStatus.OFFLINE
    enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen: Optional[str] = None
    error_message: Optional[str] = None
    
    # Settings
    resolution_width: int = 1280
    resolution_height: int = 720
    fps: int = 15
    
    # Analytics
    total_detections: int = 0
    last_detection_at: Optional[str] = None
    
    # Per-camera pose tuning (for reducing false positives)
    # Units: body_angle in degrees, velocity/accel in normalized units (relative frame size)
    pose_body_angle_thresh: float = 45.0
    pose_velocity_thresh: float = 0.02
    pose_acceleration_thresh: float = 0.02
    pose_persistence_frames: int = 3
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert enums to strings
        data['type'] = self.type.value
        data['status'] = self.status.value
        return data
    
    @staticmethod
    def from_dict(data: Dict) -> 'Camera':
        """Create Camera from dictionary"""
        # Convert string enums back to enum instances
        if isinstance(data.get('type'), str):
            data['type'] = CameraType(data['type'])
        if isinstance(data.get('status'), str):
            data['status'] = CameraStatus(data['status'])
        return Camera(**data)


class CameraManager:
    """Manages multiple cameras for the surveillance system"""
    
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize camera manager"""
        if storage_path is None:
            # Default to backend/data/cameras.json
            backend_dir = Path(__file__).parent.parent
            storage_path = backend_dir / "data" / "cameras.json"
        
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.cameras: Dict[str, Camera] = {}
        self._load_cameras()
    
    def _load_cameras(self):
        """Load cameras from storage"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for cam_dict in data.get('cameras', []):
                        cam = Camera.from_dict(cam_dict)
                        self.cameras[cam.id] = cam
                print(f"✅ Loaded {len(self.cameras)} cameras from {self.storage_path}")
            except Exception as e:
                print(f"⚠️ Error loading cameras: {e}")
                self.cameras = {}
        else:
            # Create default webcam camera
            default_cam = Camera(
                id="cam-001",
                name="Primary Webcam",
                type=CameraType.WEBCAM,
                location="Main Entrance",
                enabled=True
            )
            self.cameras[default_cam.id] = default_cam
            self._save_cameras()
            print(f"✅ Created default webcam camera: {default_cam.name}")
    
    def _save_cameras(self):
        """Save cameras to storage"""
        try:
            data = {
                'version': '1.0',
                'updated_at': datetime.now().isoformat(),
                'cameras': [cam.to_dict() for cam in self.cameras.values()]
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving cameras: {e}")
    
    def add_camera(self, camera_data: Dict) -> Camera:
        """
        Add a new camera
        
        Args:
            camera_data: Camera configuration dict
            
        Returns:
            Created Camera object
            
        Raises:
            ValueError: If camera ID already exists or validation fails
        """
        # Generate ID if not provided
        if 'id' not in camera_data or not camera_data['id']:
            # Generate unique ID
            existing_ids = set(self.cameras.keys())
            counter = len(self.cameras) + 1
            while True:
                new_id = f"cam-{counter:03d}"
                if new_id not in existing_ids:
                    camera_data['id'] = new_id
                    break
                counter += 1
        
        camera_id = camera_data['id']
        
        if camera_id in self.cameras:
            raise ValueError(f"Camera with ID {camera_id} already exists")
        
        # Validate required fields
        if 'name' not in camera_data or not camera_data['name']:
            raise ValueError("Camera name is required")
        
        if 'type' not in camera_data:
            raise ValueError("Camera type is required")
        
        # Convert type string to enum if needed
        if isinstance(camera_data.get('type'), str):
            try:
                camera_data['type'] = CameraType(camera_data['type'])
            except ValueError:
                raise ValueError(f"Invalid camera type: {camera_data['type']}")
        
        # Validate IP camera has RTSP URL
        if camera_data['type'] == CameraType.IP_CAMERA:
            if not camera_data.get('rtsp_url'):
                raise ValueError("RTSP URL is required for IP cameras")
        
        # Create camera object
        camera = Camera.from_dict(camera_data)
        
        # Add to manager
        self.cameras[camera_id] = camera
        self._save_cameras()
        
        print(f"✅ Added camera: {camera.name} ({camera.id})")
        return camera
    
    def get_camera(self, camera_id: str) -> Optional[Camera]:
        """Get camera by ID"""
        return self.cameras.get(camera_id)
    
    def get_all_cameras(self) -> List[Camera]:
        """Get all cameras"""
        return list(self.cameras.values())
    
    def get_enabled_cameras(self) -> List[Camera]:
        """Get only enabled cameras"""
        return [cam for cam in self.cameras.values() if cam.enabled]
    
    def update_camera(self, camera_id: str, updates: Dict) -> Camera:
        """
        Update camera configuration
        
        Args:
            camera_id: Camera ID to update
            updates: Dictionary of fields to update
            
        Returns:
            Updated Camera object
            
        Raises:
            ValueError: If camera not found
        """
        camera = self.cameras.get(camera_id)
        if not camera:
            raise ValueError(f"Camera {camera_id} not found")
        
        # Update allowed fields
        allowed_fields = {
            'name', 'location', 'rtsp_url', 'username', 'password',
            'enabled', 'resolution_width', 'resolution_height', 'fps',
            # Pose tuning fields
            'pose_body_angle_thresh', 'pose_velocity_thresh', 'pose_acceleration_thresh', 'pose_persistence_frames'
        }
        
        for key, value in updates.items():
            if key in allowed_fields:
                setattr(camera, key, value)
        
        self._save_cameras()
        print(f"✅ Updated camera: {camera.name} ({camera.id})")
        return camera
    
    def delete_camera(self, camera_id: str) -> bool:
        """
        Delete camera
        
        Args:
            camera_id: Camera ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if camera_id in self.cameras:
            camera = self.cameras[camera_id]
            del self.cameras[camera_id]
            self._save_cameras()
            print(f"✅ Deleted camera: {camera.name} ({camera_id})")
            return True
        return False
    
    def update_camera_status(self, camera_id: str, status: CameraStatus, 
                            error_message: Optional[str] = None):
        """Update camera online/offline status"""
        camera = self.cameras.get(camera_id)
        if camera:
            camera.status = status
            camera.last_seen = datetime.now().isoformat()
            if error_message:
                camera.error_message = error_message
            else:
                camera.error_message = None
            self._save_cameras()
    
    def record_detection(self, camera_id: str):
        """Record that a detection occurred on this camera"""
        camera = self.cameras.get(camera_id)
        if camera:
            camera.total_detections += 1
            camera.last_detection_at = datetime.now().isoformat()
            self._save_cameras()


# Global instance
_camera_manager = None


def get_camera_manager() -> CameraManager:
    """Get global camera manager instance"""
    global _camera_manager
    if _camera_manager is None:
        _camera_manager = CameraManager()
    return _camera_manager
