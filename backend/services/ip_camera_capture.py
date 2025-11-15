"""
IP Camera Capture Service
Handles RTSP streams from IP cameras, including cross-network cameras

Features:
- RTSP stream connection and management
- Automatic reconnection on failure
- Frame buffering for smooth streaming
- Thread-safe frame access
- Network bridge support for cameras on different networks

Author: AI Assistant
Date: 2025-11-09
"""

import cv2
import threading
import time
from queue import Queue
from typing import Optional, Callable, Dict, Any
import numpy as np
from .network_bridge import get_network_bridge, NetworkAccessMethod, SSHTunnelConfig, ProxyConfig


class IPCameraCapture:
    """
    Thread-safe IP camera capture with automatic reconnection
    
    Captures frames from RTSP stream in a background thread and provides
    the latest frame on demand. Handles disconnections gracefully.
    Supports cameras on different networks via SSH tunneling or proxy.
    """
    
    def __init__(self, camera_id: str, rtsp_url: str, 
                 reconnect_delay: float = 5.0,
                 max_reconnect_attempts: int = 10,
                 network_config: Optional[Dict[str, Any]] = None):
        """
        Initialize IP camera capture
        
        Args:
            camera_id: Unique camera identifier
            rtsp_url: RTSP URL (e.g., rtsp://192.168.1.100:554/stream)
            reconnect_delay: Seconds to wait before reconnecting
            max_reconnect_attempts: Maximum reconnection attempts (0 = infinite)
            network_config: Network access configuration for cross-network cameras
                {
                    'method': 'ssh_tunnel|rtsp_proxy|vpn|direct',
                    'ssh_host': 'gateway.example.com',
                    'ssh_port': 22,
                    'ssh_user': 'user',
                    'ssh_key_path': '/path/to/key',
                    'tunnel_local_port': 8554,
                    'proxy_host': 'proxy.example.com',
                    'proxy_port': 8554
                }
        """
        self.camera_id = camera_id
        self.original_rtsp_url = rtsp_url
        self.rtsp_url = rtsp_url  # Will be updated if using network bridge
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.network_config = network_config or {}
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        
        self.is_running = False
        self.is_connected = False
        self.capture_thread: Optional[threading.Thread] = None
        
        self.reconnect_count = 0
        self.last_frame_time = 0
        self.fps = 0
        self.frame_count = 0
        
        # Network bridge
        self.network_bridge = get_network_bridge()
        self._setup_network_access()
        
        # Status callbacks
        self.on_connected: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None
        self.on_error: Optional[Callable[[str], None]] = None
    
    def _setup_network_access(self):
        """Set up network access for camera (SSH tunnel, proxy, etc.)"""
        method = self.network_config.get('method', 'direct')
        
        if method == 'direct':
            # Use original URL
            self.rtsp_url = self.original_rtsp_url
            return
        
        try:
            access_method = NetworkAccessMethod(method)
            
            # Build configuration objects based on method
            ssh_config = None
            proxy_config = None
            
            if access_method == NetworkAccessMethod.SSH_TUNNEL:
                # Parse remote host and port from original RTSP URL
                # rtsp://10.50.60.21:8080/path -> remote_host=10.50.60.21, remote_port=8080
                url_parts = self.original_rtsp_url.replace('rtsp://', '').split('/')
                host_port = url_parts[0].split('@')[-1]  # Handle rtsp://user:pass@host:port
                remote_host = host_port.split(':')[0]
                remote_port = int(host_port.split(':')[1]) if ':' in host_port else 554
                
                ssh_config = SSHTunnelConfig(
                    ssh_host=self.network_config.get('ssh_host', ''),
                    ssh_port=self.network_config.get('ssh_port', 22),
                    ssh_user=self.network_config.get('ssh_user', ''),
                    ssh_key_path=self.network_config.get('ssh_key_path'),
                    local_port=self.network_config.get('tunnel_local_port', 8554),
                    remote_host=remote_host,
                    remote_port=remote_port
                )
            
            elif access_method == NetworkAccessMethod.RTSP_PROXY:
                proxy_config = ProxyConfig(
                    proxy_host=self.network_config.get('proxy_host', 'localhost'),
                    proxy_port=self.network_config.get('proxy_port', 8554),
                    proxy_path=self.network_config.get('proxy_path', '')
                )
            
            # Get accessible URL through network bridge
            self.rtsp_url = self.network_bridge.get_accessible_url(
                camera_id=self.camera_id,
                original_url=self.original_rtsp_url,
                access_method=access_method,
                ssh_config=ssh_config,
                proxy_config=proxy_config
            )
            
            print(f"🌉 Network bridge configured for camera {self.camera_id}")
            print(f"   Method: {method}")
            print(f"   Original: {self.original_rtsp_url}")
            print(f"   Accessible: {self.rtsp_url}")
            
        except Exception as e:
            print(f"⚠️ Failed to setup network access: {e}")
            print(f"   Falling back to direct connection")
            self.rtsp_url = self.original_rtsp_url
    
    def start(self):
        """Start capturing from the camera"""
        if self.is_running:
            print(f"⚠️ Camera {self.camera_id} is already running")
            return
        
        self.is_running = True
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"IPCamera-{self.camera_id}",
            daemon=True
        )
        self.capture_thread.start()
        print(f"✅ Started capture thread for camera {self.camera_id}")
    
    def stop(self):
        """Stop capturing from the camera"""
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)
        
        self._release_capture()
        print(f"🛑 Stopped capture for camera {self.camera_id}")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recent frame
        
        Returns:
            Latest frame as numpy array, or None if not available
        """
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
    
    def is_alive(self) -> bool:
        """Check if camera is connected and providing frames"""
        return self.is_connected and (time.time() - self.last_frame_time) < 5.0
    
    def _connect(self) -> bool:
        """
        Attempt to connect to the RTSP stream
        
        Returns:
            True if connected successfully
        """
        try:
            print(f"📡 Connecting to camera {self.camera_id}: {self.rtsp_url}")
            
            # OpenCV RTSP options for better compatibility
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            # Set buffer size to 1 to get latest frames (reduce latency)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set timeout for read operations (milliseconds)
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 seconds
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)  # 10 seconds
            
            if not self.cap.isOpened():
                print(f"❌ Failed to open RTSP stream for camera {self.camera_id}")
                return False
            
            # Try to read first frame to verify connection
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print(f"❌ Connected but failed to read frame from camera {self.camera_id}")
                self._release_capture()
                return False
            
            # Store first frame
            with self.frame_lock:
                self.frame = frame
            
            self.is_connected = True
            self.reconnect_count = 0
            self.last_frame_time = time.time()
            
            # Get camera properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            print(f"✅ Connected to camera {self.camera_id}: {width}x{height} @ {fps}fps")
            
            if self.on_connected:
                self.on_connected()
            
            return True
            
        except Exception as e:
            print(f"❌ Error connecting to camera {self.camera_id}: {e}")
            if self.on_error:
                self.on_error(str(e))
            return False
    
    def _release_capture(self):
        """Release the video capture object"""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                print(f"⚠️ Error releasing capture for camera {self.camera_id}: {e}")
            finally:
                self.cap = None
    
    def _capture_loop(self):
        """
        Main capture loop running in background thread
        
        Continuously reads frames from the camera and handles reconnection
        """
        print(f"🎥 Capture loop started for camera {self.camera_id}")
        
        while self.is_running:
            # Try to connect if not connected
            if not self.is_connected:
                if self.max_reconnect_attempts > 0 and self.reconnect_count >= self.max_reconnect_attempts:
                    print(f"❌ Max reconnection attempts reached for camera {self.camera_id}")
                    if self.on_error:
                        self.on_error("Max reconnection attempts reached")
                    break
                
                if self._connect():
                    continue
                else:
                    self.reconnect_count += 1
                    print(f"🔄 Reconnection attempt {self.reconnect_count} failed for camera {self.camera_id}")
                    time.sleep(self.reconnect_delay)
                    continue
            
            # Read frame
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    print(f"⚠️ Failed to read frame from camera {self.camera_id}")
                    self.is_connected = False
                    
                    if self.on_disconnected:
                        self.on_disconnected()
                    
                    self._release_capture()
                    continue
                
                # Update frame
                with self.frame_lock:
                    self.frame = frame
                
                self.frame_count += 1
                current_time = time.time()
                
                # Calculate FPS every second
                if current_time - self.last_frame_time >= 1.0:
                    elapsed = current_time - self.last_frame_time
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                
                self.last_frame_time = current_time
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)
                
            except Exception as e:
                print(f"❌ Error reading frame from camera {self.camera_id}: {e}")
                self.is_connected = False
                
                if self.on_error:
                    self.on_error(str(e))
                
                if self.on_disconnected:
                    self.on_disconnected()
                
                self._release_capture()
                time.sleep(self.reconnect_delay)
        
        # Cleanup on exit
        self._release_capture()
        print(f"🛑 Capture loop ended for camera {self.camera_id}")


class IPCameraManager:
    """Manages multiple IP camera captures"""
    
    def __init__(self):
        self.captures: dict[str, IPCameraCapture] = {}
    
    def add_camera(self, camera_id: str, rtsp_url: str, network_config: Optional[Dict[str, Any]] = None) -> IPCameraCapture:
        """
        Add and start a new IP camera capture
        
        Args:
            camera_id: Unique camera identifier
            rtsp_url: RTSP URL of the camera
            network_config: Optional network bridge configuration for cross-network cameras
        """
        if camera_id in self.captures:
            print(f"⚠️ Camera {camera_id} already exists, stopping old capture")
            self.remove_camera(camera_id)
        
        capture = IPCameraCapture(camera_id, rtsp_url, network_config=network_config)
        self.captures[camera_id] = capture
        capture.start()
        return capture
    
    def get_camera(self, camera_id: str) -> Optional[IPCameraCapture]:
        """Get camera capture by ID"""
        return self.captures.get(camera_id)
    
    def remove_camera(self, camera_id: str):
        """Stop and remove a camera capture"""
        capture = self.captures.get(camera_id)
        if capture:
            capture.stop()
            del self.captures[camera_id]
    
    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get latest frame from a camera"""
        capture = self.captures.get(camera_id)
        if capture:
            return capture.get_frame()
        return None
    
    def stop_all(self):
        """Stop all camera captures"""
        for camera_id in list(self.captures.keys()):
            self.remove_camera(camera_id)


# Global instance
_ip_camera_manager = None


def get_ip_camera_manager() -> IPCameraManager:
    """Get global IP camera manager instance"""
    global _ip_camera_manager
    if _ip_camera_manager is None:
        _ip_camera_manager = IPCameraManager()
    return _ip_camera_manager
