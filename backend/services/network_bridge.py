"""
Network Bridge Service for Cross-Network RTSP Access

Supports multiple methods to access cameras on different networks:
1. Direct connection (if routed)
2. SSH tunnel forwarding
3. RTSP proxy/relay
4. VPN connection
"""

import subprocess
import socket
import threading
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class NetworkAccessMethod(str, Enum):
    DIRECT = "direct"  # Direct RTSP connection (default)
    SSH_TUNNEL = "ssh_tunnel"  # SSH port forwarding
    RTSP_PROXY = "rtsp_proxy"  # RTSP proxy/relay server
    VPN = "vpn"  # VPN connection


@dataclass
class SSHTunnelConfig:
    """SSH tunnel configuration for remote camera access"""
    ssh_host: str  # SSH server IP/hostname
    ssh_port: int = 22
    ssh_user: str = ""
    ssh_key_path: Optional[str] = None  # Path to SSH private key
    ssh_password: Optional[str] = None  # SSH password (less secure)
    local_port: int = 8554  # Local port to forward to
    remote_host: str = ""  # Camera IP on remote network
    remote_port: int = 554  # Camera RTSP port


@dataclass
class ProxyConfig:
    """RTSP proxy configuration"""
    proxy_host: str = "localhost"
    proxy_port: int = 8554
    proxy_path: str = ""  # Path on proxy server


class NetworkBridge:
    """Manages network access to cameras on different networks"""
    
    def __init__(self):
        self.active_tunnels: Dict[str, subprocess.Popen] = {}
        self.tunnel_threads: Dict[str, threading.Thread] = {}
        
    def get_accessible_url(
        self,
        camera_id: str,
        original_url: str,
        access_method: NetworkAccessMethod = NetworkAccessMethod.DIRECT,
        ssh_config: Optional[SSHTunnelConfig] = None,
        proxy_config: Optional[ProxyConfig] = None
    ) -> str:
        """
        Get an accessible RTSP URL based on network access method
        
        Args:
            camera_id: Unique camera identifier
            original_url: Original RTSP URL (e.g., rtsp://10.50.60.21:8080/h264_ulaw.sdp)
            access_method: Network access method to use
            ssh_config: SSH tunnel configuration (if using SSH_TUNNEL)
            proxy_config: Proxy configuration (if using RTSP_PROXY)
            
        Returns:
            Accessible RTSP URL
        """
        if access_method == NetworkAccessMethod.DIRECT:
            return original_url
            
        elif access_method == NetworkAccessMethod.SSH_TUNNEL:
            if not ssh_config:
                raise ValueError("SSH tunnel requires ssh_config")
            return self._setup_ssh_tunnel(camera_id, original_url, ssh_config)
            
        elif access_method == NetworkAccessMethod.RTSP_PROXY:
            if not proxy_config:
                raise ValueError("RTSP proxy requires proxy_config")
            return self._build_proxy_url(original_url, proxy_config)
            
        elif access_method == NetworkAccessMethod.VPN:
            # VPN assumed to be already connected, return original URL
            return original_url
            
        return original_url
    
    def _setup_ssh_tunnel(
        self,
        camera_id: str,
        original_url: str,
        config: SSHTunnelConfig
    ) -> str:
        """
        Set up SSH tunnel for camera access
        
        Creates SSH tunnel: localhost:local_port -> ssh_host -> remote_host:remote_port
        Returns: rtsp://localhost:local_port/path
        """
        # Check if tunnel already exists
        if camera_id in self.active_tunnels:
            process = self.active_tunnels[camera_id]
            if process.poll() is None:  # Still running
                return self._build_tunneled_url(original_url, config.local_port)
        
        # Parse path from original URL
        # rtsp://10.50.60.21:8080/h264_ulaw.sdp -> /h264_ulaw.sdp
        path = original_url.split('/', 3)[-1] if original_url.count('/') >= 3 else ""
        
        # Build SSH command
        ssh_cmd = [
            "ssh",
            "-N",  # No remote command
            "-L", f"{config.local_port}:{config.remote_host}:{config.remote_port}",
            "-p", str(config.ssh_port),
        ]
        
        if config.ssh_key_path:
            ssh_cmd.extend(["-i", config.ssh_key_path])
        
        ssh_cmd.append(f"{config.ssh_user}@{config.ssh_host}")
        
        try:
            # Start SSH tunnel process
            process = subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )
            
            self.active_tunnels[camera_id] = process
            
            # Wait a bit for tunnel to establish
            time.sleep(2)
            
            # Check if tunnel is working
            if process.poll() is not None:
                # Process died
                stdout, stderr = process.communicate()
                raise Exception(f"SSH tunnel failed: {stderr.decode()}")
            
            return self._build_tunneled_url(original_url, config.local_port)
            
        except Exception as e:
            raise Exception(f"Failed to create SSH tunnel: {str(e)}")
    
    def _build_tunneled_url(self, original_url: str, local_port: int) -> str:
        """Build localhost URL for tunneled connection"""
        # Extract path from original URL
        parts = original_url.split('/', 3)
        path = f"/{parts[3]}" if len(parts) > 3 else ""
        
        # Extract credentials if present
        if '@' in original_url:
            # rtsp://user:pass@host:port/path -> rtsp://user:pass@localhost:port/path
            creds = original_url.split('//')[1].split('@')[0]
            return f"rtsp://{creds}@localhost:{local_port}{path}"
        
        return f"rtsp://localhost:{local_port}{path}"
    
    def _build_proxy_url(self, original_url: str, config: ProxyConfig) -> str:
        """Build URL for RTSP proxy"""
        # Extract credentials from original URL
        creds = ""
        if '@' in original_url:
            creds = original_url.split('//')[1].split('@')[0] + "@"
        
        # Build proxy URL
        path = config.proxy_path if config.proxy_path else ""
        return f"rtsp://{creds}{config.proxy_host}:{config.proxy_port}{path}"
    
    def close_tunnel(self, camera_id: str):
        """Close SSH tunnel for camera"""
        if camera_id in self.active_tunnels:
            process = self.active_tunnels[camera_id]
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            del self.active_tunnels[camera_id]
    
    def close_all_tunnels(self):
        """Close all active SSH tunnels"""
        for camera_id in list(self.active_tunnels.keys()):
            self.close_tunnel(camera_id)
    
    def test_connection(self, host: str, port: int, timeout: int = 5) -> bool:
        """Test if host:port is reachable"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False


# Singleton instance
_network_bridge = None

def get_network_bridge() -> NetworkBridge:
    """Get singleton NetworkBridge instance"""
    global _network_bridge
    if _network_bridge is None:
        _network_bridge = NetworkBridge()
    return _network_bridge
