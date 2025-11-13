import React, { useState, useEffect, useCallback } from 'react';
import MultiCameraGrid from './MultiCameraGrid';
import AlertFeedV2 from './AlertFeedV2';
import StatsPanel from './StatsPanel';
import SystemHealthMonitor from './SystemHealthMonitor';
import CameraManager from './CameraManager';
import '../styles/professional.css';

/**
 * Multi-Camera Dashboard Component
 * 
 * Displays all connected cameras in a grid with parallel real-time analysis.
 * Aggregates detections from all cameras into a unified alert feed.
 * 
 * Features:
 * - Auto-loading cameras from backend
 * - Parallel processing for all enabled cameras
 * - Unified alert system across all cameras
 * - Per-camera and aggregate statistics
 * - Camera management interface
 */
export default function MultiCameraDashboard() {
  const [cameras, setCameras] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [showCameraManager, setShowCameraManager] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [systemStats, setSystemStats] = useState({
    totalDetections: 0,
    anomalyCount: 0,
    totalFPS: 0,
    uptime: 0,
    camerasOnline: 0,
    camerasTotal: 0
  });

  const apiBase = process.env.REACT_APP_API_BASE || 'http://localhost:8000';

  // Load cameras on mount
  const loadCameras = useCallback(async () => {
    try {
      const response = await fetch(`${apiBase}/api/cameras`);
      const data = await response.json();
      
      // Handle different API response formats
      let camerasArray = [];
      if (Array.isArray(data)) {
        camerasArray = data;
      } else if (data && data.cameras && Array.isArray(data.cameras)) {
        camerasArray = data.cameras;
      } else if (data && data.success && Array.isArray(data.data)) {
        camerasArray = data.data;
      } else {
        console.warn('Unexpected camera data format:', data);
        camerasArray = [];
      }
      
      setCameras(camerasArray);
      setSystemStats(prev => ({
        ...prev,
        camerasTotal: camerasArray.length,
        camerasOnline: camerasArray.filter(c => c.enabled).length
      }));
    } catch (error) {
      console.error('Failed to load cameras:', error);
      setCameras([]); // Set empty array on error
    }
  }, [apiBase]);

  useEffect(() => {
    loadCameras();
    const interval = setInterval(loadCameras, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [loadCameras]);

  // Update uptime
  useEffect(() => {
    const startTime = Date.now();
    const interval = setInterval(() => {
      setSystemStats(prev => ({
        ...prev,
        uptime: Math.floor((Date.now() - startTime) / 1000)
      }));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Handle detection from any camera
  const handleDetection = useCallback((data) => {
    const { camera, detection, timestamp } = data;
    
    // Create alert
    const alert = {
      id: `ALERT-${camera.id}-${Date.now()}`,
      camera_id: camera.id,
      camera_name: camera.name,
      camera_location: camera.location,
      anomaly_type: detection.anomaly_type,
      severity: detection.severity,
      confidence: detection.confidence,
      explanation: detection.explanation,
      timestamp: new Date(timestamp).toISOString(),
      detection
    };
    
    // Add to alerts (keep last 100)
    setAlerts(prev => [alert, ...prev].slice(0, 100));
    
    // Update global stats
    setSystemStats(prev => ({
      ...prev,
      totalDetections: prev.totalDetections + 1,
      anomalyCount: prev.anomalyCount + 1
    }));
  }, []);

  // Handle camera selection
  const handleCameraSelect = useCallback((camera) => {
    console.log('Camera selected:', camera);
  }, []);

  // Handle feedback on alert
  const handleFeedback = async (alertId, feedback) => {
    const alert = alerts.find(a => a.id === alertId);
    if (!alert) return;

    try {
      const response = await fetch(`${apiBase}/api/apply-feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          detection_id: alert.detection.id,
          feedback,
          user: 'operator',
          camera_id: alert.camera_id
        })
      });

      if (response.ok) {
        // Update alert status
        setAlerts(prev =>
          prev.map(a =>
            a.id === alertId
              ? { ...a, feedback, feedbackTime: new Date().toISOString() }
              : a
          )
        );
      }
    } catch (error) {
      console.error('Error applying feedback:', error);
    }
  };

  // Decline all alerts
  const handleDeclineAll = async () => {
    const pendingAlerts = alerts.filter(a => !a.feedback);
    
    if (pendingAlerts.length === 0) {
      alert('No pending alerts to decline');
      return;
    }

    if (!window.confirm(`Decline all ${pendingAlerts.length} pending alerts?`)) {
      return;
    }

    try {
      const response = await fetch(`${apiBase}/api/decline-all`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user: 'operator',
          reason: 'Bulk decline'
        })
      });

      if (response.ok) {
        // Mark all as declined
        setAlerts(prev =>
          prev.map(a => ({
            ...a,
            feedback: 'decline',
            feedbackTime: new Date().toISOString()
          }))
        );
      }
    } catch (error) {
      console.error('Error declining all:', error);
    }
  };

  // Format uptime
  const formatUptime = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours}h ${minutes}m ${secs}s`;
  };

  const enabledCameras = cameras.filter(c => c.enabled);
  const pendingAlerts = alerts.filter(a => !a.feedback);

  return (
    <div className="multi-camera-dashboard" style={{ 
      height: '100vh', 
      display: 'flex', 
      flexDirection: 'column',
      background: '#0a0a0a',
      color: '#fff'
    }}>
      {/* Header */}
      <header style={{
        background: 'linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)',
        padding: '15px 30px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        borderBottom: '2px solid #00aaff',
        boxShadow: '0 2px 10px rgba(0,0,0,0.5)'
      }}>
        <div>
          <h1 style={{ margin: 0, fontSize: '24px', fontWeight: 'bold' }}>
            🎥 Multi-Camera Surveillance System
          </h1>
          <div style={{ fontSize: '12px', color: '#aaa', marginTop: '5px' }}>
            Real-time Parallel Analysis • {enabledCameras.length} Camera{enabledCameras.length !== 1 ? 's' : ''} Active
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          {/* System Stats */}
          <div style={{ textAlign: 'right', fontSize: '12px' }}>
            <div>Uptime: {formatUptime(systemStats.uptime)}</div>
            <div>Detections: {systemStats.totalDetections}</div>
            <div>Cameras: {systemStats.camerasOnline}/{systemStats.camerasTotal}</div>
          </div>

          {/* Start/Stop Analysis Button */}
          <button
            onClick={() => setIsAnalyzing(!isAnalyzing)}
            disabled={enabledCameras.length === 0}
            style={{
              background: isAnalyzing ? '#ff6600' : '#00cc66',
              color: '#fff',
              border: 'none',
              padding: '12px 24px',
              borderRadius: '5px',
              cursor: enabledCameras.length === 0 ? 'not-allowed' : 'pointer',
              fontSize: '16px',
              fontWeight: 'bold',
              opacity: enabledCameras.length === 0 ? 0.5 : 1,
              boxShadow: isAnalyzing ? '0 0 20px rgba(255,102,0,0.5)' : '0 0 20px rgba(0,204,102,0.5)',
              transition: 'all 0.3s ease'
            }}
          >
            {isAnalyzing ? '⏸️ Stop Anomaly Detection' : '▶️ Start Anomaly Detection'}
          </button>

          {/* Camera Manager Button */}
          <button
            onClick={() => setShowCameraManager(true)}
            style={{
              background: '#00aaff',
              color: '#fff',
              border: 'none',
              padding: '10px 20px',
              borderRadius: '5px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: 'bold'
            }}
          >
            ⚙️ Manage Cameras
          </button>

          {/* Decline All Button */}
          {pendingAlerts.length > 0 && (
            <button
              onClick={handleDeclineAll}
              style={{
                background: '#ff4444',
                color: '#fff',
                border: 'none',
                padding: '10px 20px',
                borderRadius: '5px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: 'bold'
              }}
            >
              🚫 Decline All ({pendingAlerts.length})
            </button>
          )}
        </div>
      </header>

      {/* Main Content */}
      <div style={{
        flex: 1,
        display: 'flex',
        overflow: 'hidden'
      }}>
        {/* Camera Grid (70%) */}
        <div style={{
          flex: '0 0 70%',
          display: 'flex',
          flexDirection: 'column',
          borderRight: '2px solid #333'
        }}>
          {enabledCameras.length > 0 ? (
            <MultiCameraGrid
              cameras={cameras}
              isAnalyzing={isAnalyzing}
              onCameraSelect={handleCameraSelect}
              onDetection={handleDetection}
            />
          ) : (
            <div style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '20px'
            }}>
              <div style={{ fontSize: '48px' }}>📹</div>
              <div style={{ fontSize: '20px', color: '#666' }}>No Cameras Configured</div>
              <button
                onClick={() => setShowCameraManager(true)}
                style={{
                  background: '#00aaff',
                  color: '#fff',
                  border: 'none',
                  padding: '15px 30px',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontSize: '16px',
                  fontWeight: 'bold'
                }}
              >
                ➕ Add Camera
              </button>
            </div>
          )}
        </div>

        {/* Right Panel (30%) */}
        <div style={{
          flex: '0 0 30%',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'auto',
          background: '#111'
        }}>
          {/* System Health Monitor */}
          <div style={{ borderBottom: '1px solid #333', padding: '15px' }}>
            <SystemHealthMonitor
              fps={systemStats.totalFPS}
              camerasOnline={systemStats.camerasOnline}
              totalDetections={systemStats.totalDetections}
            />
          </div>

          {/* Stats Panel */}
          <div style={{ borderBottom: '1px solid #333', padding: '15px' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '16px' }}>
              📊 Statistics
            </h3>
            <StatsPanel
              totalDetections={systemStats.totalDetections}
              anomalyCount={systemStats.anomalyCount}
              fps={systemStats.totalFPS}
            />
          </div>

          {/* Alert Feed */}
          <div style={{ flex: 1, padding: '15px', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '16px' }}>
              🔔 Alert Feed ({alerts.length})
            </h3>
            <AlertFeedV2
              alerts={alerts}
              onFeedback={handleFeedback}
            />
          </div>
        </div>
      </div>

      {/* Camera Manager Modal */}
      {showCameraManager && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            background: '#1a1a1a',
            borderRadius: '10px',
            maxWidth: '900px',
            maxHeight: '80vh',
            overflow: 'auto',
            boxShadow: '0 10px 50px rgba(0,0,0,0.5)'
          }}>
            <div style={{
              padding: '20px',
              borderBottom: '1px solid #333',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <h2 style={{ margin: 0 }}>Camera Management</h2>
              <button
                onClick={() => {
                  setShowCameraManager(false);
                  loadCameras(); // Reload cameras after closing
                }}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: '#fff',
                  fontSize: '24px',
                  cursor: 'pointer'
                }}
              >
                ✕
              </button>
            </div>
            <div style={{ padding: '20px' }}>
              <CameraManager onCamerasChanged={loadCameras} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
