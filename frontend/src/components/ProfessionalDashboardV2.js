import React, { useState, useCallback, useEffect } from "react";
import LiveCameraV2 from "./LiveCameraV2";
import LiveFeedV2 from "./LiveFeedV2";
import AnomalyDetailsPanel from "./AnomalyDetailsPanel";
import AlertFeedV2 from "./AlertFeedV2";
import StatsPanel from "./StatsPanel";
import SystemHealthMonitor from "./SystemHealthMonitor";
import CameraManager from "./CameraManager";

export default function ProfessionalDashboardV2() {
  const [videoStream, setVideoStream] = useState(null); // NEW: Direct stream
  const [detectionData, setDetectionData] = useState(null); // NEW: Only detection data
  const [currentDetection, setCurrentDetection] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [timeline, setTimeline] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [showCameraManager, setShowCameraManager] = useState(false);
  const [systemStats, setSystemStats] = useState({
    totalDetections: 0,
    anomalyCount: 0,
    fps: 0,
    uptime: 0,
    camerasOnline: 0,
  });

  const apiBase = process.env.REACT_APP_API_BASE || "http://localhost:8000";

  // Load cameras on mount
  useEffect(() => {
    loadCameras();
  }, []);

  // Update uptime every second
  useEffect(() => {
    const startTime = Date.now();
    const interval = setInterval(() => {
      setSystemStats((prev) => ({
        ...prev,
        uptime: Math.floor((Date.now() - startTime) / 1000),
      }));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Auto-select first enabled camera
  useEffect(() => {
    if (!selectedCamera && cameras.length > 0) {
      const firstEnabled = cameras.find(c => c.enabled);
      if (firstEnabled) {
        setSelectedCamera(firstEnabled);
      }
    }
  }, [cameras, selectedCamera]);

  const loadCameras = async () => {
    try {
      const response = await fetch(`${apiBase}/api/cameras`);
      const data = await response.json();
      if (data.success) {
        setCameras(data.cameras);
        const onlineCount = data.cameras.filter(c => c.status === "online").length;
        setSystemStats(prev => ({ ...prev, camerasOnline: onlineCount }));
      }
    } catch (error) {
      console.error("Error loading cameras:", error);
    }
  };

  const onStreamReady = useCallback((stream) => {
    setVideoStream(stream);
  }, []);

  const onDetectionData = useCallback((data) => {
    setDetectionData(data);
  }, []);

  const onAnomaly = useCallback((fusion, meta) => {
    if (!fusion) {
      setCurrentDetection(null);
      return;
    }
    // Prefer server-provided detection id when available to enable server-side feedback
    const serverId = meta?.data?.fusion?.metadata?.detection_id || fusion?.metadata?.detection_id;
    const detectionId = serverId || `DET-${Date.now()}-${Math.random()
      .toString(36)
      .substr(2, 9)}`;

    // Use camera info from WebSocket response or selected camera
    const cameraId = meta?.camera_id || selectedCamera?.id || "unknown";
    const cameraName = meta?.camera_name || selectedCamera?.name || "Unknown Camera";
    const cameraLocation = selectedCamera?.location || "Unknown Location";

    const detectionData = {
      id: detectionId,
      anomaly_type: fusion.anomaly_type,
      severity: fusion.severity,
      fusion_score: fusion.fusion_score,
      confidence: fusion.confidence,
      explanation: fusion.explanation,
      reasoning: fusion.reasoning || [],
      timestamp: meta?.timestamp || new Date().toISOString(),
      camera_id: cameraId,
      camera_name: cameraName,
      location: cameraLocation,
      frame_number: meta?.frame_number || 0,
      score_breakdown: fusion.score_breakdown || {},
      detected_objects: meta?.data?.yolo?.objects_detected || [],
      ml_prediction: meta?.data?.ml_model || {},
      motion_data: meta?.data?.motion || {},
      pose_data: meta?.data?.pose || {},
    };

    setCurrentDetection(detectionData);

    // Add to alerts
    setAlerts((prev) => [detectionData, ...prev].slice(0, 100));

    // Update stats
    setSystemStats((prev) => ({
      ...prev,
      totalDetections: prev.totalDetections + 1,
      anomalyCount: prev.anomalyCount + 1,
    }));

    // Add to timeline
    setTimeline((prev) =>
      [
        {
          id: detectionId,
          anomaly_type: fusion.anomaly_type,
          severity: fusion.severity,
          fusion_score: fusion.fusion_score,
          timestamp: detectionData.timestamp,
        },
        ...prev,
      ].slice(0, 200)
    );
  }, [selectedCamera]);

  const onNormalFrame = useCallback(() => {
    setSystemStats((prev) => ({
      ...prev,
      totalDetections: prev.totalDetections + 1,
    }));
  }, []);

  // Callback for AlertFeedV2 to sync alerts state when user takes action
  const handleAlertsChange = useCallback((updatedAlerts) => {
    setAlerts(updatedAlerts);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Top Bar - System Status */}
      <div className="flex items-center justify-between">
        <SystemHealthMonitor stats={systemStats} />
        <div className="px-4">
          <button
            onClick={() => setShowCameraManager(true)}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            <span>📹</span>
            <span>Manage Cameras</span>
          </button>
        </div>
      </div>

      {/* Camera Selector */}
      {cameras.length > 0 && (
        <div className="px-4 sm:px-6 lg:px-8 pt-4">
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-4 border border-slate-700/50">
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Select Camera
            </label>
            <select
              value={selectedCamera?.id || ""}
              onChange={(e) => {
                const camera = cameras.find(c => c.id === e.target.value);
                setSelectedCamera(camera);
              }}
              className="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {cameras.filter(c => c.enabled).map(camera => (
                <option key={camera.id} value={camera.id}>
                  {camera.type === "webcam" ? "💻" : "📹"} {camera.name} - {camera.location}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}

      {/* Stats Overview */}
      <div className="px-4 sm:px-6 lg:px-8 pt-4">
        <StatsPanel stats={systemStats} currentDetection={currentDetection} />
      </div>

      {/* Main Content Area */}
      <div className="px-4 sm:px-6 lg:px-8 py-4">
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-4">
          {/* Left Column - Live Feed (60%) */}
          <div className="xl:col-span-7 space-y-4">
            <LiveFeedV2
              videoStream={videoStream}
              detectionData={detectionData}
              status={videoStream ? "Live" : "Waiting for camera..."}
              currentDetection={currentDetection}
              timeline={timeline}
            />

            {/* Camera Controls */}
            {selectedCamera ? (
              <LiveCameraV2
                camera={selectedCamera}
                onAnomaly={onAnomaly}
                onStreamReady={onStreamReady}
                onDetectionData={onDetectionData}
                onNormalFrame={onNormalFrame}
              />
            ) : (
              <div className="bg-slate-800/50 backdrop-blur-sm rounded-lg p-8 border border-slate-700/50 text-center">
                <p className="text-slate-400">No camera selected. Please add and select a camera.</p>
              </div>
            )}
          </div>

          {/* Right Column - Details & Alerts (40%) */}
          <div className="xl:col-span-5 space-y-4">
            {/* Current Detection Details */}
            <AnomalyDetailsPanel detection={currentDetection} />

            {/* Alert Feed */}
            <AlertFeedV2 alerts={alerts} onAlertsChange={handleAlertsChange} />
          </div>
        </div>
      </div>

      {/* Camera Manager Modal */}
      {showCameraManager && (
        <CameraManager
          onClose={() => {
            setShowCameraManager(false);
            loadCameras(); // Reload cameras after closing manager
          }}
        />
      )}
    </div>
  );
}
