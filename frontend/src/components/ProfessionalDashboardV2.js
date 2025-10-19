import React, { useState, useCallback, useEffect } from "react";
import LiveCameraV2 from "./LiveCameraV2";
import LiveFeedV2 from "./LiveFeedV2";
import AnomalyDetailsPanel from "./AnomalyDetailsPanel";
import AlertFeedV2 from "./AlertFeedV2";
import StatsPanel from "./StatsPanel";
import SystemHealthMonitor from "./SystemHealthMonitor";

export default function ProfessionalDashboardV2() {
  const [videoStream, setVideoStream] = useState(null); // NEW: Direct stream
  const [detectionData, setDetectionData] = useState(null); // NEW: Only detection data
  const [currentDetection, setCurrentDetection] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [timeline, setTimeline] = useState([]);
  const [systemStats, setSystemStats] = useState({
    totalDetections: 0,
    anomalyCount: 0,
    fps: 0,
    uptime: 0,
    camerasOnline: 1,
  });

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

    const detectionId = `DET-${Date.now()}-${Math.random()
      .toString(36)
      .substr(2, 9)}`;

    const detectionData = {
      id: detectionId,
      anomaly_type: fusion.anomaly_type,
      severity: fusion.severity,
      fusion_score: fusion.fusion_score,
      confidence: fusion.confidence,
      explanation: fusion.explanation,
      reasoning: fusion.reasoning || [],
      timestamp: meta?.timestamp || new Date().toISOString(),
      camera_id: "CAM-001",
      camera_name: "Primary Surveillance Camera",
      location: "Main Entrance",
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
  }, []);

  const onNormalFrame = useCallback(() => {
    setSystemStats((prev) => ({
      ...prev,
      totalDetections: prev.totalDetections + 1,
    }));
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Top Bar - System Status */}
      <SystemHealthMonitor stats={systemStats} />

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
            <LiveCameraV2
              onAnomaly={onAnomaly}
              onStreamReady={onStreamReady}
              onDetectionData={onDetectionData}
              onNormalFrame={onNormalFrame}
            />
          </div>

          {/* Right Column - Details & Alerts (40%) */}
          <div className="xl:col-span-5 space-y-4">
            {/* Current Detection Details */}
            <AnomalyDetailsPanel detection={currentDetection} />

            {/* Alert Feed */}
            <AlertFeedV2 alerts={alerts} />
          </div>
        </div>
      </div>
    </div>
  );
}
