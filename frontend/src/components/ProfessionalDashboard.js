import React, { useState, useCallback } from "react";
import LiveCamera from "./LiveCamera";
import LiveFeed from "./LiveFeed";
import AnomalyTimeline from "./AnomalyTimeline";
import AlertFeed from "./AlertFeed";

export default function ProfessionalDashboard() {
  const [liveFrame, setLiveFrame] = useState(null);
  const [threatLevel, setThreatLevel] = useState("INFO");
  const [alerts, setAlerts] = useState([]);
  const [timeline, setTimeline] = useState([]);

  const onFrame = useCallback((base64, status) => {
    setLiveFrame({ base64, status });
  }, []);

  const onAnomaly = useCallback((fusion, meta) => {
    if (!fusion) return;
    setThreatLevel(meta?.threat_level || fusion.severity || "INFO");
    setAlerts((prev) =>
      [
        {
          anomaly_type: fusion.anomaly_type,
          severity: fusion.severity,
          fusion_score: fusion.fusion_score,
          confidence: fusion.confidence,
          explanation: fusion.explanation,
          timestamp: meta?.timestamp || new Date().toISOString(),
          title: fusion.anomaly_type,
          message: fusion.explanation,
        },
        ...prev,
      ].slice(0, 50)
    );
    setTimeline((prev) =>
      [
        {
          anomaly_type: fusion.anomaly_type,
          severity: fusion.severity,
          fusion_score: fusion.fusion_score,
          timestamp: meta?.timestamp || new Date().toISOString(),
        },
        ...prev,
      ].slice(0, 100)
    );
  }, []);

  return (
    <div className="space-y-4 sm:space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
        <div className="lg:col-span-2">
          <LiveFeed
            frameBase64={liveFrame?.base64}
            status={liveFrame?.status || "Ready"}
            threatLevel={threatLevel}
          />
        </div>
        <AnomalyTimeline timeline={timeline} />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
        <div className="lg:col-span-2 bg-gray-800 rounded-lg p-4 sm:p-6">
          <LiveCamera onAnomaly={onAnomaly} onFrame={onFrame} />
        </div>
        <AlertFeed anomalies={alerts} />
      </div>
    </div>
  );
}
