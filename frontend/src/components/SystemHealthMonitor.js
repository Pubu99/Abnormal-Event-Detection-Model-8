import React from "react";

export default function SystemHealthMonitor({ stats }) {
  const formatUptime = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours.toString().padStart(2, "0")}:${minutes
      .toString()
      .padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  const getStatusColor = () => {
    if (stats.fps >= 15) return "text-emerald-400";
    if (stats.fps >= 10) return "text-yellow-400";
    return "text-red-400";
  };

  return (
    <div className="bg-slate-900/50 backdrop-blur-sm border-b border-slate-800">
      <div className="px-4 sm:px-6 lg:px-8 py-3">
        <div className="flex flex-wrap items-center justify-between gap-4">
          {/* Left - System Info */}
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
              <span className="text-emerald-400 font-semibold text-sm">
                SYSTEM ONLINE
              </span>
            </div>
            <div className="text-slate-400 text-sm">
              Uptime:{" "}
              <span className="text-white font-mono">
                {formatUptime(stats.uptime)}
              </span>
            </div>
          </div>

          {/* Right - Performance Metrics */}
          <div className="flex items-center gap-6">
            <div className="text-slate-400 text-sm">
              FPS:{" "}
              <span className={`font-mono font-bold ${getStatusColor()}`}>
                {stats.fps.toFixed(1)}
              </span>
            </div>
            <div className="text-slate-400 text-sm">
              Cameras:{" "}
              <span className="text-cyan-400 font-semibold">
                {stats.camerasOnline}
              </span>
            </div>
            <div className="text-slate-400 text-sm">
              Detections:{" "}
              <span className="text-white font-semibold">
                {stats.totalDetections}
              </span>
            </div>
            <div className="text-slate-400 text-sm">
              Anomalies:{" "}
              <span className="text-red-400 font-semibold">
                {stats.anomalyCount}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
