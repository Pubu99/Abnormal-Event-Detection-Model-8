import React from "react";

export default function StatsPanel({ stats, currentDetection }) {
  const getThreatLevel = () => {
    if (!currentDetection) return { level: "SECURE", color: "emerald" };
    const severity = currentDetection.severity;

    const mapping = {
      CRITICAL: { level: "CRITICAL THREAT", color: "red" },
      HIGH: { level: "HIGH ALERT", color: "orange" },
      MEDIUM: { level: "WARNING", color: "yellow" },
      LOW: { level: "CAUTION", color: "blue" },
    };

    return mapping[severity] || { level: "SECURE", color: "emerald" };
  };

  const threat = getThreatLevel();

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 sm:gap-3 md:gap-4 mb-3 sm:mb-4">
      {/* Threat Level - Dynamic - User-Focused */}
      <div
        className={`bg-gradient-to-br from-${threat.color}-600/20 to-${threat.color}-900/20 border border-${threat.color}-500/50 rounded-lg sm:rounded-xl p-3 sm:p-4 md:p-5 shadow-xl backdrop-blur-sm`}
      >
        <div className="flex items-center justify-between">
          <div className="flex-1 min-w-0">
            <p
              className={`text-${threat.color}-300 text-[10px] sm:text-xs font-semibold uppercase tracking-wider truncate`}
            >
              Status
            </p>
            <p
              className={`text-${threat.color}-400 text-sm sm:text-xl md:text-2xl font-bold mt-1 sm:mt-2 truncate`}
            >
              {threat.level}
            </p>
          </div>
          <div
            className={`bg-${threat.color}-500/20 p-1.5 sm:p-2 md:p-3 rounded-full flex-shrink-0 ml-2`}
          >
            <svg
              className={`w-4 h-4 sm:w-6 sm:h-6 md:w-8 md:h-8 text-${threat.color}-400`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          </div>
        </div>
      </div>

      {/* Active Threats - What Users Care About */}
      <div className="bg-gradient-to-br from-red-600/20 to-red-900/20 border border-red-500/50 rounded-lg sm:rounded-xl p-3 sm:p-4 md:p-5 shadow-xl backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <div className="flex-1 min-w-0">
            <p className="text-red-300 text-[10px] sm:text-xs font-semibold uppercase tracking-wider truncate">
              Threats
            </p>
            <p className="text-red-400 text-sm sm:text-xl md:text-2xl font-bold mt-1 sm:mt-2">
              {stats.anomalyCount}
            </p>
            <p className="text-slate-400 text-[10px] sm:text-xs mt-0.5 sm:mt-1 truncate">
              Detected
            </p>
          </div>
          <div className="bg-red-500/20 p-1.5 sm:p-2 md:p-3 rounded-full flex-shrink-0 ml-2">
            <svg
              className="w-4 h-4 sm:w-6 sm:h-6 md:w-8 md:h-8 text-red-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
        </div>
      </div>

      {/* Cameras Online */}
      <div className="bg-gradient-to-br from-cyan-600/20 to-cyan-900/20 border border-cyan-500/50 rounded-lg sm:rounded-xl p-3 sm:p-4 md:p-5 shadow-xl backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <div className="flex-1 min-w-0">
            <p className="text-cyan-300 text-[10px] sm:text-xs font-semibold uppercase tracking-wider truncate">
              Cameras
            </p>
            <p className="text-cyan-400 text-sm sm:text-xl md:text-2xl font-bold mt-1 sm:mt-2">
              {stats.camerasOnline}
            </p>
            <p className="text-slate-400 text-[10px] sm:text-xs mt-0.5 sm:mt-1 truncate">
              Online
            </p>
          </div>
          <div className="bg-cyan-500/20 p-1.5 sm:p-2 md:p-3 rounded-full flex-shrink-0 ml-2">
            <svg
              className="w-4 h-4 sm:w-6 sm:h-6 md:w-8 md:h-8 text-cyan-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
          </div>
        </div>
      </div>

      {/* Processing Speed */}
      <div className="bg-gradient-to-br from-purple-600/20 to-purple-900/20 border border-purple-500/50 rounded-lg sm:rounded-xl p-3 sm:p-4 md:p-5 shadow-xl backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <div className="flex-1 min-w-0">
            <p className="text-purple-300 text-[10px] sm:text-xs font-semibold uppercase tracking-wider truncate">
              Speed
            </p>
            <p className="text-purple-400 text-sm sm:text-xl md:text-2xl font-bold mt-1 sm:mt-2">
              {stats.fps.toFixed(1)}
            </p>
            <p className="text-slate-400 text-[10px] sm:text-xs mt-0.5 sm:mt-1 truncate">
              FPS
            </p>
          </div>
          <div className="bg-purple-500/20 p-1.5 sm:p-2 md:p-3 rounded-full flex-shrink-0 ml-2">
            <svg
              className="w-4 h-4 sm:w-6 sm:h-6 md:w-8 md:h-8 text-purple-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}
