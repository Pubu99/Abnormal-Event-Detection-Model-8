import React from "react";

const getSeverityColor = (severity) => {
  const colors = {
    CRITICAL: {
      bg: "bg-red-500/10",
      border: "border-red-500/50",
      text: "text-red-400",
      badge: "bg-red-500",
    },
    HIGH: {
      bg: "bg-orange-500/10",
      border: "border-orange-500/50",
      text: "text-orange-400",
      badge: "bg-orange-500",
    },
    MEDIUM: {
      bg: "bg-yellow-500/10",
      border: "border-yellow-500/50",
      text: "text-yellow-400",
      badge: "bg-yellow-500",
    },
    LOW: {
      bg: "bg-blue-500/10",
      border: "border-blue-500/50",
      text: "text-blue-400",
      badge: "bg-blue-500",
    },
  };
  return colors[severity] || colors.LOW;
};

export default function AlertFeedV2({ alerts = [] }) {
  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = Math.floor((now - date) / 1000);

    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return date.toLocaleString();
  };

  // ⭐ Filter dangerous objects - User Requirement #5 ⭐
  const getDangerousObjects = (objects) => {
    if (!objects || !Array.isArray(objects)) return [];

    const dangerousKeywords = [
      "gun",
      "knife",
      "weapon",
      "pistol",
      "rifle",
      "blade",
      "firearm",
    ];
    const dangerous = [];
    const others = [];

    objects.forEach((obj) => {
      const objName = (obj.class || obj).toLowerCase();
      const isDangerous = dangerousKeywords.some((keyword) =>
        objName.includes(keyword)
      );

      if (isDangerous) {
        dangerous.push(obj);
      } else {
        others.push(obj);
      }
    });

    // Prioritize: dangerous first, then others (max 3 total)
    return [...dangerous, ...others].slice(0, 5);
  };

  if (!alerts.length) {
    return (
      <div className="bg-slate-900/50 backdrop-blur-sm rounded-lg sm:rounded-xl border border-slate-800 shadow-xl">
        <div className="px-3 sm:px-4 md:px-6 py-3 sm:py-4 border-b border-slate-800">
          <div className="flex items-center justify-between">
            <h3 className="text-white font-bold text-sm sm:text-base md:text-lg">
              THREAT ALERTS
            </h3>
            <span className="px-2 sm:px-3 py-0.5 sm:py-1 bg-emerald-500/20 border border-emerald-500/50 text-emerald-400 rounded-full text-[10px] sm:text-xs font-bold">
              ✓ SECURE
            </span>
          </div>
        </div>
        <div className="p-6 sm:p-8 md:p-12 text-center">
          <svg
            className="w-12 h-12 sm:w-14 sm:h-14 md:w-16 md:h-16 text-emerald-500 mx-auto mb-3 sm:mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
            />
          </svg>
          <p className="text-slate-400 text-xs sm:text-sm">
            No threats detected
          </p>
          <p className="text-slate-600 text-[10px] sm:text-xs mt-1">
            Monitoring active
          </p>
        </div>
      </div>
    );
  }

  const recentAlerts = alerts.slice(0, 10);
  const criticalCount = alerts.filter((a) => a.severity === "CRITICAL").length;
  const highCount = alerts.filter((a) => a.severity === "HIGH").length;

  return (
    <div className="bg-slate-900/50 backdrop-blur-sm rounded-lg sm:rounded-xl border border-slate-800 shadow-xl">
      {/* Header - Responsive */}
      <div className="px-3 sm:px-4 md:px-6 py-3 sm:py-4 border-b border-slate-800">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 sm:gap-0 mb-2 sm:mb-3">
          <h3 className="text-white font-bold text-sm sm:text-base md:text-lg">
            THREAT ALERTS
          </h3>
          <div className="flex items-center gap-1 sm:gap-2 flex-wrap">
            {criticalCount > 0 && (
              <span className="px-1.5 sm:px-2 py-0.5 sm:py-1 bg-red-500/20 border border-red-500/50 text-red-400 rounded text-[10px] sm:text-xs font-bold">
                🔴 {criticalCount}
              </span>
            )}
            {highCount > 0 && (
              <span className="px-1.5 sm:px-2 py-0.5 sm:py-1 bg-orange-500/20 border border-orange-500/50 text-orange-400 rounded text-[10px] sm:text-xs font-bold">
                ⚠️ {highCount}
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2 sm:gap-4 text-[10px] sm:text-xs text-slate-400">
          <span>Total: {alerts.length}</span>
          <span>•</span>
          <span>Recent: {Math.min(10, alerts.length)}</span>
        </div>
      </div>

      {/* Alert List - Responsive */}
      <div className="max-h-[400px] sm:max-h-[500px] md:max-h-[600px] overflow-y-auto custom-scrollbar">
        <div className="p-2 sm:p-3 md:p-4 space-y-2 sm:space-y-3">
          {recentAlerts.map((alert, idx) => {
            const colors = getSeverityColor(alert.severity);
            const prioritizedObjects = getDangerousObjects(
              alert.detected_objects
            );

            return (
              <div
                key={alert.id || idx}
                className={`${colors.bg} border ${colors.border} rounded-lg p-2 sm:p-3 md:p-4 transition-all duration-200 hover:scale-[1.01] cursor-pointer`}
              >
                {/* Header - Responsive with Navigation */}
                <div className="flex items-start justify-between mb-1 sm:mb-2 gap-2">
                  <div className="flex items-center gap-1 sm:gap-2 flex-1 min-w-0">
                    <span
                      className={`w-1.5 h-1.5 sm:w-2 sm:h-2 ${colors.badge} rounded-full animate-pulse flex-shrink-0`}
                    ></span>
                    <span
                      className={`${colors.text} font-bold text-xs sm:text-sm truncate`}
                    >
                      {alert.anomaly_type || alert.title}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <span className="text-slate-500 text-[9px] sm:text-[10px] md:text-xs whitespace-nowrap">
                      {formatTimestamp(alert.timestamp)}
                    </span>
                    {/* Navigation Icon for Details */}
                    <svg
                      className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-400 hover:text-cyan-300 cursor-pointer transition-colors"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      title="View Full Details"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M13 9l3 3m0 0l-3 3m3-3H8m13 0a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                  </div>
                </div>

                {/* ⭐ WHY DETECTED - Brief Description ⭐ */}
                {alert.reasoning && alert.reasoning.length > 0 ? (
                  <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-2 mb-2 sm:mb-3">
                    <div className="flex items-center gap-1 mb-1">
                      <svg
                        className="w-3 h-3 sm:w-4 sm:h-4 text-red-400 flex-shrink-0"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                          clipRule="evenodd"
                        />
                      </svg>
                      <span className="text-red-300 text-[9px] sm:text-[10px] md:text-xs font-bold">
                        WHY DETECTED:
                      </span>
                    </div>
                    <p className="text-red-200 text-[9px] sm:text-[10px] md:text-xs leading-relaxed">
                      {alert.reasoning[0]}
                      {alert.reasoning.length > 1 && (
                        <span className="text-red-400 ml-1">
                          (+{alert.reasoning.length - 1} more)
                        </span>
                      )}
                    </p>
                  </div>
                ) : (
                  <p className="text-slate-300 text-[10px] sm:text-xs md:text-sm mb-2 sm:mb-3 leading-relaxed line-clamp-2">
                    {alert.explanation || alert.message}
                  </p>
                )}

                {/* Metadata Grid - Responsive */}
                <div className="grid grid-cols-2 gap-1.5 sm:gap-2 md:gap-3 mb-2 sm:mb-3">
                  <div className="bg-slate-800/50 rounded px-2 sm:px-3 py-1 sm:py-2">
                    <div className="text-slate-500 text-[9px] sm:text-[10px] md:text-xs">
                      Camera
                    </div>
                    <div className="text-white text-[10px] sm:text-xs md:text-sm font-semibold truncate">
                      {alert.camera_id || "CAM-001"}
                    </div>
                  </div>
                  <div className="bg-slate-800/50 rounded px-2 sm:px-3 py-1 sm:py-2">
                    <div className="text-slate-500 text-[9px] sm:text-[10px] md:text-xs">
                      Location
                    </div>
                    <div className="text-white text-[10px] sm:text-xs md:text-sm font-semibold truncate">
                      {alert.location || "Unknown"}
                    </div>
                  </div>
                </div>

                {/* Confidence */}
                <div className="flex items-center gap-2 sm:gap-4 text-[9px] sm:text-[10px] md:text-xs mb-2">
                  <div className="flex items-center gap-1">
                    <span className="text-slate-400">Confidence:</span>
                    <span className={colors.text + " font-bold"}>
                      {((alert.confidence || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* ⭐ PRIORITIZED DANGEROUS OBJECTS ⭐ */}
                {prioritizedObjects.length > 0 && (
                  <div className="mt-2 sm:mt-3 pt-2 sm:pt-3 border-t border-slate-700/50">
                    <div className="text-slate-500 text-[9px] sm:text-[10px] md:text-xs mb-1 sm:mb-2">
                      Objects Detected:
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {prioritizedObjects.map((obj, i) => {
                        const objName = (obj.class || obj).toLowerCase();
                        const isDangerous = [
                          "gun",
                          "knife",
                          "weapon",
                          "pistol",
                          "rifle",
                          "blade",
                        ].some((danger) => objName.includes(danger));

                        return (
                          <span
                            key={i}
                            className={`px-1.5 sm:px-2 py-0.5 sm:py-1 rounded text-[9px] sm:text-[10px] md:text-xs font-semibold ${
                              isDangerous
                                ? "bg-red-500/30 border border-red-500/70 text-red-300 animate-pulse"
                                : "bg-slate-700/50 text-slate-300"
                            }`}
                          >
                            {isDangerous ? "⚠️ " : ""}
                            {obj.class || obj}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer - Responsive */}
      {alerts.length > 10 && (
        <div className="px-3 sm:px-4 md:px-6 py-2 sm:py-3 border-t border-slate-800 text-center">
          <button className="text-cyan-400 hover:text-cyan-300 text-xs sm:text-sm font-semibold transition-colors">
            View All {alerts.length} Alerts →
          </button>
        </div>
      )}
    </div>
  );
}
