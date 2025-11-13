import React, { useState, useEffect } from "react";

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

export default function AlertFeedV2({ alerts = [], onAlertsChange }) {
  const [localAlerts, setLocalAlerts] = useState(alerts || []);
  const [activeCount, setActiveCount] = useState((alerts || []).length);
  const [recentCount, setRecentCount] = useState(Math.min(10, (alerts || []).length));
  const [confirmedTotal, setConfirmedTotal] = useState(0);

  useEffect(() => {
    setLocalAlerts(alerts || []);
    setActiveCount((alerts || []).length);
    setRecentCount(Math.min(10, (alerts || []).length));
  }, [alerts]);
  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = Math.floor((now - date) / 1000);

    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return date.toLocaleString();
  };

  // API base for feedback endpoints (override with REACT_APP_API_BASE)
  const apiBase = process.env.REACT_APP_API_BASE || "http://localhost:8000";

  const resolveDetectionId = (alertObj) => {
    // Prefer server-provided detection ids embedded in metadata/fusion
    if (alertObj && alertObj.metadata && alertObj.metadata.detection_id) {
      return alertObj.metadata.detection_id;
    }

    if (alertObj && alertObj.detection_id) return alertObj.detection_id;

    if (alertObj && alertObj.meta && alertObj.meta.data && alertObj.meta.data.fusion && alertObj.meta.data.fusion.metadata && alertObj.meta.data.fusion.metadata.detection_id) {
      return alertObj.meta.data.fusion.metadata.detection_id;
    }

    if (alertObj && alertObj.fusion && alertObj.fusion.metadata && alertObj.fusion.metadata.detection_id) {
      return alertObj.fusion.metadata.detection_id;
    }

    // Fallback to frontend-generated id (may not map to backend)
    return alertObj.id || null;
  };

  const sendFeedback = async (alertObj, feedback, comment) => {
  let id = resolveDetectionId(alertObj);
    if (!id) {
      window.alert("Cannot determine detection id for this alert.");
      return;
    }

    const confirmMsg = `Are you sure you want to mark this alert as "${feedback}"? This action will be recorded.`;
    if (!window.confirm(confirmMsg)) return;

    // helper to POST feedback for a given id
    const trySend = async (idToSend) => {
      const res = await fetch(`${apiBase}/api/detections/${encodeURIComponent(idToSend)}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ feedback, comment: comment || "", user: "operator" }),
      });
      return res;
    };

    try {
      let res = await trySend(id);

      // If detection id not found on server, try normalized variants (hyphen/underscore mismatch)
      if (!res.ok && res.status === 404) {
        const alt1 = id.includes("-") ? id.replace(/-/g, "_") : null;
        const alt2 = id.includes("_") ? id.replace(/_/g, "-") : null;
        const tried = new Set([id]);
        if (alt1) tried.add(alt1);
        if (alt2) tried.add(alt2);

        // Attempt alternate forms until one succeeds
        for (const alt of [alt1, alt2]) {
          if (!alt || tried.has(alt)) continue;
          try {
            res = await trySend(alt);
            if (res.ok) {
              // use the canonical id that worked for local state updates
              id = alt; // eslint-disable-line no-param-reassign
              break;
            }
          } catch (e) {
            // ignore and continue
          }
        }
      }

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || "Failed to send feedback");
      }

      // Update local state to reflect feedback without full reload
      const idToMatch = id;
      // Parse server response for counts/orphan flag
      let respJson = {};
      try {
        respJson = await res.json();
      } catch (e) {
        // ignore
      }

      // If server indicated orphan (not matched), notify operator but still remove locally
      if (respJson.orphan) {
        console.warn('Feedback accepted as orphan (detection id not found on server).');
      }

      // Remove the alert from the local list
      setLocalAlerts((prev) => {
        const next = prev.filter((a) => resolveDetectionId(a) !== idToMatch);
        // update derived counts
        setActiveCount(next.length);
        setRecentCount(Math.min(10, next.length));
        
        // Notify parent to sync its state if callback provided
        if (onAlertsChange) {
          onAlertsChange(next);
        }
        
        return next;
      });

      // If server provided authoritative counts, prefer those
      if (respJson && typeof respJson.active_notifications === 'number') {
        setActiveCount(respJson.active_notifications);
        setRecentCount(Math.min(10, respJson.active_notifications));
      }
      if (respJson && typeof respJson.confirmed_total === 'number') {
        setConfirmedTotal(respJson.confirmed_total);
      }
    } catch (e) {
      console.error(e);
      window.alert("Failed to send feedback: " + e.message);
    }
  };

  const declineAll = async () => {
    const msg = "Are you sure you want to DECLINE ALL alerts shown? This will mark them as declined on the server.";
    if (!window.confirm(msg)) return;

    try {
      const res = await fetch(`${apiBase}/api/detections/decline-all`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirm: true, comment: "Bulk decline from UI", user: "operator" }),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || "Failed to decline all");
      }

      // Prefer authoritative counts from server response
      let rj = {};
      try {
        rj = await res.json();
      } catch (e) {
        // ignore
      }

      // Clear local alerts and update counts
      setLocalAlerts([]);
      
      // Notify parent to clear its alerts state too
      if (onAlertsChange) {
        onAlertsChange([]);
      }
      
      if (rj && typeof rj.active_notifications === 'number') {
        setActiveCount(rj.active_notifications);
        setRecentCount(Math.min(10, rj.active_notifications));
      } else {
        setActiveCount(0);
        setRecentCount(0);
      }
      if (rj && typeof rj.confirmed_total === 'number') setConfirmedTotal(rj.confirmed_total);
    } catch (e) {
      console.error(e);
      window.alert("Failed to decline all: " + e.message);
    }
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

  if (!localAlerts.length) {
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

  const recentAlerts = localAlerts.slice(0, 10);
  const criticalCount = localAlerts.filter((a) => a.severity === "CRITICAL").length;
  const highCount = localAlerts.filter((a) => a.severity === "HIGH").length;

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
            <button
              onClick={declineAll}
              className="ml-2 text-red-400 hover:text-red-300 text-xs sm:text-sm font-semibold"
              title="Decline all visible alerts"
            >
              Decline All
            </button>
          </div>
        </div>
          <div className="flex items-center gap-2 sm:gap-4 text-[10px] sm:text-xs text-slate-400">
          <span>Total: {activeCount}</span>
          <span>•</span>
          <span>Recent: {recentCount}</span>
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

                {/* Feedback actions */}
                <div className="mt-3 flex items-center gap-2">
                  <button
                    onClick={() => sendFeedback(alert, "confirm")}
                    className="px-2 py-1 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-xs font-semibold"
                  >
                    Confirm
                  </button>
                  <button
                    onClick={() => sendFeedback(alert, "not_anomaly")}
                    className="px-2 py-1 bg-yellow-600 hover:bg-yellow-500 text-white rounded text-xs font-semibold"
                  >
                    Not Anomaly
                  </button>
                  <button
                    onClick={() => sendFeedback(alert, "decline")}
                    className="px-2 py-1 bg-red-600 hover:bg-red-500 text-white rounded text-xs font-semibold"
                  >
                    Decline
                  </button>
                </div>
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
