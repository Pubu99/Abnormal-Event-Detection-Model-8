import React, { useState } from "react";

export default function AnomalyDetailsPanel({ detection }) {
  const [activeTab, setActiveTab] = useState("overview");

  if (!detection) {
    return (
      <div className="bg-slate-900/50 backdrop-blur-sm rounded-xl border border-slate-800 p-6 shadow-xl">
        <div className="text-center py-12">
          <svg
            className="w-16 h-16 text-slate-700 mx-auto mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <h3 className="text-slate-400 text-lg font-semibold mb-2">
            No Active Detection
          </h3>
          <p className="text-slate-500 text-sm">
            All systems monitoring normally
          </p>
        </div>
      </div>
    );
  }

  const getSeverityBadge = (severity) => {
    const styles = {
      CRITICAL: "bg-red-500/20 text-red-400 border-red-500/50",
      HIGH: "bg-orange-500/20 text-orange-400 border-orange-500/50",
      MEDIUM: "bg-yellow-500/20 text-yellow-400 border-yellow-500/50",
      LOW: "bg-blue-500/20 text-blue-400 border-blue-500/50",
    };
    return (
      styles[severity] || "bg-slate-500/20 text-slate-400 border-slate-500/50"
    );
  };

  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return {
      date: date.toLocaleDateString(),
      time: date.toLocaleTimeString(),
    };
  };

  const ts = formatTimestamp(detection.timestamp);

  return (
    <div className="bg-slate-900/50 backdrop-blur-sm rounded-lg sm:rounded-xl border border-slate-800 shadow-xl">
      {/* Header - Responsive */}
      <div className="px-3 sm:px-4 md:px-6 py-3 sm:py-4 border-b border-slate-800">
        <div className="flex items-center justify-between">
          <h3 className="text-white font-bold text-sm sm:text-base md:text-lg">
            THREAT DETAILS
          </h3>
          <span className="text-slate-400 text-[10px] sm:text-xs font-mono truncate max-w-[120px] sm:max-w-none">
            {detection.id}
          </span>
        </div>
      </div>

      {/* Tabs - Responsive */}
      <div className="flex border-b border-slate-800">
        {["overview", "details"].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 px-2 sm:px-3 md:px-4 py-2 sm:py-3 text-xs sm:text-sm font-semibold transition-colors ${
              activeTab === tab
                ? "text-cyan-400 border-b-2 border-cyan-400"
                : "text-slate-500 hover:text-slate-300"
            }`}
          >
            {tab.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Content - Responsive */}
      <div className="p-3 sm:p-4 md:p-6 max-h-[400px] sm:max-h-[500px] md:max-h-[600px] overflow-y-auto custom-scrollbar">
        {activeTab === "overview" && (
          <div className="space-y-3 sm:space-y-4">
            {/* ⭐ WHY DETECTED - PROMINENT SECTION - USER-FOCUSED ⭐ */}
            <div className="bg-gradient-to-br from-red-600/20 to-orange-600/20 border-2 border-red-500/70 rounded-lg sm:rounded-xl p-3 sm:p-4 md:p-5 shadow-2xl">
              <div className="flex items-center gap-2 mb-3">
                <svg
                  className="w-5 h-5 sm:w-6 sm:h-6 text-red-400 flex-shrink-0"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                    clipRule="evenodd"
                  />
                </svg>
                <h4 className="text-red-300 text-sm sm:text-base md:text-lg font-bold uppercase">
                  Why Detected as Threat
                </h4>
              </div>
              {detection.reasoning && detection.reasoning.length > 0 ? (
                <ul className="space-y-2 sm:space-y-2.5">
                  {detection.reasoning.map((reason, idx) => (
                    <li
                      key={idx}
                      className="flex items-start gap-2 text-xs sm:text-sm text-white bg-red-900/30 p-2 sm:p-3 rounded-lg border border-red-500/30"
                    >
                      <svg
                        className="w-4 h-4 sm:w-5 sm:h-5 text-red-400 mt-0.5 flex-shrink-0"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                          clipRule="evenodd"
                        />
                      </svg>
                      <span className="font-medium">{reason}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-red-200 text-xs sm:text-sm bg-red-900/30 p-2 sm:p-3 rounded-lg border border-red-500/30">
                  {detection.explanation ||
                    "Anomalous behavior pattern detected by AI system"}
                </p>
              )}
            </div>

            {/* Primary Info - User Focused */}
            <div className="bg-slate-800/50 rounded-lg sm:rounded-xl p-3 sm:p-4 border border-slate-700">
              <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3 mb-3">
                <div className="flex-1 min-w-0">
                  <h4 className="text-white font-bold text-base sm:text-lg md:text-xl mb-2 truncate">
                    {detection.anomaly_type}
                  </h4>
                  <span
                    className={`inline-block px-2 sm:px-3 py-1 rounded-full text-[10px] sm:text-xs font-bold border ${getSeverityBadge(
                      detection.severity
                    )}`}
                  >
                    {detection.severity} THREAT
                  </span>
                </div>
                <div className="text-center sm:text-right bg-cyan-900/20 rounded-lg p-2 sm:p-3 border border-cyan-500/30">
                  <div className="text-slate-400 text-[10px] sm:text-xs">
                    CONFIDENCE
                  </div>
                  <div className="text-cyan-400 text-xl sm:text-2xl font-bold">
                    {(detection.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>

            {/* Location & Time - Responsive Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 sm:gap-3 md:gap-4">
              <div className="bg-slate-800/50 rounded-lg p-3 sm:p-4 border border-slate-700">
                <div className="text-slate-400 text-[10px] sm:text-xs font-semibold mb-1 sm:mb-2">
                  📹 CAMERA
                </div>
                <div className="text-white font-semibold text-sm sm:text-base truncate">
                  {detection.camera_name}
                </div>
                <div className="text-cyan-400 text-xs sm:text-sm truncate">
                  {detection.camera_id}
                </div>
              </div>
              <div className="bg-slate-800/50 rounded-lg p-3 sm:p-4 border border-slate-700">
                <div className="text-slate-400 text-[10px] sm:text-xs font-semibold mb-1 sm:mb-2">
                  📍 LOCATION
                </div>
                <div className="text-white font-semibold text-sm sm:text-base truncate">
                  {detection.location}
                </div>
                <div className="text-slate-500 text-xs sm:text-sm">Zone A</div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 sm:gap-3 md:gap-4">
              <div className="bg-slate-800/50 rounded-lg p-3 sm:p-4 border border-slate-700">
                <div className="text-slate-400 text-[10px] sm:text-xs font-semibold mb-1 sm:mb-2">
                  📅 DATE
                </div>
                <div className="text-white font-mono text-xs sm:text-sm">
                  {ts.date}
                </div>
              </div>
              <div className="bg-slate-800/50 rounded-lg p-3 sm:p-4 border border-slate-700">
                <div className="text-slate-400 text-[10px] sm:text-xs font-semibold mb-1 sm:mb-2">
                  ⏰ TIME
                </div>
                <div className="text-white font-mono text-xs sm:text-sm">
                  {ts.time}
                </div>
              </div>
            </div>

            {/* Detected Objects - ONLY SHOW DANGEROUS/RELEVANT */}
            {detection.detected_objects &&
              detection.detected_objects.length > 0 && (
                <div className="bg-slate-800/50 rounded-lg p-3 sm:p-4 border border-slate-700">
                  <div className="text-slate-400 text-[10px] sm:text-xs font-semibold mb-2 sm:mb-3">
                    🎯 OBJECTS IN THREAT
                  </div>
                  <div className="flex flex-wrap gap-1.5 sm:gap-2">
                    {detection.detected_objects.map((obj, idx) => {
                      const objName = (obj.class || obj).toLowerCase();
                      // Prioritize dangerous objects
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
                          key={idx}
                          className={`px-2 sm:px-3 py-1 rounded-full text-[10px] sm:text-xs font-semibold ${
                            isDangerous
                              ? "bg-red-500/30 border border-red-500/70 text-red-300 animate-pulse"
                              : "bg-cyan-500/20 border border-cyan-500/50 text-cyan-400"
                          }`}
                        >
                          {obj.class || obj}
                        </span>
                      );
                    })}
                  </div>
                </div>
              )}
          </div>
        )}

        {activeTab === "details" && (
          <div className="space-y-3 sm:space-y-4">
            {/* Fusion Score Breakdown - Simplified */}
            <div className="bg-slate-800/50 rounded-lg p-3 sm:p-4 border border-slate-700">
              <div className="text-slate-400 text-[10px] sm:text-xs font-semibold mb-2 sm:mb-3">
                DETECTION METHODS USED
              </div>
              <div className="space-y-2 sm:space-y-3">
                {Object.entries(detection.score_breakdown || {}).map(
                  ([key, data]) => {
                    const score = data.score || 0;
                    const weight = data.weight || "0%";

                    return (
                      <div key={key} className="space-y-1">
                        <div className="flex justify-between text-xs sm:text-sm">
                          <span className="text-slate-300 capitalize truncate">
                            {key.replace(/_/g, " ")}
                          </span>
                          <div className="flex items-center gap-1 sm:gap-2 flex-shrink-0 ml-2">
                            <span className="text-slate-500 text-[10px] sm:text-xs">
                              {weight}
                            </span>
                            <span className="text-cyan-400 font-semibold text-xs sm:text-sm">
                              {(score * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                        <div className="w-full bg-slate-700 rounded-full h-1.5 sm:h-2">
                          <div
                            className="bg-gradient-to-r from-cyan-500 to-blue-500 h-1.5 sm:h-2 rounded-full transition-all duration-300"
                            style={{ width: `${score * 100}%` }}
                          />
                        </div>
                      </div>
                    );
                  }
                )}
              </div>
            </div>

            {/* ML Prediction - ONLY SHOW IF NOT NORMAL */}
            {detection.ml_prediction &&
              detection.ml_prediction.predicted_class &&
              !["normalvideos", "normal", "normal_videos"].includes(
                detection.ml_prediction.predicted_class.toLowerCase()
              ) && (
                <div className="bg-slate-800/50 rounded-lg p-3 sm:p-4 border border-slate-700">
                  <div className="text-slate-400 text-[10px] sm:text-xs font-semibold mb-2 sm:mb-3">
                    AI CLASSIFICATION
                  </div>
                  <div className="text-white font-semibold text-sm sm:text-base mb-2">
                    {detection.ml_prediction.predicted_class}
                  </div>
                  <div className="text-cyan-400 text-xs sm:text-sm mb-3">
                    Confidence:{" "}
                    {(detection.ml_prediction.confidence * 100).toFixed(1)}%
                  </div>
                  {detection.ml_prediction.top_3 &&
                    detection.ml_prediction.top_3.filter(
                      (pred) =>
                        !["normalvideos", "normal", "normal_videos"].includes(
                          pred.class.toLowerCase()
                        )
                    ).length > 0 && (
                      <div className="space-y-1">
                        <div className="text-slate-500 text-[10px] sm:text-xs">
                          Alternative Classifications:
                        </div>
                        {detection.ml_prediction.top_3
                          .filter(
                            (pred) =>
                              ![
                                "normalvideos",
                                "normal",
                                "normal_videos",
                              ].includes(pred.class.toLowerCase())
                          )
                          .slice(0, 3)
                          .map((pred, idx) => (
                            <div
                              key={idx}
                              className="flex justify-between text-xs sm:text-sm"
                            >
                              <span className="text-slate-400 truncate">
                                {pred.class}
                              </span>
                              <span className="text-slate-500 flex-shrink-0 ml-2">
                                {(pred.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                          ))}
                      </div>
                    )}
                </div>
              )}

            {/* Motion/Pose Data - Simplified for Users */}
            {detection.motion_data && detection.motion_data.is_unusual && (
              <div className="bg-slate-800/50 rounded-lg p-3 sm:p-4 border border-slate-700">
                <div className="text-slate-400 text-[10px] sm:text-xs font-semibold mb-2">
                  UNUSUAL MOVEMENT
                </div>
                <div className="text-red-400 font-semibold text-sm sm:text-base">
                  {detection.motion_data.anomaly_type ||
                    "Abnormal Motion Detected"}
                </div>
              </div>
            )}

            {detection.pose_data && detection.pose_data.is_anomalous && (
              <div className="bg-slate-800/50 rounded-lg p-3 sm:p-4 border border-slate-700">
                <div className="text-slate-400 text-[10px] sm:text-xs font-semibold mb-2">
                  SUSPICIOUS POSE
                </div>
                <div className="text-orange-400 font-semibold text-sm sm:text-base">
                  {detection.pose_data.anomaly_type ||
                    "Anomalous Body Position"}
                </div>
                <div className="text-slate-500 text-xs sm:text-sm mt-1">
                  {detection.pose_data.persons_detected || 0} person(s) detected
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
