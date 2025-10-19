import React from "react";

const levelColor = (lvl) =>
  ({
    CRITICAL: "bg-red-600",
    HIGH: "bg-orange-500",
    MEDIUM: "bg-yellow-500",
    LOW: "bg-cyan-500",
  }[lvl] || "bg-gray-600");

export default function AlertFeed({ anomalies = [] }) {
  if (!anomalies.length) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 sm:p-6 h-full">
        <h3 className="text-white font-semibold mb-3">Alerts</h3>
        <p className="text-gray-400 text-sm">No anomalies yet.</p>
      </div>
    );
  }
  return (
    <div className="bg-gray-800 rounded-lg p-4 sm:p-6 h-full overflow-y-auto">
      <h3 className="text-white font-semibold mb-3">Alerts</h3>
      <div className="space-y-3">
        {anomalies.map((a, idx) => (
          <div key={idx} className="bg-gray-700 rounded-lg p-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span
                  className={`inline-block w-2 h-2 rounded-full ${levelColor(
                    a.severity
                  )}`}
                ></span>
                <span className="text-white font-semibold">
                  {a.title || a.anomaly_type}
                </span>
              </div>
              <span className="text-gray-300 text-xs">
                {new Date(a.timestamp).toLocaleTimeString()}
              </span>
            </div>
            <div className="text-gray-300 text-sm mt-1">
              {a.message || a.explanation}
            </div>
            <div className="text-gray-400 text-xs mt-2 flex gap-3">
              <span>Severity: {a.severity}</span>
              {a.fusion_score != null && (
                <span>Score: {Number(a.fusion_score).toFixed(2)}</span>
              )}
              {a.confidence != null && (
                <span>Conf: {Number(a.confidence * 100).toFixed(1)}%</span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
