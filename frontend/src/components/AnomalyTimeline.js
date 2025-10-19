import React from "react";

const sevColor = (lvl) =>
  ({
    CRITICAL: "bg-red-500",
    HIGH: "bg-orange-400",
    MEDIUM: "bg-yellow-400",
    LOW: "bg-cyan-400",
  }[lvl] || "bg-gray-500");

export default function AnomalyTimeline({ timeline = [] }) {
  return (
    <div className="bg-gray-800 rounded-lg p-4 sm:p-6">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-semibold">Anomaly Timeline</h3>
        <div className="text-gray-400 text-xs">
          Last {timeline.length} events
        </div>
      </div>
      <div className="h-16 bg-gray-700 rounded flex items-end overflow-hidden">
        {timeline.map((e, idx) => (
          <div
            key={idx}
            className={`${sevColor(e.severity)} transition-all`}
            title={`${e.anomaly_type} • ${e.severity} • ${new Date(
              e.timestamp
            ).toLocaleTimeString()}`}
            style={{
              height: `${Math.min(100, (e.fusion_score || 0.7) * 100)}%`,
              width: `${Math.max(2, 100 / Math.max(1, timeline.length))}%`,
            }}
          />
        ))}
      </div>
    </div>
  );
}
