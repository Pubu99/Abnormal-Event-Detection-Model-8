import React from "react";

export default function LiveFeed({ frameBase64, status, threatLevel }) {
  const border =
    {
      CRITICAL: "border-red-600",
      HIGH: "border-orange-500",
      MEDIUM: "border-yellow-400",
      LOW: "border-cyan-400",
    }[threatLevel] || "border-gray-600";

  return (
    <div className={`bg-gray-800 rounded-lg p-2 border ${border}`}>
      <div className="flex items-center justify-between p-2">
        <h3 className="text-white font-semibold">Live Feed</h3>
        <div className="text-gray-300 text-xs">{status}</div>
      </div>
      <div className="bg-black rounded overflow-hidden">
        {frameBase64 ? (
          <img
            alt="Live"
            className="w-full object-contain"
            src={`data:image/jpeg;base64,${frameBase64}`}
          />
        ) : (
          <div className="text-gray-500 p-8 text-center">No frame yet</div>
        )}
      </div>
    </div>
  );
}
