import React, { useRef, useEffect } from "react";

export default function LiveFeedV2({
  videoStream, // NEW: Direct camera stream
  detectionData, // NEW: Only detection data from backend
  status,
  currentDetection,
  timeline,
}) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Set up video stream
  useEffect(() => {
    if (videoRef.current && videoStream) {
      videoRef.current.srcObject = videoStream;
    }
  }, [videoStream]);

  // Draw detections overlay on canvas
  useEffect(() => {
    if (!detectionData || !videoRef.current || !canvasRef.current) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      return;
    }

    const ctx = canvas.getContext("2d");

    // Match canvas INTERNAL resolution to video resolution
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Set canvas DISPLAY size to match video DISPLAY size
    const videoRect = video.getBoundingClientRect();
    canvas.style.width = `${videoRect.width}px`;
    canvas.style.height = `${videoRect.height}px`;

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw YOLO bounding boxes with tracking IDs
    if (
      detectionData.objects &&
      Array.isArray(detectionData.objects) &&
      detectionData.objects.length > 0
    ) {
      detectionData.objects.forEach((obj) => {
        // Safety check for bbox
        if (!obj.bbox || !Array.isArray(obj.bbox)) {
          return;
        }

        // YOLO format: {class: "person", confidence: 0.95, bbox: [x, y, w, h], track_id: 1}
        const [x, y, w, h] = obj.bbox;

        // Draw box
        ctx.strokeStyle = obj.is_dangerous ? "#ff0000" : "#00ff00";
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        // Draw tracking ID if available
        if (obj.track_id !== undefined) {
          ctx.fillStyle = obj.is_dangerous ? "#ff0000" : "#00ff00";
          ctx.font = "bold 14px Arial";
          ctx.fillText(`ID:${obj.track_id}`, x + 5, y + 20);
        }

        // Draw label background
        ctx.fillStyle = obj.is_dangerous ? "#ff0000" : "#00ff00";
        const label = `${obj.class || "object"} ${(
          obj.confidence * 100
        ).toFixed(0)}%`;
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(x, y - 25, textWidth + 10, 25);

        // Draw label text
        ctx.fillStyle = "#000000";
        ctx.font = "16px Arial";
        ctx.fillText(label, x + 5, y - 7);
      });
    }

    // Draw Pose keypoints and skeleton
    if (
      detectionData.poses &&
      Array.isArray(detectionData.poses) &&
      detectionData.poses.length > 0
    ) {
      ctx.strokeStyle = "#ff00ff";
      ctx.fillStyle = "#ff00ff";
      ctx.lineWidth = 2;

      detectionData.poses.forEach((pose) => {
        // Safety check for keypoints
        if (!pose.keypoints || !Array.isArray(pose.keypoints)) return;

        // Draw keypoints
        pose.keypoints.forEach((kp) => {
          // Safety check for kp properties
          if (!kp || typeof kp.x !== "number" || typeof kp.y !== "number")
            return;

          // kp format: {x: 100, y: 200, confidence: 0.9}
          if (kp.confidence > 0.5) {
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, 5, 0, 2 * Math.PI);
            ctx.fill();
          }
        });

        // Draw skeleton connections
        const connections = [
          [0, 1],
          [0, 2],
          [1, 3],
          [2, 4], // Head
          [5, 6],
          [5, 7],
          [7, 9],
          [6, 8],
          [8, 10], // Arms
          [5, 11],
          [6, 12],
          [11, 12], // Torso
          [11, 13],
          [13, 15],
          [12, 14],
          [14, 16], // Legs
        ];

        connections.forEach(([i, j]) => {
          const kp1 = pose.keypoints[i];
          const kp2 = pose.keypoints[j];
          if (
            kp1 &&
            kp2 &&
            kp1.confidence > 0.5 &&
            kp2.confidence > 0.5 &&
            typeof kp1.x === "number" &&
            typeof kp1.y === "number" &&
            typeof kp2.x === "number" &&
            typeof kp2.y === "number"
          ) {
            ctx.beginPath();
            ctx.moveTo(kp1.x, kp1.y);
            ctx.lineTo(kp2.x, kp2.y);
            ctx.stroke();
          }
        });
      });
    }
  }, [detectionData]);

  const getSeverityColor = (severity) => {
    const colors = {
      CRITICAL: "border-red-500 shadow-red-500/50",
      HIGH: "border-orange-500 shadow-orange-500/50",
      MEDIUM: "border-yellow-500 shadow-yellow-500/50",
      LOW: "border-blue-500 shadow-blue-500/50",
    };
    return colors[severity] || "border-slate-700";
  };

  const borderClass = currentDetection
    ? getSeverityColor(currentDetection.severity)
    : "border-slate-700";

  return (
    <div
      className={`bg-slate-900/50 backdrop-blur-sm rounded-lg sm:rounded-xl border-2 ${borderClass} shadow-2xl transition-all duration-300`}
    >
      {/* Header - Responsive */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between px-3 sm:px-4 md:px-6 py-3 sm:py-4 border-b border-slate-800 gap-2 sm:gap-0">
        <div className="flex items-center gap-2 sm:gap-3">
          <div className="w-2 h-2 sm:w-3 sm:h-3 bg-red-500 rounded-full animate-pulse flex-shrink-0"></div>
          <h3 className="text-white font-bold text-sm sm:text-base md:text-lg truncate">
            LIVE CAMERA FEED
          </h3>
          {currentDetection && (
            <span className="px-2 sm:px-3 py-0.5 sm:py-1 bg-red-500/20 border border-red-500/50 rounded-full text-red-400 text-[10px] sm:text-xs font-bold animate-pulse">
              🚨 THREAT
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 sm:gap-4 text-xs sm:text-sm">
          <span className="text-slate-400 font-mono">{status}</span>
          {currentDetection && (
            <span className="text-slate-400 truncate">
              <span className="text-cyan-400 font-semibold">
                {currentDetection.camera_id}
              </span>
            </span>
          )}
        </div>
      </div>

      {/* Video Feed - PROFESSIONAL: Direct stream + overlay */}
      <div className="relative bg-black">
        <div className="relative">
          {/* Native video element - smooth 30fps */}
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-auto object-contain"
          />

          {/* Canvas overlay for detections */}
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 pointer-events-none"
            style={{ objectFit: "contain" }}
          />
        </div>

        {!videoStream && (
          <div className="flex items-center justify-center h-[250px] sm:h-[350px] md:h-[480px]">
            <div className="text-center px-4">
              <div className="w-12 h-12 sm:w-16 sm:h-16 border-4 border-slate-700 border-t-cyan-500 rounded-full animate-spin mx-auto mb-3 sm:mb-4"></div>
              <p className="text-slate-500 text-sm sm:text-base md:text-lg">
                Waiting for camera feed...
              </p>
            </div>
          </div>
        )}

        {/* ⭐ ANOMALY OVERLAY - Only on detection ⭐ */}
        {currentDetection && videoStream && (
          <div className="absolute top-2 sm:top-3 md:top-4 left-2 sm:left-3 md:left-4 right-2 sm:right-3 md:right-4 bg-black/90 backdrop-blur-md border-2 border-red-500/70 rounded-lg p-2 sm:p-3 md:p-4">
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1 min-w-0">
                <div className="flex flex-wrap items-center gap-1 sm:gap-2 mb-1 sm:mb-2">
                  <span className="px-1.5 sm:px-2 py-0.5 sm:py-1 bg-red-600 text-white text-[10px] sm:text-xs font-bold rounded flex-shrink-0">
                    {currentDetection.severity}
                  </span>
                  <span className="text-white font-bold text-xs sm:text-sm md:text-base lg:text-lg truncate">
                    {currentDetection.anomaly_type}
                  </span>
                </div>
                {currentDetection.explanation && (
                  <p className="text-slate-300 text-[10px] sm:text-xs md:text-sm mb-1 sm:mb-2 line-clamp-2">
                    {currentDetection.explanation}
                  </p>
                )}
                <div className="flex flex-wrap items-center gap-2 sm:gap-3 md:gap-4 text-[9px] sm:text-[10px] md:text-xs">
                  <span className="text-slate-400">
                    Confidence:{" "}
                    <span className="text-red-400 font-bold">
                      {(currentDetection.confidence * 100).toFixed(1)}%
                    </span>
                  </span>
                  <span className="text-slate-400 truncate">
                    📍{" "}
                    <span className="text-cyan-400 font-semibold">
                      {currentDetection.location}
                    </span>
                  </span>
                </div>
              </div>
              <div className="flex-shrink-0 ml-1 sm:ml-2">
                <svg
                  className="w-8 h-8 sm:w-10 sm:h-10 md:w-12 md:h-12 text-red-500 animate-pulse"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Timeline Preview - ONLY SHOW ANOMALIES */}
      <div className="px-3 sm:px-4 md:px-6 py-3 sm:py-4 border-t border-slate-800">
        <div className="flex items-center justify-between mb-2 sm:mb-3">
          <h4 className="text-slate-400 text-[10px] sm:text-xs md:text-sm font-semibold">
            THREAT HISTORY
          </h4>
          <span className="text-slate-500 text-[9px] sm:text-[10px] md:text-xs">
            {timeline.length} threats
          </span>
        </div>
        <div className="flex gap-0.5 sm:gap-1 h-8 sm:h-10 md:h-12 items-end">
          {timeline
            .slice(0, 50)
            .reverse()
            .map((event, idx) => {
              const height = Math.max(20, event.fusion_score * 100);
              const severityColors = {
                CRITICAL: "bg-red-500",
                HIGH: "bg-orange-500",
                MEDIUM: "bg-yellow-500",
                LOW: "bg-blue-500",
              };
              const color = severityColors[event.severity] || "bg-slate-600";

              return (
                <div
                  key={event.id || idx}
                  className={`flex-1 ${color} rounded-t transition-all duration-200 hover:opacity-80 cursor-pointer`}
                  style={{ height: `${height}%` }}
                  title={`${event.anomaly_type} - ${(
                    event.fusion_score * 100
                  ).toFixed(0)}%`}
                />
              );
            })}
          {timeline.length < 50 &&
            Array.from({ length: Math.max(0, 50 - timeline.length) }).map(
              (_, idx) => (
                <div
                  key={`empty-${idx}`}
                  className="flex-1 bg-slate-800 rounded-t"
                  style={{ height: "10%" }}
                />
              )
            )}
        </div>
      </div>
    </div>
  );
}
