import React, { useState, useRef, useEffect } from "react";

function LiveCamera({ onDetection }) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [currentResult, setCurrentResult] = useState(null);
  const [detectionHistory, setDetectionHistory] = useState([]);
  const [fps, setFps] = useState(0);
  const [statusMessage, setStatusMessage] = useState("");

  // NEW: Enhanced state for professional features
  const [frameTimeline, setFrameTimeline] = useState([]); // Frame-by-frame anomaly tracking
  const [savedScreenshots, setSavedScreenshots] = useState([]); // Auto-saved screenshots
  const [fusionReasoning, setFusionReasoning] = useState([]); // Intelligent fusion reasoning
  const [showLegend, setShowLegend] = useState(true); // Color legend visibility

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const wsRef = useRef(null);
  const intervalRef = useRef(null);
  const fpsIntervalRef = useRef(null);
  const frameCountRef = useRef(0);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (fpsIntervalRef.current) {
        clearInterval(fpsIntervalRef.current);
      }
    };
  }, []);

  const startWebcam = async () => {
    try {
      setStatusMessage("Requesting camera access...");
      console.log("🎥 Requesting camera access...");

      const constraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      console.log("✅ Camera access granted!");

      if (!videoRef.current) {
        console.error("❌ Video element ref is null!");
        setStatusMessage("Error: Video element not available");
        return;
      }

      videoRef.current.srcObject = stream;
      streamRef.current = stream;

      videoRef.current.onloadedmetadata = () => {
        console.log("📹 Video metadata loaded");
        videoRef.current
          .play()
          .then(() => {
            console.log("▶️ Video playing successfully!");
            setStatusMessage("");
            setCameraActive(true);

            // Initialize overlay canvas
            if (overlayCanvasRef.current && videoRef.current) {
              overlayCanvasRef.current.width = videoRef.current.videoWidth;
              overlayCanvasRef.current.height = videoRef.current.videoHeight;
            }
          })
          .catch((playError) => {
            console.error("❌ Video play() failed:", playError);
            setStatusMessage("Error: Cannot play video");
          });
      };
    } catch (error) {
      console.error("❌ Error accessing webcam:", error);
      setStatusMessage(
        `Error: ${
          error.name === "NotAllowedError"
            ? "Camera permission denied"
            : "Could not access camera"
        }`
      );
    }
  };

  const stopCamera = () => {
    console.log("🛑 Stopping camera...");

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (fpsIntervalRef.current) {
      clearInterval(fpsIntervalRef.current);
      fpsIntervalRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraActive(false);
    setIsStreaming(false);
    setCurrentResult(null);
    setFps(0);
    frameCountRef.current = 0;
    setStatusMessage("");
  };

  const startAnalysis = () => {
    if (!cameraActive || !videoRef.current) {
      setStatusMessage("Camera not active");
      console.error("❌ Cannot start analysis: Camera not active");
      return;
    }

    console.log("🚀 Starting live analysis...");
    console.log("📡 Connecting to WebSocket: ws://localhost:8000/ws/stream");
    setStatusMessage("Connecting to analysis server...");

    // Connect WebSocket
    const ws = new WebSocket("ws://localhost:8000/ws/stream");
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("✅ WebSocket connected successfully!");
      setStatusMessage("");
      setIsStreaming(true);

      // Start FPS counter
      frameCountRef.current = 0;
      fpsIntervalRef.current = setInterval(() => {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
      }, 1000);

      // Send frames at ~10 FPS
      intervalRef.current = setInterval(() => {
        captureAndSendFrame();
      }, 100);

      console.log("🎬 Started sending frames at 10 FPS");
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);

      if (message.type === "prediction") {
        console.log("📊 Received prediction:", message);
        handlePrediction(message);
      } else if (message.error) {
        console.error("❌ Backend error:", message.error);
        setStatusMessage(`Backend error: ${message.error}`);
      }
    };

    ws.onerror = (error) => {
      console.error("❌ WebSocket error:", error);
      console.error("❌ Error details:", {
        readyState: ws.readyState,
        url: ws.url,
      });
      setStatusMessage(
        "Connection error - Check if backend is running on port 8000"
      );
    };

    ws.onclose = (event) => {
      console.log("🔌 WebSocket disconnected");
      console.log("   Close code:", event.code);
      console.log("   Close reason:", event.reason);
      console.log("   Was clean:", event.wasClean);

      setIsStreaming(false);
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      if (fpsIntervalRef.current) {
        clearInterval(fpsIntervalRef.current);
        fpsIntervalRef.current = null;
      }

      if (!event.wasClean) {
        setStatusMessage("Connection lost - Try reconnecting");
      }
    };
  };

  const stopAnalysis = () => {
    console.log("⏸️ Stopping analysis...");

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (fpsIntervalRef.current) {
      clearInterval(fpsIntervalRef.current);
      fpsIntervalRef.current = null;
    }

    setIsStreaming(false);
    setCurrentResult(null);
    setFps(0);
    frameCountRef.current = 0;
  };

  const captureAndSendFrame = () => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to JPEG and send
    canvas.toBlob(
      (blob) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64data = reader.result.split(",")[1];

          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(
              JSON.stringify({
                type: "frame",
                data: base64data,
              })
            );
            frameCountRef.current++;
          }
        };
        reader.readAsDataURL(blob);
      },
      "image/jpeg",
      0.8
    );
  };

  const handlePrediction = (message) => {
    const { data, timestamp, frame_number, anomaly_detected } = message;

    // PROFESSIONAL FUSION-BASED HANDLING
    // Only process and display if anomaly detected (fusion_score >= 0.70)
    setCurrentResult({
      ...data,
      timestamp: new Date(timestamp),
      frame_number: frame_number,
      anomaly_detected: anomaly_detected || false,
      // Fusion engine results (primary)
      fusion: data.fusion || null,
      // Professional threat assessment
      alerts: data.alerts || [],
      threat_level: data.threat_level || "NORMAL",
      is_dangerous: data.is_dangerous || false,
      summary: data.summary || "Normal activity",
      // Reference data
      ml_model: data.ml_model || {},
      yolo: data.yolo || {},
      motion: data.motion || {},
      pose: data.pose || {},
      tracking: data.tracking || {},
    });

    // Update fusion reasoning display (professional multi-modal explanation)
    if (data.fusion && data.fusion.reasoning) {
      setFusionReasoning(data.fusion.reasoning);
    }

    // Add to detection history ONLY if anomaly detected
    if (anomaly_detected && data.fusion) {
      const detection = {
        timestamp: new Date(timestamp),
        frame_number: frame_number,
        anomaly_type: data.fusion.anomaly_type,
        severity: data.fusion.severity,
        fusion_score: data.fusion.fusion_score,
        confidence: data.fusion.confidence,
        explanation: data.fusion.explanation,
      };

      setDetectionHistory((prev) => [detection, ...prev].slice(0, 50)); // Keep last 50
    }

    // NEW: Add to frame timeline
    const timelineEntry = {
      frame_number: frame_number || frameCountRef.current,
      timestamp: new Date(timestamp),
      decision: data.fusion?.final_decision || data.threat_level || "NORMAL",
      ml_prediction: data.predicted_class || "Processing...",
      confidence: data.confidence || 0,
      anomalies: {
        ml: data.is_anomaly || false,
        motion: data.motion?.is_unusual || false,
        pose: data.pose?.is_anomalous || false,
        objects: (data.objects_detected || []).length > 5,
      },
    };
    setFrameTimeline((prev) => [...prev.slice(-100), timelineEntry]); // Keep last 100 frames

    // Draw enhanced visualizations
    drawEnhancedDetections(data);

    // Add to history with enhanced data
    const historyEntry = {
      timestamp: new Date(timestamp),
      class: data.predicted_class || "Processing...",
      confidence: data.confidence || 0,
      threat_level: data.threat_level || "INFO",
      is_dangerous: data.is_dangerous || false,
      objects: data.objects_detected || [],
      dangerousObjects: data.dangerous_objects || [],
      alerts: data.alerts || [],
      motion_anomaly: data.motion?.is_unusual || false,
      pose_anomaly: data.pose?.is_anomalous || false,
      tracked_objects: data.tracking?.total_tracks || 0,
      fusion_decision: data.fusion?.final_decision || null,
    };

    setDetectionHistory((prev) => [historyEntry, ...prev].slice(0, 50));

    // NEW: Auto-capture screenshot for ANY anomaly (not just critical)
    const shouldCapture =
      data.is_dangerous ||
      data.threat_level === "CRITICAL" ||
      data.threat_level === "ABNORMAL" ||
      data.threat_level === "HIGH" ||
      data.fusion?.final_decision === "CRITICAL" ||
      data.fusion?.final_decision === "ABNORMAL";

    if (shouldCapture) {
      autoSaveScreenshot(data, frame_number, timestamp);
    }

    // Notify parent component
    if (onDetection) {
      onDetection(data);
    }
  };

  const drawEnhancedDetections = (data) => {
    if (!overlayCanvasRef.current || !videoRef.current) return;

    const canvas = overlayCanvasRef.current;
    const ctx = canvas.getContext("2d");

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Set canvas size to match video
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;

    // Get threat level colors
    const getThreatColor = (level) => {
      const colors = {
        CRITICAL: "#EF4444", // Red
        HIGH: "#F97316", // Orange
        MEDIUM: "#FBBF24", // Yellow
        LOW: "#3B82F6", // Blue
        INFO: "#10B981", // Green
      };
      return colors[level] || "#10B981";
    };

    // 1. Draw tracked objects with bounding boxes
    const trackedObjects = data.tracking?.tracked_objects || [];
    trackedObjects.forEach((obj) => {
      const [x, y, w, h] = obj.bbox;

      // Determine color based on class and threat
      let color = getThreatColor(data.threat_level || "INFO");
      if (["knife", "gun", "weapon"].includes(obj.class.toLowerCase())) {
        color = "#EF4444"; // Red for weapons
      }

      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);

      // Draw track ID and speed
      const label = `ID:${obj.track_id} ${obj.class}`;
      const speedLabel = obj.speed > 1 ? ` ${obj.speed.toFixed(1)}px/f` : "";
      const fullLabel = label + speedLabel;

      ctx.font = "bold 14px Arial";
      const textWidth = ctx.measureText(fullLabel).width;

      // Label background
      ctx.fillStyle = color;
      ctx.fillRect(x, y - 22, textWidth + 10, 22);

      // Label text
      ctx.fillStyle = "#FFFFFF";
      ctx.fillText(fullLabel, x + 5, y - 6);

      // Draw trajectory if available
      if (obj.trajectory && obj.trajectory.length > 1) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.moveTo(obj.trajectory[0][0], obj.trajectory[0][1]);
        for (let i = 1; i < obj.trajectory.length; i++) {
          ctx.lineTo(obj.trajectory[i][0], obj.trajectory[i][1]);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });

    // 2. Draw alert badges
    if (data.alerts && data.alerts.length > 0) {
      let alertY = 20;
      data.alerts.forEach((alert, index) => {
        if (index < 3) {
          // Show max 3 alerts
          const alertColor = getThreatColor(alert.level);
          const alertText = `${alert.title}`;

          ctx.font = "bold 16px Arial";
          const alertWidth = ctx.measureText(alertText).width;

          // Alert background with semi-transparency
          ctx.fillStyle = alertColor;
          ctx.globalAlpha = 0.9;
          ctx.fillRect(10, alertY, alertWidth + 60, 30);
          ctx.globalAlpha = 1.0;

          // Alert icon
          ctx.fillStyle = "#FFFFFF";
          ctx.font = "20px Arial";
          ctx.fillText("⚠", 15, alertY + 22);

          // Alert text
          ctx.font = "bold 14px Arial";
          ctx.fillText(alertText, 40, alertY + 20);

          // Confidence badge
          if (alert.confidence) {
            ctx.font = "12px Arial";
            ctx.fillText(
              `${(alert.confidence * 100).toFixed(0)}%`,
              alertWidth + 45,
              alertY + 20
            );
          }

          alertY += 40;
        }
      });
    }

    // 3. Draw motion indicator
    if (data.motion?.is_unusual) {
      const motionText = `🏃 ${data.motion.anomaly_type || "Motion Detected"}`;
      ctx.font = "bold 14px Arial";
      const motionWidth = ctx.measureText(motionText).width;

      ctx.fillStyle = "#FBBF24"; // Yellow
      ctx.globalAlpha = 0.85;
      ctx.fillRect(canvas.width - motionWidth - 20, 10, motionWidth + 15, 28);
      ctx.globalAlpha = 1.0;

      ctx.fillStyle = "#000000";
      ctx.fillText(motionText, canvas.width - motionWidth - 12, 30);
    }

    // 4. Draw pose anomaly indicator
    if (data.pose?.is_anomalous) {
      const poseText = `🤺 ${data.pose.anomaly_type || "Pose Anomaly"}`;
      ctx.font = "bold 14px Arial";
      const poseWidth = ctx.measureText(poseText).width;

      const poseY = data.motion?.is_unusual ? 50 : 10;

      ctx.fillStyle = "#F97316"; // Orange
      ctx.globalAlpha = 0.85;
      ctx.fillRect(canvas.width - poseWidth - 20, poseY, poseWidth + 15, 28);
      ctx.globalAlpha = 1.0;

      ctx.fillStyle = "#FFFFFF";
      ctx.fillText(poseText, canvas.width - poseWidth - 12, poseY + 20);
    }

    // 5. Draw threat level indicator (bottom right)
    if (data.threat_level) {
      const threatColor = getThreatColor(data.threat_level);
      const threatIcon = data.is_dangerous ? "🚨" : "✓";
      const threatText = `${threatIcon} ${data.threat_level}`;

      ctx.font = "bold 18px Arial";
      const threatWidth = ctx.measureText(threatText).width;

      ctx.fillStyle = threatColor;
      ctx.globalAlpha = 0.9;
      ctx.fillRect(
        canvas.width - threatWidth - 30,
        canvas.height - 50,
        threatWidth + 25,
        40
      );
      ctx.globalAlpha = 1.0;

      ctx.fillStyle = "#FFFFFF";
      ctx.fillText(
        threatText,
        canvas.width - threatWidth - 20,
        canvas.height - 22
      );
    }
  };

  // NEW: Auto-save screenshot with metadata (no download prompt)
  const autoSaveScreenshot = (data, frame_number, timestamp) => {
    if (!videoRef.current || !overlayCanvasRef.current) return;

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;

    // Draw video frame + overlay
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    ctx.drawImage(overlayCanvasRef.current, 0, 0, canvas.width, canvas.height);

    // Convert to blob and save to state (not download)
    canvas.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      const timestampStr = new Date(timestamp)
        .toISOString()
        .replace(/[:.]/g, "-");

      const screenshot = {
        url: url,
        blob: blob,
        filename: `frame${frame_number}_${
          data.fusion?.final_decision || data.threat_level
        }_${timestampStr}.png`,
        metadata: {
          frame_number: frame_number,
          timestamp: timestamp,
          decision: data.fusion?.final_decision || data.threat_level,
          ml_prediction: data.predicted_class,
          confidence: data.confidence,
          detected_objects: data.objects_detected || [],
          pose_anomaly: data.pose?.anomaly_type || null,
          motion_anomaly: data.motion?.anomaly_type || null,
          fusion_reasoning: data.fusion?.reasoning || [],
          fusion_score: data.fusion?.confidence || 0,
        },
      };

      setSavedScreenshots((prev) => [...prev, screenshot].slice(-50)); // Keep last 50
      console.log("📸 Screenshot auto-saved:", screenshot.filename);
    });
  };

  const captureScreenshot = (data) => {
    if (!videoRef.current || !overlayCanvasRef.current) return;

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;

    // Draw video frame
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

    // Draw overlay with bounding boxes
    ctx.drawImage(overlayCanvasRef.current, 0, 0, canvas.width, canvas.height);

    // Convert to data URL and download
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `detection_${data.predicted_class}_${timestamp}.png`;

    canvas.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);

      console.log(`📸 Screenshot saved: ${filename}`);
    });
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case "critical":
        return "text-red-500 bg-red-100";
      case "warning":
        return "text-yellow-600 bg-yellow-100";
      case "alert":
        return "text-orange-500 bg-orange-100";
      default:
        return "text-green-600 bg-green-100";
    }
  };

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case "critical":
        return "🚨";
      case "warning":
        return "⚠️";
      case "alert":
        return "⚡";
      default:
        return "✅";
    }
  };

  return (
    <div className="space-y-4">
      {/* Status Bar */}
      <div className="bg-gray-800 text-white p-4 rounded-lg flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-4">
          {/* Camera Status */}
          <div className="flex items-center space-x-2">
            <div
              className={`w-3 h-3 rounded-full ${
                cameraActive ? "bg-green-500 animate-pulse" : "bg-gray-500"
              }`}
            ></div>
            <span className="text-sm font-semibold">
              {cameraActive ? "CAMERA ACTIVE" : "CAMERA OFF"}
            </span>
          </div>

          {/* Analysis Status */}
          {cameraActive && (
            <div className="flex items-center space-x-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  isStreaming ? "bg-red-500 animate-pulse" : "bg-gray-500"
                }`}
              ></div>
              <span className="text-sm font-semibold">
                {isStreaming ? "ANALYZING" : "STANDBY"}
              </span>
            </div>
          )}

          {/* FPS Counter */}
          {isStreaming && (
            <div className="text-sm">
              <span className="text-gray-400">FPS:</span> {fps}
            </div>
          )}
        </div>

        {/* Current Detection Status */}
        {currentResult && (
          <div className="flex items-center space-x-2">
            <span className="text-2xl">
              {getSeverityIcon(currentResult.severity)}
            </span>
            <div>
              <div className="text-xs text-gray-400">Current Status</div>
              <div
                className={`text-sm font-bold ${
                  getSeverityColor(currentResult.severity).split(" ")[0]
                }`}
              >
                {currentResult.is_anomaly ? "ANOMALY DETECTED" : "NORMAL"}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Status Message */}
      {statusMessage && (
        <div className="bg-blue-100 border-l-4 border-blue-500 text-blue-700 p-3 rounded">
          <p className="text-sm">{statusMessage}</p>
        </div>
      )}

      {/* Video Display with Overlay */}
      <div
        className="relative bg-black rounded-lg overflow-hidden"
        style={{ minHeight: "400px" }}
      >
        {/* Video Element */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{
            width: "100%",
            height: "100%",
            display: cameraActive ? "block" : "none",
          }}
          className="object-contain"
        />

        {/* Overlay Canvas for Bounding Boxes */}
        <canvas
          ref={overlayCanvasRef}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            pointerEvents: "none",
            display: cameraActive ? "block" : "none",
          }}
          className="object-contain"
        />

        {/* Hidden Canvas for Frame Capture */}
        <canvas ref={canvasRef} style={{ display: "none" }} />

        {/* Placeholder when camera not active */}
        {!cameraActive && (
          <div className="w-full h-full flex items-center justify-center text-gray-500 min-h-[400px]">
            <div className="text-center">
              <svg
                className="w-24 h-24 mx-auto mb-4 opacity-50"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path d="M2 6a2 2 0 012-2h6a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6zM14.553 7.106A1 1 0 0014 8v4a1 1 0 00.553.894l2 1A1 1 0 0018 13V7a1 1 0 00-1.447-.894l-2 1z" />
              </svg>
              <p className="text-lg">No camera connected</p>
              <p className="text-sm mt-2">Click "Connect Webcam" to start</p>
            </div>
          </div>
        )}

        {/* PROFESSIONAL FUSION-BASED DETECTION OVERLAY (ANOMALY-ONLY) */}
        {currentResult &&
          isStreaming &&
          currentResult.anomaly_detected &&
          currentResult.fusion && (
            <div className="absolute bottom-4 left-4 right-4 bg-black bg-opacity-95 text-white p-4 rounded-lg shadow-2xl border-2 border-red-500">
              {/* MAIN ANOMALY ALERT */}
              <div className="grid grid-cols-3 gap-4 mb-3">
                {/* Anomaly Type */}
                <div>
                  <p className="text-xs text-gray-400">🚨 ANOMALY DETECTED</p>
                  <p className="text-lg font-bold text-red-400">
                    {currentResult.fusion.anomaly_type}
                  </p>
                  <p className="text-xs text-gray-400 mt-1">
                    {currentResult.timestamp.toLocaleTimeString()}
                  </p>
                </div>

                {/* Fusion Score */}
                <div className="text-center">
                  <p className="text-xs text-gray-400">Fusion Score</p>
                  <p
                    className={`text-2xl font-bold ${
                      currentResult.fusion.fusion_score > 0.85
                        ? "text-red-400"
                        : currentResult.fusion.fusion_score > 0.75
                        ? "text-orange-400"
                        : "text-yellow-400"
                    }`}
                  >
                    {(currentResult.fusion.fusion_score * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-gray-400">
                    {currentResult.fusion.consensus.agreement_count}/4 agreed
                  </p>
                </div>

                {/* Severity Level */}
                <div className="text-right">
                  <p className="text-xs text-gray-400">Severity</p>
                  <p
                    className={`text-lg font-bold ${
                      currentResult.fusion.severity === "CRITICAL"
                        ? "text-red-500"
                        : currentResult.fusion.severity === "HIGH"
                        ? "text-orange-500"
                        : currentResult.fusion.severity === "MEDIUM"
                        ? "text-yellow-500"
                        : "text-blue-500"
                    }`}
                  >
                    🚨 {currentResult.fusion.severity}
                  </p>
                </div>
              </div>

              {/* FUSION SCORE BREAKDOWN (Professional Weighted Display) */}
              <div className="mt-3 pt-3 border-t border-gray-700">
                <p className="text-xs text-gray-400 mb-2">
                  Multi-Modal Analysis:
                </p>
                <div className="grid grid-cols-4 gap-2 text-xs">
                  <div className="bg-purple-900 bg-opacity-30 p-2 rounded">
                    <p className="text-gray-400">ML Model (40%)</p>
                    <p className="font-bold text-purple-300">
                      {(
                        currentResult.fusion.score_breakdown.ml_model.score *
                        100
                      ).toFixed(0)}
                      %
                    </p>
                  </div>
                  <div className="bg-blue-900 bg-opacity-30 p-2 rounded">
                    <p className="text-gray-400">YOLO (25%)</p>
                    <p className="font-bold text-blue-300">
                      {(
                        currentResult.fusion.score_breakdown.yolo_objects
                          .score * 100
                      ).toFixed(0)}
                      %
                    </p>
                  </div>
                  <div className="bg-green-900 bg-opacity-30 p-2 rounded">
                    <p className="text-gray-400">Pose (20%)</p>
                    <p className="font-bold text-green-300">
                      {(
                        currentResult.fusion.score_breakdown.pose_estimation
                          .score * 100
                      ).toFixed(0)}
                      %
                    </p>
                  </div>
                  <div className="bg-yellow-900 bg-opacity-30 p-2 rounded">
                    <p className="text-gray-400">Motion (15%)</p>
                    <p className="font-bold text-yellow-300">
                      {(
                        currentResult.fusion.score_breakdown.motion_analysis
                          .score * 100
                      ).toFixed(0)}
                      %
                    </p>
                  </div>
                </div>
              </div>

              {/* Enhanced Detection Info */}
              <div className="mt-3 pt-3 border-t border-gray-700 grid grid-cols-4 gap-2 text-xs">
                {/* Objects Tracked */}
                <div>
                  <p className="text-gray-400">Tracked</p>
                  <p className="font-semibold">
                    {currentResult.tracking?.total_tracks || 0} objects
                  </p>
                </div>

                {/* Motion Status */}
                <div>
                  <p className="text-gray-400">Motion</p>
                  <p
                    className={`font-semibold ${
                      currentResult.motion?.is_unusual
                        ? "text-yellow-400"
                        : "text-green-400"
                    }`}
                  >
                    {currentResult.motion?.is_unusual
                      ? "⚠ Unusual"
                      : "✓ Normal"}
                  </p>
                </div>

                {/* Pose Status */}
                <div>
                  <p className="text-gray-400">Pose</p>
                  <p
                    className={`font-semibold ${
                      currentResult.pose?.is_anomalous
                        ? "text-orange-400"
                        : "text-green-400"
                    }`}
                  >
                    {currentResult.pose?.is_anomalous
                      ? "⚠ Anomaly"
                      : "✓ Normal"}
                  </p>
                </div>

                {/* Active Alerts */}
                <div>
                  <p className="text-gray-400">Alerts</p>
                  <p
                    className={`font-semibold ${
                      (currentResult.alerts?.length || 0) > 0
                        ? "text-red-400"
                        : "text-green-400"
                    }`}
                  >
                    {(currentResult.alerts?.length || 0) > 0
                      ? `🔔 ${currentResult.alerts.length}`
                      : "✓ None"}
                  </p>
                </div>
              </div>

              {/* FUSION REASONING (Professional Explanation) */}
              {fusionReasoning && fusionReasoning.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-700">
                  <p className="text-xs text-gray-400 mb-2">
                    🧠 Intelligent Fusion Reasoning:
                  </p>
                  <div className="space-y-1">
                    {fusionReasoning.slice(0, 4).map((reason, idx) => (
                      <div
                        key={idx}
                        className="text-xs bg-gray-800 bg-opacity-50 p-2 rounded"
                      >
                        <p className="text-gray-200">{reason}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Explanation */}
              <div className="mt-3 pt-3 border-t border-gray-700">
                <p className="text-xs text-gray-300">
                  {currentResult.fusion.explanation}
                </p>
              </div>
            </div>
          )}
      </div>

      {/* Control Buttons */}
      <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-3">
        {!cameraActive ? (
          <button
            onClick={startWebcam}
            className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg transition-colors"
          >
            📹 Connect Webcam
          </button>
        ) : (
          <>
            {!isStreaming ? (
              <button
                onClick={startAnalysis}
                className="flex-1 bg-green-600 hover:bg-green-700 text-white font-semibold py-3 rounded-lg transition-colors"
              >
                🚀 Start Analysis
              </button>
            ) : (
              <button
                onClick={stopAnalysis}
                className="flex-1 bg-yellow-600 hover:bg-yellow-700 text-white font-semibold py-3 rounded-lg transition-colors"
              >
                ⏸️ Pause Analysis
              </button>
            )}
            <button
              onClick={stopCamera}
              className="bg-red-600 hover:bg-red-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
            >
              ⏹️ Stop Camera
            </button>
          </>
        )}
      </div>

      {/* NEW: Color Legend (Collapsible) */}
      {showLegend && (
        <div className="bg-gradient-to-r from-gray-800 to-gray-900 rounded-lg shadow-lg p-4">
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-white font-semibold flex items-center">
              🎨 Color Legend
            </h3>
            <button
              onClick={() => setShowLegend(false)}
              className="text-gray-400 hover:text-white text-sm"
            >
              ✕ Hide
            </button>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            {/* Decision Levels */}
            <div className="bg-gray-700 rounded p-2">
              <div className="flex items-center mb-1">
                <div className="w-4 h-4 rounded-full bg-green-500 mr-2"></div>
                <span className="text-white font-medium">NORMAL</span>
              </div>
              <p className="text-gray-300 text-xs">No anomalies detected</p>
            </div>

            <div className="bg-gray-700 rounded p-2">
              <div className="flex items-center mb-1">
                <div className="w-4 h-4 rounded-full bg-blue-500 mr-2"></div>
                <span className="text-white font-medium">SUSPICIOUS</span>
              </div>
              <p className="text-gray-300 text-xs">Minor indicators</p>
            </div>

            <div className="bg-gray-700 rounded p-2">
              <div className="flex items-center mb-1">
                <div className="w-4 h-4 rounded-full bg-yellow-500 mr-2"></div>
                <span className="text-white font-medium">ABNORMAL</span>
              </div>
              <p className="text-gray-300 text-xs">Clear anomaly</p>
            </div>

            <div className="bg-gray-700 rounded p-2">
              <div className="flex items-center mb-1">
                <div className="w-4 h-4 rounded-full bg-red-500 mr-2"></div>
                <span className="text-white font-medium">CRITICAL</span>
              </div>
              <p className="text-gray-300 text-xs">Immediate action</p>
            </div>

            {/* Bounding Box Colors */}
            <div className="bg-gray-700 rounded p-2">
              <div className="flex items-center mb-1">
                <div className="w-4 h-4 rounded border-2 border-green-500 mr-2"></div>
                <span className="text-white font-medium">Green Box</span>
              </div>
              <p className="text-gray-300 text-xs">Normal object</p>
            </div>

            <div className="bg-gray-700 rounded p-2">
              <div className="flex items-center mb-1">
                <div className="w-4 h-4 rounded border-2 border-yellow-500 mr-2"></div>
                <span className="text-white font-medium">Yellow Box</span>
              </div>
              <p className="text-gray-300 text-xs">Suspicious activity</p>
            </div>

            <div className="bg-gray-700 rounded p-2">
              <div className="flex items-center mb-1">
                <div className="w-4 h-4 rounded border-2 border-red-500 mr-2"></div>
                <span className="text-white font-medium">Red Box</span>
              </div>
              <p className="text-gray-300 text-xs">Critical threat</p>
            </div>

            <div className="bg-gray-700 rounded p-2">
              <div className="flex items-center mb-1">
                <div className="w-4 h-4 rounded border-2 border-orange-500 mr-2"></div>
                <span className="text-white font-medium">Orange Box</span>
              </div>
              <p className="text-gray-300 text-xs">Anomalous pose</p>
            </div>
          </div>
        </div>
      )}

      {!showLegend && (
        <button
          onClick={() => setShowLegend(true)}
          className="text-blue-500 hover:text-blue-600 text-sm"
        >
          Show Color Legend
        </button>
      )}

      {/* NEW: Fusion Reasoning Panel (Below Video) */}
      {currentResult && currentResult.fusion && (
        <div className="bg-white rounded-lg shadow-lg p-4 border-l-4 border-purple-500">
          <h3 className="text-lg font-semibold mb-3 flex items-center">
            🧠 Intelligent Fusion Analysis
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Final Decision */}
            <div>
              <h4 className="text-sm font-semibold text-gray-600 mb-2">
                Final Decision
              </h4>
              <div
                className={`inline-block px-4 py-2 rounded-lg font-bold ${
                  currentResult.fusion.final_decision === "CRITICAL"
                    ? "bg-red-100 text-red-800"
                    : currentResult.fusion.final_decision === "ABNORMAL"
                    ? "bg-yellow-100 text-yellow-800"
                    : currentResult.fusion.final_decision === "SUSPICIOUS"
                    ? "bg-blue-100 text-blue-800"
                    : "bg-green-100 text-green-800"
                }`}
              >
                {currentResult.fusion.final_decision} (
                {(currentResult.fusion.confidence * 100).toFixed(1)}%)
              </div>
              {currentResult.fusion.override_applied && (
                <p className="text-xs text-red-600 mt-1">⚠️ Override Applied</p>
              )}
            </div>

            {/* Score Breakdown */}
            <div>
              <h4 className="text-sm font-semibold text-gray-600 mb-2">
                Score Breakdown
              </h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span>
                    ML Model (
                    {(currentResult.fusion.ml_weight * 100).toFixed(0)}%)
                  </span>
                  <span className="font-semibold">
                    {(
                      currentResult.fusion.score_breakdown?.ml_score * 100
                    ).toFixed(1)}
                    %
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Objects (25%)</span>
                  <span className="font-semibold">
                    {(
                      currentResult.fusion.score_breakdown?.object_score * 100
                    ).toFixed(1)}
                    %
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Pose (15%)</span>
                  <span className="font-semibold">
                    {(
                      currentResult.fusion.score_breakdown?.pose_score * 100
                    ).toFixed(1)}
                    %
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Motion (10%)</span>
                  <span className="font-semibold">
                    {(
                      currentResult.fusion.score_breakdown?.motion_score * 100
                    ).toFixed(1)}
                    %
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Reasoning */}
          <div className="mt-4 pt-4 border-t">
            <h4 className="text-sm font-semibold text-gray-600 mb-2">
              Reasoning
            </h4>
            <ul className="space-y-1">
              {fusionReasoning.map((reason, idx) => (
                <li key={idx} className="text-sm text-gray-700">
                  {reason}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* NEW: Frame Timeline */}
      {frameTimeline.length > 0 && (
        <div className="bg-white rounded-lg shadow-lg p-4">
          <h3 className="text-lg font-semibold mb-3">
            📊 Frame Timeline (Last {frameTimeline.length} frames)
          </h3>

          <div className="overflow-x-auto">
            <div
              className="flex space-x-1 pb-2"
              style={{ minWidth: "max-content" }}
            >
              {frameTimeline.map((entry, idx) => {
                const color =
                  entry.decision === "CRITICAL" || entry.decision === "HIGH"
                    ? "bg-red-500"
                    : entry.decision === "ABNORMAL" ||
                      entry.decision === "MEDIUM"
                    ? "bg-yellow-500"
                    : entry.decision === "SUSPICIOUS" ||
                      entry.decision === "LOW"
                    ? "bg-blue-500"
                    : "bg-green-500";

                const hasAnyAnomaly = Object.values(entry.anomalies).some(
                  (v) => v
                );
                const height = hasAnyAnomaly ? "h-16" : "h-8";

                return (
                  <div
                    key={idx}
                    className={`${color} ${height} w-2 rounded-t transition-all cursor-pointer hover:opacity-80`}
                    title={`Frame ${entry.frame_number}: ${entry.decision}\n${
                      entry.ml_prediction
                    } (${(entry.confidence * 100).toFixed(1)}%)\nML: ${
                      entry.anomalies.ml ? "Yes" : "No"
                    }, Motion: ${
                      entry.anomalies.motion ? "Yes" : "No"
                    }, Pose: ${entry.anomalies.pose ? "Yes" : "No"}`}
                  />
                );
              })}
            </div>
          </div>

          <div className="mt-2 text-xs text-gray-600">
            <p>
              Height indicates anomaly severity | Color indicates decision level
            </p>
            <p>
              Hover over bars for details | Last anomaly at frame{" "}
              {frameTimeline[frameTimeline.length - 1]?.frame_number}
            </p>
          </div>
        </div>
      )}

      {/* NEW: Auto-Saved Screenshots */}
      {savedScreenshots.length > 0 && (
        <div className="bg-white rounded-lg shadow-lg p-4">
          <h3 className="text-lg font-semibold mb-3">
            📸 Auto-Saved Screenshots ({savedScreenshots.length})
          </h3>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {savedScreenshots
              .slice(-8)
              .reverse()
              .map((screenshot, idx) => (
                <div
                  key={idx}
                  className="border rounded overflow-hidden hover:shadow-lg transition-shadow"
                >
                  <img
                    src={screenshot.url}
                    alt={screenshot.filename}
                    className="w-full h-32 object-cover cursor-pointer"
                    onClick={() => {
                      // Open in new tab
                      window.open(screenshot.url, "_blank");
                    }}
                  />
                  <div className="p-2 bg-gray-50">
                    <p
                      className="text-xs font-semibold truncate"
                      title={screenshot.filename}
                    >
                      Frame {screenshot.metadata.frame_number}
                    </p>
                    <p
                      className={`text-xs font-bold ${
                        screenshot.metadata.decision === "CRITICAL"
                          ? "text-red-600"
                          : screenshot.metadata.decision === "ABNORMAL"
                          ? "text-yellow-600"
                          : "text-blue-600"
                      }`}
                    >
                      {screenshot.metadata.decision}
                    </p>
                    <p className="text-xs text-gray-600 truncate">
                      {screenshot.metadata.ml_prediction}
                    </p>
                    <button
                      onClick={() => {
                        const a = document.createElement("a");
                        a.href = screenshot.url;
                        a.download = screenshot.filename;
                        a.click();
                      }}
                      className="mt-1 text-xs text-blue-500 hover:text-blue-700"
                    >
                      💾 Download
                    </button>
                  </div>
                </div>
              ))}
          </div>

          {savedScreenshots.length > 8 && (
            <p className="mt-2 text-sm text-gray-600">
              Showing last 8 of {savedScreenshots.length} screenshots
            </p>
          )}
        </div>
      )}

      {/* Detection History */}
      {detectionHistory.length > 0 && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold mb-3">Detection Log</h3>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {detectionHistory.map((entry, idx) => (
              <div
                key={idx}
                className={`p-3 rounded border-l-4 ${
                  entry.severity === "critical"
                    ? "border-red-500 bg-red-50"
                    : entry.severity === "warning"
                    ? "border-yellow-500 bg-yellow-50"
                    : entry.severity === "alert"
                    ? "border-orange-500 bg-orange-50"
                    : "border-green-500 bg-green-50"
                }`}
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">
                        {getSeverityIcon(entry.severity)}
                      </span>
                      <span className="font-bold">{entry.class}</span>
                      <span className="text-sm text-gray-500">
                        {(entry.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <p className="text-xs text-gray-600 mt-1">
                      {entry.timestamp.toLocaleString()}
                    </p>
                    {entry.objects && entry.objects.length > 0 && (
                      <p className="text-xs text-gray-700 mt-1">
                        Objects:{" "}
                        {entry.objects
                          .slice(0, 3)
                          .map((o) => o.class || o)
                          .join(", ")}
                        {entry.objects.length > 3 &&
                          ` +${entry.objects.length - 3} more`}
                      </p>
                    )}
                    {entry.dangerousObjects &&
                      entry.dangerousObjects.length > 0 && (
                        <p className="text-xs text-red-600 font-bold mt-1">
                          ⚠️ {entry.dangerousObjects.join(", ")}
                        </p>
                      )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default LiveCamera;
