import React, { useState, useEffect, useRef } from "react";

export default function LiveCameraV2({
  onAnomaly,
  onStreamReady, // NEW: Pass stream to parent
  onDetectionData, // NEW: Pass detection data to parent
  onNormalFrame,
}) {
  const [isConnected, setIsConnected] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [status, setStatus] = useState("Disconnected");
  const [fps, setFps] = useState(0);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const fpsCounterRef = useRef({ count: 0, lastTime: Date.now() });

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const startCamera = async () => {
    try {
      setStatus("Requesting camera access...");

      const constraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        const video = videoRef.current;

        // Set up event handlers BEFORE assigning srcObject
        const waitForVideo = new Promise((resolve) => {
          let resolved = false;

          const checkAndResolve = () => {
            if (
              !resolved &&
              video.videoWidth > 0 &&
              video.videoHeight > 0 &&
              video.readyState >= 2
            ) {
              resolved = true;
              resolve();
            }
          };

          video.onloadedmetadata = async () => {
            try {
              await video.play();
            } catch (err) {
              console.error("Play error:", err);
            }
            checkAndResolve();
          };

          video.onloadeddata = () => {
            checkAndResolve();
          };

          video.oncanplay = () => {
            checkAndResolve();
          };

          video.onplaying = () => {
            checkAndResolve();
          };

          // Timeout fallback
          setTimeout(() => {
            if (!resolved) {
              resolved = true;
              resolve();
            }
          }, 3000);
        });

        // Assign srcObject after handlers are set
        video.srcObject = stream;

        // Try to load and play explicitly
        video.load();
        try {
          await video.play();
        } catch (err) {
          // Will retry via event handlers
        }

        // Wait for video to be ready
        await waitForVideo;

        // Additional stabilization time
        await new Promise((resolve) => setTimeout(resolve, 500));
      }

      setIsConnected(true);
      setStatus("Camera active");

      // ⭐ PROFESSIONAL: Pass stream to parent for direct display
      if (onStreamReady) {
        onStreamReady(stream);
      }
    } catch (error) {
      console.error("Camera error:", error);
      setStatus(`Camera error: ${error.message}`);
      alert("Failed to access camera. Please check permissions.");
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    setIsConnected(false);
    setIsAnalyzing(false);
    setStatus("Disconnected");
    setFps(0);
  };

  const startAnalysis = () => {
    if (!isConnected) {
      alert("Please connect camera first");
      return;
    }

    try {
      setStatus("Connecting to analysis server...");

      const ws = new WebSocket("ws://localhost:8000/ws/stream");
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus("Analyzing...");
        setIsAnalyzing(true);

        // ⭐ PROFESSIONAL: Send frames for analysis at 15 FPS
        intervalRef.current = setInterval(() => {
          sendFrame();
        }, 67); // ~15 FPS
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // Update FPS counter
          fpsCounterRef.current.count++;
          const now = Date.now();
          if (now - fpsCounterRef.current.lastTime >= 1000) {
            setFps(fpsCounterRef.current.count);
            fpsCounterRef.current.count = 0;
            fpsCounterRef.current.lastTime = now;
          }

          if (data.type === "prediction" && data.data) {
            // ⭐ PROFESSIONAL: Only send detection data, not frame
            if (onDetectionData) {
              onDetectionData({
                objects: data.data.objects || [],
                poses: data.data.poses || [],
                motion: data.data.motion || null,
                fusion: data.data.fusion || null,
              });
            }

            // Check for anomaly
            if (data.data.fusion && data.anomaly_detected) {
              onAnomaly(data.data.fusion, data);

              // ⭐ CAPTURE SCREENSHOT only on anomaly
              captureAnomalyScreenshot(data);
            } else {
              onNormalFrame();
            }
          }
        } catch (error) {
          console.error("Error processing message:", error);
        }
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setStatus("Connection error");
      };

      ws.onclose = () => {
        setStatus("Analysis stopped");
        setIsAnalyzing(false);
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      };
    } catch (error) {
      console.error("Analysis start error:", error);
      setStatus(`Error: ${error.message}`);
    }
  };

  const stopAnalysis = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    setIsAnalyzing(false);
    setStatus("Camera active");
    setFps(0);
  };

  // ⭐ PROFESSIONAL: Capture and save screenshot on anomaly detection
  const captureAnomalyScreenshot = async (anomalyData) => {
    if (!videoRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;

    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert to blob and send to backend for saving
    canvas.toBlob(
      async (blob) => {
        if (blob) {
          const formData = new FormData();
          const timestamp = Date.now();
          const anomalyType = anomalyData.data.fusion.anomaly_type.replace(
            /\s+/g,
            "_"
          );
          const filename = `anomaly_${timestamp}_${anomalyType}.jpg`;

          formData.append("file", blob, filename);
          formData.append("anomaly_type", anomalyData.data.fusion.anomaly_type);
          formData.append("severity", anomalyData.data.fusion.severity);
          formData.append("timestamp", timestamp);

          try {
            // Send to backend to save in uploads folder
            const response = await fetch(
              "http://localhost:8000/api/save-screenshot",
              {
                method: "POST",
                body: formData,
              }
            );

            if (response.ok) {
              console.log(`Screenshot saved: ${filename}`);
            } else {
              console.error("Failed to save screenshot");
            }
          } catch (error) {
            console.error("Error saving screenshot:", error);
          }
        }
      },
      "image/jpeg",
      0.95
    );
  };

  const sendFrame = () => {
    if (!videoRef.current || !canvasRef.current || !wsRef.current) return;
    if (wsRef.current.readyState !== WebSocket.OPEN) return;

    try {
      const canvas = canvasRef.current;
      const video = videoRef.current;

      // Skip if video dimensions not ready
      if (
        video.videoWidth === 0 ||
        video.videoHeight === 0 ||
        video.readyState < 2
      ) {
        return;
      }

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const ctx = canvas.getContext("2d", {
        alpha: false, // No transparency = faster
        willReadFrequently: false, // Optimize for one-time reads
      });

      // ⭐ Use image smoothing for better quality at lower file size ⭐
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "medium";
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // ⭐ OPTIMIZED FOR SMOOTH REAL-TIME STREAMING ⭐
      // Quality 0.7 for even faster encoding (still good visual quality)
      canvas.toBlob(
        (blob) => {
          if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
            const reader = new FileReader();
            reader.onloadend = () => {
              if (wsRef.current?.readyState === WebSocket.OPEN) {
                const base64 = reader.result.split(",")[1];
                wsRef.current.send(
                  JSON.stringify({
                    type: "frame",
                    data: base64,
                  })
                );
              }
            };
            reader.readAsDataURL(blob);
          }
        },
        "image/jpeg",
        0.7 // Optimized for speed (70% quality)
      );
    } catch (error) {
      console.error("Error sending frame:", error);
    }
  };

  return (
    <div className="bg-slate-900/50 backdrop-blur-sm rounded-xl border border-slate-800 shadow-xl">
      {/* Header */}
      <div className="px-6 py-4 border-b border-slate-800">
        <div className="flex items-center justify-between">
          <h3 className="text-white font-bold text-lg">CAMERA CONTROLS</h3>
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                isConnected ? "bg-emerald-500 animate-pulse" : "bg-slate-600"
              }`}
            ></div>
            <span className="text-slate-400 text-sm">{status}</span>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="p-6">
        <div className="grid grid-cols-2 gap-4">
          {/* Camera Control */}
          {!isConnected ? (
            <button
              onClick={startCamera}
              className="bg-gradient-to-r from-emerald-600 to-emerald-700 hover:from-emerald-700 hover:to-emerald-800 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 transform hover:scale-105 shadow-lg"
            >
              <div className="flex items-center justify-center gap-2">
                <svg
                  className="w-5 h-5"
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
                <span>Start Camera</span>
              </div>
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 transform hover:scale-105 shadow-lg"
            >
              <div className="flex items-center justify-center gap-2">
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
                <span>Stop Camera</span>
              </div>
            </button>
          )}

          {/* Analysis Control */}
          {!isAnalyzing ? (
            <button
              onClick={startAnalysis}
              disabled={!isConnected}
              className={`${
                isConnected
                  ? "bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 transform hover:scale-105"
                  : "bg-slate-700 cursor-not-allowed"
              } text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 shadow-lg`}
            >
              <div className="flex items-center justify-center gap-2">
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <span>Start Analysis</span>
              </div>
            </button>
          ) : (
            <button
              onClick={stopAnalysis}
              className="bg-gradient-to-r from-orange-600 to-orange-700 hover:from-orange-700 hover:to-orange-800 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 transform hover:scale-105 shadow-lg"
            >
              <div className="flex items-center justify-center gap-2">
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <span>Pause Analysis</span>
              </div>
            </button>
          )}
        </div>

        {/* Status Info */}
        {isAnalyzing && (
          <div className="mt-4 grid grid-cols-3 gap-4">
            <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
              <div className="text-slate-400 text-xs mb-1">Processing Rate</div>
              <div className="text-cyan-400 text-xl font-bold">
                {fps.toFixed(1)} FPS
              </div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
              <div className="text-slate-400 text-xs mb-1">Connection</div>
              <div className="text-emerald-400 text-xl font-bold">ACTIVE</div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3 border border-slate-700">
              <div className="text-slate-400 text-xs mb-1">Status</div>
              <div className="text-white text-xl font-bold">LIVE</div>
            </div>
          </div>
        )}

        {/* Camera Feed - Always render for ref, show when connected */}
        <div className={`mt-4 ${isConnected ? "" : "hidden"}`}>
          <div className="relative bg-black rounded-lg overflow-hidden shadow-2xl border-2 border-slate-700">
            {/* Hidden video element - only for frame capture */}
            <video
              ref={videoRef}
              className="hidden"
              autoPlay
              playsInline
              muted
            />
            {!isAnalyzing && isConnected && (
              <div className="bg-green-600/20 border border-green-600/30 rounded-lg p-4">
                <div className="flex items-center justify-center gap-3">
                  <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-green-400 font-semibold">
                    ✓ Camera Connected - Click "Start Analysis" to begin
                  </span>
                </div>
              </div>
            )}
            {isAnalyzing && (
              <div className="bg-blue-600/20 border border-blue-600/30 rounded-lg p-4">
                <div className="flex items-center justify-center gap-3">
                  <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
                  <span className="text-blue-400 font-semibold">
                    🔴 Analysis in progress - Watch the live feed above
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Hidden Canvas for processing */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}
