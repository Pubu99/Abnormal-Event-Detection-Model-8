import React, { useState, useEffect, useRef } from "react";

export default function LiveCameraV2({
  camera, // Camera configuration object
  onAnomaly,
  onStreamReady, // NEW: Pass stream to parent
  onDetectionData, // NEW: Pass detection data to parent
  onNormalFrame,
  autoResumeOnClose = true,
  resumeDelayMs = 3000,
  maxAutoResumeAttempts = 5,
}) {
  const [isConnected, setIsConnected] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [status, setStatus] = useState("Disconnected");
  const [fps, setFps] = useState(0);
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState(null);

  // Get camera info
  const cameraId = camera?.id || "cam-001";
  const cameraName = camera?.name || "Camera";
  const cameraType = camera?.type || "webcam";
  const isWebcam = cameraType === "webcam";
  const isIPCamera = cameraType === "ip_camera";

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const fpsCounterRef = useRef({ count: 0, lastTime: Date.now() });
  const manualStopRef = useRef(false);
  const autoResumeAttemptsRef = useRef(0);
  const autoResumeTimerRef = useRef(null);

  useEffect(() => {
    // Enumerate devices on mount
    enumerateDevices();

    return () => {
      stopCamera();
      if (autoResumeTimerRef.current) {
        clearTimeout(autoResumeTimerRef.current);
        autoResumeTimerRef.current = null;
      }
    };
  }, []);

  const enumerateDevices = async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        console.warn("enumerateDevices() not supported.");
        return;
      }

      const list = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = list.filter((d) => d.kind === "videoinput");
      setDevices(videoInputs);

      // If no selection yet, pick default camera (first)
      if (!selectedDeviceId && videoInputs.length > 0) {
        setSelectedDeviceId(videoInputs[0].deviceId);
      }
    } catch (err) {
      console.error("Error enumerating devices:", err);
    }
  };

  const handleDeviceChange = (e) => {
    setSelectedDeviceId(e.target.value);
  };

  const startCamera = async () => {
    try {
      setStatus("Requesting camera access...");

      // For webcam, use getUserMedia
      if (isWebcam) {
        const constraints = {
          video: {
            width: { ideal: camera?.resolution_width || 1280 },
            height: { ideal: camera?.resolution_height || 720 },
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
      } 
      // For IP camera, no need to start camera (backend handles it)
      else if (isIPCamera) {
        setIsConnected(true);
        setStatus("IP Camera ready - Start analysis to connect");
      }
    } catch (error) {
      console.error("Camera error:", error);
      setStatus(`Camera error: ${error.message}`);
      if (isWebcam) {
        alert("Failed to access camera. Please check permissions.");
      }
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current && isWebcam) {
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

    // Clear manual stop flag (we are intentionally starting)
    manualStopRef.current = false;
    // reset attempts
    autoResumeAttemptsRef.current = 0;
    if (autoResumeTimerRef.current) {
      clearTimeout(autoResumeTimerRef.current);
      autoResumeTimerRef.current = null;
    }

    try {
      setStatus("Connecting to analysis server...");

      // Connect to camera-specific WebSocket endpoint
      const ws = new WebSocket(`ws://localhost:8000/ws/stream/${cameraId}`);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus("Analyzing...");
        setIsAnalyzing(true);

        // reset auto-resume attempts on successful connect
        autoResumeAttemptsRef.current = 0;
        if (autoResumeTimerRef.current) {
          clearTimeout(autoResumeTimerRef.current);
          autoResumeTimerRef.current = null;
        }

        // ⭐ For webcam: Send frames at configured FPS
        // ⭐ For IP camera: Backend pulls frames, no need to send
        if (isWebcam) {
          const targetFps = camera?.fps || 15;
          const frameInterval = 1000 / targetFps; // ms per frame
          
          intervalRef.current = setInterval(() => {
            sendFrame();
          }, frameInterval);
        } else {
          // For IP cameras, we don't send frames
          // Backend pulls from RTSP and sends detection results
          console.log(`🎥 IP Camera ${cameraId}: Backend handling frame capture`);
        }
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
                camera_id: data.camera_id,
                camera_name: data.camera_name,
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
              if (isWebcam) {
                captureAnomalyScreenshot(data);
              }
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

      ws.onclose = (event) => {
        console.log("🔌 WebSocket closed:", event);
        setStatus("Analysis stopped");
        setIsAnalyzing(false);
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }

        // Auto-resume if appropriate
        if (!manualStopRef.current && autoResumeOnClose) {
          const attempts = autoResumeAttemptsRef.current || 0;
          if (attempts < maxAutoResumeAttempts) {
            const delay = resumeDelayMs * Math.pow(2, attempts);
            console.warn(
              `🔁 LiveCameraV2 auto-resume in ${delay}ms (attempt ${attempts + 1}/${maxAutoResumeAttempts})`
            );
            autoResumeTimerRef.current = setTimeout(() => {
              autoResumeAttemptsRef.current = attempts + 1;
              if (isConnected) {
                console.log("🔁 LiveCameraV2 auto-resume: restarting analysis...");
                startAnalysis();
              } else {
                console.log("⏸️ LiveCameraV2 auto-resume aborted: camera disconnected");
              }
            }, delay);
          } else {
            console.error("❌ LiveCameraV2 auto-resume max attempts reached");
          }
        }
      };
    } catch (error) {
      console.error("Analysis start error:", error);
      setStatus(`Error: ${error.message}`);
    }
  };

  const stopAnalysis = () => {
    // mark manual stop so auto-resume won't restart
    manualStopRef.current = true;

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (autoResumeTimerRef.current) {
      clearTimeout(autoResumeTimerRef.current);
      autoResumeTimerRef.current = null;
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
          <div>
            <h3 className="text-white font-bold text-lg flex items-center gap-2">
              {isWebcam ? "💻" : "📹"} {cameraName}
              <span className="text-slate-400 text-sm font-normal">({cameraId})</span>
            </h3>
            <p className="text-slate-500 text-xs mt-1">
              {isWebcam ? "Webcam Camera" : "IP Camera (RTSP)"} • {camera?.location || "No location"}
            </p>
          </div>
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
            {/* Device selector */}
            <div className="col-span-2">
              <label className="text-slate-300 text-sm">Select Camera</label>
              <div className="flex items-center gap-2 mt-1">
                <select
                  value={selectedDeviceId || ""}
                  onChange={handleDeviceChange}
                  className="bg-slate-800 text-white py-2 px-3 rounded-lg w-full"
                >
                  {devices.length === 0 ? (
                    <option value="">No cameras found</option>
                  ) : (
                    devices.map((d) => (
                      <option key={d.deviceId} value={d.deviceId}>
                        {d.label || `Camera (${d.deviceId})`}
                      </option>
                    ))
                  )}
                </select>
                <button
                  onClick={enumerateDevices}
                  className="bg-slate-700 hover:bg-slate-600 text-white font-medium py-2 px-3 rounded-lg"
                  title="Refresh device list"
                >
                  Refresh
                </button>
              </div>
              <p className="text-slate-400 text-xs mt-1">If you use OBS Virtual Camera, start the Virtual Camera in OBS first, then click Refresh and select it here.</p>
            </div>

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
            {/* Hidden video element - only for webcam frame capture */}
            {isWebcam && (
              <video
                ref={videoRef}
                className="hidden"
                autoPlay
                playsInline
                muted
              />
            )}
            {!isAnalyzing && isConnected && (
              <div className="bg-green-600/20 border border-green-600/30 rounded-lg p-4">
                <div className="flex items-center justify-center gap-3">
                  <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-green-400 font-semibold">
                    ✓ {cameraName} Connected - Click "Start Analysis" to begin
                  </span>
                </div>
              </div>
            )}
            {isAnalyzing && (
              <div className="bg-blue-600/20 border border-blue-600/30 rounded-lg p-4">
                <div className="flex items-center justify-center gap-3">
                  <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
                  <span className="text-blue-400 font-semibold">
                    🔴 Analyzing {cameraName} - Watch the live feed above
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
