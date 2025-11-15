import React, { useState, useEffect, useCallback, useRef } from 'react';
import '../styles/professional.css';

// Environment variables for WebSocket URL
const WS_BASE = process.env.REACT_APP_WS_BASE || 'ws://localhost:8000';

/**
 * Multi-Camera Grid Component
 * 
 * Displays all connected cameras in a responsive grid layout
 * with parallel real-time analysis for each camera feed.
 * 
 * Features:
 * - Auto-layout grid (2x2, 3x3, 4x4 based on camera count)
 * - Individual WebSocket connection per camera
 * - Per-camera detection overlays
 * - Click to focus/expand camera
 * - Real-time status indicators
 * - Performance optimized with lazy rendering
 */
export default function MultiCameraGrid({ cameras, isAnalyzing = false, onCameraSelect, onDetection }) {
  const [gridLayout, setGridLayout] = useState({ rows: 2, cols: 2 });
  const [focusedCamera, setFocusedCamera] = useState(null);
  const [cameraStates, setCameraStates] = useState({});
  const wsConnections = useRef({});
  const videoRefs = useRef({});
  const frameCanvasRefs = useRef({});
  const overlayCanvasRefs = useRef({});
  // Per-camera track maps: { [cameraId]: { tracks: Map(trackId -> {bbox, class, lastSeen, color}), nextColorIndex } }
  const trackMapsRef = useRef({});

  const _colorForIndex = (i) => {
    const palette = [
      '#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'
    ];
    return palette[i % palette.length];
  };
  const isAnalyzingRef = useRef(isAnalyzing); // Use ref to track analyzing state

  // Update ref when isAnalyzing changes
  useEffect(() => {
    isAnalyzingRef.current = isAnalyzing;
    console.log(`🔄 Analysis state changed: ${isAnalyzing ? 'STARTED' : 'STOPPED'}`);
  }, [isAnalyzing]);

  // Calculate optimal grid layout based on camera count
  useEffect(() => {
    const enabledCameras = cameras.filter(c => c.enabled);
    const count = enabledCameras.length;
    
    let rows, cols;
    if (count <= 1) {
      rows = 1;
      cols = 1;
    } else if (count <= 4) {
      rows = 2;
      cols = 2;
    } else if (count <= 9) {
      rows = 3;
      cols = 3;
    } else if (count <= 16) {
      rows = 4;
      cols = 4;
    } else {
      // For more than 16, calculate dynamically
      cols = Math.ceil(Math.sqrt(count));
      rows = Math.ceil(count / cols);
    }
    
    setGridLayout({ rows, cols });
  }, [cameras]);

  // Initialize camera states
  useEffect(() => {
    const initialStates = {};
    cameras.filter(c => c.enabled).forEach(camera => {
      initialStates[camera.id] = {
        status: 'connecting',
        fps: 0,
        detections: 0,
        lastDetection: null,
        error: null
      };
    });
    setCameraStates(initialStates);
  }, [cameras]);

  // Render frame for IP camera
  const renderFrame = useCallback((cameraId, frameData) => {
    const canvas = frameCanvasRefs.current[cameraId];
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = () => {
      // Resize canvas to match incoming frame so overlay coordinates align
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      // Ensure overlay canvas matches size
      const overlay = overlayCanvasRefs.current[cameraId];
      if (overlay) {
        overlay.width = canvas.width;
        overlay.height = canvas.height;
      }
    };
    img.src = `data:image/jpeg;base64,${frameData}`;
  }, []);

  // Draw detection overlay on canvas
  const drawDetectionOverlay = useCallback((cameraId, detection) => {
    const canvas = overlayCanvasRefs.current[cameraId];
    if (!canvas) {
      console.warn(`No overlay canvas found for camera ${cameraId}`);
      return;
    }

    const ctx = canvas.getContext('2d');

    // Clear previous overlays so boxes don't persist
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    console.log(`🎨 Drawing detection overlay for ${cameraId}:`, detection);

  // Helper to normalize bbox into x,y,w,h in canvas pixels
    const toXYWH = (bbox) => {
      if (!bbox || bbox.length < 4) return null;
      const cw = canvas.width;
      const ch = canvas.height;
      const [a, b, c, d] = bbox;

      // If bbox values are normalized (0..1)
      if (a > 0 && a <= 1 && b > 0 && b <= 1 && c > 0 && c <= 1 && d > 0 && d <= 1) {
        return [a * cw, b * ch, c * cw, d * ch]; // x,y,w,h
      }

      // If bbox looks like x1,y1,x2,y2 (x2 > x1 and y2 > y1)
      if (c > a && d > b && (c - a) <= cw && (d - b) <= ch) {
        return [a, b, c - a, d - b];
      }

      // Otherwise assume it's x,y,w,h in pixels
      return [a, b, c, d];
    };
    // If tracking info is available, prefer tracked_objects to drive overlays
    const nowTs = Date.now();
    const trackMapEntry = trackMapsRef.current[cameraId] || { tracks: new Map(), nextColorIndex: 0 };

    const updateTrack = (track) => {
      const tid = track.track_id != null ? String(track.track_id) : null;
      if (!tid) return;
      const bbox = track.bbox || track.box || track.rect;
      const cls = track.class || track.class_name || track.label || 'obj';
      let existing = trackMapEntry.tracks.get(tid);
      if (!existing) {
        const color = _colorForIndex(trackMapEntry.nextColorIndex++);
        existing = { bbox, class: cls, lastSeen: nowTs, color };
        trackMapEntry.tracks.set(tid, existing);
      } else {
        existing.bbox = bbox;
        existing.class = cls;
        existing.lastSeen = nowTs;
      }
    };

    // Draw from tracking if available
    if (detection.tracking && Array.isArray(detection.tracking)) {
      detection.tracking.forEach(t => updateTrack(t));

      // Draw each active track
      for (const [tid, info] of trackMapEntry.tracks.entries()) {
        // Skip expired
        if (nowTs - info.lastSeen > 3000) {
          trackMapEntry.tracks.delete(tid);
          continue;
        }
        const rect = toXYWH(info.bbox);
        if (!rect) continue;
        const [x, y, w_box, h_box] = rect;

        ctx.strokeStyle = info.color || '#00ff00';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w_box, h_box);

        // Label with track id and class
        const text = `ID:${tid} ${info.class}`;
        ctx.font = 'bold 12px Arial';
        const textMetrics = ctx.measureText(text);
        const pad = 6;
        const tx = x;
        const ty = Math.max(0, y - 18);
        ctx.fillStyle = info.color || 'rgba(0,255,0,0.8)';
        ctx.fillRect(tx, ty, textMetrics.width + pad, 18);
        ctx.fillStyle = '#fff';
        ctx.fillText(text, tx + 4, ty + 13);
      }

      // persist back
      trackMapsRef.current[cameraId] = trackMapEntry;
    } else {
      // Fallback: draw raw object detections
      if (detection.objects && detection.objects.length > 0) {
        detection.objects.forEach(obj => {
          const { bbox, label, confidence, is_dangerous } = obj;
          const rect = toXYWH(bbox);
          if (!rect) return;
          const [x, y, w_box, h_box] = rect;

          // Draw bounding box
          ctx.strokeStyle = is_dangerous ? '#ff0000' : '#00ff00';
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, w_box, h_box);

          // Draw label background
          ctx.fillStyle = is_dangerous ? 'rgba(255, 0, 0, 0.8)' : 'rgba(0, 255, 0, 0.8)';
          const text = `${label || 'obj'} ${(Math.round((confidence || 0) * 100))}%`;
          ctx.font = 'bold 14px Arial';
          const textMetrics = ctx.measureText(text);
          const pad = 6;
          const tx = x;
          const ty = Math.max(0, y - 20);
          ctx.fillRect(tx, ty, textMetrics.width + pad, 20);

          // Draw label text
          ctx.fillStyle = '#fff';
          ctx.fillText(text, tx + 4, ty + 14);
        });
      }
    }

    // Draw poses (keypoints)
    if (detection.poses && detection.poses.length > 0) {
      detection.poses.forEach(pose => {
        const kps = pose.keypoints || pose.kp || [];
        ctx.fillStyle = '#ffff00';
        kps.forEach(kp => {
          // support different keypoint shapes
          const x = kp.x || kp[0];
          const y = kp.y || kp[1];
          const conf = kp.confidence || kp[2] || kp.conf || 1;
          if (conf > 0.3 && typeof x === 'number' && typeof y === 'number') {
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fill();
          }
        });
      });
    }

    console.log(`✅ Detection overlay drawn for ${cameraId}`);
  }, []);

  // Handle detection
  const handleDetection = useCallback((camera, detection) => {
    console.log(`🚨 Detection handler called for ${camera.id}:`, detection);
    
    setCameraStates(prev => ({
      ...prev,
      [camera.id]: {
        ...prev[camera.id],
        detections: (prev[camera.id]?.detections || 0) + 1,
        lastDetection: detection
      }
    }));

    // Draw detection overlay
    drawDetectionOverlay(camera.id, detection);

    // Notify parent
    if (onDetection) {
      onDetection({
        camera,
        detection,
        timestamp: Date.now()
      });
    }
  }, [onDetection, drawDetectionOverlay]);

  // Start analysis WebSocket for webcam
  const startAnalysisWebSocket = useCallback((camera, videoElement) => {
    const wsUrl = `${WS_BASE}/ws/stream/${camera.id}`;
    const ws = new WebSocket(wsUrl);
    let frameInterval = null;

    ws.onopen = () => {
      console.log(`Analysis WebSocket connected for ${camera.id}`);
      
      // Send frames periodically - only when analyzing
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      const sendFrame = () => {
        // Only send frames if analysis is active (check ref for latest value)
        if (!isAnalyzingRef.current) {
          return;
        }

        if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
          canvas.width = videoElement.videoWidth;
          canvas.height = videoElement.videoHeight;
          ctx.drawImage(videoElement, 0, 0);
          
          canvas.toBlob((blob) => {
            if (blob && ws.readyState === WebSocket.OPEN && isAnalyzingRef.current) {
              const reader = new FileReader();
              reader.onload = () => {
                ws.send(JSON.stringify({
                  type: 'frame',
                  camera_id: camera.id,
                  camera_name: camera.name,
                  data: reader.result.split(',')[1] // base64 - backend expects 'data' field
                }));
              };
              reader.readAsDataURL(blob);
            }
          }, 'image/jpeg', 0.8);
        }
      };
      
      // Start frame sending interval
      frameInterval = setInterval(sendFrame, 1000 / (camera.fps || 15));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Backend sends "prediction" type, not "detection"
        if (data.type === 'prediction') {
          console.log(`✅ Received prediction for ${camera.id}:`, data);

          // Always draw latest objects/poses to keep overlay in sync
          try {
            const detectionForOverlay = {
              objects: data.data?.objects || [],
              poses: data.data?.poses || [],
              tracking: data.data?.tracking?.tracked_objects || data.data?.tracking?.tracks || []
            };
            drawDetectionOverlay(camera.id, detectionForOverlay);
          } catch (e) {
            console.error('Error drawing overlay:', e);
          }

          // Check if anomaly detected and call handler
          if (data.anomaly_detected) {
            handleDetection(camera, {
              anomaly_type: data.data?.fusion?.anomaly_type || 'Unknown',
              severity: data.data?.fusion?.severity || 'LOW',
              confidence: data.data?.fusion?.confidence || 0,
              explanation: data.data?.summary || 'Anomaly detected',
              bbox: data.data?.fusion?.bbox,
              objects: data.data?.objects || [],
              poses: data.data?.poses || [],
              timestamp: data.timestamp
            });
          }

          // Update FPS
          if (data.frame_number) {
            setCameraStates(prev => ({
              ...prev,
              [camera.id]: { 
                ...prev[camera.id], 
                fps: Math.round(data.frame_number / ((Date.now() - startTime) / 1000))
              }
            }));
          }
        } else if (data.type === 'heartbeat') {
          console.log(`💓 Heartbeat from ${camera.id}`);
        } else if (data.error) {
          console.error(`❌ Error from backend for ${camera.id}:`, data.error);
          setCameraStates(prev => ({
            ...prev,
            [camera.id]: { ...prev[camera.id], status: 'error', error: data.error }
          }));
        }
      } catch (error) {
        console.error(`Error processing message for ${camera.id}:`, error);
      }
    };

    const startTime = Date.now();

    ws.onerror = (error) => {
      console.error(`Analysis WebSocket error for ${camera.id}:`, error);
      if (frameInterval) clearInterval(frameInterval);
    };

    ws.onclose = () => {
      console.log(`Analysis WebSocket closed for ${camera.id}`);
      if (frameInterval) clearInterval(frameInterval);
    };

    wsConnections.current[`${camera.id}_analysis`] = ws;
    
    return () => {
      if (frameInterval) clearInterval(frameInterval);
      if (ws.readyState === WebSocket.OPEN) ws.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [handleDetection, drawDetectionOverlay]);

  // Connect webcam
  const connectWebcam = useCallback(async (camera) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: camera.resolution_width || 1280 },
          height: { ideal: camera.resolution_height || 720 },
          frameRate: { ideal: camera.fps || 15 }
        }
      });

      const videoElement = videoRefs.current[camera.id];
      if (videoElement) {
        // Stop any existing stream first
        if (videoElement.srcObject) {
          videoElement.srcObject.getTracks().forEach(track => track.stop());
        }
        
        videoElement.srcObject = stream;
        
        // Wait for video to be ready before playing
        videoElement.onloadedmetadata = () => {
          videoElement.play().catch(err => {
            console.error(`Error playing video for ${camera.id}:`, err);
          });

          // Ensure overlay canvas matches video dimensions so bbox coordinates align
          try {
            const overlay = overlayCanvasRefs.current[camera.id];
            if (overlay && videoElement.videoWidth && videoElement.videoHeight) {
              overlay.width = videoElement.videoWidth;
              overlay.height = videoElement.videoHeight;
            }
          } catch (e) {
            // non-fatal
          }
        };

        setCameraStates(prev => ({
          ...prev,
          [camera.id]: { ...prev[camera.id], status: 'online' }
        }));

        // Start analysis WebSocket after video is playing
        setTimeout(() => {
          startAnalysisWebSocket(camera, videoElement);
        }, 500);
      }
    } catch (error) {
      console.error(`Error connecting webcam ${camera.id}:`, error);
      setCameraStates(prev => ({
        ...prev,
        [camera.id]: { 
          ...prev[camera.id], 
          status: 'error',
          error: error.message
        }
      }));
    }
  }, [startAnalysisWebSocket]);

  // Connect IP camera
  const connectIPCamera = useCallback((camera) => {
    const wsUrl = `${WS_BASE}/ws/stream/${camera.id}`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log(`WebSocket connected for camera ${camera.id}`);
      setCameraStates(prev => ({
        ...prev,
        [camera.id]: { ...prev[camera.id], status: 'online' }
      }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'frame') {
          // Render frame
          renderFrame(camera.id, data.frame);
          
          // Update FPS
          setCameraStates(prev => ({
            ...prev,
            [camera.id]: { ...prev[camera.id], fps: data.fps || 0 }
          }));
        } else if (data.type === 'detection' || data.type === 'prediction') {
          // Draw overlay (prediction uses data.data)
          try {
            const det = data.type === 'prediction' ? {
              objects: data.data?.objects || [],
              poses: data.data?.poses || [],
              tracking: data.data?.tracking?.tracked_objects || data.data?.tracking?.tracks || []
            } : (data.detection || {});
            drawDetectionOverlay(camera.id, det);
          } catch (e) {
            console.error('Error drawing overlay for IP camera:', e);
          }

          // If anomaly, call handler
          const isAnom = data.type === 'prediction' ? data.anomaly_detected : (data.detection?.anomaly || false);
          if (isAnom) {
            const payload = data.type === 'prediction' ? {
              anomaly_type: data.data?.fusion?.anomaly_type || 'Unknown',
              severity: data.data?.fusion?.severity || 'LOW',
              confidence: data.data?.fusion?.confidence || 0,
              explanation: data.data?.summary || 'Anomaly detected',
              bbox: data.data?.fusion?.bbox,
              objects: data.data?.objects || [],
              poses: data.data?.poses || [],
              timestamp: data.timestamp
            } : data.detection;
            handleDetection(camera, payload);
          }
        }
      } catch (error) {
        console.error(`Error processing message for ${camera.id}:`, error);
      }
    };

    ws.onerror = (error) => {
      console.error(`WebSocket error for ${camera.id}:`, error);
      setCameraStates(prev => ({
        ...prev,
        [camera.id]: { 
          ...prev[camera.id], 
          status: 'error',
          error: 'Connection error'
        }
      }));
    };

    ws.onclose = () => {
      console.log(`WebSocket closed for ${camera.id}`);
      setCameraStates(prev => ({
        ...prev,
        [camera.id]: { ...prev[camera.id], status: 'offline' }
      }));
      
      // Attempt reconnect after 5 seconds
      setTimeout(() => {
        if (cameras.find(c => c.id === camera.id && c.enabled)) {
          connectIPCamera(camera);
        }
      }, 5000);
    };

    wsConnections.current[camera.id] = ws;
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [cameras, handleDetection, renderFrame, drawDetectionOverlay]); // WS_BASE is a constant

  // Get color based on severity
  const getSeverityColor = (severity) => {
    const colors = {
      CRITICAL: '#ff0000',
      HIGH: '#ff6600',
      MEDIUM: '#ffaa00',
      LOW: '#ffdd00'
    };
    return colors[severity] || '#00ff00';
  };

  // Connect to camera WebSocket
  const connectCamera = useCallback((camera) => {
    if (camera.type === 'webcam') {
      connectWebcam(camera);
    } else if (camera.type === 'ip_camera') {
      connectIPCamera(camera);
    }
  }, [connectWebcam, connectIPCamera]);

  // Connect/disconnect cameras - cameras ALWAYS stream, analysis is controlled separately
  useEffect(() => {
    // Copy refs to variables for cleanup
    const wsConns = wsConnections.current;
    const vidRefs = videoRefs.current;

    // Connect all enabled cameras immediately for streaming
    cameras.filter(c => c.enabled).forEach(camera => {
      connectCamera(camera);
    });

    // Cleanup on unmount
    return () => {
      Object.values(wsConns).forEach(ws => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.close();
        }
      });
      
      // Stop webcam streams
      cameras.forEach(camera => {
        const videoElement = vidRefs[camera.id];
        if (videoElement && videoElement.srcObject) {
          videoElement.srcObject.getTracks().forEach(track => track.stop());
        }
      });
    };
  }, [cameras, connectCamera]);

  // Handle camera click
  const handleCameraClick = (camera) => {
    setFocusedCamera(focusedCamera?.id === camera.id ? null : camera);
    if (onCameraSelect) {
      onCameraSelect(camera);
    }
  };

  const enabledCameras = cameras.filter(c => c.enabled);

  return (
    <div className="multi-camera-grid-container">
      <div 
        className="camera-grid"
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${gridLayout.cols}, 1fr)`,
          gridTemplateRows: `repeat(${gridLayout.rows}, 1fr)`,
          gap: '10px',
          height: '100%',
          padding: '10px'
        }}
      >
        {enabledCameras.map((camera) => {
          const state = cameraStates[camera.id] || {};
          const isFocused = focusedCamera?.id === camera.id;
          
          return (
            <div
              key={camera.id}
              className={`camera-grid-item ${isFocused ? 'focused' : ''} ${state.status}`}
              onClick={() => handleCameraClick(camera)}
              style={{
                position: 'relative',
                border: `3px solid ${isFocused ? '#00aaff' : '#333'}`,
                borderRadius: '8px',
                overflow: 'hidden',
                background: '#000',
                cursor: 'pointer',
                transition: 'all 0.3s ease'
              }}
            >
              {/* Video/Canvas Display */}
              <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                {camera.type === 'webcam' ? (
                  <video
                    ref={el => videoRefs.current[camera.id] = el}
                    autoPlay
                    playsInline
                    muted
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'contain'
                    }}
                  />
                ) : (
                  <canvas
                    ref={el => frameCanvasRefs.current[camera.id] = el}
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'contain'
                    }}
                  />
                )}
                
                {/* Detection Overlay Canvas */}
                <canvas
                  ref={el => overlayCanvasRefs.current[camera.id] = el}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    pointerEvents: 'none'
                  }}
                />
              </div>

              {/* Camera Info Overlay */}
              <div
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  background: 'linear-gradient(to bottom, rgba(0,0,0,0.8), transparent)',
                  padding: '10px',
                  color: '#fff'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontSize: '14px', fontWeight: 'bold' }}>{camera.name}</div>
                    <div style={{ fontSize: '11px', color: '#aaa' }}>{camera.location}</div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div 
                      style={{
                        fontSize: '10px',
                        color: state.status === 'online' ? '#0f0' : '#f00',
                        fontWeight: 'bold'
                      }}
                    >
                      ● {state.status?.toUpperCase() || 'OFFLINE'}
                    </div>
                    <div style={{ fontSize: '10px', color: '#aaa' }}>
                      {state.fps || 0} FPS
                    </div>
                  </div>
                </div>
              </div>

              {/* Detection Count Badge */}
              {state.detections > 0 && (
                <div
                  style={{
                    position: 'absolute',
                    top: '10px',
                    right: '10px',
                    background: '#ff0000',
                    color: '#fff',
                    borderRadius: '50%',
                    width: '30px',
                    height: '30px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '12px',
                    fontWeight: 'bold'
                  }}
                >
                  {state.detections}
                </div>
              )}

              {/* Analysis Status Badge */}
              <div style={{
                position: 'absolute',
                bottom: '10px',
                left: '10px',
                padding: '5px 10px',
                borderRadius: '5px',
                fontSize: '11px',
                fontWeight: 'bold',
                background: isAnalyzing && state.status === 'online'
                  ? 'rgba(0, 255, 0, 0.9)'
                  : state.status === 'online'
                  ? 'rgba(0, 170, 255, 0.9)'
                  : state.status === 'error'
                  ? 'rgba(255, 0, 0, 0.9)'
                  : 'rgba(255, 165, 0, 0.9)',
                color: '#000'
              }}>
                {state.status === 'online'
                  ? isAnalyzing
                    ? '🔍 ANALYZING'
                    : '📹 STREAMING'
                  : state.status === 'error'
                  ? '❌ ERROR'
                  : '🔄 CONNECTING'}
              </div>

              {/* Last Detection Info */}
              {state.lastDetection && (
                <div
                  style={{
                    position: 'absolute',
                    bottom: 0,
                    left: 0,
                    right: 0,
                    background: 'rgba(0,0,0,0.8)',
                    padding: '8px',
                    color: '#fff',
                    fontSize: '11px'
                  }}
                >
                  <div style={{ fontWeight: 'bold', color: getSeverityColor(state.lastDetection.severity) }}>
                    {state.lastDetection.anomaly_type}
                  </div>
                  <div style={{ color: '#aaa' }}>
                    Confidence: {(state.lastDetection.confidence * 100).toFixed(0)}%
                  </div>
                </div>
              )}

              {/* Error Overlay */}
              {state.status === 'error' && (
                <div
                  style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    background: 'rgba(255,0,0,0.8)',
                    padding: '15px',
                    borderRadius: '8px',
                    color: '#fff',
                    textAlign: 'center',
                    maxWidth: '80%'
                  }}
                >
                  <div style={{ fontSize: '18px', marginBottom: '5px' }}>⚠️</div>
                  <div style={{ fontSize: '12px' }}>Connection Error</div>
                  {state.error && (
                    <div style={{ fontSize: '10px', marginTop: '5px', color: '#ffcccc' }}>
                      {state.error}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Grid Info */}
      <div
        style={{
          position: 'absolute',
          bottom: '20px',
          right: '20px',
          background: 'rgba(0,0,0,0.7)',
          padding: '10px 15px',
          borderRadius: '5px',
          color: '#fff',
          fontSize: '12px'
        }}
      >
        {enabledCameras.length} camera{enabledCameras.length !== 1 ? 's' : ''} • 
        {gridLayout.rows}×{gridLayout.cols} grid
      </div>
    </div>
  );
}
