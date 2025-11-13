"""
FastAPI Backend for Anomaly Detection System
Professional REST API with WebSocket support for real-time predictions
"""

from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import os
import signal
import sys
from pathlib import Path
import cv2
import numpy as np
import json
from datetime import datetime
import asyncio
import multiprocessing
import time
import threading
import subprocess


def _sanitize_for_json(obj):
    """Recursively convert common non-JSON types to JSON-serializable Python types.

    Handles numpy scalars/arrays, torch tensors, datetimes and falls back to str()
    for unknown objects. This is best-effort and should keep the websocket
    streaming stable when ML libraries return numpy.float32, tensors, etc.
    """
    # local imports to avoid hard dependency at module import time
    try:
        import numpy as _np
    except Exception:
        _np = None
    try:
        import torch as _torch
    except Exception:
        _torch = None

    # Primitives
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int, float)):
        # ensure native python float (not numpy.float32)
        if isinstance(obj, float):
            return float(obj)
        return obj

    # Numpy scalar
    if _np is not None and isinstance(obj, _np.generic):
        try:
            return obj.item()
        except Exception:
            return float(obj)

    # Numpy array
    if _np is not None and isinstance(obj, _np.ndarray):
        try:
            return obj.tolist()
        except Exception:
            return [ _sanitize_for_json(x) for x in obj ]

    # Torch tensor
    if _torch is not None and isinstance(obj, _torch.Tensor):
        try:
            if obj.numel() == 1:
                return obj.detach().cpu().item()
            return obj.detach().cpu().tolist()
        except Exception:
            try:
                return obj.detach().cpu().numpy().tolist()
            except Exception:
                return str(obj)

    # Mapping
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # ensure key is a string
            key = k if isinstance(k, str) else str(k)
            out[key] = _sanitize_for_json(v)
        return out

    # Iterable
    if isinstance(obj, (list, tuple, set)):
        return [ _sanitize_for_json(x) for x in obj ]

    # Datetime
    if isinstance(obj, datetime):
        return obj.isoformat()

    # Fallback: try .item(), then json.dumps, finally str()
    if hasattr(obj, 'item'):
        try:
            return obj.item()
        except Exception:
            pass
    try:
        # if it's already serializable, return as-is
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from inference.engine import AnomalyDetector
from core.unified_pipeline import UnifiedDetectionPipeline


# Initialize FastAPI app
app = FastAPI(
    title="Anomaly Detection API",
    description="Professional API for real-time abnormal event detection with Multi-Modal Analysis",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector = None
unified_pipeline = None
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# Pydantic models
class PredictionResponse(BaseModel):
    success: bool
    video_path: str
    anomaly_detected: bool
    max_anomaly_score: float
    predicted_class: str
    confidence: float
    top3_predictions: List[Dict]
    dangerous_objects: List[Dict]
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Initialize the anomaly detector on startup."""
    global detector, unified_pipeline
    
    print("🚀 Starting Enhanced Anomaly Detection API v3.0...")
    
    try:
        # Get absolute paths relative to project root
        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / "models" / "best_model.pth"
        config_path = project_root / "configs" / "config_research_enhanced.yaml"
        
        detector = AnomalyDetector(
            model_path=str(model_path),
            config_path=str(config_path),
            yolo_model="yolov10s.pt",
            device="cuda",
            confidence_threshold=0.7
        )

        # RL policy toggles via environment variables
        # Set RL_USE_POLICY=1 and RL_THRESHOLD=0.6 to enable
        rl_use = os.environ.get('RL_USE_POLICY', os.environ.get('USE_RL_POLICY'))
        if rl_use is not None:
            try:
                detector.use_rl_policy = str(rl_use).lower() not in ('0', 'false', 'off')
            except Exception:
                detector.use_rl_policy = False

        rl_thresh = os.environ.get('RL_THRESHOLD')
        if rl_thresh is not None:
            try:
                detector.rl_threshold = float(rl_thresh)
            except Exception:
                detector.rl_threshold = 0.5

        # Force all modalities (ML, YOLO, Pose, Motion) to contribute to
        # the fusion. This ensures the frontend "DETECTION METHODS USED"
        # panel always includes all methods and that the fusion receives
        # signals from ML and Pose even on short/early streams.
        # Toggle with environment variable FORCE_ALL_MODALITIES=0 to disable.
        force_flag = os.environ.get('FORCE_ALL_MODALITIES', '1').lower() not in ('0', 'false', 'off')
        detector.force_all_modalities = force_flag
        
        # Initialize unified pipeline with all advanced features
        print("\n🎯 Initializing Unified Detection Pipeline...")
        unified_pipeline = UnifiedDetectionPipeline(detector)
        
        # --- Auto-retrain configuration (enabled by default) ---
        # Controlled via environment variables. Defaults chosen to be conservative.
        app.state.rl_autoretrain_enabled = str(os.environ.get('RL_AUTORETRAIN_ENABLED', '1')).lower() not in ('0', 'false', 'off')
        app.state.rl_autoretrain_min_new = int(os.environ.get('RL_AUTORETRAIN_MIN_NEW', os.environ.get('RL_AUTORETRAIN_MIN', 50)))
        app.state.rl_autoretrain_interval = int(os.environ.get('RL_AUTORETRAIN_INTERVAL_SECONDS', 300))
        app.state.rl_autoretrain_niceness = int(os.environ.get('RL_AUTORETRAIN_NICENESS', 10))
        app.state.rl_autoretrain_batch_size = int(os.environ.get('RL_AUTORETRAIN_BATCH_SIZE', 16))
        app.state.rl_autoretrain_epochs = int(os.environ.get('RL_AUTORETRAIN_EPOCHS', 5))

        # last trained experience id persistence file
        DATA_DIR = Path(__file__).parent.parent / 'data'
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        app.state.rl_last_trained_file = DATA_DIR / 'rl_last_trained_id.txt'

        # lock used when swapping/loading in-memory model
        app.state.rl_model_load_lock = threading.Lock()

        # track current training process (set when spawn child)
        app.state.current_training_proc = None
        app.state.training_paused = False

        # start background retrain task if enabled
        if app.state.rl_autoretrain_enabled:
            # spawn background coroutine
            loop = asyncio.get_event_loop()
            loop.create_task(_rl_autoretrain_daemon())

        print("✅ Enhanced API Ready!")
        print("   Features: ML Model + YOLO + Motion + Pose + Tracking + Rules\n")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        print("   Make sure best_model.pth is in models/ directory")
        import traceback
        traceback.print_exc()


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "model_loaded": detector is not None,
        "device": detector.device if detector else "unknown",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check."""
    return {
        "status": "healthy" if detector else "model_not_loaded",
        "model_loaded": detector is not None,
        "device": detector.device if detector else "unknown",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_video(
    file: UploadFile = File(...),
    use_yolo: bool = True
):
    """
    Upload and analyze a video for anomalies.
    
    Args:
        file: Video file (mp4, avi, mov)
        use_yolo: Whether to use YOLO object detection
        
    Returns:
        Analysis results with anomaly detection
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Supported: mp4, avi, mov, mkv"
        )
    
    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    filepath = UPLOAD_DIR / filename
    
    try:
        # Save file
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        
        print(f"📁 Saved: {filepath}")
        
        # Run prediction
        results = detector.predict_video(str(filepath), use_yolo=use_yolo)
        
        # Get most confident prediction
        if results['predictions']:
            most_confident = max(
                results['predictions'], 
                key=lambda x: x['confidence']
            )
            
            return {
                "success": True,
                "video_path": str(filepath),
                "anomaly_detected": results['anomaly_detected'],
                "max_anomaly_score": results['max_anomaly_score'],
                "predicted_class": most_confident['predicted_class'],
                "confidence": most_confident['confidence'],
                "top3_predictions": most_confident['top3_predictions'],
                "dangerous_objects": results['dangerous_objects_detected'],
                "message": "Analysis complete"
            }
        else:
            raise HTTPException(status_code=500, detail="No predictions generated")
            
    except Exception as e:
        # Clean up file on error
        if filepath.exists():
            filepath.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    """
    Analyze a single frame/image for anomalies.
    
    Args:
        file: Image file (jpg, png)
        
    Returns:
        YOLO object detection results
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect objects
        yolo_result = detector.detect_objects(frame)
        
        return {
            "success": True,
            "objects_detected": yolo_result['objects'],
            "dangerous": yolo_result['dangerous'],
            "num_objects": len(yolo_result['objects'])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/stream/{camera_id}")
async def websocket_stream(websocket: WebSocket, camera_id: str):
    """
    Enhanced WebSocket endpoint for real-time multi-modal analysis with multi-camera support.
    Features: ML Model + YOLO + Motion + Pose + Tracking + Rules
    OPTIMIZED: Handles connection timeouts and async processing
    
    Args:
        camera_id: Camera ID to stream from (webcam sends frames, IP camera pulls from backend)
    """
    await websocket.accept()
    
    if detector is None or unified_pipeline is None:
        await websocket.send_json({
            "error": "Model not loaded"
        })
        await websocket.close()
        return
    
    # Get camera configuration
    try:
        from services.camera_manager import get_camera_manager, CameraType, CameraStatus
        camera_manager = get_camera_manager()
        camera = camera_manager.get_camera(camera_id)
        
        if not camera:
            await websocket.send_json({
                "error": f"Camera {camera_id} not found"
            })
            await websocket.close()
            return
        
        if not camera.enabled:
            await websocket.send_json({
                "error": f"Camera {camera_id} is disabled"
            })
            await websocket.close()
            return
        
        # Update camera status
        camera_manager.update_camera_status(camera_id, CameraStatus.CONNECTING)
        
    except Exception as e:
        await websocket.send_json({
            "error": f"Error loading camera config: {str(e)}"
        })
        await websocket.close()
        return
    
    print(f"🔌 WebSocket connected for camera {camera_id} ({camera.name}) - Type: {camera.type.value}")
    
    # For IP cameras, start the capture thread
    ip_capture = None
    if camera.type == CameraType.IP_CAMERA:
        try:
            from services.ip_camera_capture import get_ip_camera_manager
            ip_manager = get_ip_camera_manager()
            
            # Build network config from camera settings
            network_config = {
                'method': camera.network_access_method or 'direct',
                'ssh_host': camera.ssh_host,
                'ssh_port': camera.ssh_port,
                'ssh_user': camera.ssh_user,
                'ssh_key_path': camera.ssh_key_path,
                'tunnel_local_port': camera.tunnel_local_port,
                'proxy_host': camera.proxy_host,
                'proxy_port': camera.proxy_port
            }
            
            # Check if capture already exists, otherwise create it
            ip_capture = ip_manager.get_camera(camera_id)
            if ip_capture is None or not ip_capture.is_alive():
                print(f"🎥 Starting IP camera capture for {camera_id}")
                if network_config['method'] != 'direct':
                    print(f"🌉 Using network bridge: {network_config['method']}")
                ip_capture = ip_manager.add_camera(camera_id, camera.rtsp_url, network_config)
                
                # Wait a bit for connection
                await asyncio.sleep(1.0)
                
                if not ip_capture.is_connected:
                    camera_manager.update_camera_status(
                        camera_id, 
                        CameraStatus.ERROR,
                        "Failed to connect to RTSP stream"
                    )
                    await websocket.send_json({
                        "error": "Failed to connect to camera RTSP stream"
                    })
                    await websocket.close()
                    return
            
            camera_manager.update_camera_status(camera_id, CameraStatus.ONLINE)
            
        except Exception as e:
            print(f"❌ Error starting IP camera {camera_id}: {e}")
            camera_manager.update_camera_status(camera_id, CameraStatus.ERROR, str(e))
            await websocket.send_json({
                "error": f"Failed to start IP camera: {str(e)}"
            })
            await websocket.close()
            return
    else:
        # Webcam - frontend will send frames
        camera_manager.update_camera_status(camera_id, CameraStatus.ONLINE)
    
    frame_count = 0
    last_heartbeat = asyncio.get_event_loop().time()
    
    try:
        while True:
            try:
                # For IP cameras, pull frames from capture thread
                if camera.type == CameraType.IP_CAMERA and ip_capture:
                    frame = ip_capture.get_frame()
                    
                    if frame is None:
                        # Wait a bit for frames
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Process frame through pipeline
                    frame_count += 1
                    
                else:
                    # For webcams, wait for frames from frontend
                    # ⭐ TIMEOUT PROTECTION: Wait max 1 second for frame ⭐
                    data = await asyncio.wait_for(
                        websocket.receive_text(), 
                        timeout=1.0
                    )
                    message = json.loads(data)
                    
                    if message['type'] == 'frame':
                        # Decode base64 frame
                        import base64
                        frame_data = base64.b64decode(message['data'])
                        nparr = np.frombuffer(frame_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is None:
                            continue
                        
                        frame_count += 1
                    else:
                        continue
                
                # ⭐ CHECK CONNECTION BEFORE HEAVY PROCESSING ⭐
                if websocket.client_state.name != "CONNECTED":
                    print(f"⚠️ Client disconnected during processing for camera {camera_id}")
                    break
                
                # Process through unified pipeline
                # Pass camera_id so per-camera tuning (pose thresholds/persistence) can be applied
                result = unified_pipeline.process_frame(frame, camera_id=camera_id)
                
                # Check for pipeline errors
                if 'error' in result:
                    # ⭐ CHECK CONNECTION BEFORE SENDING ERROR ⭐
                    if websocket.client_state.name == "CONNECTED":
                        await websocket.send_json({
                            "error": f"Pipeline error: {result['error']}"
                        })
                    continue
                
                # PROFESSIONAL FUSION-BASED RESPONSE
                fusion_data = result.get('fusion', None)
                
                # Record detection for this camera if anomaly detected
                if result.get('anomaly_detected', False):
                    camera_manager.record_detection(camera_id)
            
                response = {
                    "type": "prediction",
                    "camera_id": camera_id,  # Include camera ID in response
                    "camera_name": camera.name,
                    "timestamp": result.get('timestamp', datetime.now().isoformat()),
                    "frame_number": frame_count,
                    "anomaly_detected": result.get('anomaly_detected', False),
                    "data": {
                        # FUSION ENGINE RESULTS (Primary)
                        "fusion": fusion_data,
                        
                        # Professional threat assessment
                        "threat_level": result.get('threat_level', 'NORMAL'),
                        "is_dangerous": result.get('is_dangerous', False),
                        "summary": result.get('summary', 'Normal activity'),
                        "alerts": result.get('alerts', []),
                        
                        # ⭐ RAW DETECTION DATA FOR FRONTEND OVERLAY ⭐
                        "objects": result['detections'].get('yolo', {}).get('objects', []),
                        "poses": result['detections'].get('pose', {}).get('poses', []),
                        "motion": result['detections'].get('motion', {}),
                        
                        # ML Model (for reference)
                        "ml_model": {
                            "predicted_class": result['detections'].get('ml_model', {}).get('predicted_class', 'Processing...'),
                            "confidence": result['detections'].get('ml_model', {}).get('confidence', 0.0),
                            "is_anomaly": result['detections'].get('ml_model', {}).get('is_anomaly', False),
                            "top3": result['detections'].get('ml_model', {}).get('top_3', [])
                        },
                        
                        # YOLO detections (for reference)
                        "yolo": {
                            "objects_detected": result['detections'].get('yolo', {}).get('objects', []),
                            "total_objects": result['detections'].get('yolo', {}).get('count', 0),
                            "dangerous_objects": result['detections'].get('yolo', {}).get('dangerous_objects', False)
                        },
                        
                        # Motion analysis (for reference)
                        "motion_analysis": {
                            "magnitude": result['detections'].get('motion', {}).get('magnitude', 0.0),
                            "is_unusual": result['detections'].get('motion', {}).get('is_unusual', False),
                            "anomaly_type": result['detections'].get('motion', {}).get('anomaly_type', None)
                        },
                        
                        # Pose estimation (for reference)
                        "pose_analysis": {
                            "persons_detected": result['detections'].get('pose', {}).get('persons_detected', 0),
                            "is_anomalous": result['detections'].get('pose', {}).get('is_anomalous', False),
                            "anomaly_type": result['detections'].get('pose', {}).get('anomaly_type', None)
                        },
                        
                        # Tracking (for reference)
                        "tracking": {
                            "total_tracks": result['detections'].get('tracking', {}).get('total_tracks', 0),
                            "tracked_objects": result['detections'].get('tracking', {}).get('tracked_objects', [])
                        }
                        
                        # ⭐ REMOVED: frame_base64 - frontend shows direct stream ⭐
                    }
                }
                
                # Send comprehensive fusion-based result
                # ⭐ FINAL CONNECTION CHECK BEFORE SEND ⭐
                if websocket.client_state.name == "CONNECTED":
                    await websocket.send_json(_sanitize_for_json(response))
                else:
                    print(f"⚠️ Client disconnected for camera {camera_id}, skipping send")
                    break
                
                # For IP cameras, add small delay to control frame rate
                if camera.type == CameraType.IP_CAMERA:
                    await asyncio.sleep(1.0 / camera.fps)  # Match configured FPS
                    
            except asyncio.TimeoutError:
                # ⭐ HEARTBEAT: Send ping if no data received (webcam only) ⭐
                if camera.type == CameraType.WEBCAM:
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_heartbeat > 5.0:
                        if websocket.client_state.name == "CONNECTED":
                            try:
                                await websocket.send_json(_sanitize_for_json({"type": "heartbeat", "timestamp": current_time}))
                                last_heartbeat = current_time
                            except:
                                print(f"⚠️ Heartbeat failed for camera {camera_id}, client disconnected")
                                break
                continue
            except WebSocketDisconnect:
                print(f"🔌 Client disconnected gracefully for camera {camera_id}")
                break
            except json.JSONDecodeError as e:
                print(f"⚠️ Invalid JSON received for camera {camera_id}: {e}")
                continue
                        
    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected by client for camera {camera_id}")
    except Exception as e:
        print(f"❌ WebSocket error for camera {camera_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Update camera status
        camera_manager.update_camera_status(camera_id, CameraStatus.OFFLINE)
        print(f"🔌 WebSocket session ended for camera {camera_id} (processed {frame_count} frames)")


@app.get("/api/classes")
async def get_classes():
    """Get list of anomaly classes."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": detector.class_names,
        "num_classes": len(detector.class_names)
    }


@app.delete("/api/uploads/{filename}")
async def delete_upload(filename: str):
    """Delete an uploaded video."""
    filepath = UPLOAD_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        filepath.unlink()
        return {"success": True, "message": f"Deleted {filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/detections/history")
async def get_detection_history(limit: int = 50):
    """
    Get detection history from fusion engine.
    
    Args:
        limit: Maximum number of detections to return (default: 50)
        
    Returns:
        List of recent anomaly detections with timestamps
    """
    if unified_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Use engine method that returns FusedDetection objects for formatting
        history = unified_pipeline.fusion_engine.get_detection_history(limit=limit)
        
        # Format history for frontend
        formatted_history = []
        for detection in history:
            formatted_history.append({
                "detection_id": detection.detection_id,
                "timestamp": detection.timestamp,
                "frame_number": detection.frame_number,
                "anomaly_type": detection.anomaly_type.value,
                "severity": detection.severity.value,
                "fusion_score": round(detection.fusion_score, 3),
                "confidence": round(detection.confidence, 3),
                "explanation": detection.explanation,
                "reasoning": detection.reasoning,
                "score_breakdown": {
                    "ml_score": round(detection.ml_score, 3),
                    "object_score": round(detection.object_score, 3),
                    "pose_score": round(detection.pose_score, 3),
                    "motion_score": round(detection.motion_score, 3)
                }
            })
        
        return {
            "success": True,
            "total_detections": len(formatted_history),
            "detections": formatted_history
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/debug/detection-ids")
async def debug_detection_ids(limit: int = 100):
    """Return current detection IDs from the fusion engine for debugging mismatched IDs."""
    if unified_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        ids = [d.detection_id for d in unified_pipeline.fusion_engine.get_detection_history(limit=limit)]
        return {"success": True, "count": len(ids), "detection_ids": ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/detections/statistics")
async def get_detection_statistics():
    """
    Get comprehensive detection statistics from fusion engine.
    
    Returns:
        Statistics including total detections, anomaly rate, severity distribution
    """
    if unified_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        stats = unified_pipeline.fusion_engine.get_statistics()
        
        return {
            "success": True,
            "statistics": {
                "total_detections": stats['total_detections'],
                # Align with IntelligentFusionEngine.get_statistics() return key
                "frames_processed": stats.get('total_frames_processed', 0),
                "anomaly_rate_percent": round(stats['anomaly_rate'] * 100, 2),
                "average_confidence": round(stats['average_confidence'], 3),
                "average_fusion_score": round(stats['average_fusion_score'], 3),
                "by_severity": stats['by_severity'],
                "by_anomaly_type": stats['by_type'],
                "critical_overrides": sum(1 for d in unified_pipeline.fusion_engine.detection_history 
                                        if d.critical_override),
                "consensus_detections": sum(1 for d in unified_pipeline.fusion_engine.detection_history 
                                           if d.consensus_count >= 2)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/status")
async def rl_status():
    """Return simple status about the RL agent and experience count."""
    try:
        from services.rl_agent import get_agent
        agent = get_agent()
        # Count experiences
        exp_count = 0
        try:
            import os
            exp_file = os.path.join(Path(__file__).parent.parent, 'data', 'rl_experiences.jsonl')
            if os.path.exists(exp_file):
                with open(exp_file, 'r') as f:
                    exp_count = sum(1 for _ in f)
        except Exception:
            exp_count = 0

        return {
            "success": True,
            "agent_loaded": agent is not None,
            "experience_count": exp_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _read_last_trained_id(path: Path) -> int:
    try:
        if path.exists():
            with open(path, 'r') as f:
                txt = f.read().strip()
                return int(txt or 0)
    except Exception:
        return 0
    return 0


def _write_last_trained_id(path: Path, id_value: int):
    try:
        with open(path, 'w') as f:
            f.write(str(int(id_value)))
    except Exception:
        pass


def _ensure_retrain_log_dir() -> Path:
    """Ensure and return the retrain log directory path."""
    logs_dir = Path(__file__).parent.parent / 'data' / 'retrain_logs'
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return logs_dir


def _append_retrain_log(entry: Dict[str, Any]):
    """Append a JSON line to today's retrain log file.

    Each line is a JSON object describing a retrain attempt.
    """
    try:
        logs_dir = _ensure_retrain_log_dir()
        today = datetime.utcnow().strftime('%Y-%m-%d')
        log_file = logs_dir / f'retrain_{today}.log'
        # add timestamp
        entry.setdefault('logged_at_utc', datetime.utcnow().isoformat())
        with open(log_file, 'a') as lf:
            lf.write(json.dumps(entry) + "\n")
    except Exception:
        # best-effort logging; do not raise
        pass


def _read_retrain_schedule() -> Dict[str, Any]:
    """Read retrain schedule config from data/retrain_schedule.json.

    Returns a dict with default keys if file missing. The schedule format:
    {
      "enabled": true,
      "days": [0,1,2,3,4,5,6],   # 0=Monday .. 6=Sunday (use Python weekday())
      "start_hour": 0,           # UTC hour (0-23) when training window starts
      "end_hour": 6,             # UTC hour when training window ends (exclusive)
      "min_new": 50
    }
    """
    try:
        cfg_path = Path(__file__).parent.parent / 'data' / 'retrain_schedule.json'
        if not cfg_path.exists():
            default = {
                'enabled': True,
                'days': [0,1,2,3,4,5,6],
                'start_hour': 0,
                'end_hour': 6,
                'min_new': app.state.rl_autoretrain_min_new,
                'cpu_threshold_percent': 80,
                'gpu_util_threshold_percent': 80,
                'gpu_mem_threshold_percent': 90,
                'cooldown_seconds': 300
            }
            return default
        with open(cfg_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {
            'enabled': True,
            'days': [0,1,2,3,4,5,6],
            'start_hour': 0,
            'end_hour': 6,
            'min_new': app.state.rl_autoretrain_min_new,
            'cpu_threshold_percent': 80,
            'gpu_util_threshold_percent': 80,
            'gpu_mem_threshold_percent': 90,
            'cooldown_seconds': 300
        }


def _is_within_schedule(schedule: Dict[str, Any]) -> bool:
    try:
        now = datetime.utcnow()
        wd = now.weekday()  # 0-6 Mon-Sun
        if not schedule.get('enabled', True):
            return False
        days = schedule.get('days', [0,1,2,3,4,5,6])
        if wd not in days:
            return False
        sh = int(schedule.get('start_hour', 0))
        eh = int(schedule.get('end_hour', 6))
        h = now.hour
        if sh <= eh:
            return sh <= h < eh
        else:
            # wrap-around (e.g., start 22 end 4)
            return h >= sh or h < eh
    except Exception:
        return True


def _check_system_resources(cpu_threshold: int, gpu_util_threshold: int, gpu_mem_threshold: int) -> Dict[str, Any]:
    """Return dict with resource usage and flags whether overloaded.

    Uses psutil if available, otherwise falls back to /proc or os.getloadavg.
    For GPU, tries nvidia-smi if present; otherwise returns None for GPU stats.
    """
    result = {'cpu_percent': None, 'overloaded_cpu': False, 'gpu': None, 'overloaded_gpu': False}
    try:
        # CPU
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.5)
        except Exception:
            # fallback to 1-minute load average scaled by CPU count
            try:
                load1 = os.getloadavg()[0]
                cpu = min(100.0, (load1 / max(1, os.cpu_count())) * 100.0)
            except Exception:
                cpu = 0.0
        result['cpu_percent'] = float(cpu)
        result['overloaded_cpu'] = cpu >= float(cpu_threshold)

        # GPU via nvidia-smi if available
        try:
            cmd = ['nvidia-smi', '--query-gpu=utilization.gpu,memory.total,memory.used', '--format=csv,noheader,nounits']
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=2).decode('utf-8').strip()
            gpus = []
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    util = float(parts[0])
                    mem_total = float(parts[1])
                    mem_used = float(parts[2])
                    mem_pct = (mem_used / mem_total) * 100.0 if mem_total > 0 else 0.0
                    gpus.append({'util_percent': util, 'mem_total': mem_total, 'mem_used': mem_used, 'mem_percent': mem_pct})
            if gpus:
                result['gpu'] = gpus
                # overloaded if any GPU util or mem exceeds thresholds
                overloaded_gpu = any((g['util_percent'] >= float(gpu_util_threshold) or g['mem_percent'] >= float(gpu_mem_threshold)) for g in gpus)
                result['overloaded_gpu'] = bool(overloaded_gpu)
        except Exception:
            result['gpu'] = None
            result['overloaded_gpu'] = False

    except Exception:
        pass
    return result


async def _rl_autoretrain_daemon():
    """Background coroutine that periodically checks for new experiences and
    spawns a low-priority training process to retrain the RL policy.

    Training runs in a separate process (multiprocessing.Process) where
    niceness is reduced; the parent waits for the child to finish and
    atomically swaps the model file if a tmp model is produced.
    """
    await asyncio.sleep(1.0)  # small delay to let startup finish
    try:
        import services.rl_agent as rl_mod
        last_file = app.state.rl_last_trained_file
        min_new = int(app.state.rl_autoretrain_min_new)
        interval = int(app.state.rl_autoretrain_interval)
        niceness = int(app.state.rl_autoretrain_niceness)
        batch_size = int(app.state.rl_autoretrain_batch_size)
        epochs = int(app.state.rl_autoretrain_epochs)

        last_id = _read_last_trained_id(last_file)

        while True:
            try:
                await asyncio.sleep(interval)
                agent = rl_mod.get_agent()
                if agent is None:
                    continue

                current_max = agent.get_max_experience_id()
                if current_max - int(last_id) < int(min_new):
                    # not enough new experiences yet
                    continue

                # prepare tmp model path
                ts = int(time.time())
                tmp_model = str(Path(rl_mod.MODEL_FILE).with_suffix(f".retrain.{ts}.tmp"))

                # spawn child process to do training with lowered niceness
                start_ts = time.time()
                proc = multiprocessing.Process(target=rl_mod._train_process_target, args=(epochs, batch_size, niceness, tmp_model))
                # register current proc on app.state so admin endpoints can control it
                app.state.current_training_proc = proc
                app.state.training_paused = False
                proc.start()

                # wait for child to finish without blocking event loop
                while proc.is_alive():
                    await asyncio.sleep(1.0)

                proc.join(timeout=1.0)
                # clear current proc
                try:
                    app.state.current_training_proc = None
                    app.state.training_paused = False
                except Exception:
                    pass

                # If child produced a tmp model file, evaluate metrics and perform atomic swap with validation/rollback
                if os.path.exists(tmp_model) and os.path.getsize(tmp_model) > 0:
                    metrics_path = tmp_model + '.metrics.json'
                    new_metrics = None
                    try:
                        if os.path.exists(metrics_path):
                            with open(metrics_path, 'r') as mf:
                                new_metrics = json.load(mf)
                    except Exception:
                        new_metrics = None

                    # load previous metrics if present
                    metrics_file = Path(rl_mod.DATA_DIR) / 'rl_last_metrics.json' if hasattr(rl_mod, 'DATA_DIR') else Path(__file__).parent.parent / 'data' / 'rl_last_metrics.json'
                    prev_metrics = None
                    try:
                        if metrics_file.exists():
                            with open(metrics_file, 'r') as pf:
                                prev_metrics = json.load(pf)
                    except Exception:
                        prev_metrics = None

                    # validation tolerance (allow slight regressions)
                    val_tol = float(os.environ.get('RL_VALIDATION_TOLERANCE', '0.01'))

                    # decide whether to swap using configurable validation rules
                    # rules may be provided via RL_VALIDATION_RULES env var as JSON, e.g.
                    # {"f1": 0.001, "accuracy": 0.0} meaning require f1 improvement >= 0.001 and accuracy >= 0.0
                    rules_raw = os.environ.get('RL_VALIDATION_RULES')
                    policy = os.environ.get('RL_VALIDATION_POLICY', 'all').lower()  # 'all' or 'any'
                    global_min_improve = float(os.environ.get('RL_VALIDATION_MIN_IMPROVEMENT', '0.0'))
                    accept_swap = True
                    try:
                        rules = None
                        if rules_raw:
                            try:
                                rules = json.loads(rules_raw)
                            except Exception:
                                rules = None

                        if not new_metrics:
                            accept_swap = False
                        elif not prev_metrics:
                            # no previous metrics -> accept if new_metrics exist
                            accept_swap = True
                        else:
                            # build per-metric decisions
                            decisions = []
                            if rules:
                                for m, min_delta in rules.items():
                                    prev_v = float(prev_metrics.get(m, 0.0) or 0.0)
                                    new_v = float(new_metrics.get(m, 0.0) or 0.0)
                                    # improvement = new - prev
                                    decisions.append((new_v - prev_v) >= float(min_delta))
                            else:
                                # default behavior: require improvement in f1 by at least global_min_improve
                                prev_v = float(prev_metrics.get('f1', 0.0) or 0.0)
                                new_v = float(new_metrics.get('f1', 0.0) or 0.0)
                                decisions.append((new_v - prev_v) >= global_min_improve)

                            if policy == 'all':
                                accept_swap = all(decisions) if decisions else True
                            elif policy == 'any':
                                accept_swap = any(decisions) if decisions else True
                            else:
                                # unknown policy -> default to all
                                accept_swap = all(decisions) if decisions else True
                    except Exception:
                        accept_swap = False

                    # perform atomic swap with backup and safe in-memory reload
                    existing_model = rl_mod.MODEL_FILE
                    ts_now = int(time.time())
                    backup_path = existing_model + f".bak.{ts_now}"
                    swapped = False
                    try:
                        # backup current model if exists
                        if os.path.exists(existing_model):
                            try:
                                os.rename(existing_model, backup_path)
                            except Exception:
                                # best-effort backup
                                pass

                        # move new model into place
                        try:
                            os.replace(tmp_model, existing_model)
                        except Exception:
                            try:
                                os.rename(tmp_model, existing_model)
                            except Exception:
                                # could not move new model
                                raise

                        if accept_swap:
                            # reload model into running agent with lock
                            try:
                                load_lock = app.state.rl_model_load_lock
                                with load_lock:
                                    agent = rl_mod.get_agent()
                                    agent.load()
                                swapped = True
                            except Exception:
                                # load failed; attempt rollback
                                swapped = False
                        else:
                            swapped = False

                        if not swapped:
                            # rollback: restore backup if present
                            try:
                                if os.path.exists(backup_path):
                                    if os.path.exists(existing_model):
                                        try:
                                            os.remove(existing_model)
                                        except Exception:
                                            pass
                                    os.replace(backup_path, existing_model)
                            except Exception:
                                pass
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        # try to restore backup
                        try:
                            if os.path.exists(backup_path) and not os.path.exists(existing_model):
                                os.replace(backup_path, existing_model)
                        except Exception:
                            pass
                    finally:
                        # update last trained id only on successful swap
                        if swapped:
                            last_id = current_max
                            _write_last_trained_id(last_file, last_id)
                            try:
                                # persist new metrics
                                if new_metrics:
                                    with open(metrics_file, 'w') as pf:
                                        json.dump(new_metrics, pf)
                            except Exception:
                                pass
                        else:
                            # leave tmp files for inspection; do not update last_id
                            pass

                        # Write a retrain run log entry (auto daemon)
                        try:
                            entry = {
                                'type': 'auto',
                                'start_ts': start_ts,
                                'end_ts': time.time(),
                                'duration_seconds': time.time() - start_ts,
                                'min_new_required': min_new,
                                'experience_max_id': current_max,
                                'previous_trained_id': int(last_id),
                                'epochs': epochs,
                                'batch_size': batch_size,
                                'niceness': niceness,
                                'tmp_model': tmp_model,
                                'accepted_swap': bool(swapped),
                                'accepted_by_validation': bool(accept_swap),
                                'new_metrics': new_metrics,
                                'prev_metrics': prev_metrics
                            }
                            _append_retrain_log(entry)
                        except Exception:
                            pass

            except asyncio.CancelledError:
                break
            except Exception:
                # swallow errors in daemon to keep it running
                import traceback
                traceback.print_exc()
                await asyncio.sleep(interval)
    except Exception:
        # if module import or other fatal error, log and stop daemon
        import traceback
        traceback.print_exc()
        return


@app.post("/api/rl/train")
async def rl_train(payload: dict = {}):
    """Trigger RL training on stored experiences.

    payload may contain: epochs (int), batch_size (int)
    """
    try:
        from services.rl_agent import get_agent
        agent = get_agent()
        if agent is None:
            raise HTTPException(status_code=500, detail="RL agent not available")

        epochs = int(payload.get('epochs', 5))
        batch_size = int(payload.get('batch_size', 32))
        # run synchronous (manual) training on the calling thread — keep lightweight
        start_ts = time.time()
        num = agent.train(epochs=epochs, batch_size=batch_size)
        duration = time.time() - start_ts

        # log manual training run
        try:
            entry = {
                'type': 'manual',
                'start_ts': start_ts,
                'end_ts': time.time(),
                'duration_seconds': duration,
                'epochs': epochs,
                'batch_size': batch_size,
                'samples_used': int(num)
            }
            _append_retrain_log(entry)
        except Exception:
            pass

        return {"success": True, "samples_used": num}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/experiences")
async def rl_experiences(limit: int = 100):
    """Return recent RL experiences for inspection."""
    try:
        from services.rl_agent import get_agent
        agent = get_agent()
        if agent is None:
            return {"success": False, "message": "RL agent not available"}
        exps = agent.get_experiences(limit=limit)
        return {"success": True, "count": len(exps), "experiences": exps}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/schedule")
async def rl_get_schedule():
    """Return the current retrain schedule from disk (or defaults)."""
    try:
        sched = _read_retrain_schedule()
        return {"success": True, "schedule": sched}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rl/schedule")
async def rl_update_schedule(payload: dict):
    """Update retrain schedule (writes backend/data/retrain_schedule.json)."""
    try:
        cfg_path = Path(__file__).parent.parent / 'data' / 'retrain_schedule.json'
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        # Basic validation: ensure JSON-serializable
        json.dumps(payload)
        with open(cfg_path, 'w') as f:
            json.dump(payload, f, indent=2)
        return {"success": True, "wrote": str(cfg_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rl/logs")
async def rl_get_logs(date: Optional[str] = None, limit: int = 200):
    """Return retrain log entries for a given UTC date (YYYY-MM-DD)."""
    try:
        logs_dir = _ensure_retrain_log_dir()
        if date is None:
            date = datetime.utcnow().strftime('%Y-%m-%d')
        file_path = logs_dir / f'retrain_{date}.log'
        if not file_path.exists():
            return {"success": True, "count": 0, "entries": []}
        entries = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
        return {"success": True, "count": len(entries), "entries": entries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/anomaly-history")
async def anomaly_history(limit: int = 100, date: Optional[str] = None):
    """Return confirmed anomalies stored in the persistent DB."""
    try:
        from services.anomaly_store import get_confirmed_detections
        entries = get_confirmed_detections(limit=limit, date=date)
        return {"success": True, "count": len(entries), "entries": entries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rl/pause")
async def rl_pause():
    """Pause the currently running training process (if any)."""
    try:
        proc = getattr(app.state, 'current_training_proc', None)
        if not proc:
            return {"success": False, "message": "No training process running"}
        pid = getattr(proc, 'pid', None)
        if not pid:
            return {"success": False, "message": "Process not started or pid unknown"}
        try:
            os.kill(pid, signal.SIGSTOP)
            app.state.training_paused = True
            return {"success": True, "paused": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rl/resume")
async def rl_resume():
    """Resume a paused training process (if any)."""
    try:
        proc = getattr(app.state, 'current_training_proc', None)
        if not proc:
            return {"success": False, "message": "No training process running"}
        pid = getattr(proc, 'pid', None)
        if not pid:
            return {"success": False, "message": "Process not started or pid unknown"}
        try:
            os.kill(pid, signal.SIGCONT)
            app.state.training_paused = False
            return {"success": True, "resumed": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detections/clear")
async def clear_detection_history():
    """Clear detection history (useful for testing/debugging)."""
    if unified_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        unified_pipeline.fusion_engine.detection_history.clear()
        unified_pipeline.fusion_engine.frames_processed = 0
        # Also clear any decline_all suppression so notifications may re-appear normally
        try:
            unified_pipeline.fusion_engine.clear_decline_all_suppression()
        except Exception:
            pass
        
        return {
            "success": True,
            "message": "Detection history cleared"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detections/{detection_id}/feedback")
async def post_detection_feedback(detection_id: str, payload: dict):
    """
    Record user feedback for a given detection.
    payload keys:
      - feedback: 'confirm' | 'not_anomaly' | 'decline'
      - comment: optional string
      - user: optional string
    """
    if unified_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    feedback = payload.get('feedback')
    comment = payload.get('comment')
    user = payload.get('user')

    if feedback not in ('confirm', 'not_anomaly', 'decline'):
        raise HTTPException(status_code=400, detail="Invalid feedback value")

    try:
        # pre-check whether the id likely exists (use similar matching heuristics)
        import re
        found = False
        for det in unified_pipeline.fusion_engine.get_detection_history(limit=200):
            if det.detection_id == detection_id:
                found = True
                break
        if not found:
            alt1 = detection_id.replace('-', '_') if '-' in detection_id else detection_id
            alt2 = detection_id.replace('_', '-') if '_' in detection_id else detection_id
            compact = re.sub(r'[^A-Za-z0-9]', '', detection_id)
            tokens = re.split(r'[-_]', detection_id)
            last = tokens[-1] if tokens else ''
            for det in unified_pipeline.fusion_engine.get_detection_history(limit=200):
                if det.detection_id == alt1 or det.detection_id == alt2:
                    found = True
                    break
                det_compact = re.sub(r'[^A-Za-z0-9]', '', det.detection_id)
                if det_compact == compact:
                    found = True
                    break
                if last and last in det.detection_id:
                    found = True
                    break

        ok = unified_pipeline.fusion_engine.apply_feedback(detection_id, feedback, comment=comment, user=user)

        # compute updated counts
        active_notifications = len(unified_pipeline.fusion_engine.detection_history)
        try:
            from services.anomaly_store import get_confirmed_detections
            confirmed_total = len(get_confirmed_detections(limit=10000))
        except Exception:
            confirmed_total = 0

        return {"success": True, "detection_id": detection_id, "feedback": feedback,
                "active_notifications": active_notifications,
                "confirmed_total": confirmed_total,
                "orphan": not found}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detections/decline-all")
async def post_decline_all(payload: dict):
    """
    Bulk-decline detections. Requires explicit confirm flag to avoid accidental mass actions.
    payload keys:
      - confirm: True/False (required)
      - detection_ids: optional list of ids (if omitted, all detections will be marked declined)
      - comment: optional string
      - user: optional string
    """
    if unified_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    confirm = payload.get('confirm', False)
    if not confirm:
        raise HTTPException(status_code=400, detail="Confirmation required to decline all detections")

    detection_ids = payload.get('detection_ids')
    comment = payload.get('comment')
    user = payload.get('user')

    try:
        updated = unified_pipeline.fusion_engine.decline_all(detection_ids=detection_ids, comment=comment, user=user)
        # return updated counts too
        active_notifications = len(unified_pipeline.fusion_engine.detection_history)
        try:
            from services.anomaly_store import get_confirmed_detections
            confirmed_total = len(get_confirmed_detections(limit=10000))
        except Exception:
            confirmed_total = 0

        # Indicate whether any per-id suppression is active (suppressed ids exist)
        suppression_active = bool(getattr(unified_pipeline.fusion_engine, 'suppressed_ids', None))
        return {"success": True, "updated": updated, "active_notifications": active_notifications, "confirmed_total": confirmed_total, "suppression_active": suppression_active}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/save-screenshot")
async def save_screenshot(file: UploadFile = File(...)):
    """
    Save anomaly screenshot to uploads folder
    """
    try:
        # Create screenshots directory
        screenshots_dir = UPLOAD_DIR / "screenshots"
        screenshots_dir.mkdir(exist_ok=True)
        
        # Save file
        file_path = screenshots_dir / file.filename
        contents = await file.read()
        
        with open(file_path, 'wb') as f:
            f.write(contents)
        
        print(f"✅ Screenshot saved: {file.filename}")
        
        return {
            "success": True,
            "filename": file.filename,
            "path": str(file_path)
        }
        
    except Exception as e:
        print(f"❌ Error saving screenshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# CAMERA MANAGEMENT API
# ============================================================================

@app.get("/api/cameras")
async def get_cameras():
    """Get all configured cameras"""
    try:
        from services.camera_manager import get_camera_manager
        camera_manager = get_camera_manager()
        cameras = camera_manager.get_all_cameras()
        return {
            "success": True,
            "cameras": [cam.to_dict() for cam in cameras]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cameras/{camera_id}")
async def get_camera(camera_id: str):
    """Get specific camera by ID"""
    try:
        from services.camera_manager import get_camera_manager
        camera_manager = get_camera_manager()
        camera = camera_manager.get_camera(camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        return {
            "success": True,
            "camera": camera.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cameras")
async def add_camera(camera_data: dict):
    """
    Add a new camera
    
    Request body:
    {
        "name": "Camera Name",
        "type": "webcam" | "ip_camera",
        "location": "Location description",
        "rtsp_url": "rtsp://..." (required for IP cameras),
        "username": "optional",
        "password": "optional",
        "resolution_width": 1280,
        "resolution_height": 720,
        "fps": 15
    }
    """
    try:
        from services.camera_manager import get_camera_manager
        camera_manager = get_camera_manager()
        camera = camera_manager.add_camera(camera_data)
        return {
            "success": True,
            "message": f"Camera {camera.name} added successfully",
            "camera": camera.to_dict()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/cameras/{camera_id}")
async def update_camera(camera_id: str, updates: dict):
    """
    Update camera configuration
    
    Allowed updates:
    - name, location
    - rtsp_url, username, password (for IP cameras)
    - enabled
    - resolution_width, resolution_height, fps
    """
    try:
        from services.camera_manager import get_camera_manager
        camera_manager = get_camera_manager()
        camera = camera_manager.update_camera(camera_id, updates)
        return {
            "success": True,
            "message": f"Camera {camera.name} updated successfully",
            "camera": camera.to_dict()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/cameras/{camera_id}")
async def delete_camera(camera_id: str):
    """Delete a camera"""
    try:
        from services.camera_manager import get_camera_manager
        camera_manager = get_camera_manager()
        success = camera_manager.delete_camera(camera_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        return {
            "success": True,
            "message": f"Camera {camera_id} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cameras/{camera_id}/test")
async def test_camera_connection(camera_id: str):
    """Test camera connection (especially for IP cameras)"""
    try:
        from services.camera_manager import get_camera_manager, CameraType, CameraStatus
        camera_manager = get_camera_manager()
        camera = camera_manager.get_camera(camera_id)
        
        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        if camera.type == CameraType.WEBCAM:
            return {
                "success": True,
                "status": "online",
                "message": "Webcam camera - test connection from browser"
            }
        
        elif camera.type == CameraType.IP_CAMERA:
            # Test RTSP connection
            import cv2
            try:
                camera_manager.update_camera_status(camera_id, CameraStatus.CONNECTING)
                
                # Try to open RTSP stream
                cap = cv2.VideoCapture(camera.rtsp_url)
                
                if not cap.isOpened():
                    camera_manager.update_camera_status(
                        camera_id, 
                        CameraStatus.ERROR,
                        "Failed to connect to RTSP stream"
                    )
                    return {
                        "success": False,
                        "status": "error",
                        "message": "Failed to connect to RTSP stream"
                    }
                
                # Try to read one frame
                ret, frame = cap.read()
                cap.release()
                
                if not ret or frame is None:
                    camera_manager.update_camera_status(
                        camera_id,
                        CameraStatus.ERROR,
                        "Connected but failed to read frames"
                    )
                    return {
                        "success": False,
                        "status": "error",
                        "message": "Connected but failed to read frames"
                    }
                
                # Success!
                camera_manager.update_camera_status(camera_id, CameraStatus.ONLINE)
                return {
                    "success": True,
                    "status": "online",
                    "message": "Successfully connected to IP camera",
                    "frame_size": {"width": frame.shape[1], "height": frame.shape[0]}
                }
                
            except Exception as e:
                camera_manager.update_camera_status(
                    camera_id,
                    CameraStatus.ERROR,
                    str(e)
                )
                return {
                    "success": False,
                    "status": "error",
                    "message": f"Connection error: {str(e)}"
                }
        
        return {
            "success": False,
            "status": "error",
            "message": "Unknown camera type"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the API server."""
    print("="*70)
    print("🚀 ANOMALY DETECTION API SERVER")
    print("="*70)
    print("\nStarting server on http://localhost:8000")
    print("API docs available at http://localhost:8000/docs")
    print("\nPress CTRL+C to stop\n")
    
    # ⭐ OPTIMIZED WEBSOCKET CONFIGURATION ⭐
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        ws_ping_interval=20.0,  # Send ping every 20 seconds
        ws_ping_timeout=20.0,   # Wait 20 seconds for pong
        timeout_keep_alive=30   # Keep connection alive for 30 seconds
    )


if __name__ == "__main__":
    main()
