"""
FastAPI Backend for Anomaly Detection System
Professional REST API with WebSocket support for real-time predictions
"""

from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import json
from datetime import datetime
import asyncio

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
            yolo_model="yolov8n.pt",
            device="cuda",
            confidence_threshold=0.7
        )
        
        # Initialize unified pipeline with all advanced features
        print("\n🎯 Initializing Unified Detection Pipeline...")
        unified_pipeline = UnifiedDetectionPipeline(detector)
        
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


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Enhanced WebSocket endpoint for real-time multi-modal analysis.
    Features: ML Model + YOLO + Motion + Pose + Tracking + Rules
    """
    await websocket.accept()
    
    if detector is None or unified_pipeline is None:
        await websocket.send_json({
            "error": "Model not loaded"
        })
        await websocket.close()
        return
    
    print("🔌 WebSocket connected - Enhanced Pipeline Active")
    frame_count = 0
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_text()
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
                
                # Process through unified pipeline
                result = unified_pipeline.process_frame(frame)
                
                # Check for pipeline errors
                if 'error' in result:
                    await websocket.send_json({
                        "error": f"Pipeline error: {result['error']}"
                    })
                    continue
                
                # PROFESSIONAL FUSION-BASED RESPONSE
                # Only send when anomaly detected (fusion_score >= 0.70)
                fusion_data = result.get('fusion', None)
                
                response = {
                    "type": "prediction",
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
                        "motion": {
                            "magnitude": result['detections'].get('motion', {}).get('magnitude', 0.0),
                            "is_unusual": result['detections'].get('motion', {}).get('is_unusual', False),
                            "anomaly_type": result['detections'].get('motion', {}).get('anomaly_type', None)
                        },
                        
                        # Pose estimation (for reference)
                        "pose": {
                            "persons_detected": result['detections'].get('pose', {}).get('persons_detected', 0),
                            "is_anomalous": result['detections'].get('pose', {}).get('is_anomalous', False),
                            "anomaly_type": result['detections'].get('pose', {}).get('anomaly_type', None)
                        },
                        
                        # Tracking (for reference)
                        "tracking": {
                            "total_tracks": result['detections'].get('tracking', {}).get('total_tracks', 0),
                            "tracked_objects": result['detections'].get('tracking', {}).get('tracked_objects', [])
                        },
                        
                        # Annotated frame
                        "frame_base64": result.get('frame_base64', '')
                    }
                }
                
                # Send comprehensive fusion-based result
                await websocket.send_json(response)
                        
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔌 WebSocket disconnected")


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
                "frames_processed": stats['frames_processed'],
                "anomaly_rate_percent": round(stats['anomaly_rate'] * 100, 2),
                "average_confidence": round(stats['avg_confidence'], 3),
                "average_fusion_score": round(stats['avg_fusion_score'], 3),
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


@app.post("/api/detections/clear")
async def clear_detection_history():
    """Clear detection history (useful for testing/debugging)."""
    if unified_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        unified_pipeline.fusion_engine.detection_history.clear()
        unified_pipeline.fusion_engine.frames_processed = 0
        
        return {
            "success": True,
            "message": "Detection history cleared"
        }
        
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
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
