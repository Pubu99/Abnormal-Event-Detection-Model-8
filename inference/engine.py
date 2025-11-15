"""
Professional Inference Engine for Anomaly Detection System
Integrates: Your trained model + YOLO object detection + OpenCV processing

Features:
- Video frame extraction and preprocessing
- YOLO object detection (guns, knives, fire, etc.)
- Temporal sequence prediction
- Confidence scoring
- Alert generation
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np      
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import threading
import queue
import multiprocessing
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
try:
    # Try to import YOLOv10 (if installed as a separate package or via ultralytics-yolov10)
    # Some distributions expose a similar API; we handle both cases gracefully.
    from ultralytics import YOLO as YOLOv10
    _YOLOV10_AVAILABLE = True
except Exception:
    _YOLOV10_AVAILABLE = False
import sys
from collections import deque
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.research_model import create_research_model

def _yolo_process_main(yolo_model_path, yolo_fast_imgsz, yolo_imgsz, use_fp16, yolo_mode, in_q, out_q):
    """Process entrypoint for a separate YOLO worker.

    Loads a YOLO model inside the child process and listens on in_q for frames
    (numpy arrays). It returns a simplified detection dict on out_q.
    """
    try:
        # Import inside process
        import cv2
        import numpy as np
        try:
            from ultralytics import YOLO as _YOLO
        except Exception:
            _YOLO = None

        if _YOLO is None:
            # Nothing to do
            return

        # Load YOLO model inside the process
        yolo = _YOLO(yolo_model_path)

        while True:
            frame = in_q.get()
            if frame is None:
                break

            try:
                h, w = frame.shape[:2]
                max_dim = 640
                scale = min(max_dim / max(w, h), 1.0)
                if scale < 1.0:
                    frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
                else:
                    frame_resized = frame

                imgsz = yolo_fast_imgsz if yolo_mode == 'detect' else yolo_imgsz
                # Prefer predict/detect for speed when requested
                try:
                    predict_kwargs = {'conf': 0.35, 'iou': 0.5, 'imgsz': imgsz}
                    results = None
                    if hasattr(yolo, 'predict'):
                        results = yolo.predict(frame_resized, **predict_kwargs)[0]
                    else:
                        results = yolo(frame_resized, **predict_kwargs)[0]
                except Exception:
                    try:
                        results = yolo(frame_resized)[0]
                    except Exception:
                        results = None

                detections = {'objects': [], 'boxes': [], 'confidences': [], 'track_ids': [], 'dangerous': False}
                if results is not None:
                    names_map = getattr(results, 'names', None) or getattr(yolo, 'names', None) or {}
                    for box in results.boxes:
                        try:
                            class_id = int(box.cls[0]) if hasattr(box, 'cls') else int(box.data[0][-1])
                            confidence = float(box.conf[0]) if hasattr(box, 'conf') else float(box.data[0][-2])
                            track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else None
                            class_name = str(names_map.get(class_id, str(class_id))).lower()
                            if hasattr(box, 'xyxy'):
                                bbox = box.xyxy[0].cpu().numpy()
                            else:
                                arr = box.data[0].cpu().numpy()
                                bbox = arr[:4]
                            if scale < 1.0:
                                bbox = bbox / scale
                            detections['objects'].append(class_name)
                            detections['boxes'].append(bbox.tolist())
                            detections['confidences'].append(float(confidence))
                            detections['track_ids'].append(track_id)
                            if any(danger in class_name for danger in {'knife','gun','rifle','pistol','weapon','fire','smoke','explosion'}):
                                detections['dangerous'] = True
                        except Exception:
                            continue

                # Put the simple dict back to main process
                try:
                    out_q.put_nowait(detections)
                except Exception:
                    # If queue is full or error, try blocking put
                    try:
                        out_q.put(detections, timeout=0.1)
                    except Exception:
                        pass
            except Exception:
                # ignore and continue
                continue
    except Exception:
        return


class AnomalyDetector:
    """
    Professional anomaly detection engine with multi-modal analysis.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "configs/config_research_enhanced.yaml",
        yolo_model: str = "yolov8n.pt",
        device: str = "cuda",
        sequence_length: int = 16,
        confidence_threshold: float = 0.7,
        ml_influence: float = 0.2,
        ml_ignore_below_confidence: float = 0.3,
        ml_min_confidence_for_vote: float = 0.6,
        # Fast-mode options to prioritize throughput over max accuracy
        fast_mode: bool = False,
        fast_image_size: int = 160,
        fast_sequence_length: int = 8,
        yolo_fast_imgsz: int = 320,
        # yolo_mode: 'track' (default, provides tracking but slower) or 'detect' (single-frame detection - faster)
        yolo_mode: str = 'track',
        # default YOLO imgsz for non-fast mode (kept small to favor speed)
        yolo_imgsz: int = 640,
        # Use a separate process for YOLO inference (recommended when you want
        # to isolate YOLO work, reduce Python GIL impact, and avoid blocking)
        yolo_process: bool = False,
    ):
        """
        Initialize the anomaly detection system.
        
        Args:
            model_path: Path to trained .pth model
            config_path: Path to config file
            yolo_model: YOLO model variant (yolov8n, yolov8s, yolov8m)
            device: 'cuda' or 'cpu'
            sequence_length: Number of frames for temporal analysis
            confidence_threshold: Minimum confidence for anomaly alert
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        # ML dampening and thresholds (reduce model influence on final fusion)
        # ml_influence: fraction [0..1] of raw ML anomaly score that will be used
        # ml_ignore_below_confidence: if model confidence < this, ignore ML score
        # ml_min_confidence_for_vote: confidence at which ML casts a full vote
        self.ml_influence = ml_influence
        self.ml_ignore_below_confidence = ml_ignore_below_confidence
        self.ml_min_confidence_for_vote = ml_min_confidence_for_vote
        self.ml_enabled = True
        # Fast-mode settings
        self.fast_mode = bool(fast_mode)
        self.fast_image_size = int(fast_image_size)
        self.fast_sequence_length = int(fast_sequence_length)
        self.yolo_fast_imgsz = int(yolo_fast_imgsz)
        self.yolo_mode = str(yolo_mode).lower()
        self.yolo_imgsz = int(yolo_imgsz)
        self.yolo_process = bool(yolo_process)
        # YOLO caching to avoid running heavy detector every frame
        self.yolo_min_interval = 0.08  # seconds between YOLO runs when caching (default ~12 FPS)
        self._last_yolo_time = 0.0
        self._last_yolo_result = None
        self._yolo_lock = threading.Lock()
        self._yolo_queue = None
        self._yolo_thread = None

        # If fast_mode is enabled, prefer the fast sequence length to reduce ML latency
        if self.fast_mode:
            # Note: reducing sequence length may affect ML accuracy; this is a runtime trade-off
            self.sequence_length = int(self.fast_sequence_length)
            # Start async YOLO worker to decouple detection latency from pipeline
            try:
                self._yolo_queue = queue.Queue(maxsize=2)
                self._yolo_thread = threading.Thread(target=self._yolo_worker, daemon=True)
                self._yolo_thread.start()
            except Exception:
                # If threading isn't available or fails, fall back to synchronous YOLO
                self._yolo_queue = None
                self._yolo_thread = None
        # Optionally start a separate YOLO process worker (better isolation)
        self._yolo_process = None
        self._yolo_in_q = None
        self._yolo_out_q = None
        if self.yolo_process:
            try:
                ctx = multiprocessing.get_context('spawn')
                self._yolo_in_q = ctx.Queue(maxsize=2)
                self._yolo_out_q = ctx.Queue(maxsize=2)
                # Start process with model path from provided yolo_model
                self._yolo_process = ctx.Process(
                    target=_yolo_process_main,
                    args=(yolo_model, self.yolo_fast_imgsz, self.yolo_imgsz, self.use_fp16, self.yolo_mode, self._yolo_in_q, self._yolo_out_q),
                    daemon=True
                )
                self._yolo_process.start()
            except Exception:
                self._yolo_process = None
                self._yolo_in_q = None
                self._yolo_out_q = None
        
        print("🚀 Initializing Anomaly Detection System...")
        print(f"   Device: {self.device}")
        
        # Load configuration (simple load without validation for inference)
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        print(f"📄 Loading config from {config_path}...")
        self.config = OmegaConf.load(config_path)
        
        # Load your trained model
        print(f"📦 Loading trained model from {model_path}...")
        self.model = create_research_model(self.config, device=self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle compiled model state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            state_dict = {key.replace('_orig_mod.', ''): value 
                         for key, value in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("   ✅ Model loaded successfully")

        # Performance: enable FP16 and GPU optimizations when available
        self.use_fp16 = False
        try:
            if torch.cuda.is_available():
                print("🚀 GPU available - enabling FP16 and optimizations...")
                # convert model to half precision
                self.model = self.model.half()
                self.use_fp16 = True
                # cuDNN autotuner
                torch.backends.cudnn.benchmark = True
                # channels last memory format can improve throughput
                try:
                    self.model.to(memory_format=torch.channels_last)
                except Exception:
                    pass

                # Try to compile model (PyTorch 2.x) - optional
                try:
                    # Helpful logging for Dynamo failures (useful when reporting bugs)
                    os.environ.setdefault('TORCHDYNAMO_VERBOSE', '1')
                    os.environ.setdefault('TORCH_LOGS', '+dynamo')

                    # NOTE: torch.compile / Dynamo can fail when model parameters
                    # are in FP16 but some inputs or internal ops are still float.
                    # To avoid the common "Input type (float) and bias type (c10::Half)"
                    # mismatch we *skip* compilation when running the model in FP16.
                    if not self.use_fp16:
                        self.model = torch.compile(self.model, mode='reduce-overhead', fullgraph=True)
                        print("   ✅ Model compiled with torch.compile")
                    else:
                        print("   ⚠️ Skipping torch.compile for FP16 model to avoid Dynamo dtype issues")
                except Exception as e:
                    print(f"   ⚠️ torch.compile not available or failed: {e}")

                # Warmup with a dummy input (if sequence_length reasonable)
                try:
                    dummy_t = torch.randn(1, max(1, min(self.sequence_length, 8)), 3, 224, 224, device=self.device)
                    if self.use_fp16:
                        dummy_t = dummy_t.half()
                    with torch.no_grad():
                        _ = self.model(dummy_t)
                    print("   ✅ Model warmup complete")
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ GPU optimization setup failed: {e}")
        
        # Load YOLO for object detection
        print(f"� Loading object detector: preferring YOLOv10 when available...")
        self.detector_name = None
        self.yolo = None
        try:
            if _YOLOV10_AVAILABLE and ("yolov10" in yolo_model.lower() or yolo_model.lower().startswith("yolov10")):
                # Explicit yolov10 model string
                self.yolo = YOLOv10(yolo_model)
                self.detector_name = "YOLOv10"
            elif _YOLOV10_AVAILABLE:
                # Allow default v10 small model if user didn't specify
                # You can change to yolov10n if you prefer nano
                self.yolo = YOLOv10("yolov10s.pt")
                self.detector_name = "YOLOv10"
            else:
                # Fallback to YOLOv8 via ultralytics
                self.yolo = YOLO(yolo_model)
                self.detector_name = "YOLOv8"
            print(f"   ✅ {self.detector_name} loaded successfully")
        except Exception as e:
            print(f"   ❌ Failed to load requested detector ({yolo_model}): {e}")
            print("   ↩️ Falling back to YOLOv8n")
            self.yolo = YOLO("yolov8n.pt")
            self.detector_name = "YOLOv8"
        
        # Class names
        self.class_names = [
            'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
            'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting',
            'Stealing', 'Vandalism', 'NormalVideos'
        ]
        
        # Dangerous objects that trigger immediate alerts
        self.dangerous_objects = {
            'knife', 'gun', 'rifle', 'pistol', 'weapon', 
            'fire', 'smoke', 'explosion'
        }
        
        # Transform pipeline (image size adapts when fast_mode enabled)
        image_size = 224 if not self.fast_mode else self.fast_image_size
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        # Frame buffer for temporal sequences
        self.frame_buffer = deque(maxlen=sequence_length)

        # Inference options
        # Test-time augmentation is expensive; disable when in fast_mode
        self.enable_tta = False if self.fast_mode else True
        # When True, prefer speed over extra TTA runs (fast_mode enables this)
        self.fast_inference = bool(self.fast_mode)
        print("✅ Anomaly Detection System Ready!\n")

    def _adjust_ml_score(self, raw_score: float, confidence: float) -> float:
        """
        Adjust the raw ML anomaly score according to configured influence & thresholds.

        Rules (simple, transparent):
        - If ML disabled, return 0.0
        - If confidence < ml_ignore_below_confidence -> return 0.0 (ignore low-confidence)
        - If confidence between ignore and min_vote, scale linearly from 0..0.5 of ml_influence
        - If confidence >= min_vote, use full ml_influence factor

        Returns the adjusted anomaly score (0..1, typically smaller than raw_score).
        """
        if not getattr(self, 'ml_enabled', True):
            return 0.0

        # Safety clamps
        raw_score = float(np.clip(raw_score, 0.0, 1.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))

        if confidence < self.ml_ignore_below_confidence:
            return 0.0

        if confidence < self.ml_min_confidence_for_vote:
            # Partial credit: map confidence in [ignore_below, min_vote) -> scale in [0, 0.5]
            denom = max(self.ml_min_confidence_for_vote - self.ml_ignore_below_confidence, 1e-6)
            frac = (confidence - self.ml_ignore_below_confidence) / denom
            scale = max(0.0, min(1.0, frac)) * 0.5
        else:
            scale = 1.0

        adjusted = raw_score * float(self.ml_influence) * scale
        # Keep within sane bounds
        return float(np.clip(adjusted, 0.0, 1.0))

    def _yolo_worker(self):
        """Background worker that consumes frames and updates the latest YOLO result."""
        while True:
            try:
                if self._yolo_queue is None:
                    break
                frame = self._yolo_queue.get()
                # None is a sentinel to stop the worker
                if frame is None:
                    break

                # Run the detection (reuse logic similar to detect_objects)
                try:
                    h, w = frame.shape[:2]
                    max_dim = 640
                    scale = min(max_dim / max(w, h), 1.0)
                    if scale < 1.0:
                        frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    else:
                        frame_resized = frame
                    # Prefer a lightweight detect-only call when configured for speed.
                    try:
                        # choose imgsz based on fast_mode / configured value
                        imgsz = self.yolo_fast_imgsz if getattr(self, 'fast_mode', False) else int(getattr(self, 'yolo_imgsz', 640))
                        if self.yolo_mode == 'detect' or getattr(self, 'fast_mode', False):
                            predict_kwargs = {'conf': 0.35, 'iou': 0.5, 'imgsz': imgsz}
                            if self.use_fp16:
                                predict_kwargs.update({'half': True})
                            # ultralytics supports .predict or calling the model directly.
                            if hasattr(self.yolo, 'predict'):
                                results = self.yolo.predict(frame_resized, **predict_kwargs)[0]
                            else:
                                results = self.yolo(frame_resized, **predict_kwargs)[0]
                        else:
                            track_kwargs = dict(persist=True, verbose=False)
                            track_kwargs.update({'conf': 0.35, 'iou': 0.5, 'imgsz': imgsz})
                            if self.use_fp16:
                                track_kwargs.update({'half': True})
                            results = self.yolo.track(frame_resized, **track_kwargs)[0]
                    except Exception:
                        # If the preferred API fails, try a simple call and continue.
                        try:
                            results = self.yolo(frame_resized)[0]
                        except Exception:
                            results = None
                    now = cv2.getTickCount() / cv2.getTickFrequency()
                    with self._yolo_lock:
                        self._last_yolo_result = results
                        self._last_yolo_time = now
                except Exception:
                    # ignore worker errors but continue
                    continue
            except Exception:
                # Ensure worker loop doesn't die silently
                continue
    
    def extract_frames(
        self, 
        video_path: str, 
        sample_rate: int = 2
    ) -> List[np.ndarray]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
            sample_rate: Extract every Nth frame
            
        Returns:
            List of frames (BGR format)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"📹 Processing video: {Path(video_path).name}")
        print(f"   Total frames: {total_frames}, FPS: {fps}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        print(f"   ✅ Extracted {len(frames)} frames")
        
        return frames
    
    def detect_objects(self, frame: np.ndarray, camera_id: Optional[str] = None) -> Dict:
        """
        Detect and track objects using YOLO with persistent tracking.
        
        Args:
            frame: Input frame (BGR)
            camera_id: Optional camera identifier for ROI-/camera-specific filters
            
        Returns:
            Dict with detected objects, bounding boxes, and tracking IDs
        """
        # ⭐ Use track() with downscaling + FP16 when available for speed
        # Downscale frame to max dimension 640 to reduce compute
        h, w = frame.shape[:2]
        max_dim = 640
        scale = min(max_dim / max(w, h), 1.0)
        if scale < 1.0:
            frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            frame_resized = frame

        # Prefer separate process worker when enabled
        if getattr(self, 'yolo_process', False) and getattr(self, '_yolo_in_q', None) is not None:
            try:
                try:
                    self._yolo_in_q.put_nowait(frame.copy())
                except Exception:
                    pass
                try:
                    proc_res = self._yolo_out_q.get_nowait()
                    now = cv2.getTickCount() / cv2.getTickFrequency()
                    with self._yolo_lock:
                        self._last_yolo_result = proc_res
                        self._last_yolo_time = now
                    return self._post_filter_detections(proc_res, (h, w), camera_id)
                except Exception:
                    with self._yolo_lock:
                        if self._last_yolo_result is not None:
                            return self._post_filter_detections(self._last_yolo_result, (h, w), camera_id)
            except Exception:
                pass

        # Prefer thread worker for low-latency when fast_mode
        if getattr(self, 'fast_mode', False) and getattr(self, '_yolo_queue', None) is not None:
            try:
                try:
                    self._yolo_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass
                with self._yolo_lock:
                    if self._last_yolo_result is not None:
                        # thread worker stores ultralytics Results object
                        raw = self._convert_yolo_results(self._last_yolo_result, scale)
                        return self._post_filter_detections(raw, (h, w), camera_id)
            except Exception:
                pass

        # Build kwargs for track - some backends accept imgsz/conf/iou/half
        track_kwargs = dict(persist=True, verbose=False)
        try:
            imgsz = self.yolo_fast_imgsz if getattr(self, 'fast_mode', False) else int(getattr(self, 'yolo_imgsz', 640))
            track_kwargs.update({'conf': 0.35, 'iou': 0.5, 'imgsz': imgsz})
            if self.use_fp16:
                track_kwargs.update({'half': True})

            # Simple caching to avoid running YOLO on every frame
            now = cv2.getTickCount() / cv2.getTickFrequency()
            with self._yolo_lock:
                if self._last_yolo_result is not None and (now - float(self._last_yolo_time)) < float(getattr(self, 'yolo_min_interval', 0.08)):
                    results = self._last_yolo_result
                else:
                    results = self.yolo.track(frame_resized, **track_kwargs)[0]
                    self._last_yolo_result = results
                    self._last_yolo_time = now
        except Exception:
            results = self.yolo.track(frame_resized, persist=True, verbose=False)[0]

        # If the result is already a simplified dict (from process worker), use it
        if isinstance(results, dict):
            return self._post_filter_detections(results, (h, w), camera_id)

        # Otherwise convert and post-filter
        raw = self._convert_yolo_results(results, scale)
        return self._post_filter_detections(raw, (h, w), camera_id)

    def _convert_yolo_results(self, results_obj, scale: float) -> Dict:
        """Convert raw ultralytics results object to the detections dict (used for cached results)."""
        detections = {
            'objects': [],
            'boxes': [],
            'confidences': [],
            'track_ids': [],
            'dangerous': False
        }
        for box in results_obj.boxes:
            class_id = int(box.cls[0]) if hasattr(box, 'cls') else int(box.data[0][-1])
            confidence = float(box.conf[0]) if hasattr(box, 'conf') else float(box.data[0][-2])
            track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else None
            names_map = getattr(results_obj, 'names', None) or getattr(self.yolo, 'names', None) or {}
            class_name = str(names_map.get(class_id, str(class_id))).lower()
            if hasattr(box, 'xyxy'):
                bbox = box.xyxy[0].cpu().numpy()
            else:
                arr = box.data[0].cpu().numpy()
                bbox = arr[:4]
            if scale < 1.0:
                bbox = bbox / scale

            detections['objects'].append(class_name)
            detections['boxes'].append(bbox)
            detections['confidences'].append(confidence)
            detections['track_ids'].append(track_id)
            if any(danger in class_name for danger in self.dangerous_objects):
                detections['dangerous'] = True

        return detections

    # --------------------
    # False-positive reduction utilities
    # --------------------
    def _get_yolo_filter_config(self) -> Dict:
        """Assemble YOLO post-filtering configuration from self.config with safe defaults."""
        # Defaults chosen to reduce common false positives while keeping people/vehicles
        defaults = {
            'class_thresholds': {
                # people/vehicles
                'person': 0.55,
                'car': 0.55, 'truck': 0.55, 'bus': 0.55, 'motorcycle': 0.55, 'bicycle': 0.55,
                # weapons and hazardous
                'knife': 0.70, 'scissors': 0.70, 'gun': 0.75, 'rifle': 0.75, 'pistol': 0.75, 'weapon': 0.75,
                'fire': 0.80, 'smoke': 0.80, 'explosion': 0.85,
                # bags (prone to FP)
                'backpack': 0.65, 'suitcase': 0.65, 'handbag': 0.65
            },
            # If set, only keep these classes; else accept any but with thresholds above
            'class_whitelist': [
                'person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle',
                'knife', 'scissors', 'gun', 'rifle', 'pistol', 'weapon',
                'fire', 'smoke', 'explosion', 'backpack', 'suitcase', 'handbag'
            ],
            'use_whitelist': True,
            # Geometry filters
            'min_box_area_ratio': 0.0002,  # drop boxes < 0.02% of frame area
            'max_aspect_ratio': 4.0,       # drop extremely skinny boxes
            'min_size_px': 8,              # min width/height in pixels
            # Temporal confirmation
            'require_persistence_frames': 2  # require track to persist this many frames if below 0.85 conf
        }
        try:
            cfg = dict(defaults)
            yolo_cfg = (self.config.get('yolo') or {}) if hasattr(self, 'config') else {}
            # merge nested dicts
            if 'class_thresholds' in yolo_cfg:
                cfg['class_thresholds'].update(dict(yolo_cfg['class_thresholds']))
            for k in ['class_whitelist', 'use_whitelist', 'min_box_area_ratio', 'max_aspect_ratio', 'min_size_px', 'require_persistence_frames']:
                if k in yolo_cfg:
                    cfg[k] = yolo_cfg[k]
        except Exception:
            cfg = defaults
        return cfg

    def _post_filter_detections(self, det: Dict, frame_hw: Tuple[int, int], camera_id: Optional[str]) -> Dict:
        """Apply confidence/geometry/persistence filters to reduce false positives."""
        try:
            H, W = frame_hw
            area = float(H * W)
            cfg = self._get_yolo_filter_config()
            class_thr = cfg['class_thresholds']
            whitelist = set(cfg['class_whitelist']) if cfg.get('use_whitelist', False) else None
            min_area = float(cfg['min_box_area_ratio']) * area
            max_ar = float(cfg['max_aspect_ratio'])
            min_px = int(cfg['min_size_px'])
            persist_needed = int(cfg['require_persistence_frames'])

            # Initialize track persistence cache
            if not hasattr(self, '_track_persistence'):
                self._track_persistence = {}
                self._frame_counter = 0
            self._frame_counter += 1

            f_objects, f_boxes, f_confs, f_ids = [], [], [], []
            dangerous = False

            for cls, box, conf, tid in zip(det.get('objects', []), det.get('boxes', []), det.get('confidences', []), det.get('track_ids', [])):
                try:
                    cname = str(cls).lower()
                    # Whitelist filtering
                    if whitelist is not None and cname not in whitelist:
                        continue

                    # Geometry checks
                    x1, y1, x2, y2 = map(float, box)
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    if w < min_px or h < min_px:
                        continue
                    box_area = w * h
                    if box_area < min_area:
                        continue
                    ar = (w / max(h, 1e-6)) if h > 0 else 999.0
                    if ar > max_ar or (1.0/ar) > max_ar:
                        continue

                    # Confidence threshold by class with fallback
                    base_thr = class_thr.get(cname, 0.60)
                    score = float(conf)

                    # Temporal persistence: allow slightly lower conf if the track persisted
                    ok = False
                    if score >= max(0.85, base_thr):
                        ok = True
                    else:
                        # update persistence counter for this track id (if available)
                        if tid is not None:
                            st = self._track_persistence.get(tid, {'last_seen': self._frame_counter - 100, 'count': 0})
                            if self._frame_counter - st['last_seen'] <= 2:
                                st['count'] = min(st['count'] + 1, persist_needed + 2)
                            else:
                                st['count'] = 1
                            st['last_seen'] = self._frame_counter
                            self._track_persistence[tid] = st
                            if st['count'] >= persist_needed and score >= (base_thr - 0.05):
                                ok = True
                        else:
                            # No tracking id; be stricter
                            ok = score >= base_thr

                    if not ok:
                        continue

                    f_objects.append(cname)
                    f_boxes.append([x1, y1, x2, y2])
                    f_confs.append(score)
                    f_ids.append(tid)
                    if any(d in cname for d in self.dangerous_objects):
                        dangerous = True
                except Exception:
                    continue

            return {
                'objects': f_objects,
                'boxes': f_boxes,
                'confidences': f_confs,
                'track_ids': f_ids,
                'dangerous': dangerous
            }
        except Exception:
            # On any failure, return the original detections to avoid breaking pipeline
            return det

    def yolo_infer_frame(self, frame: np.ndarray, mode: Optional[str] = None) -> Optional[Dict]:
        """Synchronous helper to run one YOLO inference on a frame.

        mode: 'detect' or 'track' or None (use self.yolo_mode)
        Returns the raw ultralytics Results object or None on failure.
        """
        if self.yolo is None:
            return None

        mode = (mode or self.yolo_mode or 'track').lower()
        h, w = frame.shape[:2]
        max_dim = 640
        scale = min(max_dim / max(w, h), 1.0)
        if scale < 1.0:
            frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            frame_resized = frame

        try:
            imgsz = self.yolo_fast_imgsz if getattr(self, 'fast_mode', False) else int(getattr(self, 'yolo_imgsz', 640))
            if mode == 'detect' or getattr(self, 'fast_mode', False):
                predict_kwargs = {'conf': 0.35, 'iou': 0.5, 'imgsz': imgsz}
                if self.use_fp16:
                    predict_kwargs.update({'half': True})
                if hasattr(self.yolo, 'predict'):
                    results = self.yolo.predict(frame_resized, **predict_kwargs)[0]
                else:
                    results = self.yolo(frame_resized, **predict_kwargs)[0]
            else:
                track_kwargs = dict(persist=True, verbose=False)
                track_kwargs.update({'conf': 0.35, 'iou': 0.5, 'imgsz': imgsz})
                if self.use_fp16:
                    track_kwargs.update({'half': True})
                results = self.yolo.track(frame_resized, **track_kwargs)[0]
            return results
        except Exception:
            try:
                return self.yolo(frame_resized)[0]
            except Exception:
                return None

    def benchmark_yolo(self, runs: int = 20, warmup: int = 3, mode: str = 'detect') -> Dict:
        """Benchmark YOLO on a synthetic frame and return timing stats.

        This is a simple micro-benchmark: it creates a random image with a
        representative size and runs the chosen YOLO API repeatedly.
        """
        import time

        # Make a synthetic frame (RGB) of moderate size
        frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)

        # Warmup runs
        for _ in range(warmup):
            _ = self.yolo_infer_frame(frame, mode=mode)

        timings = []
        for _ in range(runs):
            t0 = time.time()
            _ = self.yolo_infer_frame(frame, mode=mode)
            t1 = time.time()
            timings.append(t1 - t0)

        timings = np.array(timings)
        stats = {
            'mode': mode,
            'runs': int(runs),
            'mean_s': float(timings.mean()),
            'median_s': float(np.median(timings)),
            'p95_s': float(np.percentile(timings, 95)),
            'min_s': float(timings.min()),
            'max_s': float(timings.max())
        }
        print("YOLO benchmark:", stats)
        return stats
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess single frame for model input.
        
        Args:
            frame: BGR frame from OpenCV
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        transformed = self.transform(image=frame_rgb)
        frame_tensor = transformed['image']
        
        return frame_tensor
    
    def create_sequence(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Create temporal sequence from frames.
        
        Args:
            frames: List of frames
            
        Returns:
            Sequence tensor (1, T, C, H, W)
        """
        sequence = []
        for frame in frames:
            frame_tensor = self.preprocess_frame(frame)
            sequence.append(frame_tensor)
        
        # Stack into sequence
        sequence_tensor = torch.stack(sequence)  # (T, C, H, W)
        sequence_tensor = sequence_tensor.unsqueeze(0)  # (1, T, C, H, W)

        # Move to device and ensure dtype matches model precision
        sequence_tensor = sequence_tensor.to(self.device)
        # Prefer channels_last for better GPU throughput if model uses it
        try:
            sequence_tensor = sequence_tensor.to(memory_format=torch.channels_last)
        except Exception:
            pass
        if getattr(self, 'use_fp16', False):
            try:
                sequence_tensor = sequence_tensor.half()
            except Exception:
                # If half-cast fails for any reason, fall back to full precision
                pass

        return sequence_tensor
    
    @torch.no_grad()
    def predict_sequence(self, sequence: torch.Tensor) -> Dict:
        """
        Predict anomaly for a sequence of frames.
        
        Args:
            sequence: Preprocessed sequence tensor
            
        Returns:
            Prediction results with confidence scores
        """
        # Use autocast for mixed precision when model is not already in FP16.
        # If model is already half (use_fp16=True) we skip autocast.
        if torch.cuda.is_available() and not getattr(self, 'use_fp16', False):
            try:
                from torch.cuda.amp import autocast
                with autocast(enabled=True):
                    outputs = self.model(sequence)
            except Exception:
                outputs = self.model(sequence)
        else:
            outputs = self.model(sequence)
        
        # Get classification predictions
        class_logits = outputs['class_logits']
        class_probs = F.softmax(class_logits, dim=1)[0]  # (14,)
        
        # Get predicted class and confidence
        confidence, pred_class = torch.max(class_probs, dim=0)
        confidence = confidence.item()
        pred_class = pred_class.item()
        
        # Get top 3 predictions
        top3_conf, top3_idx = torch.topk(class_probs, k=3)
        top3_predictions = [
            {
                'class': self.class_names[idx],
                'confidence': conf.item()
            }
            for idx, conf in zip(top3_idx, top3_conf)
        ]
        
        # Determine if anomaly
        is_anomaly = pred_class != 13  # 13 = NormalVideos

        # Raw ML anomaly score (1 - P(Normal)) and adjusted score used downstream
        raw_anomaly_score = float(1.0 - class_probs[13].item())
        adjusted_anomaly_score = self._adjust_ml_score(raw_anomaly_score, confidence)

        return {
            'predicted_class': self.class_names[pred_class],
            'confidence': confidence,
            'is_anomaly': is_anomaly,
            'ml_raw_anomaly_score': raw_anomaly_score,
            'ml_adjusted_anomaly_score': adjusted_anomaly_score,
            # For backward compatibility downstream we set 'anomaly_score' to the adjusted value
            'anomaly_score': adjusted_anomaly_score,
            'top3_predictions': top3_predictions,
            'all_confidences': {
                name: class_probs[i].item() 
                for i, name in enumerate(self.class_names)
            }
        }

    @torch.no_grad()
    def predict_sequence_tta(self, frames: List[np.ndarray], do_flip: bool = True) -> Dict:
        """
        Predict using simple Test-Time Augmentation (horizontal flip).
        This runs the model on the original sequence and on the horizontally
        flipped sequence (if enabled) and averages softmax probabilities.
        """
        probs_list = []

        variants = [frames]
        if do_flip and not self.fast_inference:
            flipped = [cv2.flip(f, 1) for f in frames]
            variants.append(flipped)

        for var in variants:
            seq = self.create_sequence(var)
            # model call - reuse same autocast behavior as predict_sequence
            if torch.cuda.is_available() and not getattr(self, 'use_fp16', False):
                try:
                    from torch.cuda.amp import autocast
                    with autocast(enabled=True):
                        out = self.model(seq)
                except Exception:
                    out = self.model(seq)
            else:
                out = self.model(seq)

            logits = out['class_logits']
            probs = F.softmax(logits, dim=1)[0]
            probs_list.append(probs)

        # Average probabilities across variants
        avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)

        # Build result dict similar to predict_sequence
        confidence, pred_class = torch.max(avg_probs, dim=0)
        top3_conf, top3_idx = torch.topk(avg_probs, k=3)

        top3_predictions = [
            {'class': self.class_names[int(idx)], 'confidence': float(conf)}
            for idx, conf in zip(top3_idx.tolist(), top3_conf.tolist())
        ]

        is_anomaly = int(pred_class.item()) != 13

        raw_anomaly_score = float(1.0 - float(avg_probs[13].item()))
        adjusted_anomaly_score = self._adjust_ml_score(raw_anomaly_score, float(confidence.item()))

        return {
            'predicted_class': self.class_names[int(pred_class.item())],
            'confidence': float(confidence.item()),
            'is_anomaly': bool(is_anomaly),
            'ml_raw_anomaly_score': raw_anomaly_score,
            'ml_adjusted_anomaly_score': adjusted_anomaly_score,
            'anomaly_score': adjusted_anomaly_score,
            'top3_predictions': top3_predictions,
            'all_confidences': {
                name: float(avg_probs[i].item()) for i, name in enumerate(self.class_names)
            }
        }
    
    def predict_video(
        self, 
        video_path: str,
        use_yolo: bool = True,
        stride: int = 8
    ) -> Dict:
        """
        Analyze complete video for anomalies.
        
        Args:
            video_path: Path to video file
            use_yolo: Whether to use YOLO object detection
            stride: Stride for sliding window
            
        Returns:
            Complete analysis results
        """
        print(f"\n{'='*70}")
        print(f"🎥 ANALYZING VIDEO: {Path(video_path).name}")
        print(f"{'='*70}\n")
        
        # Extract frames
        frames = self.extract_frames(video_path, sample_rate=2)
        
        if len(frames) < self.sequence_length:
            raise ValueError(
                f"Video too short. Need at least {self.sequence_length} frames, "
                f"got {len(frames)}"
            )
        
        # Analyze sequences with sliding window
        results = {
            'video_path': video_path,
            'total_frames': len(frames),
            'predictions': [],
            'dangerous_objects_detected': [],
            'max_anomaly_score': 0.0,
            'anomaly_detected': False
        }
        
        print(f"🔍 Analyzing {len(frames) - self.sequence_length + 1} sequences...\n")
        
        for i in range(0, len(frames) - self.sequence_length + 1, stride):
            sequence_frames = frames[i:i + self.sequence_length]
            
            # YOLO detection on middle frame
            yolo_result = None
            if use_yolo:
                middle_frame = sequence_frames[self.sequence_length // 2]
                yolo_result = self.detect_objects(middle_frame)
                
                if yolo_result['dangerous']:
                    results['dangerous_objects_detected'].append({
                        'frame_index': i + self.sequence_length // 2,
                        'objects': yolo_result['objects']
                    })
            
            # Model prediction (with optional TTA)
            if getattr(self, 'enable_tta', False):
                prediction = self.predict_sequence_tta(sequence_frames)
            else:
                sequence_tensor = self.create_sequence(sequence_frames)
                prediction = self.predict_sequence(sequence_tensor)
            
            # Add YOLO info
            prediction['yolo_objects'] = yolo_result['objects'] if yolo_result else []
            prediction['dangerous_objects'] = yolo_result['dangerous'] if yolo_result else False
            prediction['frame_range'] = (i, i + self.sequence_length)
            
            results['predictions'].append(prediction)
            
            # Update max anomaly score
            if prediction['anomaly_score'] > results['max_anomaly_score']:
                results['max_anomaly_score'] = prediction['anomaly_score']
        
        # Determine overall anomaly
        anomaly_count = sum(1 for p in results['predictions'] if p['is_anomaly'])
        anomaly_ratio = anomaly_count / len(results['predictions'])
        
        results['anomaly_detected'] = (
            anomaly_ratio > 0.3 or  # 30% of sequences show anomaly
            results['max_anomaly_score'] > self.confidence_threshold or
            len(results['dangerous_objects_detected']) > 0
        )
        
        results['anomaly_ratio'] = anomaly_ratio
        
        # Summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print analysis summary."""
        print(f"\n{'='*70}")
        print(f"📊 ANALYSIS SUMMARY")
        print(f"{'='*70}\n")
        
        if results['anomaly_detected']:
            print("🚨 ANOMALY DETECTED!")
        else:
            print("✅ NO ANOMALY DETECTED")
        
        print(f"\nMetrics:")
        print(f"   Max Anomaly Score: {results['max_anomaly_score']:.2%}")
        print(f"   Anomaly Ratio: {results['anomaly_ratio']:.2%}")
        
        if results['dangerous_objects_detected']:
            print(f"\n⚠️  Dangerous Objects Detected:")
            for det in results['dangerous_objects_detected']:
                print(f"   Frame {det['frame_index']}: {', '.join(det['objects'])}")
        
        # Most confident prediction
        most_confident = max(results['predictions'], key=lambda x: x['confidence'])
        print(f"\nMost Confident Prediction:")
        print(f"   Class: {most_confident['predicted_class']}")
        print(f"   Confidence: {most_confident['confidence']:.2%}")
        
        print(f"\n{'='*70}\n")


def main():
    """Test the inference engine."""
    # Initialize detector
    detector = AnomalyDetector(
        model_path="models/best_model.pth",
        yolo_model="yolov8n.pt",
        device="cuda"
    )
    
    # Test on a video
    video_path = "test_video.mp4"  # Replace with your test video
    
    if Path(video_path).exists():
        results = detector.predict_video(video_path, use_yolo=True)
        
        # Print detailed results
        print("Top 3 Predictions for first sequence:")
        for pred in results['predictions'][0]['top3_predictions']:
            print(f"   {pred['class']}: {pred['confidence']:.2%}")
    else:
        print(f"❌ Test video not found: {video_path}")
        print("   Place a test video in the project root to test the system")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench-yolo', action='store_true', help='Run a small YOLO benchmark (detect vs track)')
    parser.add_argument('--bench-runs', type=int, default=6, help='Number of timed runs for benchmark')
    parser.add_argument('--bench-mode', choices=['detect', 'track'], default='detect', help='YOLO API to benchmark')
    args = parser.parse_args()

    if args.bench_yolo:
        det = AnomalyDetector(model_path="models/best_model.pth", yolo_model="yolov8n.pt", device="cuda", fast_mode=True)
        det.benchmark_yolo(runs=args.bench_runs, warmup=1, mode=args.bench_mode)
    else:
        main()


# Note: module-level CLI handling is above; do not call main() again here.
