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
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
import sys
from collections import deque
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.research_model import create_research_model


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
        confidence_threshold: float = 0.7
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
        
        # Load YOLO for object detection
        print(f"🎯 Loading YOLO model ({yolo_model})...")
        self.yolo = YOLO(yolo_model)
        print("   ✅ YOLO loaded successfully")
        
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
        
        # Transform pipeline
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        # Frame buffer for temporal sequences
        self.frame_buffer = deque(maxlen=sequence_length)
        
        print("✅ Anomaly Detection System Ready!\n")
    
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
    
    def detect_objects(self, frame: np.ndarray) -> Dict:
        """
        Detect objects using YOLO.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Dict with detected objects and bounding boxes
        """
        results = self.yolo(frame, verbose=False)[0]
        
        detections = {
            'objects': [],
            'boxes': [],
            'confidences': [],
            'dangerous': False
        }
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = results.names[class_id].lower()
            bbox = box.xyxy[0].cpu().numpy()
            
            detections['objects'].append(class_name)
            detections['boxes'].append(bbox)
            detections['confidences'].append(confidence)
            
            # Check for dangerous objects
            if any(danger in class_name for danger in self.dangerous_objects):
                detections['dangerous'] = True
        
        return detections
    
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
        
        return sequence_tensor.to(self.device)
    
    @torch.no_grad()
    def predict_sequence(self, sequence: torch.Tensor) -> Dict:
        """
        Predict anomaly for a sequence of frames.
        
        Args:
            sequence: Preprocessed sequence tensor
            
        Returns:
            Prediction results with confidence scores
        """
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
        
        return {
            'predicted_class': self.class_names[pred_class],
            'confidence': confidence,
            'is_anomaly': is_anomaly,
            'anomaly_score': 1.0 - class_probs[13].item(),  # 1 - P(Normal)
            'top3_predictions': top3_predictions,
            'all_confidences': {
                name: class_probs[i].item() 
                for i, name in enumerate(self.class_names)
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
            
            # Model prediction
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
    main()
