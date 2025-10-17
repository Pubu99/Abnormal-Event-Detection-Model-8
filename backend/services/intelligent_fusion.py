"""
Intelligent Multi-Modal Fusion System
Aligns ML model predictions with complementary detection methods
Professional weighted scoring and conflict resolution

Author: AI Assistant (Pro Implementation)
Date: 2025-10-17
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class AnomalyLevel(Enum):
    """Unified anomaly classification"""
    NORMAL = 0
    SUSPICIOUS = 1
    ABNORMAL = 2
    CRITICAL = 3


@dataclass
class FusionResult:
    """Intelligent fusion decision"""
    final_decision: AnomalyLevel
    confidence: float
    reasoning: List[str]
    ml_weight: float
    context_weight: float
    override_applied: bool
    frame_number: int
    timestamp: str
    
    # Detailed breakdown
    ml_score: float
    object_score: float
    pose_score: float
    motion_score: float
    
    # Supporting evidence
    detected_objects: List[str]
    pose_anomalies: List[str]
    motion_anomalies: List[str]


class IntelligentFusionEngine:
    """
    Professional multi-modal fusion system that intelligently combines:
    1. ML Model predictions (your trained 99.38% accuracy model)
    2. Object detection (YOLO - weapons, dangerous objects)
    3. Pose estimation (abnormal human poses)
    4. Motion analysis (crowd panic, unusual movements)
    
    Uses weighted scoring with intelligent override logic:
    - Critical objects (weapons) → Immediate CRITICAL
    - ML model has primary authority for learned patterns
    - Context methods provide verification and early warnings
    - Conflicts resolved through weighted voting and rules
    """
    
    def __init__(self):
        # Weights for fusion (adjustable based on confidence in each module)
        self.weights = {
            'ml_model': 0.50,      # Primary - your trained model
            'objects': 0.25,        # High priority - dangerous objects
            'pose': 0.15,          # Supporting - abnormal poses
            'motion': 0.10         # Supporting - unusual motion
        }
        
        # Critical objects that override ML model
        self.critical_objects = {
            'knife', 'gun', 'weapon', 'fire', 'smoke', 
            'bomb', 'explosive', 'blade'
        }
        
        # Anomaly class patterns from your training
        self.violent_classes = {
            'Abuse', 'Arrest', 'Arson', 'Assault', 'Fighting', 
            'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
        }
        
    def fuse_detections(self,
                       ml_result: Optional[Dict],
                       yolo_objects: List[str],
                       pose_result: Dict,
                       motion_result: Dict,
                       frame_number: int,
                       timestamp: str) -> FusionResult:
        """
        Intelligent fusion of all detection sources
        
        Returns unified decision with reasoning
        """
        reasoning = []
        override_applied = False
        
        # 1. Check for CRITICAL overrides (weapons detected)
        critical_detected = self._check_critical_objects(yolo_objects)
        if critical_detected:
            reasoning.append(f"🚨 CRITICAL: Dangerous object detected - {', '.join(critical_detected)}")
            override_applied = True
            
            return FusionResult(
                final_decision=AnomalyLevel.CRITICAL,
                confidence=0.99,
                reasoning=reasoning,
                ml_weight=0.0,
                context_weight=1.0,
                override_applied=True,
                frame_number=frame_number,
                timestamp=timestamp,
                ml_score=0.0,
                object_score=1.0,
                pose_score=pose_result.get('confidence', 0.0) if pose_result.get('is_anomalous') else 0.0,
                motion_score=motion_result.get('confidence', 0.0) if motion_result.get('is_unusual') else 0.0,
                detected_objects=critical_detected,
                pose_anomalies=[pose_result.get('anomaly_type')] if pose_result.get('is_anomalous') else [],
                motion_anomalies=[motion_result.get('anomaly_type')] if motion_result.get('is_unusual') else []
            )
        
        # 2. Calculate individual scores
        ml_score = self._calculate_ml_score(ml_result)
        object_score = self._calculate_object_score(yolo_objects)
        pose_score = self._calculate_pose_score(pose_result)
        motion_score = self._calculate_motion_score(motion_result)
        
        # 3. Weighted fusion
        weighted_score = (
            ml_score * self.weights['ml_model'] +
            object_score * self.weights['objects'] +
            pose_score * self.weights['pose'] +
            motion_score * self.weights['motion']
        )
        
        # 4. Apply intelligent rules
        final_decision, confidence, reasoning = self._apply_fusion_rules(
            weighted_score, ml_result, yolo_objects, 
            pose_result, motion_result
        )
        
        # 5. Build supporting evidence
        detected_objects = yolo_objects if object_score > 0.3 else []
        pose_anomalies = [pose_result.get('anomaly_type')] if pose_result.get('is_anomalous') else []
        motion_anomalies = [motion_result.get('anomaly_type')] if motion_result.get('is_unusual') else []
        
        return FusionResult(
            final_decision=final_decision,
            confidence=confidence,
            reasoning=reasoning,
            ml_weight=self.weights['ml_model'],
            context_weight=sum([self.weights['objects'], self.weights['pose'], self.weights['motion']]),
            override_applied=override_applied,
            frame_number=frame_number,
            timestamp=timestamp,
            ml_score=ml_score,
            object_score=object_score,
            pose_score=pose_score,
            motion_score=motion_score,
            detected_objects=detected_objects,
            pose_anomalies=pose_anomalies,
            motion_anomalies=motion_anomalies
        )
    
    def _check_critical_objects(self, objects: List[str]) -> List[str]:
        """Check if any critical objects present"""
        return [obj for obj in objects if obj.lower() in self.critical_objects]
    
    def _calculate_ml_score(self, ml_result: Optional[Dict]) -> float:
        """Calculate ML model anomaly score"""
        if not ml_result:
            return 0.0
        
        # Use anomaly_score if available, otherwise derive from confidence
        if ml_result.get('is_anomaly'):
            return ml_result.get('anomaly_score', ml_result.get('confidence', 0.0))
        return 0.0
    
    def _calculate_object_score(self, objects: List[str]) -> float:
        """Calculate object detection anomaly score"""
        if not objects:
            return 0.0
        
        # Suspicious objects (person in restricted zones, unusual items)
        suspicious_count = len([obj for obj in objects if obj in ['person', 'backpack', 'suitcase']])
        
        if suspicious_count > 10:  # Crowd
            return 0.6
        elif suspicious_count > 5:
            return 0.4
        elif suspicious_count > 0:
            return 0.2
        
        return 0.0
    
    def _calculate_pose_score(self, pose_result: Dict) -> float:
        """Calculate pose anomaly score"""
        if not pose_result.get('is_anomalous'):
            return 0.0
        
        confidence = pose_result.get('confidence', 0.5)
        anomaly_type = pose_result.get('anomaly_type', '')
        
        # High-risk poses
        if anomaly_type in ['fighting', 'weapon_handling']:
            return confidence * 1.0
        elif anomaly_type in ['falling', 'distress']:
            return confidence * 0.8
        
        return confidence * 0.5
    
    def _calculate_motion_score(self, motion_result: Dict) -> float:
        """Calculate motion anomaly score"""
        if not motion_result.get('is_unusual'):
            return 0.0
        
        confidence = motion_result.get('confidence', 0.5)
        anomaly_type = motion_result.get('anomaly_type', '')
        
        # High-risk motion patterns
        if anomaly_type in ['crowd_panic', 'rapid_dispersal']:
            return confidence * 1.0
        elif anomaly_type in ['loitering', 'unusual_pattern']:
            return confidence * 0.6
        
        return confidence * 0.3
    
    def _apply_fusion_rules(self, 
                           weighted_score: float,
                           ml_result: Optional[Dict],
                           yolo_objects: List[str],
                           pose_result: Dict,
                           motion_result: Dict) -> Tuple[AnomalyLevel, float, List[str]]:
        """
        Apply intelligent decision rules
        
        Returns: (decision, confidence, reasoning)
        """
        reasoning = []
        
        # Rule 1: Critical threshold (≥ 0.8)
        if weighted_score >= 0.8:
            reasoning.append(f"🚨 High anomaly score: {weighted_score:.2f}")
            
            if ml_result and ml_result.get('is_anomaly'):
                reasoning.append(f"ML Model: {ml_result['predicted_class']} ({ml_result['confidence']:.1%})")
            
            return AnomalyLevel.CRITICAL, weighted_score, reasoning
        
        # Rule 2: Abnormal threshold (≥ 0.5)
        elif weighted_score >= 0.5:
            # Check for consensus
            anomaly_count = sum([
                1 if ml_result and ml_result.get('is_anomaly') else 0,
                1 if len(yolo_objects) > 5 else 0,
                1 if pose_result.get('is_anomalous') else 0,
                1 if motion_result.get('is_unusual') else 0
            ])
            
            if anomaly_count >= 2:  # At least 2 sources agree
                reasoning.append(f"⚠️ Multiple sources detected anomaly (Score: {weighted_score:.2f})")
                
                if ml_result and ml_result.get('is_anomaly'):
                    reasoning.append(f"ML: {ml_result['predicted_class']} ({ml_result['confidence']:.1%})")
                if pose_result.get('is_anomalous'):
                    reasoning.append(f"Pose: {pose_result['anomaly_type']}")
                if motion_result.get('is_unusual'):
                    reasoning.append(f"Motion: {motion_result['anomaly_type']}")
                
                return AnomalyLevel.ABNORMAL, weighted_score, reasoning
            else:
                reasoning.append(f"⚠️ Suspicious activity detected (Score: {weighted_score:.2f})")
                return AnomalyLevel.SUSPICIOUS, weighted_score * 0.8, reasoning
        
        # Rule 3: Suspicious threshold (≥ 0.3)
        elif weighted_score >= 0.3:
            reasoning.append(f"ℹ️ Minor anomaly indicators (Score: {weighted_score:.2f})")
            
            # List what triggered it
            if ml_result and ml_result.get('is_anomaly'):
                reasoning.append(f"ML flagged: {ml_result['predicted_class']}")
            if pose_result.get('is_anomalous'):
                reasoning.append(f"Pose: {pose_result['anomaly_type']}")
            if motion_result.get('is_unusual'):
                reasoning.append(f"Motion: {motion_result['anomaly_type']}")
            
            return AnomalyLevel.SUSPICIOUS, weighted_score, reasoning
        
        # Rule 4: Normal
        else:
            reasoning.append(f"✅ Normal activity (Score: {weighted_score:.2f})")
            return AnomalyLevel.NORMAL, 1.0 - weighted_score, reasoning
