# Post-Training Runtime Guide

A clear, technical walkthrough of what happens after model training: how frames enter the system, how each modality analyzes them, how decisions are fused, how alerts are generated, and what the backend returns to the frontend in real time.

Last updated: 2025-10-19

## Architecture diagram (post-training runtime)

```mermaid
flowchart LR
  subgraph FE[Frontend (React)]
    CAM[Camera Stream]
    UI[Live UI + Overlays]
    CAM -->|Frames (base64)| WS
  end

  WS([WebSocket ws://localhost:8000/ws/stream]) --> API

  subgraph API[FastAPI Backend]
    DP[UnifiedDetectionPipeline]
  end

  API --> DP

  subgraph Mods[Detection Modalities]
    YOLO[YOLO Objects\n+Tracking IDs]
    POSE[Pose Estimation\n(OpenPose/MediaPipe)]
    MOTION[Motion Analysis\n(Optical Flow, MOG2)]
    TRACK[Centroid Tracking\n+Speed Analyzer]
    ML[Trained Model\n(Temporal Sequence Classifier)]
  end

  DP --> YOLO
  DP --> POSE
  DP --> MOTION
  DP --> TRACK
  DP --> ML

  YOLO --> FUSION
  POSE --> FUSION
  MOTION --> FUSION
  TRACK --> FUSION
  ML --> FUSION

  subgraph FX[Intelligent Fusion]
    FUSION[Fusion Engine\n(Weights: ML 40, Obj 25, Pose 20, Motion 15)\nConsensus + Critical Overrides]
  end

  FUSION --> RULES

  subgraph RE[Rule Engine]
    RULES[Context & Zone Rules\n(weapon/person, crowd, zones, flow)]
  end

  RULES --> OUT{Threat Level\n+Alerts + Summary}
  OUT -->|JSON messages| FE
```

Note: If your Markdown viewer doesn’t render Mermaid, this diagram shows the flow: Frontend streams frames over WebSocket to FastAPI → UnifiedDetectionPipeline fans out to modalities (YOLO, Pose, Motion, Tracking/Speed, Trained Model) → Intelligent Fusion combines signals with weights and safety overrides → Rule Engine adds context-aware alerts → Backend returns a structured JSON result (threat level, alerts, details) back to the frontend.

## What runs after training

Once the model is trained, two main runtime layers work together:

- Core inference engine
  - Loads your trained weights (`models/best_model.pth`) and config (`configs/config_research_enhanced.yaml`).
  - Runs YOLO object detection with persistent tracking IDs.
  - Builds temporal sequences of frames and makes anomaly predictions with the trained model.
- Unified detection pipeline
  - Adds motion analysis, pose estimation, tracker + speed analysis, a context-aware rule engine, and a professional multi-modal fusion engine.

The backend exposes these capabilities via a FastAPI service (REST + WebSocket) for batch video analysis and real-time streaming.

## End-to-end data flow (frame to alert)

1. Frame ingestion

- Batch mode (video): A video file is uploaded; frames are sampled and grouped into temporal sequences.
- Real-time mode (WebSocket): The frontend streams image frames (as base64). Each frame is decoded and processed immediately.

2. YOLO object detection + tracking

- YOLO detects objects per frame and assigns stable track IDs (persisted across frames) so movement, speed, and direction can be computed.
- Outputs include detected class names, bounding boxes, confidences, and optional track IDs.

3. Temporal sequence prediction (trained model)

- Frames are buffered into sequences (default length: 16).
- When the buffer is full, the trained model produces a classification result with a confidence and top-3 classes.
- “Normal” is considered the non-anomalous class; any other class indicates a type of abnormal event (e.g., Shooting, Robbery, Fighting, etc.).

4. Motion analysis

- Optical flow and background subtraction summarize the scene’s motion magnitude, direction, and active regions.
- The service raises flags such as rapid movement, crowd panic, abandoned object, or loitering based on thresholds and motion history.

5. Pose estimation

- If a pose backend is available (OpenPose preferred; MediaPipe fallback), human keypoints are analyzed for signatures like falling, distress, fighting, or unusual poses.
- The result is summarized as persons detected, anomaly type (if any), and confidence.

6. Centroid tracking + speed

- A lightweight tracker maintains identities over time, updates trajectory, and computes per-object speed and direction.
- A speed analyzer flags fast-moving persons and speeding vehicles.

7. Intelligent fusion (single final decision)

- The fusion engine combines all modalities with tuned weights: ML model (40%), YOLO objects (25%), pose (20%), motion (15%).
- It reports anomalies only when the fused score ≥ 0.70, keeping output focused on actual threats.
- Critical overrides: clearly dangerous objects (e.g., guns, knives, fire) trigger an immediate CRITICAL alert even if other signals are weak.
- Special-case boosts: genuine “person falling” scenarios are detected by consensus between pose, motion, and a present person object, ensuring they cross the anomaly threshold.

8. Rule engine (context-aware alerts)

- Adds additional alerts based on the scene context: person-with-weapon, dangerous objects, crowd density, vehicle near a crowd, pose/motion anomalies, restricted zone violations, abnormal crowd flow, sudden object appearance/disappearance, etc.
- Zone-based rules work when polygonal zones are configured; otherwise they remain inactive.

9. Visualization (optional, runtime-only)

- Bounding boxes and alert overlays are rendered on the frame. For real-time streaming, the backend focuses on structured JSON messages; the encoded frame can be enabled if required.

10. Outputs returned to the frontend

- Summary: threat level (LOW/MEDIUM/HIGH/CRITICAL), human-readable explanation, anomaly flag.
- Fusion details: anomaly type, severity, fused score, confidence, weighted score breakdown, consensus info.
- Alerts: list of rule-based and fusion alerts, each with level, title, message, confidence, timestamp, and basic metadata.
- Raw detections for overlay: objects (with boxes), motion summary, pose summary, tracked objects and speeds.

## The runtime components that make this work

- Inference engine

  - Loads the trained model and a YOLO detector.
  - Maintains a frame buffer to form sequences for temporal prediction.
  - Exposes functions for object detection and sequence classification.

- Motion analyzer

  - Uses a background model and optical flow to detect crowd panic, rapid movement, abandoned objects, or loitering.
  - Keeps short-term history to identify statistical outliers (z-score based).

- Pose estimator

  - Detects and interprets human keypoints. Flags “falling”, “distress”, “fighting”, “unusual pose”.
  - Draws pose overlays only when anomalies need highlighting.

- Object tracker + speed analyzer

  - Tracks detected objects via centroids. Computes trajectory, speed, and direction.
  - Flags running persons and speeding vehicles.

- Rule engine

  - Implements practical rules that map patterns to alerts (e.g., person + weapon → CRITICAL, crowd density > threshold → MEDIUM, dangerous objects → CRITICAL).
  - Supports zone-based policies (restricted areas, tripwires, speed zones) when zones are defined.

- Intelligent fusion engine

  - Computes a single fused decision per frame considering all modalities.
  - Applies consensus bonuses when multiple modalities agree.
  - Escalates severity for inherently dangerous detections (weapons, fire, explosion) via critical overrides.

- FastAPI backend
  - REST endpoints for health, classes, video analysis, frame analysis, and detection history/statistics.
  - WebSocket endpoint for real-time streaming with backpressure/heartbeat protections.

## How the fusion decision is formed

- Inputs considered per frame

  - ML model: predicted class and confidence (ignored if the class is “Normal”).
  - YOLO: object types, counts (especially persons and weapons), bounding boxes.
  - Pose: anomaly type and confidence (falling, fighting, distress, unusual pose).
  - Motion: magnitude and anomaly type (crowd panic, rapid movement, abandoned object, loitering).

- Scoring and thresholding

  - Weighted sum of modality scores (ML 40%, Objects 25%, Pose 20%, Motion 15%).
  - Consensus bonus is added when ≥2 modalities indicate abnormality.
  - Anomaly emitted only if fused score ≥ 0.70; otherwise the scene is considered normal and not reported.

- Severity mapping

  - Critical: weapons, fire/explosion, multiple weapons.
  - High: robbery, assault, fighting, road accidents, burglary, crowd panic.
  - Medium: abuse, vandalism, shoplifting, high crowd density, suspicious behavior, person falling, rapid movement, abandoned object.
  - Low: loitering, arrest, unusual pose, person lying down, crowd flow anomaly.

- Output content
  - Anomaly type and severity, fused score, confidence, reasoning list and concise explanation.
  - Score breakdown per modality and whether a critical override occurred.

## Rule engine: what it adds on top

Active rule highlights (out-of-the-box):

- Person with weapon: immediate CRITICAL alert.
- Multiple weapons: CRITICAL alert summarizing weapon types.
- Dangerous object detected (fire/smoke/explosion): CRITICAL alert.
- Crowd density above threshold: MEDIUM alert with counts.
- Motion anomalies: configurable severity for panic, rapid movement, abandoned object, loitering.
- Pose anomalies: configurable severity for fighting, falling, distress, suspicious pose.
- Vehicle in a crowd: MEDIUM alert for safety monitoring.
- Abnormal crowd flow: MEDIUM alert for movement against normal flow or chaotic variance.
- Sudden object appearance/disappearance: LOW informational alert.

Zone-based rules (require configured zones):

- Restricted/high-security zone violations by persons.
- Running in restricted zones.
- Large object presence in restricted zones.
- Vehicle speeding within speed-limit zones.
- Virtual tripwire crossing events.

Note: A zone manager is instantiated by default, but no zones exist unless you define them. Once zones are added (polygons/lines), these spatial rules become active automatically.

## Real-time API and outputs

Backend interface (high level):

- Health and class info: basic service and model status.
- Upload video for analysis: runs the batch pipeline and returns aggregate metrics, top predictions, dangerous object occurrences, and anomaly ratio.
- Analyze a single frame (image): returns object detection summary.
- WebSocket stream: primary real-time path. The client sends frames; the server returns a structured message each cycle with the fusion result, threat level, alerts, and raw detection summaries suitable for UI overlay.
- Detections history and statistics: retrieve recent fused detections and calculated metrics (counts by severity/type, averages, anomaly rate, override/consensus counts) and clear history when needed.

Operational safeguards:

- Heartbeats if no frames are received for a short period.
- Connection-state checks before heavy processing and send operations.
- Conservative JPEG encoding for visual overlays (kept optional to favor JSON-only real-time updates).

## Concrete “what happens” examples (no code)

1. Weapon appears in the scene

- YOLO detects a “gun” or “knife”; the critical override triggers immediately.
- Fusion reports a CRITICAL anomaly of type “Weapon Detected” (or “Multiple Weapons” if more than one) with maximal fused score.
- Rule engine adds a matching CRITICAL alert, typically associated with the weapon’s bounding box; if persons are nearby, that context is included.
- Threat level is set to CRITICAL; the frontend highlights the alert prominently.

2. Person falling

- Pose detects a falling signature; motion confirms rapid downward movement; a person is present.
- Even if the ML model is neutral, fusion adds a strong boost to guarantee the overall score crosses the anomaly threshold.
- Fusion outputs “Person Falling” at MEDIUM severity by default (configurable), with reasoning that pose and motion agreed.
- Rule engine adds a HIGH or MEDIUM alert depending on the pose mapping; the UI surfaces this clearly with a timestamp.

3. High crowd density

- YOLO detects many persons; object scoring reaches its maximum.
- Fusion elevates the score and emits a “High Crowd Density” anomaly once the person count passes the threshold, even without other signals.
- Rule engine adds a MEDIUM “High Crowd Density” alert with the count and configured threshold for context.

4. Fighting or violent confrontation

- Pose shows aggressive arm configurations and high temporal variance; the ML model often also predicts “Fighting” with high confidence; motion is turbulent.
- Multiple modalities agree; the consensus bonus pushes the fused score well above threshold.
- Fusion reports a HIGH-severity anomaly (or CRITICAL depending on context) with reasoning that lists pose, objects, and ML prediction.
- Rule engine adds a matching HIGH alert with brief explanation and any relevant objects.

5. Dangerous object without people

- YOLO detects “fire” or “smoke”; even without persons present the scene is dangerous.
- Critical override triggers a CRITICAL alert; fusion emits a CRITICAL anomaly with a clear explanation.
- The frontend receives a top-priority alert advising immediate action.

6. Vehicle moving through a crowd

- YOLO finds a vehicle and several persons; tracker computes vehicle speed.
- Rule engine emits a MEDIUM alert (“Vehicle in Crowd”) and, if speeding is detected or zones are configured, a speeding violation.
- Fusion may classify the scene as “Suspicious Behavior” or “Crowd Panic” depending on motion patterns and ML evidence.

7. Restricted zone violation (with zones configured)

- A person’s bounding box center enters a restricted or high-security polygon.
- Rule engine emits a HIGH “Restricted Zone Violation” alert, tagging the zone ID and type.
- Fusion considers this context in the overall severity; the frontend highlights the spatial violation.

## Practical thresholds and knobs

- Fusion threshold: 0.70 default for anomaly-only output. Raise it to be stricter, lower it to be more sensitive.
- Modality weights: ML 40%, Objects 25%, Pose 20%, Motion 15%. Tune to prioritize certain sensors.
- Crowd density: default threshold 15 persons. Adjust per camera and field of view.
- Loitering/motion: stationary region frames (>150 for abandoned; >900 for loitering) and motion magnitude thresholds control sensitivity.
- Speed: person running and vehicle speeding thresholds in the speed analyzer; speed-limit zones can enforce stricter values.
- Pose backend: OpenPose offers richer keypoints if installed; MediaPipe provides a strong fallback. Disable pose if the environment lacks these dependencies.

## What the frontend can rely on

- A consistent real-time message structure containing:
  - Whether an anomaly is detected and its severity.
  - A concise summary and explanation suitable for alert banners.
  - A list of alerts with levels, messages, and timestamps for a timeline view.
  - Minimal raw detection data (objects, motion, pose, tracking) to power overlays.
- History and statistics endpoints to populate dashboards: totals, rates, distribution by severity/type, and aggregate confidence/fused score.

## How to enable zone-based rules (spatial policies)

- Define polygons (zones) and lines (tripwires) for your camera’s view in a setup path of your choosing.
- Once zones are added to the zone manager at runtime, the rule engine’s zone-based checks become active immediately (restricted/high-security checks, running in zones, large-object-in-zone, speed limit violations, tripwire crossing).
- You can draw zones on outgoing frames for operator context; use transparent overlays and labels to keep the view clean.

## Operational notes

- The backend gracefully handles client timeouts and disconnects in real time.
- Visualization rendering is optimized and can be omitted for bandwidth; the core value lies in the structured fusion result and alert stream.
- Detection history is recorded in memory and exposed via APIs for UI dashboards; it can be cleared for testing sessions.

## Where to look in the codebase (by purpose)

- Core inference and fusion runtime: backend/core and backend/services.
- API entrypoint and routes: backend/api.
- Trained model integration and video analysis helpers: inference/engine.
- Configurations and training research details: configs and docs.

## Summary

After training, your system operates as a multi-modal real-time detector. Each frame goes through object detection, temporal classification, motion analysis, pose estimation, and tracking. A professional fusion engine consolidates these signals into one clear decision, augmented by a rule engine for context-aware alerts. The backend serves this as clean, structured outputs suitable for a responsive frontend UI and operational dashboards.
