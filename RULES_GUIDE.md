# Smart Rule Engine - Detection Rules Guide

## Overview

The rule engine provides intelligent alerts by combining ML predictions with object detection and motion analysis. All rules work automatically without configuration.

---

## Active Detection Rules

### 🔴 CRITICAL Alerts

#### 1. Person with Weapon

- **Detection:** Person detected + Weapon (knife, gun, scissors) in same frame
- **Alert Level:** CRITICAL (Red)
- **How it works:** YOLO detects both person and weapon objects simultaneously
- **Example:** "Person with KNIFE detected! Immediate action required."

#### 2. Multiple Weapons

- **Detection:** 2 or more weapons detected in frame
- **Alert Level:** CRITICAL (Red)
- **How it works:** Counts weapon-class objects from YOLO detections
- **Example:** "2 weapons detected: knife, gun"

#### 3. Dangerous Objects

- **Detection:** Fire, explosion, smoke, or bomb detected
- **Alert Level:** CRITICAL (Red)
- **How it works:** YOLO identifies dangerous objects
- **Example:** "Dangerous condition: fire detected!"

#### 4. ML Model Anomalies (High Severity)

- **Detection:** ML model predicts: Shooting, Explosion, Robbery, Arson
- **Alert Level:** CRITICAL (Red)
- **How it works:** BiLSTM-Transformer model classifies video sequence
- **Example:** "ML Model detected Shooting with 95.3% confidence"

---

### 🟠 HIGH Alerts

#### 5. Fighting/Assault Detection

- **Detection:** ML model predicts: Fighting, Assault, Abuse
- **Alert Level:** HIGH (Orange)
- **How it works:** Pose estimation + ML model classification
- **Example:** "Abnormal pose detected: fighting detected"

#### 6. Road Accidents

- **Detection:** ML model predicts: RoadAccidents
- **Alert Level:** HIGH (Orange)
- **How it works:** ML model with vehicle object context
- **Example:** "ML Model detected RoadAccidents with objects: car, truck"

#### 7. Burglary/Stealing

- **Detection:** ML model predicts: Burglary, Stealing
- **Alert Level:** HIGH (Orange)
- **How it works:** ML model analyzes suspicious behavior patterns
- **Example:** "ML Model detected Stealing with 87.2% confidence"

---

### 🟡 MEDIUM Alerts

#### 8. Crowd Density

- **Detection:** More than 15 people in frame
- **Alert Level:** MEDIUM (Yellow)
- **How it works:** YOLO counts 'person' class detections
- **Threshold:** 15 people (configurable)
- **Example:** "23 people detected. Monitoring for crowd control."

#### 9. Abnormal Crowd Flow

- **Detection:** People moving in chaotic/opposite directions
- **Alert Level:** MEDIUM (Yellow)
- **How it works:** Analyzes velocity vectors of multiple people
- **Threshold:** High variance in movement angles
- **Example:** "Abnormal crowd movement pattern detected"

#### 10. Vandalism/Shoplifting

- **Detection:** ML model predicts: Vandalism, Shoplifting
- **Alert Level:** MEDIUM (Yellow)
- **How it works:** ML model identifies suspicious retail behavior
- **Example:** "ML Model detected Shoplifting"

---

### 🔵 LOW/INFO Alerts

#### 11. Loitering Detection

- **Detection:** Person stationary in same location > 30 seconds
- **Alert Level:** LOW (Blue)
- **How it works:** Tracks person positions over time
- **Threshold:** 30 seconds (configurable)
- **Example:** "Person loitering for 45 seconds"

#### 12. Sudden Appearance/Disappearance

- **Detection:** Object appears without tracking history
- **Alert Level:** LOW (Blue)
- **How it works:** Compares detection with tracking database
- **Example:** "Person suddenly appeared in scene"

#### 13. Motion Anomalies

- **Detection:** Unusual motion patterns (rapid movement, crowd panic)
- **Alert Level:** Context-based
- **How it works:** Optical flow + background subtraction analysis
- **Example:** "Unusual motion pattern detected: crowd panic"

---

## How the System Works

### Detection Pipeline

```
Video Frame
    ↓
1. YOLO Object Detection → [person, knife, car, ...]
    ↓
2. ML Model Classification → [Shooting, Fighting, Normal, ...]
    ↓
3. Motion Analysis → [speed, direction, optical flow]
    ↓
4. Pose Estimation → [fighting pose, fallen person]
    ↓
5. Rule Engine → Combine all signals
    ↓
ALERTS (with severity, location, confidence)
```

### Multi-Signal Fusion

- **YOLO:** Detects what objects are present
- **ML Model:** Classifies the action/event happening
- **Motion:** Analyzes how things are moving
- **Pose:** Detects human body positions
- **Rules:** Combines everything for smart alerts

---

## Future Enhancements (Zone-Based Rules)

These rules are implemented but disabled. Requires zone configuration:

### Zone-Based Detection (Coming Soon)

1. **Running in Restricted Areas** - Detect fast movement in sensitive zones
2. **Large Objects in Restricted Areas** - Flag unauthorized bags in secure zones
3. **Vehicle Speeding** - Speed violations in parking/pedestrian areas
4. **Tripwire Crossing** - Virtual boundary violations

**To Enable:**

1. Define zones using `zone_manager.add_zone()`
2. Uncomment zone rules in `rule_engine.py`
3. Configure zone types (RESTRICTED, SPEED_LIMIT, NO_LOITERING, etc.)

---

## Configuration

### Adjustable Thresholds

Located in `backend/services/rule_engine.py`:

```python
crowd_density_threshold = 15        # Max people before alert
loitering_time_seconds = 30.0       # Seconds before loitering alert
running_speed_threshold = 15.0      # px/frame for running detection
vehicle_speed_limit = 20.0          # px/frame for speeding
large_object_size_threshold = 5000  # pixels² for large object
```

### Alert Levels

- **CRITICAL (Red):** Immediate threat - requires instant action
- **HIGH (Orange):** High risk - urgent attention needed
- **MEDIUM (Yellow):** Attention needed - monitor situation
- **LOW (Blue):** Monitor - low priority
- **INFO (Green):** Normal activity - informational

---

## Testing

Run the test suite to verify all rules:

```bash
python test_enhanced_rules.py
```

Current test results:

- ✅ Multiple Weapons Detection
- ✅ Person + Weapon Detection
- ✅ Original 8 Rules Working
- ✅ Zone Infrastructure Ready
- ✅ Crowd Flow Detection
- ✅ Sudden Appearance Detection

---

## Summary

**Total Active Rules:** 13 rules (no configuration needed)
**Future Rules:** 4 zone-based rules (requires setup)
**Detection Methods:** 6 modalities (ML, YOLO, Motion, Pose, Tracking, Rules)
**Alert Levels:** 5 severity levels (CRITICAL → INFO)

The system works out-of-the-box with your existing ML model and YOLO detection!
