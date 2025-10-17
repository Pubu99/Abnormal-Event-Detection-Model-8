import sys
import numpy as np
sys.path.append('backend')

from services.rule_engine import RuleEngine
from services.zone_manager import RestrictedZoneManager, ZoneType

# Setup
zm = RestrictedZoneManager()
zm.add_zone('test', [(100, 100), (400, 100), (400, 400), (100, 400)], ZoneType.RESTRICTED)
re = RuleEngine(zone_manager=zm)

# Test detection
det = [{'class': 'person', 'bbox': (200, 200, 250, 350), 'confidence': 0.95, 'velocity': (50, 0)}]

# Add speed manually to bypass any issues
det_with_speed = det.copy()
det_with_speed[0]['speed'] = 50.0

print(f"Zone manager zones: {len(zm)}")
print(f"Zone manager has zones: {len(re.zone_manager)}")
print(f"Running threshold: {re.running_speed_threshold}")
print(f"Detection: {det_with_speed[0]}")

# Check conditions
obj = det_with_speed[0]
print(f"\nCondition checks:")
print(f"  obj.get('class'): {obj.get('class')}")
print(f"  obj.get('class') == 'person': {obj.get('class') == 'person'}")
print(f"  obj.get('speed', 0): {obj.get('speed', 0)}")
print(f"  speed > threshold: {obj.get('speed', 0) > re.running_speed_threshold}")
print(f"  Combined condition: {obj.get('class') == 'person' and obj.get('speed', 0) > re.running_speed_threshold}")

# Test bbox in zone
is_in, zone_id = zm.is_bbox_in_zone((200, 200, 250, 350), zone_type=ZoneType.RESTRICTED)
print(f"\nIs bbox in zone: {is_in}, zone: {zone_id}")

# Manually call the method parts
print(f"\nManual check inside method:")
if obj.get('class') == 'person' and obj.get('speed', 0) > re.running_speed_threshold:
    print("  First condition passed")
    bbox = obj.get('bbox')
    if bbox:
        print(f"  Bbox exists: {bbox}")
        is_in_zone, zone_id = re.zone_manager.is_bbox_in_zone(bbox, zone_type=ZoneType.RESTRICTED)
        print(f"  is_in_zone: {is_in_zone}, zone_id: {zone_id}")

# Test the specific rule
alert = re.check_running_in_restricted_zone(det_with_speed)
print(f"\nRunning alert: {alert}")

# Test full evaluate
result = re.evaluate(det)
print(f"\nTotal alerts from evaluate: {len(result.alerts)}")
for a in result.alerts:
    print(f"  - {a.title}: {a.message}")
