"""
Zone Configuration Example
Define restricted zones, tripwires, and speed limits for your camera views

Author: AI Assistant
Date: 2025-10-17
"""

from services.zone_manager import RestrictedZoneManager, ZoneType


def setup_default_zones(zone_manager: RestrictedZoneManager):
    """
    Setup default zones for testing
    Adjust coordinates based on your camera view
    
    Args:
        zone_manager: RestrictedZoneManager instance
    """
    
    # Example Zone 1: Entrance (Restricted Area)
    zone_manager.add_zone(
        'entrance',
        polygon=[
            (100, 200),   # Top-left
            (400, 200),   # Top-right
            (400, 500),   # Bottom-right
            (100, 500)    # Bottom-left
        ],
        zone_type=ZoneType.RESTRICTED
    )
    
    # Example Zone 2: Server Room (High Security)
    zone_manager.add_zone(
        'server_room',
        polygon=[
            (500, 100),
            (800, 100),
            (800, 400),
            (500, 400)
        ],
        zone_type=ZoneType.HIGH_SECURITY
    )
    
    # Example Zone 3: Loading Dock (No Loitering)
    zone_manager.add_zone(
        'loading_dock',
        polygon=[
            (50, 550),
            (300, 550),
            (300, 700),
            (50, 700)
        ],
        zone_type=ZoneType.NO_LOITERING
    )
    
    # Example Zone 4: Speed Limit Zone (Parking Lot)
    zone_manager.add_zone(
        'parking_lot',
        polygon=[
            (600, 500),
            (1200, 500),
            (1200, 720),
            (600, 720)
        ],
        zone_type=ZoneType.SPEED_LIMIT,
        speed_limit=15.0  # px/frame
    )
    
    # Example Tripwire 1: Exit Gate
    zone_manager.add_tripwire(
        'exit_gate',
        line=((200, 300), (600, 300)),  # Horizontal line
        direction='both'  # Alert on both directions
    )
    
    # Example Tripwire 2: Perimeter Fence
    zone_manager.add_tripwire(
        'perimeter_fence',
        line=((100, 100), (100, 700)),  # Vertical line
        direction='forward'  # Alert only when crossing from left to right
    )
    
    print(f"✅ Configured {len(zone_manager)} zones")
    return zone_manager


def setup_custom_zones(zone_manager: RestrictedZoneManager, camera_config: dict):
    """
    Setup zones from camera configuration
    
    Args:
        zone_manager: RestrictedZoneManager instance
        camera_config: Dictionary with zone definitions
        
    Example camera_config:
        {
            'zones': [
                {
                    'id': 'main_entrance',
                    'type': 'restricted',
                    'polygon': [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                },
                ...
            ],
            'tripwires': [
                {
                    'id': 'gate_1',
                    'line': [(x1, y1), (x2, y2)],
                    'direction': 'both'
                },
                ...
            ]
        }
    """
    # Add zones
    for zone_config in camera_config.get('zones', []):
        zone_type_map = {
            'restricted': ZoneType.RESTRICTED,
            'high_security': ZoneType.HIGH_SECURITY,
            'no_loitering': ZoneType.NO_LOITERING,
            'speed_limit': ZoneType.SPEED_LIMIT,
            'public': ZoneType.PUBLIC
        }
        
        zone_manager.add_zone(
            zone_config['id'],
            zone_config['polygon'],
            zone_type=zone_type_map.get(zone_config['type'], ZoneType.PUBLIC),
            **zone_config.get('metadata', {})
        )
    
    # Add tripwires
    for tripwire_config in camera_config.get('tripwires', []):
        zone_manager.add_tripwire(
            tripwire_config['id'],
            tripwire_config['line'],
            direction=tripwire_config.get('direction', 'both')
        )
    
    print(f"✅ Configured {len(zone_manager)} zones from camera config")
    return zone_manager


# Example usage:
if __name__ == "__main__":
    # Create zone manager
    zm = RestrictedZoneManager()
    
    # Setup default zones
    setup_default_zones(zm)
    
    # Or setup from config
    camera_config = {
        'zones': [
            {
                'id': 'test_zone',
                'type': 'restricted',
                'polygon': [(200, 200), (400, 200), (400, 400), (200, 400)]
            }
        ],
        'tripwires': [
            {
                'id': 'test_tripwire',
                'line': [(100, 300), (500, 300)],
                'direction': 'both'
            }
        ]
    }
    
    zm2 = RestrictedZoneManager()
    setup_custom_zones(zm2, camera_config)
