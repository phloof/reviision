"""Zone management helper for zone-aware analytics"""
import threading
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ZoneManager:
    """Thread-safe access to rectangular zones stored in database"""

    def __init__(self, database):
        self._db = database
        self._lock = threading.RLock()
        self._zones: Dict[str, List[Dict]] = {}
        self.reload()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def reload(self):
        """Pull zones from DB into memory cache"""
        with self._lock:
            try:
                # fetch for all cameras
                cameras = ['default']  # Start with default; extend if multi-cam later
                zones_global: Dict[str, List[Dict]] = {}
                for cam in cameras:
                    zones_global[cam] = self._db.get_zones(cam)
                self._zones = zones_global
                logger.debug("Zone cache reloaded: %s", self._zones)
            except Exception as e:
                logger.error("Failed to reload zones: %s", e)

    def get_zones(self, camera_id: str = 'default') -> List[Dict]:
        with self._lock:
            return list(self._zones.get(camera_id, []))

    def point_to_zone(self, camera_id: str, x: int, y: int) -> int:
        """Return zone id containing point or 0 if none"""
        with self._lock:
            for zone in self._zones.get(camera_id, []):
                if zone['x1'] <= x <= zone['x2'] and zone['y1'] <= y <= zone['y2']:
                    return zone['id']
            return 0

    # ------------------------------------------------------------------
    # CRUD (these call DB then reload cache)
    # ------------------------------------------------------------------
    def create_zone(self, name: str, x1: int, y1: int, x2: int, y2: int, camera_id: str = 'default') -> int:
        zone_id = self._db.create_zone(name, x1, y1, x2, y2, camera_id)
        self.reload()
        return zone_id

    def update_zone(self, zone_id: int, **fields):
        self._db.update_zone(zone_id, **fields)
        self.reload()

    def delete_zone(self, zone_id: int):
        self._db.delete_zone(zone_id)
        self.reload() 