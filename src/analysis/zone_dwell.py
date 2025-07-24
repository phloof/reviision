"""Zone-aware Dwell Time Analyzer

Tracks dwell time per person per rectangular zone using ZoneManager.
A dwell begins when a track appears in a zone (including zone 0 for outside zones)
and ends when the person leaves the zone, switches zones, or times out.
"""
import time
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)

class ZoneDwellTimeAnalyzer:
    """Compute dwell times for tracks with zone support"""

    def __init__(self, config: dict, zone_manager, database=None, camera_id: str = 'default'):
        self.config = config
        self.zone_manager = zone_manager
        self.db = database
        self.camera_id = camera_id
        # parameters
        self.min_dwell_time = config.get('min_dwell_time', 2.0)
        self.max_inactive_time = config.get('max_inactive_time', 10.0)
        # Runtime state
        # person_id -> { zone_id, start_time, last_update }
        self.active: Dict[int, Dict] = {}
        # Completed records for quick retrieval in session
        self.completed: List[Dict] = []
        logger.info("ZoneDwellTimeAnalyzer initialised for camera %s", camera_id)

    # ------------------------------------------------------------------
    def update(self, tracks: List[dict]):
        """Call every frame with list of tracked people.
        Each track must have keys: 'id', 'bbox' (x1,y1,x2,y2).
        """
        now = time.time()
        current_ids = set()
        for tr in tracks:
            pid = tr['id']
            bbox = tr['bbox']
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            zone_id = self.zone_manager.point_to_zone(self.camera_id, cx, cy)
            current_ids.add(pid)
            if pid not in self.active:
                # new dwell
                self._start_dwell(pid, zone_id, now)
            else:
                state = self.active[pid]
                if zone_id != state['zone_id']:
                    # switched zones → close previous, start new
                    self._end_dwell(pid, now)
                    self._start_dwell(pid, zone_id, now)
                else:
                    # same zone; update last seen
                    state['last_update'] = now
        # Handle disappeared persons
        inactive_ids = [pid for pid, st in self.active.items() if pid not in current_ids and now - st['last_update'] > self.max_inactive_time]
        for pid in inactive_ids:
            self._end_dwell(pid, now)
        return self.completed

    # ------------------------------------------------------------------
    def _start_dwell(self, person_id: int, zone_id: int, timestamp: float):
        self.active[person_id] = {
            'zone_id': zone_id,
            'start_time': timestamp,
            'last_update': timestamp
        }
        logger.debug("Start dwell person %s in zone %s", person_id, zone_id)

    def _end_dwell(self, person_id: int, timestamp: float):
        state = self.active.pop(person_id, None)
        if not state:
            return
        duration = timestamp - state['start_time']
        if duration < self.min_dwell_time:
            logger.debug("Discard short dwell person %s duration %.2f", person_id, duration)
            return
        record = {
            'person_id': person_id,
            'zone_id': state['zone_id'],
            'start_time': state['start_time'],
            'end_time': timestamp,
            'duration': duration
        }
        self.completed.append(record)
        # Persist to DB if available
        if self.db:
            try:
                self.db.store_dwell_time(person_id, state['zone_id'], datetime.fromtimestamp(state['start_time']), datetime.fromtimestamp(timestamp))
            except Exception as e:
                logger.error("Failed to store dwell time: %s", e)
        logger.debug("End dwell person %s zone %s duration %.2fs", person_id, state['zone_id'], duration)

    # ------------------------------------------------------------------
    def get_stats(self):
        if not self.completed:
            return {}
        total = len(self.completed)
        durations = [r['duration'] for r in self.completed]
        return {
            'total_dwells': total,
            'avg_duration': sum(durations)/total,
            'min_duration': min(durations),
            'max_duration': max(durations)
        } 