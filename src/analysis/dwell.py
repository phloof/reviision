"""
Dwell Time Analyzer class for Retail Analytics System
"""

import time
import logging
import numpy as np
import cv2
from collections import defaultdict

logger = logging.getLogger(__name__)

class DwellTimeAnalyzer:
    """
    Dwell time analyzer that measures how long customers stay in specific areas
    
    This class tracks the time customers spend in defined zones and provides
    analytics on dwell times across different areas of the store.
    """
    
    def __init__(self, config):
        """
        Initialize the dwell time analyzer with the provided configuration
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        
        # Dwell time parameters
        self.min_dwell_time = config.get('min_dwell_time', 2.0)  # Minimum seconds to consider a dwell
        self.max_inactive_time = config.get('max_inactive_time', 10.0)  # Maximum seconds with no updates before ending dwell
        self.zones = config.get('zones', {})
        
        # Get zones from zone_grid if available and zones not defined
        if not self.zones and 'zone_grid' in config:
            rows = config['zone_grid'].get('rows', 3)
            cols = config['zone_grid'].get('cols', 3)
            width = config['zone_grid'].get('width', 640)
            height = config['zone_grid'].get('height', 480)
            self._create_zone_grid(rows, cols, width, height)
        
        # Point of interest areas (specific products or displays)
        self.points_of_interest = config.get('points_of_interest', {})
        
        # Storage for dwell data
        self.current_dwells = {}  # person_id -> {zone_id: {start_time, last_update, position}}
        self.completed_dwells = defaultdict(list)  # zone_id -> list of {person_id, start_time, end_time, duration}
        self.person_history = defaultdict(list)  # person_id -> list of all dwell records
        
        logger.info(f"Dwell time analyzer initialized with {len(self.zones)} zones and {len(self.points_of_interest)} points of interest")
    
    def _create_zone_grid(self, rows, cols, width, height):
        """
        Create a grid of zones
        
        Args:
            rows (int): Number of rows in the grid
            cols (int): Number of columns in the grid
            width (int): Frame width
            height (int): Frame height
        """
        self.zones = {}
        
        cell_width = width // cols
        cell_height = height // rows
        
        for row in range(rows):
            for col in range(cols):
                zone_id = f"zone_{row}_{col}"
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = (col + 1) * cell_width
                y2 = (row + 1) * cell_height
                
                self.zones[zone_id] = {
                    'name': f"Zone {row+1}-{col+1}",
                    'bounds': (x1, y1, x2, y2),
                    'color': (
                        (col * 50) % 255,
                        (row * 50) % 255,
                        ((row + col) * 75) % 255
                    )
                }
    
    def update(self, tracks, frame_shape=None):
        """
        Update dwell time data with new tracking information
        
        Args:
            tracks (list): List of person tracks with IDs and bounding boxes
            frame_shape (tuple, optional): Frame dimensions (height, width)
            
        Returns:
            dict: Updated dwell time data by zone
        """
        if not tracks:
            return self.completed_dwells
        
        current_time = time.time()
        
        # Update zones if needed and frame shape provided
        if not self.zones and frame_shape:
            height, width = frame_shape[:2]
            self._create_zone_grid(3, 3, width, height)
        
        # Update active tracks
        active_person_ids = set()
        for track in tracks:
            person_id = track['id']
            bbox = track['bbox']
            active_person_ids.add(person_id)
            
            # Calculate center point of the person
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            # Check which zone(s) the person is in
            person_zones = self._get_zones(center_x, center_y)
            
            # Update dwell times for each zone
            for zone_id in person_zones:
                self._update_dwell(person_id, zone_id, current_time, (center_x, center_y))
            
            # End dwells for zones the person is no longer in
            if person_id in self.current_dwells:
                zones_to_end = [z for z in self.current_dwells[person_id] if z not in person_zones]
                for zone_id in zones_to_end:
                    self._end_dwell(person_id, zone_id, current_time)
        
        # Check for inactive dwells
        all_person_ids = set(self.current_dwells.keys())
        inactive_person_ids = all_person_ids - active_person_ids
        
        for person_id in inactive_person_ids:
            inactive_time = current_time - max(
                dwell['last_update'] for dwell in self.current_dwells[person_id].values()
            )
            
            if inactive_time > self.max_inactive_time:
                # End all dwells for this inactive person
                for zone_id in list(self.current_dwells[person_id].keys()):
                    self._end_dwell(person_id, zone_id, current_time)
        
        return self.completed_dwells
    
    def _get_zones(self, x, y):
        """
        Determine which zone(s) a point is in
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            list: List of zone IDs that contain the point
        """
        zones = []
        
        # Check regular zones
        for zone_id, zone_info in self.zones.items():
            x1, y1, x2, y2 = zone_info['bounds']
            if x1 <= x <= x2 and y1 <= y <= y2:
                zones.append(zone_id)
        
        # Check points of interest
        for poi_id, poi_info in self.points_of_interest.items():
            if 'radius' in poi_info:
                # Circular POI
                center_x, center_y = poi_info['center']
                radius = poi_info['radius']
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if distance <= radius:
                    zones.append(poi_id)
            elif 'bounds' in poi_info:
                # Rectangular POI
                x1, y1, x2, y2 = poi_info['bounds']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    zones.append(poi_id)
        
        return zones
    
    def _update_dwell(self, person_id, zone_id, timestamp, position):
        """
        Update dwell time information for a person in a zone
        
        Args:
            person_id (int): Person track ID
            zone_id (str): Zone ID
            timestamp (float): Current timestamp
            position (tuple): (x, y) position of the person
        """
        # Initialize person's dwell data if not exists
        if person_id not in self.current_dwells:
            self.current_dwells[person_id] = {}
        
        # If this is a new dwell in this zone, create it
        if zone_id not in self.current_dwells[person_id]:
            self.current_dwells[person_id][zone_id] = {
                'start_time': timestamp,
                'last_update': timestamp,
                'positions': [position]
            }
        else:
            # Update existing dwell
            self.current_dwells[person_id][zone_id]['last_update'] = timestamp
            self.current_dwells[person_id][zone_id]['positions'].append(position)
            
            # Limit the number of positions stored to prevent memory issues
            max_positions = 100
            if len(self.current_dwells[person_id][zone_id]['positions']) > max_positions:
                self.current_dwells[person_id][zone_id]['positions'] = self.current_dwells[person_id][zone_id]['positions'][-max_positions:]
    
    def _end_dwell(self, person_id, zone_id, timestamp):
        """
        End a dwell period and record the data
        
        Args:
            person_id (int): Person track ID
            zone_id (str): Zone ID
            timestamp (float): Current timestamp
        """
        if person_id in self.current_dwells and zone_id in self.current_dwells[person_id]:
            dwell_data = self.current_dwells[person_id][zone_id]
            start_time = dwell_data['start_time']
            duration = timestamp - start_time
            
            # Only record dwells that meet the minimum time threshold
            if duration >= self.min_dwell_time:
                # Calculate average position
                positions = dwell_data['positions']
                avg_x = sum(p[0] for p in positions) / len(positions)
                avg_y = sum(p[1] for p in positions) / len(positions)
                
                # Record completed dwell
                dwell_record = {
                    'person_id': person_id,
                    'zone_id': zone_id,
                    'start_time': start_time,
                    'end_time': timestamp,
                    'duration': duration,
                    'avg_position': (avg_x, avg_y)
                }
                
                self.completed_dwells[zone_id].append(dwell_record)
                self.person_history[person_id].append(dwell_record)
                
                logger.debug(f"Person {person_id} dwelled in {zone_id} for {duration:.2f} seconds")
            
            # Remove from current dwells
            del self.current_dwells[person_id][zone_id]
            
            # Clean up if no more active dwells for this person
            if not self.current_dwells[person_id]:
                del self.current_dwells[person_id]
    
    def get_zone_dwell_stats(self, zone_id=None):
        """
        Get dwell time statistics for a specific zone or all zones
        
        Args:
            zone_id (str, optional): Zone ID to get statistics for
            
        Returns:
            dict: Dwell time statistics
        """
        if zone_id is not None:
            dwells = self.completed_dwells.get(zone_id, [])
            return self._calculate_dwell_stats(dwells, zone_id)
        
        # Calculate for all zones
        all_stats = {}
        for zone_id, dwells in self.completed_dwells.items():
            all_stats[zone_id] = self._calculate_dwell_stats(dwells, zone_id)
        
        return all_stats
    
    def _calculate_dwell_stats(self, dwells, zone_id):
        """
        Calculate statistics for a list of dwell records
        
        Args:
            dwells (list): List of dwell records
            zone_id (str): Zone ID
            
        Returns:
            dict: Statistics dictionary
        """
        if not dwells:
            return {
                'zone_id': zone_id,
                'total_dwells': 0,
                'unique_visitors': 0,
                'avg_duration': 0,
                'min_duration': 0,
                'max_duration': 0,
                'total_dwell_time': 0
            }
        
        durations = [d['duration'] for d in dwells]
        unique_visitors = len(set(d['person_id'] for d in dwells))
        
        stats = {
            'zone_id': zone_id,
            'total_dwells': len(dwells),
            'unique_visitors': unique_visitors,
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'total_dwell_time': sum(durations)
        }
        
        zone_name = None
        if zone_id in self.zones:
            zone_name = self.zones[zone_id].get('name', zone_id)
        elif zone_id in self.points_of_interest:
            zone_name = self.points_of_interest[zone_id].get('name', zone_id)
        
        if zone_name:
            stats['zone_name'] = zone_name
        
        return stats
    
    def get_person_dwell_history(self, person_id):
        """
        Get the dwell history for a specific person
        
        Args:
            person_id (int): Person track ID
            
        Returns:
            list: List of dwell records for the person
        """
        return self.person_history.get(person_id, [])
    
    def get_active_dwellers(self, zone_id=None):
        """
        Get list of people currently dwelling in a zone
        
        Args:
            zone_id (str, optional): Zone ID to check
            
        Returns:
            list: List of person IDs currently in the zone
        """
        if zone_id is not None:
            return [
                person_id for person_id, zones in self.current_dwells.items()
                if zone_id in zones
            ]
        
        # Return all active dwellers
        return list(self.current_dwells.keys())
    
    def get_dwell_heatmap_data(self):
        """
        Get data for generating a dwell time heatmap
        
        Returns:
            list: List of dwell points with intensity
        """
        heatmap_data = []
        
        # Include completed dwells
        for zone_dwells in self.completed_dwells.values():
            for dwell in zone_dwells:
                heatmap_data.append({
                    'position': dwell['avg_position'],
                    'intensity': dwell['duration'],
                    'active': False
                })
        
        # Include active dwells
        for person_id, zones in self.current_dwells.items():
            for zone_id, dwell in zones.items():
                if dwell['positions']:
                    # Calculate average of current positions
                    positions = dwell['positions']
                    avg_x = sum(p[0] for p in positions) / len(positions)
                    avg_y = sum(p[1] for p in positions) / len(positions)
                    
                    current_duration = time.time() - dwell['start_time']
                    if current_duration >= self.min_dwell_time:
                        heatmap_data.append({
                            'position': (avg_x, avg_y),
                            'intensity': current_duration,
                            'active': True
                        })
        
        return heatmap_data
    
    def visualize_dwells(self, frame, show_zones=True, show_active_only=False):
        """
        Visualize dwell times on the frame
        
        Args:
            frame (numpy.ndarray): Current video frame
            show_zones (bool): Whether to draw zone boundaries
            show_active_only (bool): Whether to show only active dwells
            
        Returns:
            numpy.ndarray: Frame with dwell visualization
        """
        if frame is None:
            return frame
        
        # Create a copy of the frame
        output_frame = frame.copy()
        
        # Draw zones if requested
        if show_zones:
            # Draw regular zones
            for zone_id, zone_info in self.zones.items():
                x1, y1, x2, y2 = zone_info['bounds']
                color = zone_info.get('color', (0, 255, 0))
                
                # Get active dwellers count for this zone
                active_count = len([
                    p for p, zones in self.current_dwells.items()
                    if zone_id in zones
                ])
                
                # Get completed dwell stats
                completed_dwells = len(self.completed_dwells.get(zone_id, []))
                
                # Adjust color based on activity
                if active_count > 0:
                    # Make more intense for active dwells
                    color = (0, 0, 255)  # Red for active
                
                # Draw zone boundary
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw zone name and stats
                zone_name = zone_info.get('name', zone_id)
                label = f"{zone_name}: {active_count} active, {completed_dwells} total"
                cv2.putText(output_frame, label, (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw points of interest
            for poi_id, poi_info in self.points_of_interest.items():
                color = poi_info.get('color', (255, 255, 0))
                name = poi_info.get('name', poi_id)
                
                # Get active dwellers count
                active_count = len([
                    p for p, zones in self.current_dwells.items()
                    if poi_id in zones
                ])
                
                # Adjust color based on activity
                if active_count > 0:
                    color = (0, 255, 255)  # Yellow for active
                
                if 'radius' in poi_info:
                    # Draw circular POI
                    center = poi_info['center']
                    radius = poi_info['radius']
                    cv2.circle(output_frame, center, radius, color, 2)
                    
                    # Draw name and stats
                    label = f"{name}: {active_count} active"
                    cv2.putText(output_frame, label, 
                                (center[0] - radius, center[1] - radius - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                elif 'bounds' in poi_info:
                    # Draw rectangular POI
                    x1, y1, x2, y2 = poi_info['bounds']
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw name and stats
                    label = f"{name}: {active_count} active"
                    cv2.putText(output_frame, label, (x1 + 5, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw active dwells
        for person_id, zones in self.current_dwells.items():
            for zone_id, dwell in zones.items():
                if not dwell['positions']:
                    continue
                
                # Get the most recent position
                x, y = dwell['positions'][-1]
                
                # Calculate dwell time so far
                current_duration = time.time() - dwell['start_time']
                
                # Draw dwell indicator
                if current_duration >= self.min_dwell_time:
                    # Size of circle proportional to dwell time
                    radius = min(50, int(5 + current_duration / 2))
                    cv2.circle(output_frame, (int(x), int(y)), radius, (0, 0, 255), 2)
                    
                    # Draw dwell time
                    label = f"ID: {person_id} - {current_duration:.1f}s"
                    cv2.putText(output_frame, label, (int(x) + 10, int(y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw completed dwells if requested
        if not show_active_only:
            # Get recent completed dwells
            recent_time = time.time() - 30  # Show dwells from last 30 seconds
            for zone_dwells in self.completed_dwells.values():
                for dwell in zone_dwells:
                    if dwell['end_time'] >= recent_time:
                        x, y = dwell['avg_position']
                        duration = dwell['duration']
                        
                        # Draw smaller indicator for completed dwells
                        radius = min(30, int(3 + duration / 3))
                        cv2.circle(output_frame, (int(x), int(y)), radius, (0, 255, 0), 1)
        
        return output_frame 