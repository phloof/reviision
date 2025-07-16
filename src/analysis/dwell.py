"""
Dwell Time Analyzer class for Retail Analytics System
"""

import time
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class DwellTimeAnalyzer:
    """
    Dwell time analyzer that measures how long customers stay in the monitored area
    
    This class tracks the time customers spend in the general area and provides
    analytics on dwell times without zone-based segmentation.
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
        
        # Storage for dwell data
        self.current_dwells = {}  # person_id -> {start_time, last_update, positions}
        self.completed_dwells = []  # List of completed dwell records
        self.person_history = defaultdict(list)  # person_id -> list of all dwell records
        
        logger.info("Dwell time analyzer initialized (zone-free mode)")
    
    def update(self, tracks, frame_shape=None):
        """
        Update dwell time data with new tracking information
        
        Args:
            tracks (list): List of person tracks with IDs and bounding boxes
            frame_shape (tuple, optional): Frame dimensions (height, width)
            
        Returns:
            list: Updated dwell time data
        """
        if not tracks:
            return self.completed_dwells
        
        current_time = time.time()
        
        # Update active tracks
        active_person_ids = set()
        for track in tracks:
            person_id = track['id']
            bbox = track['bbox']
            active_person_ids.add(person_id)
            
            # Calculate center point of the person
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            # Update dwell time for this person
            self._update_dwell(person_id, current_time, (center_x, center_y))
        
        # Check for inactive dwells and end them
        all_person_ids = set(self.current_dwells.keys())
        inactive_person_ids = all_person_ids - active_person_ids
        
        for person_id in inactive_person_ids:
            if person_id in self.current_dwells:
                inactive_time = current_time - self.current_dwells[person_id]['last_update']
                if inactive_time > self.max_inactive_time:
                    self._end_dwell(person_id, current_time)
        
        return self.completed_dwells
    
    def _update_dwell(self, person_id, timestamp, position):
        """
        Update dwell time information for a person
        
        Args:
            person_id (int): Person track ID
            timestamp (float): Current timestamp
            position (tuple): (x, y) position of the person
        """
        # If this is a new dwell, create it
        if person_id not in self.current_dwells:
            self.current_dwells[person_id] = {
                'start_time': timestamp,
                'last_update': timestamp,
                'positions': [position]
            }
        else:
            # Update existing dwell
            self.current_dwells[person_id]['last_update'] = timestamp
            self.current_dwells[person_id]['positions'].append(position)
            
            # Limit the number of positions stored to prevent memory issues
            max_positions = 100
            if len(self.current_dwells[person_id]['positions']) > max_positions:
                self.current_dwells[person_id]['positions'] = self.current_dwells[person_id]['positions'][-max_positions:]
    
    def _end_dwell(self, person_id, timestamp):
        """
        End a dwell period and record the data
        
        Args:
            person_id (int): Person track ID
            timestamp (float): Current timestamp
        """
        if person_id in self.current_dwells:
            dwell_data = self.current_dwells[person_id]
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
                    'start_time': start_time,
                    'end_time': timestamp,
                    'duration': duration,
                    'avg_position': (avg_x, avg_y)
                }
                
                self.completed_dwells.append(dwell_record)
                self.person_history[person_id].append(dwell_record)
                
                logger.debug(f"Person {person_id} dwelled for {duration:.2f} seconds")
            
            # Remove from current dwells
            del self.current_dwells[person_id]
    
    def get_dwell_stats(self):
        """
        Get overall dwell time statistics
        
        Returns:
            dict: Dwell time statistics
        """
        if not self.completed_dwells:
            return {
                'total_dwells': 0,
                'unique_visitors': 0,
                'avg_duration': 0,
                'min_duration': 0,
                'max_duration': 0,
                'total_dwell_time': 0
            }
        
        durations = [d['duration'] for d in self.completed_dwells]
        unique_visitors = len(set(d['person_id'] for d in self.completed_dwells))
        
        stats = {
            'total_dwells': len(self.completed_dwells),
            'unique_visitors': unique_visitors,
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'total_dwell_time': sum(durations)
        }
        
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
    
    def get_active_dwellers(self):
        """
        Get list of people currently dwelling
        
        Returns:
            list: List of person IDs currently dwelling
        """
        return list(self.current_dwells.keys())
    
    def get_dwell_heatmap_data(self):
        """
        Get data for generating a dwell time heatmap
        
        Returns:
            list: List of dwell points with intensity
        """
        heatmap_data = []
        
        # Include completed dwells
        for dwell in self.completed_dwells:
            heatmap_data.append({
                'position': dwell['avg_position'],
                'intensity': dwell['duration'],
                'active': False
            })
        
        # Include active dwells
        for person_id, dwell in self.current_dwells.items():
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
    
    def visualize_dwells(self, frame, show_active_only=False):
        """
        Visualize dwell times on the frame
        
        Args:
            frame (numpy.ndarray): Current video frame
            show_active_only (bool): Whether to show only active dwells
            
        Returns:
            numpy.ndarray: Frame with dwell visualizations
        """
        import cv2
        
        # Create a copy to avoid modifying the original frame
        viz_frame = frame.copy()
        
        # Draw completed dwells (unless showing active only)
        if not show_active_only:
            for dwell in self.completed_dwells:
                x, y = dwell['avg_position']
                duration = dwell['duration']
                
                # Color intensity based on duration
                intensity = min(1.0, duration / 60.0)  # Normalize to 1 minute
                color = (0, int(255 * intensity), int(255 * (1 - intensity)))
                
                cv2.circle(viz_frame, (int(x), int(y)), 8, color, -1)
                cv2.putText(viz_frame, f"{duration:.1f}s", (int(x) + 10, int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw active dwells
        current_time = time.time()
        for person_id, dwell in self.current_dwells.items():
            if dwell['positions']:
                positions = dwell['positions']
                avg_x = sum(p[0] for p in positions) / len(positions)
                avg_y = sum(p[1] for p in positions) / len(positions)
                
                duration = current_time - dwell['start_time']
                if duration >= self.min_dwell_time:
                    # Active dwells in green
                    cv2.circle(viz_frame, (int(avg_x), int(avg_y)), 10, (0, 255, 0), -1)
                    cv2.putText(viz_frame, f"{duration:.1f}s", (int(avg_x) + 12, int(avg_y)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return viz_frame
    
    def cleanup_old_data(self, max_age_seconds=3600):
        """
        Clean up old dwell data to prevent memory issues
        
        Args:
            max_age_seconds (int): Maximum age of data to keep in seconds
        """
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds
        
        # Clean up completed dwells
        self.completed_dwells = [
            dwell for dwell in self.completed_dwells
            if dwell['end_time'] > cutoff_time
        ]
        
        # Clean up person history
        for person_id in list(self.person_history.keys()):
            self.person_history[person_id] = [
                dwell for dwell in self.person_history[person_id]
                if dwell['end_time'] > cutoff_time
            ]
            
            # Remove empty histories
            if not self.person_history[person_id]:
                del self.person_history[person_id] 