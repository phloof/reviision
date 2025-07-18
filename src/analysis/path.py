"""
Path Analysis Module for Retail Analytics System

This module provides comprehensive customer path tracking, analysis, and storage
to understand customer movement patterns and optimize store layout.
"""

import time
import logging
import numpy as np
import cv2
import math
import hashlib
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import json

try:
    from sklearn.cluster import DBSCAN
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class PathAnalyzer:
    """
    Advanced path analyzer for tracking customer movement patterns
    
    This class provides comprehensive path tracking, pattern recognition,
    spatial clustering, and performance analytics for customer behavior analysis.
    """
    
    def __init__(self, config, database=None):
        """
        Initialize the path analyzer with configuration
        
        Args:
            config (dict): Configuration dictionary
            database: Database connection for persistent storage
        """
        self.config = config
        self.database = database
        
        # Path tracking parameters
        self.min_path_length = config.get('min_path_length', 5)
        self.max_path_gap = config.get('max_path_gap', 2.0)  # seconds
        self.simplification_tolerance = config.get('simplification_tolerance', 5.0)  # pixels
        self.speed_calculation_window = config.get('speed_window', 3)  # frames
        self.min_movement_threshold = config.get('min_movement_threshold', 10)  # pixels
        
        # Pattern recognition settings
        self.pattern_detection = config.get('enable_pattern_detection', True)
        self.clustering_enabled = config.get('enable_clustering', True)
        self.min_cluster_size = config.get('min_cluster_size', 3)
        self.cluster_eps = config.get('cluster_eps', 50.0)  # DBSCAN epsilon
        
        # Storage settings
        self.storage_interval = config.get('storage_interval', 1.0)  # seconds
        self.path_retention_days = config.get('retention_days', 30)
        self.enable_db_storage = config.get('enable_db_storage', True)
        self.batch_size = config.get('batch_size', 100)
        
        # Zone detection settings
        self.enable_zone_detection = config.get('enable_zone_detection', True)
        self.zones = config.get('zones', {})
        
        # Active path tracking
        self.active_paths = {}  # person_id -> path_data
        self.completed_paths = []  # List of completed path segments
        self.pending_storage = []  # Buffer for batch storage
        self.last_storage_time = time.time()
        
        # Performance tracking
        self.path_stats = {
            'total_paths': 0,
            'active_paths': 0,
            'avg_path_length': 0,
            'avg_speed': 0,
            'common_patterns': 0
        }
        
        logger.info(f"PathAnalyzer initialized with {len(self.zones)} zones, "
                   f"clustering: {self.clustering_enabled}, "
                   f"pattern detection: {self.pattern_detection}")
    
    def update_paths(self, tracked_people, timestamp=None):
        """
        Update path data for all currently tracked people
        
        Args:
            tracked_people (list): List of tracked person objects
            timestamp (float, optional): Current timestamp
            
        Returns:
            dict: Updated path information
        """
        if timestamp is None:
            timestamp = time.time()
        
        current_person_ids = set()
        
        # Process each tracked person
        for person in tracked_people:
            person_id = person.get('id')
            if not person_id:
                continue
                
            current_person_ids.add(person_id)
            
            # Extract position data
            bbox = person.get('bbox', [0, 0, 0, 0])
            confidence = person.get('confidence', 0.5)
            
            # Calculate center position
            if len(bbox) >= 4:
                center_x = (bbox[0] + bbox[2]) / 2 if len(bbox) == 4 else bbox[0] + bbox[2] / 2
                center_y = (bbox[1] + bbox[3]) / 2 if len(bbox) == 4 else bbox[1] + bbox[3] / 2
            else:
                center_x, center_y = person.get('center', (0, 0))
            
            # Update or create path for this person
            self._update_person_path(person_id, center_x, center_y, timestamp, confidence, person)
        
        # End paths for people who are no longer tracked
        inactive_persons = set(self.active_paths.keys()) - current_person_ids
        for person_id in inactive_persons:
            self._end_person_path(person_id, timestamp)
        
        # Store pending data if interval reached
        if timestamp - self.last_storage_time >= self.storage_interval:
            self._store_pending_data()
            self.last_storage_time = timestamp
        
        # Update statistics
        self._update_path_stats()
        
        return {
            'active_paths': len(self.active_paths),
            'completed_paths': len(self.completed_paths),
            'pending_storage': len(self.pending_storage),
            'path_stats': self.path_stats.copy()
        }
    
    def _update_person_path(self, person_id, x, y, timestamp, confidence, person_data):
        """
        Update path data for a specific person
        
        Args:
            person_id: Person track ID
            x, y (float): Current position
            timestamp (float): Current timestamp
            confidence (float): Detection confidence
            person_data (dict): Additional person information
        """
        position = (float(x), float(y))
        
        # Initialize new path if needed
        if person_id not in self.active_paths:
            session_id = f"{person_id}_{int(timestamp)}"
            self.active_paths[person_id] = {
                'session_id': session_id,
                'person_id': person_id,
                'positions': deque(maxlen=1000),  # Limit memory usage
                'timestamps': deque(maxlen=1000),
                'confidences': deque(maxlen=1000),
                'start_time': timestamp,
                'last_update': timestamp,
                'total_distance': 0.0,
                'speeds': deque(maxlen=self.speed_calculation_window),
                'sequence_number': 0,
                'zone_visits': [],
                'demographics': person_data.get('demographics', {})
            }
            logger.debug(f"Started new path for person {person_id}")
        
        path_data = self.active_paths[person_id]
        
        # Check for significant gap in tracking
        time_gap = timestamp - path_data['last_update']
        if time_gap > self.max_path_gap:
            # End current path and start new one
            self._end_person_path(person_id, path_data['last_update'])
            # Recursively call to start new path
            self._update_person_path(person_id, x, y, timestamp, confidence, person_data)
            return
        
        # Calculate movement from last position
        if path_data['positions']:
            last_pos = path_data['positions'][-1]
            distance = self._calculate_distance(last_pos, position)
            
            # Only add point if significant movement occurred
            if distance >= self.min_movement_threshold:
                # Calculate speed
                time_diff = timestamp - path_data['timestamps'][-1]
                if time_diff > 0:
                    speed = distance / time_diff  # pixels per second
                    path_data['speeds'].append(speed)
                
                # Add position to path
                path_data['positions'].append(position)
                path_data['timestamps'].append(timestamp)
                path_data['confidences'].append(confidence)
                path_data['total_distance'] += distance
                path_data['sequence_number'] += 1
                
                # Detect zone if enabled
                if self.enable_zone_detection:
                    zone = self._detect_zone(x, y)
                    if zone and (not path_data['zone_visits'] or 
                               path_data['zone_visits'][-1] != zone):
                        path_data['zone_visits'].append(zone)
                
                # Add to storage buffer
                self._add_to_storage_buffer(path_data, position, timestamp, confidence)
        else:
            # First position
            path_data['positions'].append(position)
            path_data['timestamps'].append(timestamp)
            path_data['confidences'].append(confidence)
            path_data['sequence_number'] = 1
            
            # Detect initial zone
            if self.enable_zone_detection:
                zone = self._detect_zone(x, y)
                if zone:
                    path_data['zone_visits'].append(zone)
        
        path_data['last_update'] = timestamp
    
    def _end_person_path(self, person_id, timestamp):
        """
        End and process a person's path
        
        Args:
            person_id: Person track ID
            timestamp (float): End timestamp
        """
        if person_id not in self.active_paths:
            return
        
        path_data = self.active_paths[person_id]
        
        # Only process paths that meet minimum length requirement
        if len(path_data['positions']) >= self.min_path_length:
            # Calculate final path metrics
            path_metrics = self._calculate_path_metrics(path_data)
            
            # Create completed path record
            completed_path = {
                'person_id': person_id,
                'session_id': path_data['session_id'],
                'start_time': path_data['start_time'],
                'end_time': timestamp,
                'duration': timestamp - path_data['start_time'],
                'positions': list(path_data['positions']),
                'total_distance': path_data['total_distance'],
                'avg_speed': path_metrics['avg_speed'],
                'max_speed': path_metrics['max_speed'],
                'path_complexity': path_metrics['complexity'],
                'zone_visits': path_data['zone_visits'],
                'demographics': path_data['demographics'],
                'point_count': len(path_data['positions'])
            }
            
            self.completed_paths.append(completed_path)
            
            # Store simplified path segment
            if self.enable_db_storage and self.database:
                self._store_path_segment(completed_path)
            
            self.path_stats['total_paths'] += 1
            logger.debug(f"Completed path for person {person_id}: "
                        f"{len(path_data['positions'])} points, "
                        f"{path_data['total_distance']:.1f}px distance")
        
        # Remove from active paths
        del self.active_paths[person_id]
    
    def _add_to_storage_buffer(self, path_data, position, timestamp, confidence):
        """
        Add point to storage buffer for batch processing
        
        Args:
            path_data (dict): Current path data
            position (tuple): Current position
            timestamp (float): Current timestamp
            confidence (float): Detection confidence
        """
        if not self.enable_db_storage:
            return
        
        # Calculate current speed and direction
        current_speed = 0.0
        direction_angle = None
        movement_type = 'stationary'
        
        if len(path_data['speeds']) > 0:
            current_speed = path_data['speeds'][-1]
            
            # Classify movement type based on speed
            if current_speed < 5:  # pixels/second
                movement_type = 'stationary'
            elif current_speed < 20:
                movement_type = 'walking'
            else:
                movement_type = 'running'
        
        # Calculate direction if we have previous positions
        if len(path_data['positions']) >= 2:
            prev_pos = path_data['positions'][-2]
            direction_angle = self._calculate_direction(prev_pos, position)
        
        # Detect current zone
        zone_name = None
        if self.enable_zone_detection:
            zone_name = self._detect_zone(position[0], position[1])
        
        # Calculate path complexity (simplified)
        path_complexity = self._calculate_local_complexity(path_data['positions'])
        
        # Create storage point
        storage_point = {
            'person_id': path_data['person_id'],
            'session_id': path_data['session_id'],
            'sequence_number': path_data['sequence_number'],
            'x_position': int(position[0]),
            'y_position': int(position[1]),
            'timestamp': datetime.fromtimestamp(timestamp),
            'confidence': confidence,
            'movement_type': movement_type,
            'speed': current_speed,
            'direction_angle': direction_angle,
            'zone_name': zone_name,
            'path_complexity': path_complexity
        }
        
        self.pending_storage.append(storage_point)
    
    def _store_pending_data(self):
        """Store pending path points to database in batch"""
        if not self.pending_storage or not self.database:
            return
        
        try:
            # Store in batches to avoid overwhelming database
            batch_size = min(self.batch_size, len(self.pending_storage))
            batch = self.pending_storage[:batch_size]
            
            success = self.database.store_path_points(batch)
            if success:
                # Remove stored points from buffer
                self.pending_storage = self.pending_storage[batch_size:]
                logger.debug(f"Stored {batch_size} path points to database")
            else:
                logger.warning("Failed to store path points to database")
                
        except Exception as e:
            logger.error(f"Error storing path points: {e}")
    
    def _store_path_segment(self, completed_path):
        """
        Store simplified path segment to database
        
        Args:
            completed_path (dict): Completed path data
        """
        if not self.database:
            return
        
        try:
            positions = completed_path['positions']
            if len(positions) < 2:
                return
            
            # Use first and last positions for segment
            start_pos = positions[0]
            end_pos = positions[-1]
            
            # Determine segment type
            segment_type = self._classify_segment_type(positions)
            
            # Store segment
            self.database.store_path_segment(
                person_id=completed_path['person_id'],
                session_id=completed_path['session_id'],
                start_time=datetime.fromtimestamp(completed_path['start_time']),
                end_time=datetime.fromtimestamp(completed_path['end_time']),
                start_x=int(start_pos[0]),
                start_y=int(start_pos[1]),
                end_x=int(end_pos[0]),
                end_y=int(end_pos[1]),
                total_distance=completed_path['total_distance'],
                avg_speed=completed_path['avg_speed'],
                segment_type=segment_type,
                point_count=completed_path['point_count']
            )
            
        except Exception as e:
            logger.error(f"Error storing path segment: {e}")
    
    def _calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    
    def _calculate_direction(self, pos1, pos2):
        """Calculate direction angle between two positions in degrees"""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        angle = math.atan2(dy, dx) * 180 / math.pi
        return angle if angle >= 0 else angle + 360
    
    def _calculate_path_metrics(self, path_data):
        """
        Calculate comprehensive path metrics
        
        Args:
            path_data (dict): Path data
            
        Returns:
            dict: Calculated metrics
        """
        speeds = list(path_data['speeds'])
        positions = list(path_data['positions'])
        
        metrics = {
            'avg_speed': np.mean(speeds) if speeds else 0.0,
            'max_speed': max(speeds) if speeds else 0.0,
            'min_speed': min(speeds) if speeds else 0.0,
            'speed_variance': np.var(speeds) if speeds else 0.0,
            'complexity': self._calculate_path_complexity(positions),
            'linearity': self._calculate_path_linearity(positions),
            'direction_changes': self._count_direction_changes(positions)
        }
        
        return metrics
    
    def _calculate_path_complexity(self, positions):
        """
        Calculate path complexity using total distance vs straight-line distance ratio
        
        Args:
            positions (list): List of (x, y) positions
            
        Returns:
            float: Complexity score (1.0 = straight line, higher = more complex)
        """
        if len(positions) < 2:
            return 0.0
        
        # Calculate total path distance
        total_distance = 0.0
        for i in range(1, len(positions)):
            total_distance += self._calculate_distance(positions[i-1], positions[i])
        
        # Calculate straight-line distance
        straight_distance = self._calculate_distance(positions[0], positions[-1])
        
        if straight_distance == 0:
            return 0.0
        
        complexity = total_distance / straight_distance
        return complexity
    
    def _calculate_local_complexity(self, positions, window_size=5):
        """
        Calculate local path complexity for current position
        
        Args:
            positions (deque): Recent positions
            window_size (int): Number of recent positions to consider
            
        Returns:
            float: Local complexity score
        """
        if len(positions) < 3:
            return 0.0
        
        # Use last window_size positions
        recent_positions = list(positions)[-window_size:]
        return self._calculate_path_complexity(recent_positions)
    
    def _calculate_path_linearity(self, positions):
        """
        Calculate how linear the path is using least squares fitting
        
        Args:
            positions (list): List of (x, y) positions
            
        Returns:
            float: Linearity score (0-1, 1 = perfectly linear)
        """
        if len(positions) < 3:
            return 1.0
        
        try:
            # Extract x and y coordinates
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            # Calculate correlation coefficient
            if SCIPY_AVAILABLE:
                correlation, _ = pearsonr(x_coords, y_coords)
                return abs(correlation) if not np.isnan(correlation) else 0.5
            else:
                # Simple linear fit without scipy
                x_mean = np.mean(x_coords)
                y_mean = np.mean(y_coords)
                
                numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_coords, y_coords))
                denominator = math.sqrt(
                    sum((x - x_mean)**2 for x in x_coords) * 
                    sum((y - y_mean)**2 for y in y_coords)
                )
                
                return abs(numerator / denominator) if denominator > 0 else 0.5
                
        except Exception as e:
            logger.debug(f"Error calculating linearity: {e}")
            return 0.5
    
    def _count_direction_changes(self, positions, threshold=45):
        """
        Count significant direction changes in path
        
        Args:
            positions (list): List of (x, y) positions
            threshold (float): Minimum angle change to count as direction change
            
        Returns:
            int: Number of direction changes
        """
        if len(positions) < 3:
            return 0
        
        changes = 0
        prev_angle = None
        
        for i in range(2, len(positions)):
            angle = self._calculate_direction(positions[i-1], positions[i])
            
            if prev_angle is not None:
                angle_diff = abs(angle - prev_angle)
                # Handle angle wraparound
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                if angle_diff >= threshold:
                    changes += 1
            
            prev_angle = angle
        
        return changes
    
    def _classify_segment_type(self, positions):
        """
        Classify the type of path segment
        
        Args:
            positions (list): List of positions
            
        Returns:
            str: Segment type ('linear', 'curved', 'zigzag', 'stationary')
        """
        if len(positions) < 3:
            return 'linear'
        
        complexity = self._calculate_path_complexity(positions)
        linearity = self._calculate_path_linearity(positions)
        direction_changes = self._count_direction_changes(positions)
        
        # Classify based on metrics
        if complexity < 1.1 and linearity > 0.9:
            return 'linear'
        elif direction_changes > len(positions) * 0.3:
            return 'zigzag'
        elif complexity > 1.5:
            return 'curved'
        else:
            return 'linear'
    
    def _detect_zone(self, x, y):
        """
        Detect which zone the position falls into
        
        Args:
            x, y (float): Position coordinates
            
        Returns:
            str: Zone name or None
        """
        for zone_name, zone_data in self.zones.items():
            if 'bounds' in zone_data:
                x1, y1, x2, y2 = zone_data['bounds']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return zone_name
        return None
    
    def _update_path_stats(self):
        """Update path statistics"""
        self.path_stats['active_paths'] = len(self.active_paths)
        
        if self.completed_paths:
            distances = [path['total_distance'] for path in self.completed_paths]
            speeds = [path['avg_speed'] for path in self.completed_paths]
            
            self.path_stats['avg_path_length'] = np.mean(distances)
            self.path_stats['avg_speed'] = np.mean(speeds)
        
        # Update common patterns count
        if self.clustering_enabled:
            self.path_stats['common_patterns'] = len(self.get_common_patterns())
    
    def get_common_patterns(self, min_frequency=3):
        """
        Get common path patterns using clustering
        
        Args:
            min_frequency (int): Minimum frequency to consider a pattern common
            
        Returns:
            list: List of common patterns
        """
        if not SCIPY_AVAILABLE or not self.clustering_enabled:
            return []
        
        try:
            # Get path data for clustering
            if len(self.completed_paths) < min_frequency:
                return []
            
            # Extract features for clustering (start/end positions)
            features = []
            path_refs = []
            
            for path in self.completed_paths:
                positions = path['positions']
                if len(positions) >= 2:
                    start_pos = positions[0]
                    end_pos = positions[-1]
                    
                    # Create feature vector
                    feature = [start_pos[0], start_pos[1], end_pos[0], end_pos[1]]
                    features.append(feature)
                    path_refs.append(path)
            
            if len(features) < min_frequency:
                return []
            
            # Perform DBSCAN clustering
            clustering = DBSCAN(eps=self.cluster_eps, min_samples=min_frequency)
            cluster_labels = clustering.fit_predict(features)
            
            # Group paths by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                if label != -1:  # Ignore noise points
                    clusters[label].append(path_refs[i])
            
            # Create common patterns
            common_patterns = []
            for cluster_id, paths in clusters.items():
                if len(paths) >= min_frequency:
                    # Calculate representative path
                    avg_features = np.mean([features[i] for i, label in enumerate(cluster_labels) 
                                          if label == cluster_id], axis=0)
                    
                    pattern = {
                        'pattern_id': f"pattern_{cluster_id}",
                        'frequency': len(paths),
                        'avg_start': (avg_features[0], avg_features[1]),
                        'avg_end': (avg_features[2], avg_features[3]),
                        'avg_distance': np.mean([p['total_distance'] for p in paths]),
                        'avg_speed': np.mean([p['avg_speed'] for p in paths]),
                        'avg_duration': np.mean([p['duration'] for p in paths]),
                        'paths': paths
                    }
                    common_patterns.append(pattern)
            
            # Sort by frequency
            common_patterns.sort(key=lambda x: x['frequency'], reverse=True)
            
            return common_patterns
            
        except Exception as e:
            logger.error(f"Error finding common patterns: {e}")
            return []
    
    def get_path_data_for_person(self, person_id, hours=24):
        """
        Get path data for a specific person
        
        Args:
            person_id: Person track ID
            hours (int): Hours to look back
            
        Returns:
            dict: Path data for person
        """
        if not self.database:
            # Return active path data if available
            if person_id in self.active_paths:
                path_data = self.active_paths[person_id]
                return {
                    'person_id': person_id,
                    'session_id': path_data['session_id'],
                    'positions': list(path_data['positions']),
                    'timestamps': list(path_data['timestamps']),
                    'total_distance': path_data['total_distance'],
                    'zone_visits': path_data['zone_visits']
                }
            return {}
        
        try:
            # Get from database
            paths = self.database.get_person_paths(person_id, hours=hours)
            
            # Group by session
            sessions = defaultdict(list)
            for point in paths:
                sessions[point['session_id']].append(point)
            
            return {
                'person_id': person_id,
                'sessions': dict(sessions),
                'total_sessions': len(sessions),
                'total_points': len(paths)
            }
            
        except Exception as e:
            logger.error(f"Error getting path data for person {person_id}: {e}")
            return {}
    
    def get_analytics_summary(self):
        """
        Get comprehensive analytics summary
        
        Returns:
            dict: Analytics summary
        """
        summary = {
            'path_stats': self.path_stats.copy(),
            'active_tracking': {
                'active_paths': len(self.active_paths),
                'pending_storage': len(self.pending_storage),
                'completed_paths': len(self.completed_paths)
            },
            'configuration': {
                'min_path_length': self.min_path_length,
                'storage_interval': self.storage_interval,
                'clustering_enabled': self.clustering_enabled,
                'pattern_detection': self.pattern_detection,
                'zones_configured': len(self.zones)
            }
        }
        
        # Add common patterns if available
        if self.clustering_enabled:
            patterns = self.get_common_patterns()
            summary['common_patterns'] = {
                'count': len(patterns),
                'top_patterns': patterns[:5]  # Top 5 patterns
            }
        
        return summary
    
    def cleanup_old_data(self):
        """Clean up old path data based on retention policy"""
        if not self.database:
            return
        
        try:
            deleted_count = self.database.cleanup_old_paths(self.path_retention_days)
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old path records")
                
        except Exception as e:
            logger.error(f"Error cleaning up old path data: {e}")
    
    def force_storage(self):
        """Force immediate storage of all pending data"""
        if self.pending_storage:
            logger.info(f"Force storing {len(self.pending_storage)} pending path points")
            self._store_pending_data()
    
    def get_zone_statistics(self, hours=24):
        """
        Get zone visit statistics
        
        Args:
            hours (int): Hours to analyze
            
        Returns:
            dict: Zone statistics
        """
        zone_stats = defaultdict(lambda: {
            'visits': 0,
            'unique_visitors': set(),
            'avg_dwell_time': 0,
            'total_time': 0
        })
        
        # Analyze completed paths
        cutoff_time = time.time() - (hours * 3600)
        
        for path in self.completed_paths:
            if path['start_time'] >= cutoff_time:
                for zone in path['zone_visits']:
                    zone_stats[zone]['visits'] += 1
                    zone_stats[zone]['unique_visitors'].add(path['person_id'])
        
        # Convert sets to counts and calculate averages
        result = {}
        for zone, stats in zone_stats.items():
            result[zone] = {
                'visits': stats['visits'],
                'unique_visitors': len(stats['unique_visitors']),
                'popularity_score': stats['visits'] / max(1, len(stats['unique_visitors']))
            }
        
        return result 