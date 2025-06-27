"""
Path Analyzer module for Retail Analytics System
"""

import time
import logging
import numpy as np
import cv2
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

class PathAnalyzer:
    """
    Path analyzer that tracks and analyzes customer movement paths through the store
    
    This class maintains a history of customer paths, calculates common routes,
    analyzes zone transitions, and provides insights into traffic flows.
    """
    
    def __init__(self, config):
        """
        Initialize the path analyzer with the provided configuration
        
        Args:
            config (dict): Configuration dictionary containing path analysis parameters
        """
        self.config = config
        
        # Path tracking parameters
        self.max_path_history = config.get('max_path_history', 100)  # Maximum paths to store
        self.max_path_time = config.get('max_path_time', 3600)  # Maximum time to store a path (seconds)
        self.min_path_points = config.get('min_path_points', 5)  # Minimum points for a valid path
        self.path_smoothing = config.get('path_smoothing', True)  # Whether to smooth paths
        self.smoothing_window = config.get('smoothing_window', 3)  # Points to average for smoothing
        
        # Path storage
        self.active_paths = {}  # person_id -> list of path points [(x, y, timestamp, zone_id), ...]
        self.completed_paths = []  # List of completed path dictionaries
        self.last_cleanup_time = time.time()
        self.cleanup_interval = config.get('cleanup_interval', 60)  # Seconds between cleanups
        
        # Zone transition tracking
        self.zone_transitions = defaultdict(int)  # (source_zone, target_zone) -> count
        self.zone_dwell_times = defaultdict(list)  # zone_id -> list of dwell times
        
        # Heatmap for path density visualization
        self.frame_shape = config.get('frame_shape', (720, 1280))  # (height, width)
        self.heatmap_resolution = config.get('heatmap_resolution', 0.25)  # Downsample factor
        self.path_density = None
        self._initialize_path_density()
        
        # Common path clustering
        self.cluster_paths = config.get('cluster_paths', True)
        self.min_cluster_size = config.get('min_cluster_size', 3)
        self.common_paths = []  # List of common paths found through clustering
        
        logger.info("Path analyzer initialized")
    
    def _initialize_path_density(self):
        """Initialize the path density heatmap"""
        h, w = self.frame_shape
        h_resized = int(h * self.heatmap_resolution)
        w_resized = int(w * self.heatmap_resolution)
        self.path_density = np.zeros((h_resized, w_resized), dtype=np.float32)
        self.path_density_shape = (h_resized, w_resized)
        logger.debug(f"Path density heatmap initialized with shape {self.path_density_shape}")
    
    def update_position(self, person_id, position, timestamp, zone_id=None):
        """
        Update the position of a person in their path
        
        Args:
            person_id (str): Unique identifier for the person
            position (tuple): (x, y) coordinates
            timestamp (float): Current timestamp
            zone_id (str, optional): Current zone identifier
        """
        if person_id not in self.active_paths:
            self.active_paths[person_id] = []
        
        # Check if person already has points, if so check zone transition
        if self.active_paths[person_id] and zone_id is not None:
            last_zone = self.active_paths[person_id][-1][3]
            if last_zone != zone_id and last_zone is not None:
                # Zone transition occurred
                self.zone_transitions[(last_zone, zone_id)] += 1
                
                # Calculate dwell time in previous zone
                zone_entry_time = None
                for i in range(len(self.active_paths[person_id])-1, -1, -1):
                    if self.active_paths[person_id][i][3] != last_zone:
                        zone_entry_time = self.active_paths[person_id][i+1][2]
                        break
                
                if zone_entry_time is None and len(self.active_paths[person_id]) > 0:
                    zone_entry_time = self.active_paths[person_id][0][2]
                
                if zone_entry_time is not None:
                    dwell_time = timestamp - zone_entry_time
                    self.zone_dwell_times[last_zone].append(dwell_time)
        
        # Add new position
        x, y = position
        self.active_paths[person_id].append((x, y, timestamp, zone_id))
        
        # Update path density map
        if 0 <= x < self.frame_shape[1] and 0 <= y < self.frame_shape[0]:
            x_heatmap = int(x * self.heatmap_resolution)
            y_heatmap = int(y * self.heatmap_resolution)
            
            if 0 <= x_heatmap < self.path_density_shape[1] and 0 <= y_heatmap < self.path_density_shape[0]:
                self.path_density[y_heatmap, x_heatmap] += 1.0
        
        # Periodically cleanup paths
        current_time = time.time()
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            self._cleanup_paths(current_time)
            self.last_cleanup_time = current_time
    
    def mark_exit(self, person_id):
        """
        Mark a person as having exited the scene, completing their path
        
        Args:
            person_id (str): Unique identifier for the person
        
        Returns:
            bool: True if the path was completed, False otherwise
        """
        if person_id not in self.active_paths:
            return False
        
        path = self.active_paths[person_id]
        
        # Only process paths with sufficient points
        if len(path) >= self.min_path_points:
            # Calculate path metrics
            timestamps = [p[2] for p in path]
            duration = timestamps[-1] - timestamps[0]
            
            # Calculate total distance using vectorized operations
            coordinates = np.array([(p[0], p[1]) for p in path])
            distances = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
            distance = np.sum(distances)
            
            # Get unique zones visited
            zones_visited = set(p[3] for p in path if p[3] is not None)
            
            # Apply path smoothing if enabled
            smoothed_path = self._smooth_path(path) if self.path_smoothing else path
            
            # Store completed path
            completed_path = {
                'person_id': person_id,
                'path': smoothed_path,
                'start_time': timestamps[0],
                'end_time': timestamps[-1],
                'duration': duration,
                'distance': distance,
                'zones_visited': list(zones_visited),
                'num_points': len(path)
            }
            
            self.completed_paths.append(completed_path)
            
            # Limit storage
            if len(self.completed_paths) > self.max_path_history:
                self.completed_paths = self.completed_paths[-self.max_path_history:]
        
        # Remove from active paths
        del self.active_paths[person_id]
        return True
    
    def _smooth_path(self, path):
        """
        Apply smoothing to a path to reduce noise
        
        Args:
            path (list): List of path points [(x, y, timestamp, zone_id), ...]
        
        Returns:
            list: Smoothed path
        """
        if len(path) < self.smoothing_window:
            return path
        
        smoothed_path = []
        half_window = self.smoothing_window // 2
        
        for i in range(len(path)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(path), i + half_window + 1)
            window = path[start_idx:end_idx]
            
            avg_x = sum(p[0] for p in window) / len(window)
            avg_y = sum(p[1] for p in window) / len(window)
            
            # Keep original timestamp and zone_id
            smoothed_path.append((avg_x, avg_y, path[i][2], path[i][3]))
        
        return smoothed_path
    
    def _cleanup_paths(self, current_time):
        """
        Clean up old paths and perform maintenance tasks
        
        Args:
            current_time (float): Current timestamp
        """
        # Check for timed-out active paths
        timed_out_ids = []
        for person_id, path in self.active_paths.items():
            if not path:
                timed_out_ids.append(person_id)
                continue
                
            last_update_time = path[-1][2]
            if current_time - last_update_time > self.config.get('inactive_timeout', 10):
                # Person has been inactive, consider them exited
                self.mark_exit(person_id)
                timed_out_ids.append(person_id)
        
        # Remove timed out paths
        for person_id in timed_out_ids:
            if person_id in self.active_paths:
                del self.active_paths[person_id]
        
        # Clean up old completed paths
        if self.max_path_time > 0 and self.completed_paths:
            valid_paths = []
            for path_data in self.completed_paths:
                if current_time - path_data['end_time'] <= self.max_path_time:
                    valid_paths.append(path_data)
            
            self.completed_paths = valid_paths
        
        # Update common paths if enabled
        if self.cluster_paths and len(self.completed_paths) >= self.min_cluster_size:
            self._update_common_paths()
    
    def _update_common_paths(self):
        """
        Update common paths by clustering completed paths
        """
        if len(self.completed_paths) < self.min_cluster_size:
            return
        
        # Simple path clustering by comparing zone sequences
        zone_sequences = {}
        
        for path_data in self.completed_paths:
            # Extract zone sequence
            zone_seq = []
            for point in path_data['path']:
                zone = point[3]
                if zone is not None and (not zone_seq or zone_seq[-1] != zone):
                    zone_seq.append(zone)
            
            if not zone_seq:
                continue
                
            # Convert to string for hashing
            zone_key = '->'.join(str(z) for z in zone_seq)
            if zone_key not in zone_sequences:
                zone_sequences[zone_key] = {
                    'sequence': zone_seq,
                    'count': 0,
                    'total_duration': 0,
                    'paths': []
                }
            
            zone_sequences[zone_key]['count'] += 1
            zone_sequences[zone_key]['total_duration'] += path_data['duration']
            zone_sequences[zone_key]['paths'].append(path_data)
        
        # Sort by count
        common_paths = sorted(
            [v for k, v in zone_sequences.items() if v['count'] >= self.min_cluster_size],
            key=lambda x: x['count'],
            reverse=True
        )
        
        # Calculate average metrics for each common path
        for path_group in common_paths:
            path_group['avg_duration'] = path_group['total_duration'] / path_group['count']
            path_group['avg_distance'] = sum(p['distance'] for p in path_group['paths']) / path_group['count']
        
        self.common_paths = common_paths[:10]  # Keep top 10 common paths
    
    def get_path_density_heatmap(self, normalize=True, blur_size=15):
        """
        Get the path density heatmap
        
        Args:
            normalize (bool): Whether to normalize the heatmap (0-255)
            blur_size (int): Size of Gaussian blur kernel to apply
        
        Returns:
            numpy.ndarray: Path density heatmap
        """
        if self.path_density is None:
            self._initialize_path_density()
            return np.zeros(self.path_density_shape, dtype=np.uint8)
        
        # Create a copy of the density map
        heatmap = self.path_density.copy()
        
        # Apply Gaussian blur
        if blur_size > 0:
            heatmap = cv2.GaussianBlur(heatmap, (blur_size, blur_size), 0)
        
        # Normalize if requested
        if normalize:
            if np.max(heatmap) > 0:
                heatmap = np.clip(heatmap, 0, np.percentile(heatmap[heatmap > 0], 95))
                heatmap = (heatmap * 255.0 / np.max(heatmap)).astype(np.uint8)
            else:
                heatmap = np.zeros(self.path_density_shape, dtype=np.uint8)
        
        return heatmap
    
    def get_path_overlay(self, frame):
        """
        Get an overlay visualization of paths on the frame
        
        Args:
            frame (numpy.ndarray): Current video frame
        
        Returns:
            numpy.ndarray: Frame with path visualization overlaid
        """
        # Resize original frame
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw active paths
        for person_id, path in self.active_paths.items():
            if len(path) < 2:
                continue
                
            # Draw path line
            points = np.array([(int(p[0]), int(p[1])) for p in path], dtype=np.int32)
            cv2.polylines(overlay, [points], False, (0, 255, 0), 2)
            
            # Draw current position
            cv2.circle(overlay, (int(path[-1][0]), int(path[-1][1])), 5, (0, 0, 255), -1)
        
        # Apply heatmap as overlay
        heatmap = self.get_path_density_heatmap(normalize=True)
        
        # Resize heatmap to frame size
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to color heatmap
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # Create mask for non-zero areas of heatmap
        mask = (heatmap_resized > 10).astype(np.uint8)
        mask_expanded = np.expand_dims(mask, axis=2)
        mask_rgb = np.repeat(mask_expanded, 3, axis=2)
        
        # Alpha blend
        alpha = 0.5
        blended = np.where(
            mask_rgb > 0,
            cv2.addWeighted(overlay, 1-alpha, heatmap_color, alpha, 0),
            overlay
        )
        
        return blended
    
    def get_active_paths(self):
        """
        Get currently active paths
        
        Returns:
            dict: Dictionary of active paths by person ID
        """
        return self.active_paths
    
    def get_completed_paths(self, limit=None):
        """
        Get completed paths
        
        Args:
            limit (int, optional): Maximum number of paths to return
        
        Returns:
            list: List of completed path dictionaries
        """
        if limit is None or limit <= 0:
            return self.completed_paths
        
        return self.completed_paths[-limit:]
    
    def get_zone_transitions(self):
        """
        Get zone transition counts
        
        Returns:
            dict: Dictionary of transition counts by (source_zone, target_zone)
        """
        return dict(self.zone_transitions)
    
    def get_common_paths(self):
        """
        Get common path patterns
        
        Returns:
            list: List of common path dictionaries
        """
        return self.common_paths
    
    def get_zone_dwell_stats(self):
        """
        Get dwell time statistics for each zone
        
        Returns:
            dict: Dictionary of zone ID to dwell time statistics
        """
        stats = {}
        
        for zone_id, dwell_times in self.zone_dwell_times.items():
            if not dwell_times:
                continue
                
            stats[zone_id] = {
                'avg_dwell': np.mean(dwell_times),
                'min_dwell': np.min(dwell_times),
                'max_dwell': np.max(dwell_times),
                'median_dwell': np.median(dwell_times),
                'total_visits': len(dwell_times)
            }
        
        return stats
    
    def get_zone_popularity(self):
        """
        Get zone popularity based on visit frequency
        
        Returns:
            dict: Dictionary of zone ID to visit count
        """
        zone_visits = {}
        
        # Count zone appearances in active paths
        for path in self.active_paths.values():
            zones = set(point[3] for point in path if point[3] is not None)
            for zone in zones:
                zone_visits[zone] = zone_visits.get(zone, 0) + 1
        
        # Count zone appearances in completed paths
        for path_data in self.completed_paths:
            for zone in path_data['zones_visited']:
                if zone is not None:
                    zone_visits[zone] = zone_visits.get(zone, 0) + 1
        
        return zone_visits
    
    def get_traffic_flow_map(self, frame_shape=None):
        """
        Generate a traffic flow map showing movement vectors
        
        Args:
            frame_shape (tuple, optional): Shape of frame (height, width)
        
        Returns:
            numpy.ndarray: Traffic flow map
        """
        if frame_shape is None:
            frame_shape = self.frame_shape
        
        h, w = frame_shape[:2]
        flow_map = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw flow arrows for completed paths
        for path_data in self.completed_paths:
            path = path_data['path']
            if len(path) < 5:
                continue
            
            # Sample points along the path for cleaner visualization
            sample_rate = max(1, len(path) // 10)
            sampled_points = [path[i] for i in range(0, len(path), sample_rate)]
            if len(sampled_points) < 2:
                continue
            
            # Draw line for overall path
            points = np.array([(int(p[0]), int(p[1])) for p in path], dtype=np.int32)
            cv2.polylines(flow_map, [points], False, (100, 100, 100), 1)
            
            # Draw arrows for direction
            for i in range(1, len(sampled_points)):
                x1, y1 = int(sampled_points[i-1][0]), int(sampled_points[i-1][1])
                x2, y2 = int(sampled_points[i][0]), int(sampled_points[i][1])
                
                # Skip if out of bounds
                if not (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
                    continue
                
                # Draw arrow
                cv2.arrowedLine(flow_map, (x1, y1), (x2, y2), (0, 255, 0), 2, tipLength=0.3)
        
        return flow_map
    
    def get_zone_transition_graph(self):
        """
        Generate a graph representation of zone transitions
        
        Returns:
            tuple: (nodes, edges) where nodes is a dict of zone IDs to visit counts
                and edges is a list of (source, target, count) tuples
        """
        # Get zone popularity
        zone_popularity = self.get_zone_popularity()
        nodes = {zone: count for zone, count in zone_popularity.items() if zone is not None}
        
        # Get transitions
        edges = []
        for (source, target), count in self.zone_transitions.items():
            if source is not None and target is not None:
                edges.append((source, target, count))
        
        # Sort edges by count
        edges.sort(key=lambda x: x[2], reverse=True)
        
        return nodes, edges
    
    def export_path_data(self):
        """
        Export path data in a format suitable for external analysis
        
        Returns:
            dict: Dictionary containing exportable path data
        """
        export_data = {
            'active_paths': {
                pid: [(p[0], p[1], p[2], p[3]) for p in path]
                for pid, path in self.active_paths.items()
            },
            'completed_paths': self.completed_paths,
            'zone_transitions': {
                f"{src}->{tgt}": count
                for (src, tgt), count in self.zone_transitions.items()
            },
            'zone_dwell_stats': self.get_zone_dwell_stats(),
            'common_paths': self.common_paths,
            'timestamp': time.time()
        }
        
        return export_data
    
    def reset_path_density(self):
        """Reset the path density heatmap"""
        self._initialize_path_density()
        logger.debug("Path density heatmap reset")
    
    def visualize_common_paths(self, frame_shape=None):
        """
        Visualize common paths on an image
        
        Args:
            frame_shape (tuple, optional): Shape of frame (height, width)
        
        Returns:
            numpy.ndarray: Visualization of common paths
        """
        if frame_shape is None:
            frame_shape = self.frame_shape
        
        h, w = frame_shape[:2]
        viz = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Define colors for different paths
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 0),    # Maroon
            (0, 128, 0),    # Green
            (0, 0, 128),    # Navy
            (128, 128, 0)   # Olive
        ]
        
        # Draw common paths with different colors
        for i, path_group in enumerate(self.common_paths[:len(colors)]):
            color = colors[i]
            
            # Draw all paths in this group
            for path_data in path_group['paths'][:3]:  # Draw max 3 paths per group
                path = path_data['path']
                points = np.array([(int(p[0]), int(p[1])) for p in path], dtype=np.int32)
                cv2.polylines(viz, [points], False, color, 2)
            
            # Draw a label for the path
            if path_group['paths']:
                sample_path = path_group['paths'][0]['path']
                if sample_path:
                    x, y = int(sample_path[0][0]), int(sample_path[0][1])
                    if 0 <= x < w and 0 <= y < h:
                        cv2.putText(viz, f"Path {i+1}: {path_group['count']} occurrences", 
                                   (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return viz
    
    def generate_path_statistics_plot(self):
        """
        Generate a plot with path statistics
        
        Returns:
            matplotlib.figure.Figure: Figure with path statistics
        """
        plt.figure(figsize=(10, 8))
        
        # Plot zone popularity
        zone_popularity = self.get_zone_popularity()
        if zone_popularity:
            plt.subplot(221)
            zones = list(zone_popularity.keys())
            counts = list(zone_popularity.values())
            
            # Sort by popularity
            sorted_idx = np.argsort(counts)[::-1]
            zones = [zones[i] for i in sorted_idx]
            counts = [counts[i] for i in sorted_idx]
            
            plt.bar(zones[:10], counts[:10])
            plt.title('Zone Popularity')
            plt.xlabel('Zone ID')
            plt.ylabel('Visit Count')
            plt.xticks(rotation=45)
        
        # Plot zone dwell times
        zone_dwell_stats = self.get_zone_dwell_stats()
        if zone_dwell_stats:
            plt.subplot(222)
            zones = list(zone_dwell_stats.keys())
            avg_dwell = [stats['avg_dwell'] for stats in zone_dwell_stats.values()]
            
            # Sort by average dwell time
            sorted_idx = np.argsort(avg_dwell)[::-1]
            zones = [zones[i] for i in sorted_idx]
            avg_dwell = [avg_dwell[i] for i in sorted_idx]
            
            plt.bar(zones[:10], avg_dwell[:10])
            plt.title('Average Dwell Time by Zone')
            plt.xlabel('Zone ID')
            plt.ylabel('Dwell Time (s)')
            plt.xticks(rotation=45)
        
        # Plot zone transitions
        transitions = {f"{src}->{tgt}": count 
                      for (src, tgt), count in self.zone_transitions.items()
                      if src is not None and tgt is not None}
        
        if transitions:
            plt.subplot(212)
            labels = list(transitions.keys())
            values = list(transitions.values())
            
            # Sort by frequency
            sorted_idx = np.argsort(values)[::-1]
            labels = [labels[i] for i in sorted_idx]
            values = [values[i] for i in sorted_idx]
            
            plt.bar(labels[:15], values[:15])
            plt.title('Top Zone Transitions')
            plt.xlabel('Transition')
            plt.ylabel('Count')
            plt.xticks(rotation=90)
        
        plt.tight_layout()
        return plt.gcf() 