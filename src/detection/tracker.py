"""
Person Tracker class for Retail Analytics System
"""

import time
import logging
import numpy as np
import cv2
from collections import defaultdict
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

class PersonTracker:
    """
    Person tracker using Kalman filtering
    
    This class tracks persons across frames by associating detections
    with existing tracks using a combination of IoU matching and Kalman filtering.
    """
    
    def __init__(self, config):
        """
        Initialize the person tracker with the provided configuration
        
        Args:
            config (dict): Tracker configuration dictionary
        """
        self.config = config
        
        # Tracking parameters
        self.max_age = config.get('max_age', 30)  # Max frames to keep a track alive without matching
        self.min_hits = config.get('min_hits', 3)  # Min detections before a track is confirmed
        self.iou_threshold = config.get('iou_threshold', 0.3)  # IoU threshold for matching
        
        # State variables
        self.tracks = []
        self.track_id_count = 0
        
        # Colors for visualization
        self.colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)
        
        logger.info(f"Person tracker initialized with IoU threshold {self.iou_threshold}")
    
    def update(self, detections, frame=None):
        """
        Update the tracker with new detections
        
        Args:
            detections (list): List of detection dictionaries with 'bbox' and 'confidence'
            frame (numpy.ndarray, optional): Current frame for feature extraction
            
        Returns:
            list: List of active tracks with id, bbox, etc.
        """
        # Convert detections to format expected by tracker
        detection_boxes = np.array([self._bbox_to_xyah(d['bbox']) for d in detections]) if detections else np.empty((0, 4))
        
        # Get predicted locations from existing tracks
        track_boxes = np.array([track.predict()[0] for track in self.tracks])
        
        # Match detections to tracks
        matched, unmatched_dets, unmatched_tracks = [], [], []
        if len(track_boxes) > 0 and len(detection_boxes) > 0:
            matched, unmatched_dets, unmatched_tracks = self._associate_detections_to_tracks(
                detection_boxes, track_boxes, self.iou_threshold)
        else:
            unmatched_dets = list(range(len(detection_boxes)))
            unmatched_tracks = list(range(len(self.tracks)))
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(detection_boxes[det_idx])
            self.tracks[track_idx].last_detection = detections[det_idx]
            self.tracks[track_idx].hits += 1
            self.tracks[track_idx].age = 0
        
        # Mark unmatched tracks for removal
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].age += 1
        
        # Remove dead tracks
        self.tracks = [track for track in self.tracks if track.age < self.max_age]
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._initiate_track(detection_boxes[det_idx], detections[det_idx])
        
        # Get active tracks (confirmed by multiple detections)
        active_tracks = []
        for track in self.tracks:
            if track.is_confirmed() and track.age < 1:
                track_info = {
                    'id': track.id,
                    'bbox': self._xyah_to_bbox(track.state[0]),
                    'confidence': track.last_detection.get('confidence', 1.0) if track.last_detection else 1.0,
                    'age': track.age,
                    'hits': track.hits
                }
                active_tracks.append(track_info)
        
        return active_tracks
    
    def _initiate_track(self, bbox, detection):
        """
        Initialize a new track
        
        Args:
            bbox (numpy.ndarray): Bounding box in [x, y, a, h] format
            detection (dict): Original detection dictionary
        """
        self.track_id_count += 1
        new_track = Track(
            id=self.track_id_count,
            bbox=bbox,
            max_age=self.max_age,
            min_hits=self.min_hits
        )
        new_track.last_detection = detection
        self.tracks.append(new_track)
    
    def _associate_detections_to_tracks(self, detections, tracks, threshold):
        """
        Associate detections to existing tracks using IoU
        
        Args:
            detections (numpy.ndarray): Detection boxes in [x, y, a, h] format
            tracks (numpy.ndarray): Track boxes in [x, y, a, h] format
            threshold (float): IoU threshold for matching
            
        Returns:
            tuple: (matched, unmatched_detections, unmatched_tracks)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        # Compute IoU cost matrix
        cost_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1 - self._iou(self._xyah_to_bbox(track), self._xyah_to_bbox(det))
        
        # Hungarian algorithm for optimal assignment
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches by threshold
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_detections = list(range(len(detections)))
        
        for track_idx, det_idx in zip(track_indices, det_indices):
            if cost_matrix[track_idx, det_idx] <= 1 - threshold:
                matches.append((track_idx, det_idx))
                unmatched_tracks.remove(track_idx)
                unmatched_detections.remove(det_idx)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _bbox_to_xyah(self, bbox):
        """
        Convert [x1, y1, x2, y2] box to [x, y, aspect_ratio, height] format
        
        Args:
            bbox (tuple): Bounding box in [x1, y1, x2, y2] format
            
        Returns:
            numpy.ndarray: Box in [x, y, a, h] format
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        aspect_ratio = w / h if h > 0 else 1.0
        return np.array([x, y, aspect_ratio, h])
    
    def _xyah_to_bbox(self, xyah):
        """
        Convert [x, y, aspect_ratio, height] to [x1, y1, x2, y2] format
        
        Args:
            xyah (numpy.ndarray): Box in [x, y, a, h] format
            
        Returns:
            tuple: Bounding box in [x1, y1, x2, y2] format
        """
        x, y, aspect_ratio, h = xyah
        w = aspect_ratio * h
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        return (x1, y1, x2, y2)
    
    def _iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            bbox1 (tuple): First bounding box in [x1, y1, x2, y2] format
            bbox2 (tuple): Second bounding box in [x1, y1, x2, y2] format
            
        Returns:
            float: IoU score
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area
    
    def visualize_tracks(self, frame, tracks):
        """
        Draw tracks on the frame
        
        Args:
            frame (numpy.ndarray): Input frame
            tracks (list): List of track dictionaries
            
        Returns:
            numpy.ndarray: Frame with tracks drawn
        """
        if frame is None or not tracks:
            return frame
        
        # Create a copy of the frame
        output_frame = frame.copy()
        
        # Draw tracks
        for track in tracks:
            track_id = track['id']
            bbox = track['bbox']
            
            # Get color for this track ID
            color = self.colors[track_id % len(self.colors)].tolist()
            
            # Draw rectangle around person
            cv2.rectangle(output_frame, 
                          (bbox[0], bbox[1]), 
                          (bbox[2], bbox[3]), 
                          color, 2)
            
            # Draw ID
            cv2.putText(output_frame, 
                        f"ID: {track_id}", 
                        (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        color, 
                        2)
        
        return output_frame


class Track:
    """
    Track class for maintaining state of a tracked person
    
    This class uses a Kalman filter to predict and update the state of a tracked person.
    """
    
    def __init__(self, id, bbox, max_age=30, min_hits=3):
        """
        Initialize a track
        
        Args:
            id (int): Unique ID for this track
            bbox (numpy.ndarray): Initial bounding box in [x, y, a, h] format
            max_age (int): Maximum age before track is removed
            min_hits (int): Minimum hits before track is confirmed
        """
        self.id = id
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.max_age = max_age
        self.min_hits = min_hits
        self.last_detection = None
        
        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.eye(8)
        self.kf.F[0, 4] = 1.0  # x' = x + dx
        self.kf.F[1, 5] = 1.0  # y' = y + dy
        self.kf.F[2, 6] = 1.0  # a' = a + da
        self.kf.F[3, 7] = 1.0  # h' = h + dh
        
        # Measurement function (we only observe position, not velocity)
        self.kf.H = np.zeros((4, 8))
        self.kf.H[0, 0] = 1.0  # x
        self.kf.H[1, 1] = 1.0  # y
        self.kf.H[2, 2] = 1.0  # aspect ratio
        self.kf.H[3, 3] = 1.0  # height
        
        # Measurement noise
        self.kf.R = np.diag([10.0, 10.0, 1.0, 10.0])
        
        # Process noise
        self.kf.Q = np.eye(8) * 0.1
        
        # Initial state
        self.kf.x = np.zeros(8)
        self.kf.x[:4] = bbox
        
        # Initial state covariance
        self.kf.P = np.eye(8) * 100.0
    
    def predict(self):
        """
        Predict the state of the track
        
        Returns:
            tuple: (predicted_state, covariance)
        """
        if self.kf.x[3] <= 0:
            self.kf.x[3] = 0.1  # Ensure height is positive
        
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        
        return self.kf.x[:4], self.kf.P
    
    def update(self, bbox):
        """
        Update the state of the track with a new detection
        
        Args:
            bbox (numpy.ndarray): New bounding box in [x, y, a, h] format
        """
        self.time_since_update = 0
        self.kf.update(bbox)
    
    def get_state(self):
        """
        Get the current state of the track
        
        Returns:
            numpy.ndarray: Current state [x, y, a, h]
        """
        return self.kf.x[:4]
    
    def is_confirmed(self):
        """
        Check if this track is confirmed (has enough hits)
        
        Returns:
            bool: True if track is confirmed, False otherwise
        """
        return self.hits >= self.min_hits
    
    @property
    def state(self):
        """Get the current state and covariance"""
        return self.kf.x, self.kf.P 