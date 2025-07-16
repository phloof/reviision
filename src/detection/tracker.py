"""
Person Tracker class for Retail Analytics System with Enhanced Re-identification
"""

import time
import logging
import numpy as np
import cv2
from collections import defaultdict
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

logger = logging.getLogger(__name__)

class PersonTracker:
    """
    Enhanced Person tracker with feature-based re-identification

    This class tracks persons across frames by associating detections
    with existing tracks using IoU matching, Kalman filtering, and visual features.
    """
    
    def __init__(self, config):
        """
        Initialize the person tracker with enhanced re-identification features

        Args:
            config (dict): Tracker configuration dictionary
        """
        self.config = config
        
        # Enhanced tracking parameters
        self.max_age = config.get('max_age', 45)  # Increased for better re-identification
        self.min_hits = config.get('min_hits', 5)  # Increased to reduce false positives
        self.iou_threshold = config.get('iou_threshold', 0.2)  # Slightly lower for better matching
        self.distance_threshold = config.get('distance_threshold', 150)  # Max pixel distance for matching

        # Re-identification parameters
        self.feature_weight = config.get('feature_weight', 0.4)  # Weight for feature similarity
        self.position_weight = config.get('position_weight', 0.6)  # Weight for position similarity
        self.min_track_confidence = config.get('min_track_confidence', 0.3)  # Minimum confidence for stable tracks
        self.confirmation_frames = config.get('confirmation_frames', 8)  # Frames needed to confirm person in DB

        # Re-identification settings
        self.reid_enabled = config.get('reid_enabled', True)
        self.reid_distance_threshold = config.get('reid_distance_threshold', 300)  # Max distance for re-identification
        self.reid_feature_threshold = config.get('reid_feature_threshold', 0.7)  # Feature similarity threshold
        self.reid_max_age = config.get('reid_max_age', 90)  # Max age for re-identification attempts

        # State variables
        self.tracks = []
        self.track_id_count = 0
        self.lost_tracks = []  # Store recently lost tracks for re-identification
        self.track_features = {}  # Store visual features for each track
        self.track_demographics = {}  # Store demographic info for each track

        # Colors for visualization
        self.colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)
        
        logger.info(f"Enhanced person tracker initialized with IoU threshold {self.iou_threshold}, min_hits {self.min_hits}")

    def _extract_visual_features(self, frame, bbox):
        """
        Extract visual features from person crop for re-identification

        Args:
            frame: Full frame image
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Feature vector for re-identification
        """
        try:
            x1, y1, x2, y2 = bbox
            # Ensure valid coordinates
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return None

            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                return None

            # Resize to standard size for consistent features
            person_crop = cv2.resize(person_crop, (64, 128))

            # Extract color histogram features
            hist_b = cv2.calcHist([person_crop], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([person_crop], [1], None, [32], [0, 256])
            hist_r = cv2.calcHist([person_crop], [2], None, [32], [0, 256])

            # Normalize histograms
            hist_b = hist_b.flatten() / (hist_b.sum() + 1e-7)
            hist_g = hist_g.flatten() / (hist_g.sum() + 1e-7)
            hist_r = hist_r.flatten() / (hist_r.sum() + 1e-7)

            # Extract texture features using LBP-like approach
            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)

            # Simple texture features - edge density in different regions
            edges = cv2.Canny(gray, 50, 150)

            # Divide image into regions and calculate edge density
            h, w = edges.shape
            regions = []
            for i in range(4):  # 4 vertical regions
                for j in range(2):  # 2 horizontal regions
                    region = edges[i*h//4:(i+1)*h//4, j*w//2:(j+1)*w//2]
                    edge_density = np.sum(region) / (region.size * 255)
                    regions.append(edge_density)

            # Combine all features
            features = np.concatenate([hist_b, hist_g, hist_r, regions])

            return features

        except Exception as e:
            logger.debug(f"Error extracting visual features: {e}")
            return None

    def _calculate_feature_similarity(self, features1, features2):
        """
        Calculate similarity between two feature vectors

        Args:
            features1, features2: Feature vectors

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        if features1 is None or features2 is None:
            return 0.0

        try:
            # Cosine similarity
            similarity = cosine_similarity([features1], [features2])[0][0]
            return max(0.0, similarity)  # Ensure non-negative
        except:
            return 0.0

    def _find_reid_candidate(self, detection, frame):
        """
        Find potential re-identification candidate from lost tracks

        Args:
            detection: New detection
            frame: Current frame

        Returns:
            Track ID if re-identification successful, None otherwise
        """
        if not self.reid_enabled or not self.lost_tracks:
            return None

        # Extract features from current detection
        current_features = self._extract_visual_features(frame, detection['bbox'])
        if current_features is None:
            return None

        best_match = None
        best_score = 0.0
        current_time = time.time()

        # Check against recently lost tracks
        for lost_track in self.lost_tracks[:]:  # Use slice to avoid modification during iteration
            # Skip if too old
            if current_time - lost_track['lost_time'] > self.reid_max_age:
                self.lost_tracks.remove(lost_track)
                continue

            # Calculate positional distance
            lost_center = lost_track['last_center']
            current_center = detection['center']
            distance = np.sqrt((lost_center[0] - current_center[0])**2 +
                             (lost_center[1] - current_center[1])**2)

            if distance > self.reid_distance_threshold:
                continue

            # Calculate feature similarity
            lost_features = self.track_features.get(lost_track['id'])
            if lost_features is None:
                continue

            feature_sim = self._calculate_feature_similarity(current_features, lost_features)

            # Combined score (distance and features)
            distance_score = max(0, 1 - distance / self.reid_distance_threshold)
            combined_score = 0.3 * distance_score + 0.7 * feature_sim

            if combined_score > best_score and combined_score > self.reid_feature_threshold:
                best_score = combined_score
                best_match = lost_track

        if best_match:
            # Re-activate the track
            logger.info(f"Re-identified person {best_match['id']} with score {best_score:.3f}")
            self.lost_tracks.remove(best_match)
            return best_match['id']

        return None

    def update(self, detections, frame=None):
        """
        Update the tracker with new detections and enhanced re-identification

        Args:
            detections (list): List of detection dictionaries with 'bbox' and 'confidence'
            frame (numpy.ndarray, optional): Current frame for feature extraction
            
        Returns:
            list: List of active tracks with id, bbox, etc.
        """
        if not detections:
            detections = []

        # Convert detections to format expected by tracker
        detection_boxes = np.array([self._bbox_to_xyah(d['bbox']) for d in detections]) if detections else np.empty((0, 4))
        
        # Get predicted locations from existing tracks
        track_boxes = np.array([track.predict()[0] for track in self.tracks])
        
        # Enhanced association with feature matching
        matched, unmatched_dets, unmatched_tracks = self._enhanced_associate_detections_to_tracks(
            detections, detection_boxes, track_boxes, frame)

        # Update matched tracks
        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(detection_boxes[det_idx])
            self.tracks[track_idx].last_detection = detections[det_idx]
            self.tracks[track_idx].hits += 1
            self.tracks[track_idx].age = 0
            self.tracks[track_idx].confidence_history.append(detections[det_idx]['confidence'])

            # Update visual features
            if frame is not None:
                features = self._extract_visual_features(frame, detections[det_idx]['bbox'])
                if features is not None:
                    track_id = self.tracks[track_idx].id
                    if track_id in self.track_features:
                        # Exponential moving average for feature updates
                        self.track_features[track_id] = 0.7 * self.track_features[track_id] + 0.3 * features
                    else:
                        self.track_features[track_id] = features

        # Handle unmatched tracks
        tracks_to_remove = []
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].age += 1
            if self.tracks[track_idx].age >= self.max_age:
                # Move to lost tracks for potential re-identification
                if self.tracks[track_idx].is_confirmed():
                    lost_track = {
                        'id': self.tracks[track_idx].id,
                        'last_center': self._xyah_to_center(self.tracks[track_idx].state[0]),
                        'lost_time': time.time(),
                        'confidence': np.mean(self.tracks[track_idx].confidence_history[-10:])
                    }
                    self.lost_tracks.append(lost_track)
                    logger.debug(f"Track {self.tracks[track_idx].id} moved to lost tracks")
                tracks_to_remove.append(track_idx)

        # Remove dead tracks
        for idx in sorted(tracks_to_remove, reverse=True):
            del self.tracks[idx]

        # Handle unmatched detections with re-identification
        for det_idx in unmatched_dets:
            # Try re-identification first
            reid_id = self._find_reid_candidate(detections[det_idx], frame) if frame is not None else None

            if reid_id is not None:
                # Re-activate existing track
                self._reactivate_track(detection_boxes[det_idx], detections[det_idx], reid_id, frame)
            else:
                # Create new track
                self._initiate_track(detection_boxes[det_idx], detections[det_idx], frame)

        # Clean up old lost tracks
        current_time = time.time()
        self.lost_tracks = [track for track in self.lost_tracks
                           if current_time - track['lost_time'] < self.reid_max_age]

        # Get active tracks (confirmed by multiple detections)
        active_tracks = []
        for track in self.tracks:
            if track.is_confirmed() and track.age < 1:
                # Only include tracks with sufficient confidence
                avg_confidence = np.mean(track.confidence_history[-10:])
                if avg_confidence >= self.min_track_confidence:
                    track_info = {
                        'id': track.id,
                        'bbox': self._xyah_to_bbox(track.state[0]),
                        'confidence': avg_confidence,
                        'age': track.age,
                        'hits': track.hits,
                        'frames_tracked': track.hits,
                        'is_confirmed': track.hits >= self.confirmation_frames
                    }
                    active_tracks.append(track_info)

        return active_tracks
    
    def _enhanced_associate_detections_to_tracks(self, detections, detection_boxes, track_boxes, frame):
        """
        Enhanced association using IoU and visual features
        """
        if len(self.tracks) == 0 or len(detection_boxes) == 0:
            return [], list(range(len(detection_boxes))), list(range(len(self.tracks)))

        # Compute cost matrix with IoU and features
        cost_matrix = np.zeros((len(self.tracks), len(detection_boxes)))

        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                # IoU component
                iou = self._iou(self._xyah_to_bbox(track_boxes[i]), detection['bbox'])
                iou_cost = 1 - iou

                # Feature component
                feature_cost = 1.0
                if frame is not None and track.id in self.track_features:
                    detection_features = self._extract_visual_features(frame, detection['bbox'])
                    if detection_features is not None:
                        feature_sim = self._calculate_feature_similarity(
                            self.track_features[track.id], detection_features)
                        feature_cost = 1 - feature_sim

                # Combined cost
                cost_matrix[i, j] = (self.position_weight * iou_cost +
                                   self.feature_weight * feature_cost)

        # Hungarian algorithm for optimal assignment
        track_indices, det_indices = linear_sum_assignment(cost_matrix)

        # Filter matches by threshold
        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detection_boxes)))

        for track_idx, det_idx in zip(track_indices, det_indices):
            if cost_matrix[track_idx, det_idx] <= 1 - self.iou_threshold:
                matches.append((track_idx, det_idx))
                unmatched_tracks.remove(track_idx)
                unmatched_detections.remove(det_idx)

        return matches, unmatched_detections, unmatched_tracks

    def _reactivate_track(self, bbox, detection, track_id, frame):
        """
        Re-activate a track that was previously lost
        """
        new_track = Track(
            id=track_id,
            bbox=bbox,
            max_age=self.max_age,
            min_hits=self.min_hits
        )
        new_track.last_detection = detection
        new_track.hits = self.confirmation_frames  # Start with confirmed status
        new_track.confidence_history = [detection['confidence']]

        # Update features
        if frame is not None:
            features = self._extract_visual_features(frame, detection['bbox'])
            if features is not None:
                self.track_features[track_id] = features

        self.tracks.append(new_track)
        logger.info(f"Re-activated track {track_id}")

    def _initiate_track(self, bbox, detection, frame=None):
        """
        Initialize a new track with enhanced features

        Args:
            bbox (numpy.ndarray): Bounding box in [x, y, a, h] format
            detection (dict): Original detection dictionary
            frame (numpy.ndarray, optional): Current frame for feature extraction
        """
        self.track_id_count += 1
        new_track = Track(
            id=self.track_id_count,
            bbox=bbox,
            max_age=self.max_age,
            min_hits=self.min_hits
        )
        new_track.last_detection = detection
        new_track.confidence_history = [detection['confidence']]

        # Extract and store visual features
        if frame is not None:
            features = self._extract_visual_features(frame, detection['bbox'])
            if features is not None:
                self.track_features[self.track_id_count] = features

        self.tracks.append(new_track)

    def _associate_detections_to_tracks(self, detections, tracks, threshold):
        """
        Associate detections to existing tracks using IoU (legacy method)

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


class Track:
    """
    Enhanced Track class for maintaining state of a tracked person with confidence tracking
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
        self.confidence_history = []

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
