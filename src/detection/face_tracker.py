"""
Face Tracker class for Retail Analytics System with Face Embedding-based Re-identification
"""

import time
import logging
import numpy as np
import cv2
from collections import defaultdict
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class FaceTracker:
    """
    Enhanced Face tracker with face embedding-based re-identification

    This class tracks faces across frames by associating detections
    with existing tracks using IoU matching, Kalman filtering, and face embeddings.
    """
    
    def __init__(self, config):
        """
        Initialize the face tracker with face embedding features

        Args:
            config (dict): Tracker configuration dictionary
        """
        self.config = config
        
        # ReID configuration (more lenient thresholds for better performance)
        self.reid_enabled = config.get('reid_enabled', True)
        self.reid_max_age = config.get('reid_max_age', 180)  # Longer time to keep lost tracks
        self.reid_distance_threshold = config.get('reid_distance_threshold', 300)  # Increased for better coverage
        self.reid_embedding_threshold = config.get('reid_embedding_threshold', 0.45)  # Less strict similarity
        self.reid_max_lost_tracks = config.get('reid_max_lost_tracks', 50)  # More tracks to remember

        # Tracking parameters optimized for faces
        self.max_age = config.get('max_age', 5)
        self.min_hits = config.get('min_hits', 1)  # Lower for better face detection
        self.iou_threshold = config.get('iou_threshold', 0.25)  # Lower for face boxes
        self.embedding_similarity_threshold = config.get('embedding_similarity_threshold', 0.4)  # Less strict

        # Face-specific settings
        self.face_quality_threshold = config.get('face_quality_threshold', 0.3)  # Lower for more faces
        self.embedding_cache_size = config.get('embedding_cache_size', 1000)
        
        # Enhanced embedding parameters for better ReID
        self.embedding_weight = config.get('embedding_weight', 0.8)  # Higher weight for embeddings
        self.position_weight = config.get('position_weight', 0.2)  # Lower weight for position  
        self.min_track_confidence = config.get('min_track_confidence', 0.3)  # Lower for more detections
        self.confirmation_frames = config.get('confirmation_frames', 3)  # Faster confirmation
        self.distance_threshold = config.get('distance_threshold', 150)  # Larger for face movement
        self.embedding_update_rate = config.get('embedding_update_rate', 0.3)  # Rate for embedding updates

        # State variables
        self.tracks = []
        self.track_id_count = 0
        self.lost_tracks = []  # Store recently lost tracks for re-identification
        self.track_embeddings = {}  # Store face embeddings for each track
        self.track_qualities = {}  # Store quality scores for each track

        # Colors for visualization
        self.colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)
        
        logger.info(f"Face tracker initialized with embedding threshold {self.reid_embedding_threshold}, "
                   f"IoU threshold {self.iou_threshold}, min_hits {self.min_hits}")

    def _calculate_embedding_similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity between two face embeddings

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0

        try:
            # Normalize embeddings for better similarity calculation
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Cosine similarity for face embeddings
            similarity = np.dot(embedding1, embedding2)
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
        except:
            return 0.0

    def _find_reid_candidate(self, detection):
        """
        Find potential re-identification candidate from lost tracks using face embeddings

        Args:
            detection: New face detection with embedding

        Returns:
            Track ID if re-identification successful, None otherwise
        """
        if not self.reid_enabled or not self.lost_tracks:
            return None

        current_embedding = detection.get('embedding')
        if current_embedding is None:
            return None

        best_match = None
        best_score = 0.0
        current_time = time.time()

        # Check against recently lost tracks
        for lost_track in self.lost_tracks[:]:
            # Skip if too old
            if current_time - lost_track['lost_time'] > self.reid_max_age:
                self.lost_tracks.remove(lost_track)
                continue

            # Calculate positional distance
            lost_center = lost_track['last_center']
            current_center = self._bbox_to_center(detection['bbox'])
            distance = np.sqrt((lost_center[0] - current_center[0])**2 +
                             (lost_center[1] - current_center[1])**2)

            if distance > self.reid_distance_threshold:
                continue

            # Calculate face embedding similarity
            lost_embedding = self.track_embeddings.get(lost_track['id'])
            if lost_embedding is None:
                continue

            embedding_sim = self._calculate_embedding_similarity(current_embedding, lost_embedding)

            # Enhanced combined score with configurable weighting for better ReID
            distance_score = max(0, 1 - distance / self.reid_distance_threshold)
            combined_score = (distance_score * self.position_weight) + (embedding_sim * self.embedding_weight)

            # More lenient threshold for better ReID performance
            if combined_score > best_score and embedding_sim > self.reid_embedding_threshold:
                best_score = combined_score
                best_match = lost_track
                
            # Log potential matches for debugging
            if embedding_sim > self.reid_embedding_threshold * 0.8:  # Log near matches
                logger.debug(f"ReID candidate track {lost_track['id']}: embedding_sim={embedding_sim:.3f}, "
                           f"distance={distance:.1f}, combined_score={combined_score:.3f}")

        if best_match:
            # Re-activate the track
            logger.info(f"Re-identified face {best_match['id']} with embedding similarity {best_score:.3f}")
            self.lost_tracks.remove(best_match)
            return best_match['id']

        # Clean up old lost tracks periodically to prevent memory buildup
        if len(self.lost_tracks) > self.reid_max_lost_tracks:
            self.lost_tracks = sorted(self.lost_tracks, key=lambda x: x['lost_time'], reverse=True)[:self.reid_max_lost_tracks]
            logger.debug(f"Cleaned up lost tracks, now keeping {len(self.lost_tracks)} tracks")

        return None

    def update(self, detections, frame=None):
        """
        Update the tracker with new face detections

        Args:
            detections (list): List of face detection dictionaries with 'bbox', 'confidence', 'embedding'
            frame (numpy.ndarray, optional): Current frame (not used for face tracking)
            
        Returns:
            list: List of active tracks with id, bbox, etc.
        """
        if not detections:
            detections = []

        # Filter detections by quality
        quality_detections = [
            det for det in detections 
            if det.get('quality_score', 1.0) >= self.face_quality_threshold
        ]

        # Convert detections to format expected by tracker
        detection_boxes = np.array([self._bbox_to_xyah(d['bbox']) for d in quality_detections]) if quality_detections else np.empty((0, 4))
        
        # Get predicted locations from existing tracks
        track_boxes = np.array([track.predict()[0] for track in self.tracks])
        
        # Enhanced association with face embedding matching
        matched, unmatched_dets, unmatched_tracks = self._enhanced_associate_detections_to_tracks(
            quality_detections, detection_boxes, track_boxes)

        # Update matched tracks
        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(detection_boxes[det_idx])
            self.tracks[track_idx].last_detection = quality_detections[det_idx]
            self.tracks[track_idx].hits += 1
            self.tracks[track_idx].age = 0
            self.tracks[track_idx].confidence_history.append(quality_detections[det_idx]['confidence'])

            # Update face embedding with exponential moving average
            track_id = self.tracks[track_idx].id
            new_embedding = quality_detections[det_idx].get('embedding')
            if new_embedding is not None:
                if track_id in self.track_embeddings:
                    # Exponential moving average for embedding updates
                    old_embedding = self.track_embeddings[track_id]
                    updated_embedding = (1 - self.embedding_update_rate) * old_embedding + self.embedding_update_rate * new_embedding
                    # Re-normalize the updated embedding
                    self.track_embeddings[track_id] = updated_embedding / np.linalg.norm(updated_embedding)
                else:
                    self.track_embeddings[track_id] = new_embedding

            # Update quality scores
            quality_score = quality_detections[det_idx].get('quality_score', 1.0)
            if track_id in self.track_qualities:
                # Moving average for quality
                self.track_qualities[track_id] = 0.8 * self.track_qualities[track_id] + 0.2 * quality_score
            else:
                self.track_qualities[track_id] = quality_score

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
                        'confidence': np.mean(self.tracks[track_idx].confidence_history[-5:])
                    }
                    self.lost_tracks.append(lost_track)
                    logger.debug(f"Face track {self.tracks[track_idx].id} moved to lost tracks")
                tracks_to_remove.append(track_idx)

        # Remove dead tracks
        for idx in sorted(tracks_to_remove, reverse=True):
            del self.tracks[idx]

        # Handle unmatched detections with re-identification
        for det_idx in unmatched_dets:
            # Try re-identification first
            reid_id = self._find_reid_candidate(quality_detections[det_idx])

            if reid_id is not None:
                # Re-activate existing track
                self._reactivate_track(detection_boxes[det_idx], quality_detections[det_idx], reid_id)
            else:
                # Create new track
                self._initiate_track(detection_boxes[det_idx], quality_detections[det_idx])

        # Clean up old lost tracks and embeddings
        current_time = time.time()
        self.lost_tracks = [track for track in self.lost_tracks
                           if current_time - track['lost_time'] < self.reid_max_age]

        # Clean up embeddings for tracks that no longer exist
        active_track_ids = {track.id for track in self.tracks}
        lost_track_ids = {track['id'] for track in self.lost_tracks}
        all_active_ids = active_track_ids | lost_track_ids
        
        # Remove embeddings and qualities for tracks that are completely gone
        self.track_embeddings = {k: v for k, v in self.track_embeddings.items() if k in all_active_ids}
        self.track_qualities = {k: v for k, v in self.track_qualities.items() if k in all_active_ids}

        # Get active tracks (confirmed by multiple detections)
        active_tracks = []
        for track in self.tracks:
            if track.is_confirmed() and track.age < 1:
                # Only include tracks with sufficient confidence and quality
                avg_confidence = np.mean(track.confidence_history[-5:])
                quality_score = self.track_qualities.get(track.id, 1.0)
                
                if avg_confidence >= self.min_track_confidence:
                    track_info = {
                        'id': track.id,
                        'bbox': self._xyah_to_bbox(track.state[0]),
                        'confidence': avg_confidence,
                        'quality_score': quality_score,
                        'age': track.age,
                        'hits': track.hits,
                        'frames_tracked': track.hits,
                        'is_confirmed': track.hits >= self.confirmation_frames,
                        'embedding': self.track_embeddings.get(track.id)
                    }
                    active_tracks.append(track_info)

        return active_tracks
    
    def _enhanced_associate_detections_to_tracks(self, detections, detection_boxes, track_boxes):
        """
        Enhanced association using IoU and face embeddings
        """
        if len(self.tracks) == 0 or len(detection_boxes) == 0:
            return [], list(range(len(detection_boxes))), list(range(len(self.tracks)))

        # Compute cost matrix with IoU and face embeddings
        cost_matrix = np.zeros((len(self.tracks), len(detection_boxes)))

        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                # IoU component
                iou = self._iou(self._xyah_to_bbox(track_boxes[i]), detection['bbox'])
                iou_cost = 1 - iou

                # Face embedding component
                embedding_cost = 1.0
                track_embedding = self.track_embeddings.get(track.id)
                detection_embedding = detection.get('embedding')
                
                if track_embedding is not None and detection_embedding is not None:
                    embedding_sim = self._calculate_embedding_similarity(track_embedding, detection_embedding)
                    embedding_cost = 1 - embedding_sim

                # Face size consistency component (lightweight check)
                size_cost = 0.0
                if hasattr(track, 'last_detection') and track.last_detection:
                    last_bbox = track.last_detection['bbox']
                    last_size = min(last_bbox[2] - last_bbox[0], last_bbox[3] - last_bbox[1])
                    current_size = detection.get('face_size', 0)
                    if last_size > 0 and current_size > 0:
                        size_ratio = abs(current_size - last_size) / max(last_size, current_size)
                        size_cost = min(0.5, size_ratio * 0.5)  # Reduced impact

                # Enhanced combined cost with improved weighting for face tracking
                cost_matrix[i, j] = (
                    self.position_weight * iou_cost +
                    self.embedding_weight * embedding_cost +
                    0.1 * size_cost  # Minimal size weight
                )

        # Hungarian algorithm for optimal assignment
        track_indices, det_indices = linear_sum_assignment(cost_matrix)

        # Filter matches by threshold
        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detection_boxes)))

        for track_idx, det_idx in zip(track_indices, det_indices):
            # Enhanced threshold check optimized for face tracking
            cost = cost_matrix[track_idx, det_idx]
            
            # Calculate individual components for better decision making
            iou = self._iou(self._xyah_to_bbox(track_boxes[track_idx]), detections[det_idx]['bbox'])
            
            # More lenient acceptance criteria for faces
            embedding_sim = 0.0
            track_embedding = self.track_embeddings.get(self.tracks[track_idx].id)
            detection_embedding = detections[det_idx].get('embedding')
            if track_embedding is not None and detection_embedding is not None:
                embedding_sim = self._calculate_embedding_similarity(track_embedding, detection_embedding)
            
            # Accept if either IoU is decent OR embedding similarity is high
            accept_match = (
                cost <= 0.8 and  # Overall cost threshold
                (iou >= self.iou_threshold or embedding_sim >= self.embedding_similarity_threshold)
            )
            
            if accept_match:
                matches.append((track_idx, det_idx))
                unmatched_tracks.remove(track_idx)
                logger.debug(f"Matched track {self.tracks[track_idx].id}: IoU={iou:.3f}, "
                           f"embedding_sim={embedding_sim:.3f}, cost={cost:.3f}")
                unmatched_detections.remove(det_idx)

        return matches, unmatched_detections, unmatched_tracks

    def _reactivate_track(self, bbox, detection, track_id):
        """
        Re-activate a track that was previously lost
        """
        new_track = FaceTrack(
            id=track_id,
            bbox=bbox,
            max_age=self.max_age,
            min_hits=self.min_hits
        )
        new_track.last_detection = detection
        new_track.hits = self.confirmation_frames  # Start with confirmed status
        new_track.confidence_history = [detection['confidence']]

        # Re-use existing embedding
        embedding = detection.get('embedding')
        if embedding is not None:
            self.track_embeddings[track_id] = embedding

        quality_score = detection.get('quality_score', 1.0)
        self.track_qualities[track_id] = quality_score

        self.tracks.append(new_track)
        logger.info(f"Re-activated face track {track_id}")

    def _initiate_track(self, bbox, detection):
        """
        Initialize a new face track

        Args:
            bbox (numpy.ndarray): Bounding box in [x, y, a, h] format
            detection (dict): Original detection dictionary
        """
        self.track_id_count += 1
        new_track = FaceTrack(
            id=self.track_id_count,
            bbox=bbox,
            max_age=self.max_age,
            min_hits=self.min_hits
        )
        new_track.last_detection = detection
        new_track.confidence_history = [detection['confidence']]

        # Store face embedding and quality
        embedding = detection.get('embedding')
        if embedding is not None:
            self.track_embeddings[self.track_id_count] = embedding

        quality_score = detection.get('quality_score', 1.0)
        self.track_qualities[self.track_id_count] = quality_score

        self.tracks.append(new_track)

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
            xyah (numpy.ndarray): Box in [x, y, a, h] format (may contain velocity components)

        Returns:
            tuple: Bounding box in [x1, y1, x2, y2] format
        """
        # Extract only position components (first 4 elements) from state vector
        if len(xyah) > 4:
            xyah = xyah[:4]  # Take only [x, y, a, h], ignore velocity components
            
        x, y, aspect_ratio, h = xyah
        w = aspect_ratio * h
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        return (x1, y1, x2, y2)

    def _bbox_to_center(self, bbox):
        """
        Get center point of bounding box

        Args:
            bbox (tuple): Bounding box in [x1, y1, x2, y2] format

        Returns:
            tuple: Center point (x, y)
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _xyah_to_center(self, xyah):
        """
        Get center point from xyah format

        Args:
            xyah (numpy.ndarray): Box in [x, y, a, h] format

        Returns:
            tuple: Center point (x, y)
        """
        return (xyah[0], xyah[1])

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

    def get_track_embeddings(self):
        """
        Get all track embeddings for external use
        
        Returns:
            dict: Dictionary mapping track IDs to embeddings
        """
        return self.track_embeddings.copy()


class FaceTrack:
    """
    Face Track class for maintaining state of a tracked face
    """
    
    def __init__(self, id, bbox, max_age=30, min_hits=3):
        """
        Initialize a face track
        
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

        # Initialize Kalman filter (adjusted for face tracking)
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
        
        # Measurement noise (lower for faces as they're smaller and more precise)
        self.kf.R = np.diag([5.0, 5.0, 0.5, 5.0])
        
        # Process noise (lower for faces as they move less erratically)
        self.kf.Q = np.eye(8) * 0.05
        
        # Initial state
        self.kf.x = np.zeros(8)
        self.kf.x[:4] = bbox
        
        # Initial state covariance (higher uncertainty for new faces)
        self.kf.P = np.eye(8) * 50.0
    
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