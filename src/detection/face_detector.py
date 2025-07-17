"""
Face Detector class using InsightFace for Retail Analytics System
"""

import time
import logging
import numpy as np
import cv2
from pathlib import Path

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Face detector class using InsightFace model
    
    This class handles the detection of faces in frames using InsightFace.
    It provides methods for detecting faces and extracting face embeddings for ReID.
    """
    
    def __init__(self, config):
        """
        Initialize the face detector with the provided configuration
        
        Args:
            config (dict): Detector configuration dictionary
        """
        self.config = config
        self.model_name = config.get('model_name', 'buffalo_l')
        self.model_dir = config.get('model_dir', './models')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.min_face_size = config.get('min_face_size', 30)  # Minimum face size in pixels
        self.max_face_size = config.get('max_face_size', 1000)  # Maximum face size in pixels
        self.det_size = config.get('det_size', (640, 640))  # Detection input size
        self.ctx_id = config.get('ctx_id', 0)  # GPU context ID (-1 for CPU)
        
        # Performance settings
        self.detection_interval = config.get('detection_interval', 1)  # Run detection every N frames
        self.frame_count = 0
        
        # Quality control
        self.enable_quality_check = config.get('enable_quality_check', True)
        self.min_quality_score = config.get('min_quality_score', 0.3)
        
        # Initialize the model
        self._load_model()
        
        logger.info(f"Face detector initialized with confidence threshold {self.confidence_threshold}")
    
    def _load_model(self):
        """
        Load the InsightFace model
        
        Raises:
            ImportError: If InsightFace is not available
            RuntimeError: If the model fails to load
        """
        if not INSIGHTFACE_AVAILABLE:
            error_msg = "InsightFace not available. Please install with: pip install insightface"
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        try:
            # Initialize InsightFace app with detection and recognition modules
            self.face_app = FaceAnalysis(
                name=self.model_name,
                root=self.model_dir,
                allowed_modules=['detection', 'recognition']
            )
            
            # Prepare the model with specified context and detection size
            self.face_app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
            
            logger.info(f"InsightFace model '{self.model_name}' loaded from {self.model_dir}")
            
        except Exception as e:
            error_msg = f"Failed to load InsightFace model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def detect(self, frame):
        """
        Detect faces in the given frame
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            list: List of detected faces with bounding boxes, confidence scores, and embeddings
                 Each item is a dict with: 'bbox', 'confidence', 'embedding', 'landmarks'
        """
        if frame is None:
            return []
        
        # Skip frames if detection interval is set
        self.frame_count += 1
        if self.detection_interval > 1 and (self.frame_count % self.detection_interval) != 0:
            return []
        
        try:
            start_time = time.time()
            
            # Run face detection and recognition
            faces = self.face_app.get(frame)
            
            # Extract and format detections
            detections = []
            for face in faces:
                # Get bounding box
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                
                # Get confidence score
                confidence = float(face.det_score)
                
                # Filter by confidence threshold
                if confidence < self.confidence_threshold:
                    continue
                
                # Filter by face size
                face_width = x2 - x1
                face_height = y2 - y1
                face_size = min(face_width, face_height)
                
                if face_size < self.min_face_size or face_size > self.max_face_size:
                    continue
                
                # Quality check
                if self.enable_quality_check:
                    quality_score = self._assess_face_quality(frame, bbox, face)
                    if quality_score < self.min_quality_score:
                        continue
                else:
                    quality_score = 1.0
                
                # Get face embedding for ReID
                embedding = face.embedding
                if embedding is not None:
                    # Normalize embedding
                    embedding = embedding / np.linalg.norm(embedding)
                
                # Get landmarks if available
                landmarks = face.kps if hasattr(face, 'kps') else None
                
                # Format detection
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'embedding': embedding,
                    'landmarks': landmarks,
                    'quality_score': quality_score,
                    'face_size': face_size
                }
                detections.append(detection)
            
            elapsed_time = time.time() - start_time
            if detections:
                logger.debug(f"Detected {len(detections)} faces in {elapsed_time:.4f} seconds")
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            return []
    
    def _assess_face_quality(self, frame, bbox, face):
        """
        Assess the quality of a detected face for demographic analysis
        
        Args:
            frame (numpy.ndarray): Input frame
            bbox (numpy.ndarray): Face bounding box
            face: InsightFace detection object
            
        Returns:
            float: Quality score (0-1, higher is better)
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Extract face region
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                return 0.0
            
            # Calculate quality factors
            quality_factors = []
            
            # 1. Face size factor (larger faces are generally better)
            face_size = min(x2 - x1, y2 - y1)
            size_factor = min(1.0, face_size / 100.0)  # Normalize to 100px
            quality_factors.append(size_factor)
            
            # 2. Sharpness factor (using Laplacian variance)
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            sharpness_factor = min(1.0, laplacian_var / 500.0)  # Normalize
            quality_factors.append(sharpness_factor)
            
            # 3. Brightness factor (avoid too dark or too bright faces)
            mean_brightness = np.mean(gray_face)
            brightness_factor = 1.0 - abs(mean_brightness - 128) / 128.0
            quality_factors.append(brightness_factor)
            
            # 4. Pose factor (use landmarks if available for frontal face check)
            pose_factor = 1.0  # Default to good pose
            if hasattr(face, 'kps') and face.kps is not None:
                landmarks = face.kps
                # Simple frontal face check using eye positions
                if len(landmarks) >= 2:
                    left_eye = landmarks[0]
                    right_eye = landmarks[1]
                    eye_distance = np.linalg.norm(left_eye - right_eye)
                    face_width = x2 - x1
                    
                    # Eyes should be roughly 1/3 of face width apart for frontal faces
                    expected_ratio = 0.3
                    actual_ratio = eye_distance / face_width if face_width > 0 else 0
                    pose_factor = 1.0 - abs(actual_ratio - expected_ratio) / expected_ratio
                    pose_factor = max(0.0, min(1.0, pose_factor))
            
            quality_factors.append(pose_factor)
            
            # Combine quality factors (weighted average)
            weights = [0.2, 0.3, 0.2, 0.3]  # size, sharpness, brightness, pose
            quality_score = sum(f * w for f, w in zip(quality_factors, weights))
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.debug(f"Error assessing face quality: {e}")
            return 0.5  # Default medium quality
    
    def visualize_detections(self, frame, detections, show_embeddings=False):
        """
        Draw bounding boxes around detected faces
        
        Args:
            frame (numpy.ndarray): Input frame
            detections (list): List of detection dictionaries
            show_embeddings (bool): Whether to show embedding info
            
        Returns:
            numpy.ndarray: Frame with bounding boxes drawn
        """
        if frame is None or not detections:
            return frame
        
        # Create a copy of the frame
        output_frame = frame.copy()
        
        # Draw bounding boxes
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            quality_score = detection.get('quality_score', 1.0)
            face_size = detection.get('face_size', 0)
            
            # Color based on quality (green = high quality, red = low quality)
            if quality_score > 0.7:
                color = (0, 255, 0)  # Green
            elif quality_score > 0.4:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            # Draw rectangle around face
            cv2.rectangle(output_frame, 
                          (bbox[0], bbox[1]), 
                          (bbox[2], bbox[3]), 
                          color, 2)
            
            # Draw label with confidence and quality
            label = f"Face: {confidence:.2f} | Q: {quality_score:.2f}"
            if show_embeddings and detection.get('embedding') is not None:
                label += f" | E: {len(detection['embedding'])}"
            
            # Add face size info
            label += f" | {face_size}px"
            
            cv2.putText(output_frame, 
                        label, 
                        (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4, 
                        color, 
                        1)
            
            # Draw landmarks if available
            if detection.get('landmarks') is not None:
                landmarks = detection['landmarks']
                for landmark in landmarks:
                    cv2.circle(output_frame, 
                             (int(landmark[0]), int(landmark[1])), 
                             2, color, -1)
        
        return output_frame
    
    def get_face_embeddings(self, detections):
        """
        Extract face embeddings from detections for ReID
        
        Args:
            detections (list): List of detection dictionaries
            
        Returns:
            list: List of normalized face embeddings
        """
        embeddings = []
        for detection in detections:
            embedding = detection.get('embedding')
            if embedding is not None:
                embeddings.append(embedding)
        
        return embeddings
    
    def filter_high_quality_faces(self, detections, min_quality=0.5):
        """
        Filter detections to only include high-quality faces suitable for demographics
        
        Args:
            detections (list): List of detection dictionaries
            min_quality (float): Minimum quality threshold
            
        Returns:
            list: Filtered list of high-quality face detections
        """
        return [
            det for det in detections 
            if det.get('quality_score', 0) >= min_quality
        ] 