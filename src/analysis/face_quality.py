"""
Face Quality Assessment Module for Retail Analytics System

This module provides comprehensive face quality scoring to select the best
face snapshots for each person for optimal demographic analysis.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any
import math

logger = logging.getLogger(__name__)

class FaceQualityScorer:
    """
    Advanced face quality scorer that evaluates multiple factors to determine
    the best face snapshot for demographic analysis
    """
    
    def __init__(self, config=None):
        """
        Initialize the face quality scorer
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        
        # Quality assessment weights
        self.weights = {
            'size': self.config.get('size_weight', 0.25),          # Face size importance
            'clarity': self.config.get('clarity_weight', 0.30),     # Image sharpness
            'frontality': self.config.get('frontality_weight', 0.20), # Face angle
            'lighting': self.config.get('lighting_weight', 0.15),   # Lighting quality
            'detection_conf': self.config.get('detection_weight', 0.10) # Detection confidence
        }
        
        # Quality thresholds
        self.min_face_size = self.config.get('min_face_size', 48)
        self.optimal_face_size = self.config.get('optimal_face_size', 112)
        self.max_face_size = self.config.get('max_face_size', 300)
        
        logger.info("Face quality scorer initialized")
    
    def calculate_quality_score(self, face_img: np.ndarray, bbox: Tuple[int, int, int, int], 
                               detection_confidence: float = 1.0, landmarks=None) -> float:
        """
        Calculate comprehensive quality score for a face image
        
        Args:
            face_img (np.ndarray): Face image array
            bbox (tuple): Bounding box (x1, y1, x2, y2)
            detection_confidence (float): Detection confidence score
            landmarks (np.ndarray): Optional facial landmarks
            
        Returns:
            float: Quality score between 0 and 1
        """
        try:
            if face_img is None or face_img.size == 0:
                return 0.0
            
            # Calculate individual quality metrics
            size_score = self._calculate_size_score(bbox)
            clarity_score = self._calculate_clarity_score(face_img)
            frontality_score = self._calculate_frontality_score(face_img, landmarks)
            lighting_score = self._calculate_lighting_score(face_img)
            
            # Weighted combination
            quality_score = (
                self.weights['size'] * size_score +
                self.weights['clarity'] * clarity_score +
                self.weights['frontality'] * frontality_score +
                self.weights['lighting'] * lighting_score +
                self.weights['detection_conf'] * detection_confidence
            )
            
            # Ensure score is in valid range
            quality_score = max(0.0, min(1.0, quality_score))
            
            logger.debug(f"Face quality: size={size_score:.3f}, clarity={clarity_score:.3f}, "
                        f"frontality={frontality_score:.3f}, lighting={lighting_score:.3f}, "
                        f"final={quality_score:.3f}")
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error calculating face quality score: {e}")
            return 0.0
    
    def _calculate_size_score(self, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate quality score based on face size"""
        try:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            face_size = min(width, height)
            
            if face_size < self.min_face_size:
                return 0.0
            elif face_size >= self.optimal_face_size:
                # Perfect size or larger - gradually decrease for very large faces
                if face_size <= self.max_face_size:
                    return 1.0
                else:
                    # Penalize extremely large faces
                    penalty = min(0.5, (face_size - self.max_face_size) / self.max_face_size)
                    return max(0.5, 1.0 - penalty)
            else:
                # Scale linearly from min to optimal
                return (face_size - self.min_face_size) / (self.optimal_face_size - self.min_face_size)
                
        except Exception as e:
            logger.debug(f"Error calculating size score: {e}")
            return 0.0
    
    def _calculate_clarity_score(self, face_img: np.ndarray) -> float:
        """Calculate image clarity/sharpness score using Laplacian variance"""
        try:
            # Convert to grayscale if needed
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img
            
            # Calculate Laplacian variance (higher = sharper)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 range (empirically determined thresholds)
            min_sharpness = 50    # Very blurry
            max_sharpness = 1000  # Very sharp
            
            if laplacian_var <= min_sharpness:
                return 0.0
            elif laplacian_var >= max_sharpness:
                return 1.0
            else:
                return (laplacian_var - min_sharpness) / (max_sharpness - min_sharpness)
                
        except Exception as e:
            logger.debug(f"Error calculating clarity score: {e}")
            return 0.5  # Default middle value
    
    def _calculate_frontality_score(self, face_img: np.ndarray, landmarks=None) -> float:
        """Calculate frontality score (how straight-on the face is)"""
        try:
            if landmarks is not None and len(landmarks) >= 5:
                # Use landmarks for precise frontality calculation
                return self._frontality_from_landmarks(landmarks)
            else:
                # Fallback to image-based estimation
                return self._frontality_from_image(face_img)
                
        except Exception as e:
            logger.debug(f"Error calculating frontality score: {e}")
            return 0.7  # Default good value
    
    def _frontality_from_landmarks(self, landmarks: np.ndarray) -> float:
        """Calculate frontality from facial landmarks"""
        try:
            # Assuming 5-point landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
            if len(landmarks) < 5:
                return 0.7
            
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
            
            # Calculate eye distance ratio
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            nose_x = nose[0]
            
            # Perfect frontality would have nose centered between eyes
            eye_distance = abs(right_eye[0] - left_eye[0])
            nose_offset = abs(nose_x - eye_center_x)
            
            if eye_distance == 0:
                return 0.7
            
            # Normalize offset by eye distance
            offset_ratio = nose_offset / (eye_distance / 2)
            
            # Convert to frontality score (lower offset = higher score)
            frontality = max(0.0, 1.0 - offset_ratio)
            
            return min(1.0, frontality)
            
        except Exception as e:
            logger.debug(f"Error in landmark-based frontality: {e}")
            return 0.7
    
    def _frontality_from_image(self, face_img: np.ndarray) -> float:
        """Estimate frontality from image symmetry"""
        try:
            # Convert to grayscale
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img
            
            h, w = gray.shape
            
            # Split face into left and right halves
            mid = w // 2
            left_half = gray[:, :mid]
            right_half = gray[:, mid:]
            
            # Flip right half and resize to match left half
            right_flipped = cv2.flip(right_half, 1)
            
            # Resize if dimensions don't match
            if left_half.shape != right_flipped.shape:
                min_width = min(left_half.shape[1], right_flipped.shape[1])
                left_half = left_half[:, :min_width]
                right_flipped = right_flipped[:, :min_width]
            
            # Calculate correlation between halves
            correlation = cv2.matchTemplate(left_half, right_flipped, cv2.TM_CCOEFF_NORMED)
            symmetry_score = correlation[0, 0]
            
            # Normalize to 0-1 range
            return max(0.0, min(1.0, (symmetry_score + 1) / 2))
            
        except Exception as e:
            logger.debug(f"Error in image-based frontality: {e}")
            return 0.7
    
    def _calculate_lighting_score(self, face_img: np.ndarray) -> float:
        """Calculate lighting quality score"""
        try:
            # Convert to grayscale if needed
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img
            
            # Calculate statistics
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Optimal brightness range (0-255 scale)
            optimal_min = 60
            optimal_max = 200
            
            # Brightness score
            if optimal_min <= mean_brightness <= optimal_max:
                brightness_score = 1.0
            elif mean_brightness < optimal_min:
                brightness_score = mean_brightness / optimal_min
            else:  # mean_brightness > optimal_max
                brightness_score = max(0.0, 1.0 - (mean_brightness - optimal_max) / (255 - optimal_max))
            
            # Contrast score (standard deviation indicates contrast)
            optimal_contrast = 40  # Good contrast level
            contrast_score = min(1.0, std_brightness / optimal_contrast)
            
            # Combined lighting score
            lighting_score = (brightness_score + contrast_score) / 2
            
            return max(0.0, min(1.0, lighting_score))
            
        except Exception as e:
            logger.debug(f"Error calculating lighting score: {e}")
            return 0.5
    
    def should_update_face(self, current_quality: float, new_quality: float, 
                          improvement_threshold: float = 0.1) -> bool:
        """
        Determine if a new face snapshot should replace the current one
        
        Args:
            current_quality (float): Quality score of current face
            new_quality (float): Quality score of new face
            improvement_threshold (float): Minimum improvement needed
            
        Returns:
            bool: True if new face should replace current
        """
        return new_quality > current_quality + improvement_threshold
    
    def get_quality_grade(self, quality_score: float) -> str:
        """
        Convert quality score to human-readable grade
        
        Args:
            quality_score (float): Quality score 0-1
            
        Returns:
            str: Quality grade (A-F)
        """
        if quality_score >= 0.9:
            return "A"
        elif quality_score >= 0.8:
            return "B"
        elif quality_score >= 0.7:
            return "C"
        elif quality_score >= 0.6:
            return "D"
        elif quality_score >= 0.5:
            return "E"
        else:
            return "F" 