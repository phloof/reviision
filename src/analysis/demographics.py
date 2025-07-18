"""
Simplified Demographic Analyzer using InsightFace primary with DeepFace fallback
"""

import os
import time
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

logger = logging.getLogger(__name__)

class DemographicAnalyzer:
    """
    Simplified Demographic analyzer using InsightFace as primary with DeepFace as fallback
    
    This class uses InsightFace buffalo_l model as the primary method for demographic analysis,
    with DeepFace as a simple fallback when InsightFace fails or no face is detected.
    """
    
    def __init__(self, config):
        """
        Initialize the demographic analyzer with the provided configuration
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        
        # Basic parameters
        self.min_face_size = config.get('min_face_size', 32)
        self.detection_interval = config.get('detection_interval', 5)
        self.optimized = config.get('one_time_demographics', True)
        
        # Person demographics cache for optimization
        self.person_demographics = {}
        self.frame_count = 0
        
        # Initialize models
        self._init_insightface_model()
        
        logger.info(f"Simplified demographic analyzer initialized - InsightFace: {INSIGHTFACE_AVAILABLE}, DeepFace: {DEEPFACE_AVAILABLE}")
    
    def _init_insightface_model(self):
        """Initialize InsightFace buffalo_l model"""
        try:
            if not INSIGHTFACE_AVAILABLE:
                logger.warning("InsightFace not available. Will use DeepFace fallback only.")
                self.face_app = None
                return
                
            # Use buffalo_l model for best accuracy
            model_dir = self.config.get('model_dir', './models')
            self.face_app = FaceAnalysis(
                name='buffalo_l', 
                root=model_dir,
                allowed_modules=['detection', 'recognition', 'genderage']
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace buffalo_l model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace model: {e}")
            self.face_app = None

    def analyze(self, person_img, person_id=None):
        """
        Analyze demographics using InsightFace primary with DeepFace fallback
        
        Args:
            person_img (numpy.ndarray): Person image
            person_id (int): Optional person track ID for caching
            
        Returns:
            dict: Demographic information
        """
        try:
            if person_img is None or person_img.size == 0:
                logger.debug("No person image provided for demographic analysis")
                return self._get_default_demographics()
            
            # Check cache for optimization
            if person_id is not None and self.optimized and person_id in self.person_demographics:
                logger.debug(f"Using cached demographics for person {person_id}")
                return self.person_demographics[person_id]
            
            # Skip analysis based on detection interval
            self.frame_count += 1
            if person_id is not None and self.frame_count % self.detection_interval != 0:
                if person_id in self.person_demographics:
                    return self.person_demographics[person_id]
            
            # Try InsightFace first (primary method)
            demographics = None
            if self.face_app is not None:
                demographics = self._analyze_with_insightface(person_img)
                if demographics and demographics.get('confidence', 0) > 0:
                    logger.debug(f"InsightFace analysis successful: {demographics.get('gender', 'unknown')} {demographics.get('age', 'unknown')}")
            
            # Fallback to DeepFace if InsightFace failed
            if not demographics or demographics.get('confidence', 0) == 0:
                if DEEPFACE_AVAILABLE:
                    logger.debug("InsightFace failed, trying DeepFace fallback")
                    demographics = self._analyze_with_deepface(person_img)
                    if demographics and demographics.get('confidence', 0) > 0:
                        logger.debug(f"DeepFace analysis successful: {demographics.get('gender', 'unknown')} {demographics.get('age', 'unknown')}")
                else:
                    logger.debug("DeepFace not available, using default demographics")
            
            # Final fallback to default
            if not demographics or demographics.get('confidence', 0) == 0:
                demographics = self._get_default_demographics()
            
            # Cache result for optimization
            if person_id is not None:
                self.person_demographics[person_id] = demographics
                logger.debug(f"Cached demographics for person {person_id}")
            
            return demographics
            
        except Exception as e:
            logger.error(f"Error in demographic analysis: {e}")
            return self._get_default_demographics()

    def _analyze_with_insightface(self, img):
        """
        Analyze demographics using InsightFace buffalo_l model
        
        Args:
            img (numpy.ndarray): Input image
            
        Returns:
            dict: Demographic information or None if failed
        """
        try:
            if self.face_app is None:
                return None
            
            # Convert BGR to RGB for InsightFace
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            
            # Detect faces
            faces = self.face_app.get(img_rgb)
            
            if not faces:
                logger.debug("No faces detected by InsightFace")
                return None
            
            # Use the largest face
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            
            # Extract demographics
            age = int(face.age) if hasattr(face, 'age') and face.age is not None else None
            gender_code = getattr(face, 'gender', None)
            
            # Map gender code to text with better reliability
            if gender_code is not None:
                # InsightFace gender codes: 0 = female, 1 = male
                if gender_code == 1:
                    gender = 'male'
                elif gender_code == 0:
                    gender = 'female'
                else:
                    gender = 'unknown'
                    
                logger.debug(f"InsightFace gender detection: code={gender_code} -> {gender}")
            else:
                gender = 'unknown'
            
            # Use detection score as confidence with minimum threshold
            confidence = float(face.det_score) if hasattr(face, 'det_score') else 0.5
            # Boost confidence for successful face detection
            confidence = max(confidence, 0.4)  # Ensure minimum confidence for detected faces
            
            # Calculate age group
            age_group = self._calculate_age_group(age) if age is not None else 'unknown'
            
            return {
                'age': age or 30,
                'age_group': age_group,
                'gender': gender,
                'race': 'unknown',
                'emotion': 'neutral',
                'confidence': confidence,
                'timestamp': time.time(),
                'analysis_method': 'insightface',
                'det_score': confidence
            }
            
        except Exception as e:
            logger.warning(f"InsightFace analysis failed: {e}")
            return None
    
    def _analyze_with_deepface(self, img):
        """
        Analyze demographics using DeepFace as fallback
        
        Args:
            img (numpy.ndarray): Input image
            
        Returns:
            dict: Demographic information or None if failed
        """
        try:
            if not DEEPFACE_AVAILABLE:
                return None
            
            # DeepFace analysis
            results = DeepFace.analyze(
                img_path=img,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            
            # Process results
            if isinstance(results, list):
                result = results[0]
            else:
                result = results
            
            age = result.get('age', 30)
            gender_result = result.get('dominant_gender', 'unknown')
            emotion = result.get('dominant_emotion', 'neutral')
            
            # Normalize gender result to match our format
            gender = gender_result.lower() if gender_result else 'unknown'
            if gender in ['man', 'male']:
                gender = 'male'
            elif gender in ['woman', 'female']:
                gender = 'female'
            else:
                gender = 'unknown'
                
            logger.debug(f"DeepFace gender detection: raw='{gender_result}' -> normalized='{gender}'")
            
            # Calculate age group
            age_group = self._calculate_age_group(age)
            
            # Improved confidence based on successful analysis and gender confidence
            base_confidence = 0.6  # Default confidence for DeepFace
            gender_confidence = result.get('gender', {}).get(gender_result, 0) if isinstance(result.get('gender', {}), dict) else 0
            
            # Combine base confidence with gender-specific confidence
            if gender_confidence > 0:
                confidence = min(0.9, base_confidence + (gender_confidence / 100 * 0.3))
            else:
                confidence = base_confidence
            
            return {
                'age': int(age),
                'age_group': age_group,
                'gender': gender,
                'race': 'unknown',
                'emotion': emotion,
                'confidence': confidence,
                'timestamp': time.time(),
                'analysis_method': 'deepface',
                'det_score': confidence
            }
            
        except Exception as e:
            logger.warning(f"DeepFace analysis failed: {e}")
            return None
    
    def _calculate_age_group(self, age):
        """Calculate age group from age"""
        if age is None:
            return 'unknown'
        
        if age < 18:
            return 'under_18'
        elif age < 25:
            return '18_24'
        elif age < 35:
            return '25_34'
        elif age < 45:
            return '35_44'
        elif age < 55:
            return '45_54'
        elif age < 65:
            return '55_64'
        else:
            return '65_plus'
    
    def _get_default_demographics(self):
        """Get default demographics when all analysis methods fail"""
        return {
            'age': 30,
            'age_group': '25_34',
            'gender': 'unknown',
            'race': 'unknown',
            'emotion': 'neutral',
            'confidence': 0.1,  # Low confidence for defaults
            'timestamp': time.time(),
            'analysis_method': 'default',
            'det_score': 0.1
        }
    
    def get_person_demographics(self, person_id):
        """Get stored demographics for a person"""
        return self.person_demographics.get(person_id, self._get_default_demographics())
    
    def update_person_demographics(self, person_id, new_demographics):
        """Update stored demographics for a person"""
        if person_id in self.person_demographics:
            existing = self.person_demographics[person_id]
            # Only update if new data has higher confidence
            if new_demographics.get('confidence', 0) > existing.get('confidence', 0):
                self.person_demographics[person_id] = new_demographics
                logger.debug(f"Updated demographics for person {person_id}")
        else:
            self.person_demographics[person_id] = new_demographics 