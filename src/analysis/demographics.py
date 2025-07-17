"""
Demographic Analyzer class for Retail Analytics System
"""

import os
import time
import logging
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from deepface import DeepFace
import numpy as np
from scipy.spatial.distance import cosine
from collections import deque
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

logger = logging.getLogger(__name__)

class DemographicAnalyzer:
    """
    Demographic analyzer that detects age, gender, race, and emotion from faces
    
    This class uses deep learning models to extract demographic information
    from detected faces in the retail environment.
    """
    
    def __init__(self, config):
        """
        Initialize the demographic analyzer with the provided configuration
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        
        # Analysis parameters
        self.min_face_size = config.get('min_face_size', 50)
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.detection_interval = config.get('detection_interval', 30)  # Frames between analysis
        self.analysis_distance = config.get('analysis_distance', 5)  # Don't re-analyze same person
        self.use_insightface = config.get('use_insightface', True)  # Use InsightFace for face detection
        self.optimized = config.get('one_time_demographics', True)
        self.embedding_threshold = config.get('embedding_similarity_threshold', 0.4)  # Stricter default
        self.embedding_normalize = config.get('embedding_normalize', True)
        self.embedding_window_size = config.get('embedding_window_size', 100)
        self.embeddings = deque(maxlen=self.embedding_window_size)  # rolling window: list of dicts {person_id, embedding, demographics}
        self.face_match_backend = config.get('face_match_backend', 'model')
        self.face_match_threshold = config.get('face_match_threshold', 0.8)
        
        # Tracking parameters
        self.person_demographics = {}  # Dictionary to store demographics by person ID
        self.frame_count = 0
        
        # Initialize face detection/analysis models
        self._init_models()
        
        logger.info(f"Demographic analyzer initialized")
    
    def _init_models(self):
        """Initialize face detection and analysis models"""
        try:
            # Initialize face_app to None by default
            self.face_app = None
            
            if self.use_insightface and INSIGHTFACE_AVAILABLE:
                # InsightFace for face detection and recognition
                self.face_app = FaceAnalysis(name='buffalo_l', 
                                           root=self.config.get('model_dir', './models'),
                                           allowed_modules=['detection', 'recognition'])
                self.face_app.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("InsightFace models loaded successfully")
            
            # Set up DeepFace models
            models = {
                'emotion': self.config.get('emotion_model', 'Emotion'),
                'age': self.config.get('age_model', 'Age'),
                'gender': self.config.get('gender_model', 'Gender'),
                'race': self.config.get('race_model', 'Race')
            }
            
            # Pre-load models
            backends = {
                'emotion': self.config.get('emotion_backend', 'opencv'),
                'age': self.config.get('age_backend', 'opencv'),
                'gender': self.config.get('gender_backend', 'opencv'), 
                'race': self.config.get('race_backend', 'opencv')
            }
            
            self.models = models
            self.backends = backends
            
            logger.info("DeepFace models configured successfully")
            
        except Exception as e:
            logger.error(f"Error initializing demographic analysis models: {e}")
            raise
    
    def process(self, frame, tracks):
        """
        Process face tracks to extract demographic information for tracked faces
        
        Args:
            frame (numpy.ndarray): Current video frame
            tracks (list): List of face tracks with IDs, bounding boxes, and quality scores
            
        Returns:
            dict: Updated demographic information by person ID
        """
        if frame is None or not tracks:
            return self.person_demographics
        
        # Skip frames to improve performance
        self.frame_count += 1
        if self.frame_count % self.detection_interval != 0:
            return self.person_demographics
        
        try:
            # Process each tracked face
            for track in tracks:
                person_id = track['id']
                bbox = track['bbox']
                quality_score = track.get('quality_score', 1.0)
                
                # Skip if we already have good demographic data for this person
                if person_id in self.person_demographics and self.person_demographics[person_id].get('confidence', 0) > 0.9:
                    continue
                
                # Skip low quality faces
                if quality_score < self.confidence_threshold:
                    continue
                
                # Extract face image directly from face bounding box
                face_img = self._extract_face_from_bbox(frame, bbox)
                if face_img is None:
                    continue
                
                # Skip small faces
                h, w = face_img.shape[:2]
                if h < self.min_face_size or w < self.min_face_size:
                    continue
                
                # Analyze demographics
                demographics = self.analyze(face_img, person_id)
                if demographics:
                    # Update or create demographic entry
                    if person_id not in self.person_demographics:
                        self.person_demographics[person_id] = demographics
                    else:
                        # Average with existing data for smoother results
                        self._update_demographics(person_id, demographics)
            
            return self.person_demographics
            
        except Exception as e:
            logger.error(f"Error in demographic analysis: {e}")
            return self.person_demographics
    
    def _extract_face_from_bbox(self, frame, bbox):
        """
        Extract face image directly from face bounding box
        
        Args:
            frame (numpy.ndarray): Current video frame
            bbox (tuple): Face bounding box (x1, y1, x2, y2)
            
        Returns:
            numpy.ndarray: Face image or None if extraction fails
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # Check if bounding box is valid
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Extract face region directly
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return None
                
            return face_img
                
        except Exception as e:
            logger.error(f"Error extracting face from bbox: {e}")
            return None

    def _extract_face(self, frame, bbox):
        """
        Legacy method for extracting face from person bounding box (kept for compatibility)
        
        Args:
            frame (numpy.ndarray): Current video frame
            bbox (tuple): Bounding box (x1, y1, x2, y2)
            
        Returns:
            numpy.ndarray: Face image or None if no face found
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Extract person region with margin
            margin = int((y2 - y1) * 0.1)  # 10% margin
            y_start = max(0, y1 - margin)
            y_end = min(frame.shape[0], y2 + margin)
            x_start = max(0, x1 - margin)
            x_end = min(frame.shape[1], x2 + margin)
            
            person_img = frame[y_start:y_end, x_start:x_end]
            
            if self.use_insightface:
                # Use InsightFace for face detection
                faces = self.face_app.get(person_img)
                if faces:
                    # Sort by face size and take the largest
                    faces = sorted(faces, key=lambda x: x.bbox[2] * x.bbox[3], reverse=True)
                    bbox = faces[0].bbox.astype(int)
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    face_img = person_img[y1:y2, x1:x2]
                    return face_img
                else:
                    # Fallback: assume upper 1/3 of person is face
                    h = y_end - y_start
                    face_img = person_img[:h//3, :]
                    return face_img
            else:
                # Fallback: assume upper 1/3 of person is face
                h = y_end - y_start
                face_img = person_img[:h//3, :]
                return face_img
                
        except Exception as e:
            logger.error(f"Error extracting face: {e}")
            return None
    
    def _analyze_demographics(self, face_img):
        """
        Enhanced demographic analysis with ensemble age prediction for improved accuracy
        
        Args:
            face_img (numpy.ndarray): Face image
            
        Returns:
            dict: Demographic information (age, gender, race, emotion) with improved age accuracy
        """
        try:
            start_time = time.time()
            
            # Ensemble age prediction with multiple backends for better accuracy
            age_predictions = []
            primary_result = None
            
            # Try primary backend (dlib) for enhanced age estimation
            try:
                primary_results = DeepFace.analyze(
                    img_path=face_img,
                    actions=['age', 'gender', 'race', 'emotion'],
                    enforce_detection=False,
                    detector_backend=self.backends.get('age', 'dlib'),
                    silent=True
                )
                primary_result = primary_results[0] if isinstance(primary_results, list) else primary_results
                if 'age' in primary_result:
                    age_predictions.append(primary_result['age'])
                logger.debug(f"Primary age prediction: {primary_result.get('age', 'N/A')}")
            except Exception as e:
                logger.debug(f"Primary backend failed: {e}")
            
            # Try fallback backend for additional age estimation
            fallback_backend = self.config.get('fallback_age_backend', 'mtcnn')
            if fallback_backend and fallback_backend != self.backends.get('age'):
                try:
                    fallback_results = DeepFace.analyze(
                        img_path=face_img,
                        actions=['age'],
                        enforce_detection=False,
                        detector_backend=fallback_backend,
                        silent=True
                    )
                    fallback_result = fallback_results[0] if isinstance(fallback_results, list) else fallback_results
                    if 'age' in fallback_result:
                        age_predictions.append(fallback_result['age'])
                    logger.debug(f"Fallback age prediction: {fallback_result.get('age', 'N/A')}")
                except Exception as e:
                    logger.debug(f"Fallback backend failed: {e}")
            
            # If no predictions, use original backend as last resort
            if not age_predictions and not primary_result:
                try:
                    original_results = DeepFace.analyze(
                        img_path=face_img,
                        actions=['age', 'gender', 'race', 'emotion'],
                        enforce_detection=False,
                        detector_backend='opencv',  # Original fallback
                        silent=True
                    )
                    primary_result = original_results[0] if isinstance(original_results, list) else original_results
                    if 'age' in primary_result:
                        age_predictions.append(primary_result['age'])
                except Exception as e:
                    logger.warning(f"All age prediction backends failed: {e}")
                    raise e
            
            # Calculate ensemble age prediction
            if age_predictions:
                # Remove outliers and calculate weighted average
                valid_ages = [age for age in age_predictions if isinstance(age, (int, float)) and 1 <= age <= 100]
                if valid_ages:
                    if len(valid_ages) == 1:
                        final_age = valid_ages[0]
                    else:
                        # Weighted average with outlier removal
                        ages_array = np.array(valid_ages)
                        mean_age = np.mean(ages_array)
                        std_age = np.std(ages_array)
                        # Remove outliers beyond 1.5 standard deviations
                        filtered_ages = ages_array[np.abs(ages_array - mean_age) <= 1.5 * std_age]
                        final_age = np.mean(filtered_ages) if len(filtered_ages) > 0 else mean_age
                    
                    final_age = max(1, min(100, float(final_age)))
                    logger.debug(f"Ensemble age prediction: {final_age} (from {len(valid_ages)} models)")
                else:
                    final_age = 25.0  # Default fallback
            else:
                final_age = float(primary_result.get('age', 25)) if primary_result else 25.0
            
            # Use primary result for other attributes or fallback
            if not primary_result:
                logger.warning("No successful analysis results, using fallback")
                return {
                    'age': final_age,
                    'age_group': self._calculate_age_group(final_age),
                    'gender': 'unknown',
                    'race': 'unknown',
                    'emotion': 'neutral',
                    'confidence': 0.1,
                    'analysis_method': 'fallback_enhanced'
                }
            
            # Handle gender parsing robustly
            gender = primary_result.get('dominant_gender', primary_result.get('gender', 'unknown'))
            if isinstance(gender, dict):
                gender = max(gender.keys(), key=lambda k: gender[k]) if gender else 'unknown'
            
            # Calculate confidence based on age prediction consistency
            base_confidence = 0.7
            if len(age_predictions) > 1:
                ages_std = np.std(age_predictions) if age_predictions else 0
                if ages_std < 5:  # Ages within 5 years are considered consistent
                    base_confidence = min(1.0, base_confidence + 0.1)
            
            demographics = {
                'age': final_age,
                'gender': gender,
                'race': primary_result.get('dominant_race', 'unknown'),
                'race_scores': primary_result.get('race', {}),
                'emotion': primary_result.get('dominant_emotion', 'neutral'),
                'emotion_scores': primary_result.get('emotion', {}),
                'confidence': base_confidence,
                'timestamp': time.time(),
                'analysis_count': 1,
                'analysis_method': 'enhanced_ensemble',
                'age_predictions': age_predictions,  # For debugging
                'age_consistency': len(age_predictions) > 1 and np.std(age_predictions) < 5 if age_predictions else False
            }
            
            # Calculate age group from enhanced age
            demographics['age_group'] = self._calculate_age_group(demographics['age'])
            
            elapsed_time = time.time() - start_time
            logger.debug(f"Enhanced demographic analysis completed in {elapsed_time:.4f} seconds (ensemble age: {demographics['age']})")
            
            return demographics
            
        except Exception as e:
            logger.warning(f"Error in enhanced demographics analysis: {e}")
            # Return fallback demographics instead of None
            return {
                'age': 25,
                'age_group': '25-34',
                'gender': 'unknown',
                'race': 'unknown',
                'emotion': 'neutral',
                'confidence': 0.1,
                'analysis_method': 'error_fallback'
            }
    
    def _update_demographics(self, person_id, new_data):
        """
        Update demographic information for a person with new data
        
        Args:
            person_id (int): Person track ID
            new_data (dict): New demographic information
        """
        existing = self.person_demographics[person_id]
        count = existing['analysis_count']
        new_count = count + 1
        
        # Simple weighted average for age
        age_weight = 0.7  # Weight for existing data
        existing['age'] = (existing['age'] * age_weight) + (new_data['age'] * (1 - age_weight))
        
        # Keep most confident gender
        if new_data.get('confidence', 0) > existing.get('confidence', 0):
            existing['gender'] = new_data['gender']
            existing['confidence'] = new_data['confidence']
        
        # Average race and emotion scores
        for race, score in new_data['race_scores'].items():
            if 'race_scores' not in existing:
                existing['race_scores'] = {}
            if race not in existing['race_scores']:
                existing['race_scores'][race] = score
            else:
                existing['race_scores'][race] = (existing['race_scores'][race] * count + score) / new_count
        
        for emotion, score in new_data['emotion_scores'].items():
            if 'emotion_scores' not in existing:
                existing['emotion_scores'] = {}
            if emotion not in existing['emotion_scores']:
                existing['emotion_scores'][emotion] = score
            else:
                existing['emotion_scores'][emotion] = (existing['emotion_scores'][emotion] * count + score) / new_count
        
        # Update dominant race and emotion
        if 'race_scores' in existing:
            existing['race'] = max(existing['race_scores'].items(), key=lambda x: x[1])[0]
        
        if 'emotion_scores' in existing:
            existing['emotion'] = max(existing['emotion_scores'].items(), key=lambda x: x[1])[0]
        
        # Update count and timestamp
        existing['analysis_count'] = new_count
        existing['timestamp'] = time.time()
    
    def get_demographics(self, person_id=None):
        """
        Get demographic information for a specific person or all persons
        
        Args:
            person_id (int, optional): Person track ID
            
        Returns:
            dict: Demographic information for the person or all persons
        """
        if person_id is not None:
            return self.person_demographics.get(person_id, {})
        return self.person_demographics
    
    def get_aggregate_demographics(self):
        """
        Get aggregate demographic statistics across all tracked persons
        
        Returns:
            dict: Aggregate demographic statistics
        """
        if not self.person_demographics:
            return {}
        
        stats = {
            'total_persons': len(self.person_demographics),
            'age': {
                'avg': 0,
                'min': float('inf'),
                'max': 0
            },
            'gender': {
                'Male': 0,
                'Female': 0
            },
            'race': defaultdict(int),
            'emotion': defaultdict(int)
        }
        
        # Calculate statistics
        for person_id, demo in self.person_demographics.items():
            # Age stats
            age = demo.get('age', 0)
            stats['age']['avg'] += age
            stats['age']['min'] = min(stats['age']['min'], age)
            stats['age']['max'] = max(stats['age']['max'], age)
            
            # Gender stats
            gender = demo.get('gender')
            if gender:
                stats['gender'][gender] += 1
            
            # Race stats
            race = demo.get('race')
            if race:
                stats['race'][race] += 1
            
            # Emotion stats
            emotion = demo.get('emotion')
            if emotion:
                stats['emotion'][emotion] += 1
        
        # Calculate averages
        if stats['total_persons'] > 0:
            stats['age']['avg'] /= stats['total_persons']
        
        # Convert defaultdicts to regular dicts
        stats['race'] = dict(stats['race'])
        stats['emotion'] = dict(stats['emotion'])
        
        # Add percentages
        total = stats['total_persons']
        if total > 0:
            stats['gender_pct'] = {k: (v / total) * 100 for k, v in stats['gender'].items()}
            stats['race_pct'] = {k: (v / total) * 100 for k, v in stats['race'].items()}
            stats['emotion_pct'] = {k: (v / total) * 100 for k, v in stats['emotion'].items()}
        
        return stats
    
    def clear_old_data(self, max_age_seconds=300):
        """
        Clear demographic data older than the specified age
        
        Args:
            max_age_seconds (int): Maximum age in seconds before data is cleared
        """
        current_time = time.time()
        to_remove = []
        
        for person_id, demo in self.person_demographics.items():
            if current_time - demo.get('timestamp', 0) > max_age_seconds:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.person_demographics[person_id]
        
        if to_remove:
            logger.debug(f"Cleared demographic data for {len(to_remove)} persons")
    
    def visualize_demographics(self, frame, tracks):
        """
        Visualize demographic information on the frame
        
        Args:
            frame (numpy.ndarray): Current video frame
            tracks (list): List of face tracks with IDs and quality scores
            
        Returns:
            numpy.ndarray: Frame with demographic information visualized
        """
        if frame is None or not tracks:
            return frame
        
        # Create a copy of the frame
        output_frame = frame.copy()
        
        # Draw demographic information for each face
        for track in tracks:
            person_id = track['id']
            bbox = track['bbox']
            quality_score = track.get('quality_score', 1.0)
            
            # Skip if no demographic data
            if person_id not in self.person_demographics:
                continue
            
            demo = self.person_demographics[person_id]
            
            # Create label with demographic info
            label = f"ID: {person_id} | "
            label += f"Age: {int(demo.get('age', 0))} | "
            label += f"{demo.get('gender', 'Unknown')} | "
            label += f"{demo.get('race', 'Unknown')} | "
            label += f"{demo.get('emotion', 'Unknown')}"
            label += f" | Q: {quality_score:.2f}"
            
            # Get position for text (above the face)
            text_x = bbox[0]
            text_y = bbox[1] - 10
            
            # Color based on quality and demographic confidence
            demo_confidence = demo.get('confidence', 0.5)
            if quality_score > 0.7 and demo_confidence > 0.8:
                color = (0, 255, 0)  # Green for high quality
            elif quality_score > 0.4 and demo_confidence > 0.5:
                color = (0, 255, 255)  # Yellow for medium quality
            else:
                color = (0, 0, 255)  # Red for low quality
            
            # Draw background rectangle for text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(output_frame, 
                          (text_x, text_y - text_size[1] - 10), 
                          (text_x + text_size[0], text_y), 
                          (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(output_frame, label, (text_x, text_y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return output_frame 

    def get_face_embedding(self, face_img):
        if not self.face_app:
            return None
        faces = self.face_app.get(face_img)
        if faces:
            emb = faces[0].embedding
            if self.embedding_normalize:
                emb = emb / np.linalg.norm(emb)
            return emb
        return None

    def _verify_faces(self, face_img1, face_img2):
        # Use InsightFace verification if available
        if hasattr(self, 'face_app') and hasattr(self.face_app, 'model') and hasattr(self.face_app.model, 'verify'):
            try:
                # Returns (is_same, score)
                is_same, score = self.face_app.model.verify(face_img1, face_img2)
                return score
            except Exception as e:
                logger.warning(f"InsightFace verification failed: {e}")
                return 0.0
        # Fallback: always return 0.0 (not same)
        return 0.0

    def find_matching_person(self, embedding, face_img):
        best_pid = None
        best_score = float('-inf') if self.face_match_backend == 'model' else float('inf')
        for entry in self.embeddings:
            if self.face_match_backend == 'model':
                # Use dedicated face verification model
                score = self._verify_faces(face_img, entry['demographics'].get('face_img'))
                if score > self.face_match_threshold and score > best_score:
                    best_score = score
                    best_pid = entry['person_id']
            else:
                # Use cosine similarity
                score = 1 - cosine(embedding, entry['embedding'])
                if score > self.face_match_threshold and score > best_score:
                    best_score = score
                    best_pid = entry['person_id']
        if best_pid is not None:
            logger.info(f"ReID: Matched to person {best_pid} with similarity {best_score:.3f} using {self.face_match_backend}")
        return best_pid

    def analyze(self, face_img, person_id=None):
        if not self.optimized or not self.face_app:
            demographics = self._analyze_demographics(face_img)
            return demographics if demographics is not None else self._get_fallback_demographics_with_quality(0.0)
        
        embedding = self.get_face_embedding(face_img)
        if embedding is None:
            demographics = self._analyze_demographics(face_img)
            return demographics if demographics is not None else self._get_fallback_demographics_with_quality(0.0)
        
        match_id = self.find_matching_person(embedding, face_img)
        if match_id is not None:
            logger.info(f"ReID: Duplicate detected, merging with person {match_id}")
            # Safely find matching demographics
            matching_entry = next((e for e in self.embeddings if e['person_id'] == match_id), None)
            if matching_entry and 'demographics' in matching_entry:
                return matching_entry['demographics']
            else:
                logger.warning(f"Could not find demographics for matched person {match_id}")
                
        demographics = self._analyze_demographics(face_img)
        if demographics is None:
            demographics = self._get_fallback_demographics_with_quality(0.0)
            
        if person_id is None:
            person_id = len(self.embeddings) + 1
        
        # Ensure demographics is a valid dict before adding face_img
        if isinstance(demographics, dict):
            demographics['face_img'] = face_img  # Store for future verification
            self.embeddings.append({'person_id': person_id, 'embedding': embedding, 'demographics': demographics})
        else:
            logger.error(f"Demographics is not a dict: {type(demographics)}")
            demographics = self._get_fallback_demographics_with_quality(0.0)
            
        return demographics
    
    def analyze_with_quality_check(self, person_id, face_img, bbox, detection_confidence, 
                                  landmarks=None, database=None, face_snapshot_manager=None):
        """
        Enhanced analyze method with quality assessment and one-time demographic analysis
        
        Args:
            person_id (int): Person track ID
            face_img (np.ndarray): Face image
            bbox (tuple): Face bounding box
            detection_confidence (float): Detection confidence
            landmarks: Optional facial landmarks
            database: Database connection for checking existing demographics
            face_snapshot_manager: Face snapshot manager for quality assessment
            
        Returns:
            dict: Demographic analysis results with quality info
        """
        try:
            # Check if person already has good demographics
            if database and database.has_good_demographics(person_id):
                logger.debug(f"Person {person_id} already has good demographics, skipping analysis")
                # Return cached demographics from database
                cached_demographics = self._get_cached_demographics_from_db(person_id, database)
                if cached_demographics:
                    cached_demographics['analysis_method'] = 'cached'
                    return cached_demographics
            
            # Process face snapshot and check quality
            if face_snapshot_manager:
                snapshot_result = face_snapshot_manager.process_face_snapshot(
                    person_id, face_img, bbox, detection_confidence, landmarks
                )
                
                quality_score = snapshot_result.get('quality_score', 0.0)
                
                # Only analyze if quality is good enough
                if quality_score < face_snapshot_manager.min_quality_threshold:
                    logger.debug(f"Face quality too low for analysis: {quality_score:.3f}")
                    return self._get_fallback_demographics_with_quality(quality_score)
                
                # Perform demographic analysis only on high-quality faces
                demographics = self.analyze(face_img, person_id)
                
                # Ensure demographics is a valid dict
                if not isinstance(demographics, dict):
                    logger.error(f"analyze() returned invalid type: {type(demographics)}")
                    demographics = self._get_fallback_demographics_with_quality(quality_score)
                
                # Add quality information to demographics
                demographics['quality_score'] = quality_score
                demographics['quality_grade'] = face_snapshot_manager.quality_scorer.get_quality_grade(quality_score)
                demographics['face_stored'] = snapshot_result.get('stored', False)
                demographics['is_primary_face'] = snapshot_result.get('is_primary', False)
                demographics['analysis_method'] = 'enhanced_quality'
                
                return demographics
            else:
                # Fallback to standard analysis
                demographics = self.analyze(face_img, person_id)
                
                # Ensure demographics is a valid dict
                if not isinstance(demographics, dict):
                    logger.error(f"analyze() returned invalid type: {type(demographics)}")
                    demographics = self._get_fallback_demographics_with_quality(0.0)
                
                demographics['analysis_method'] = 'standard'
                return demographics
                
        except Exception as e:
            logger.error(f"Error in enhanced demographic analysis: {e}")
            return self._get_fallback_demographics_with_quality(0.0)
    
    def _get_cached_demographics_from_db(self, person_id, database):
        """Get cached demographics from database"""
        try:
            # Get the most recent high-confidence demographics
            demographics_records = database.get_demographics(person_id=person_id)
            if not demographics_records:
                return None
            
            # Find the best record
            best_record = None
            best_confidence = 0.0
            
            for record in demographics_records:
                confidence = record.get('confidence', 0.0) if isinstance(record, dict) else record[9]  # confidence column
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_record = record
            
            if best_record:
                if isinstance(best_record, dict):
                    return {
                        'age': best_record.get('age'),
                        'age_group': self._calculate_age_group(best_record.get('age', 25)),
                        'gender': best_record.get('gender', 'unknown'),
                        'race': best_record.get('race', 'unknown'),
                        'emotion': best_record.get('emotion', 'neutral'),
                        'confidence': best_record.get('confidence', 0.0),
                        'analysis_method': 'database_cached'
                    }
                else:
                    # Handle tuple format from database
                    age = best_record[4] if len(best_record) > 4 else 25
                    return {
                        'age': age,
                        'age_group': self._calculate_age_group(age),
                        'gender': 'unknown',  # Would need join to lookup tables
                        'race': 'unknown',
                        'emotion': 'neutral',
                        'confidence': best_confidence,
                        'analysis_method': 'database_cached'
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached demographics from database: {e}")
            return None
    
    def _get_fallback_demographics_with_quality(self, quality_score):
        """Get fallback demographics with quality information"""
        return {
            'age': 25,
            'age_group': '25-34',
            'gender': 'unknown',
            'race': 'unknown',
            'emotion': 'neutral',
            'confidence': 0.1,
            'quality_score': quality_score,
            'analysis_method': 'fallback'
        }
    
    def _calculate_age_group(self, age):
        """Calculate age group from age"""
        if age < 13:
            return "0-12"
        elif age < 18:
            return "13-17"
        elif age < 25:
            return "18-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        elif age < 55:
            return "45-54"
        elif age < 65:
            return "55-64"
        else:
            return "65+" 