"""
Web services module for ReViision
Contains business logic for web routes to keep them clean
"""

import logging
import cv2
import numpy as np
import base64
import os
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    from filterpy.kalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False

# Import advanced demographic analysis
try:
    from deepface import DeepFace
    from insightface.app import FaceAnalysis
    ADVANCED_MODELS_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Advanced demographic models (DeepFace, InsightFace) loaded successfully")
except ImportError as e:
    ADVANCED_MODELS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Advanced demographic models not available: {e}")

logger = logging.getLogger(__name__)


class EnhancedDemographicAnalyzer:
    """
    Enhanced demographic analyzer using DeepFace and InsightFace for accurate analysis
    """
    
    def __init__(self):
        self.models_loaded = False
        self.face_app = None
        self.deepface_models = {
            'emotion': 'emotion',
            'age': 'age', 
            'gender': 'gender',
            'race': 'race'
        }
        
        if ADVANCED_MODELS_AVAILABLE:
            self._load_models()
    
    def _load_models(self):
        """Load InsightFace and DeepFace models"""
        try:
            # Load InsightFace buffalo_l model
            model_dir = Path('./models')
            if not model_dir.exists():
                model_dir.mkdir(parents=True)
                
            self.face_app = FaceAnalysis(
                name='buffalo_l',
                root=str(model_dir),
                allowed_modules=['detection', 'recognition']
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            
            self.models_loaded = True
            logger.info("Enhanced demographic models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading enhanced demographic models: {e}")
            self.models_loaded = False
    
    def analyze_demographics(self, person_img: np.ndarray) -> Dict[str, Any]:
        """
        Analyze demographics using advanced models
        
        Args:
            person_img: Person image array
            
        Returns:
            Dictionary with demographic information
        """
        if not ADVANCED_MODELS_AVAILABLE:
            logger.debug("Advanced models not available, using fallback")
            return self._fallback_analysis(person_img)
            
        if not self.models_loaded:
            logger.debug("Models not loaded, using fallback")
            return self._fallback_analysis(person_img)
        
        if person_img is None or person_img.size == 0:
            logger.debug("No person image provided, using fallback")
            return self._fallback_analysis(person_img)
        
        logger.debug(f"Starting demographic analysis on image: {person_img.shape}")
        
        try:
            # Extract face using InsightFace
            face_img = self._extract_face_insightface(person_img)
            if face_img is None:
                logger.debug("Face extraction failed, using fallback analysis")
                return self._fallback_analysis(person_img)
            
            logger.debug(f"Face extracted successfully, analyzing with DeepFace: {face_img.shape}")
            
            # Analyze with DeepFace
            demographics = self._analyze_with_deepface(face_img)
            
            logger.debug(f"DeepFace analysis completed: {demographics.get('gender', 'unknown')} {demographics.get('age_group', 'unknown')} conf: {demographics.get('confidence', 0):.2f}")
            
            return demographics
            
        except Exception as e:
            logger.warning(f"Error in enhanced demographic analysis: {e}")
            return self._fallback_analysis(person_img)
    
    def _enhance_image_quality(self, img: np.ndarray) -> np.ndarray:
        """Enhance image quality for better face analysis"""
        try:
            # Apply denoising
            enhanced = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            
            # Improve contrast and brightness
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            lab_planes = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab_planes[0] = clahe.apply(lab_planes[0])
            enhanced = cv2.merge(lab_planes)
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Sharpen image
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            return enhanced
        except Exception as e:
            logger.debug(f"Image enhancement failed: {e}")
            return img

    def _extract_face_insightface(self, person_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract face using InsightFace detection with quality enhancement"""
        try:
            if self.face_app is None:
                logger.debug("InsightFace face_app not available")
                return None
            
            # Ensure minimum image size
            if person_img.shape[0] < 32 or person_img.shape[1] < 32:
                logger.debug(f"Person image too small for face detection: {person_img.shape}")
                return None
            
            # Enhance image quality before detection
            enhanced_img = self._enhance_image_quality(person_img)
                
            # Try face detection on enhanced image first
            faces = self.face_app.get(enhanced_img)
            if not faces:
                # Fallback to original image
                logger.debug("No faces on enhanced image, trying original")
                faces = self.face_app.get(person_img)
                source_img = person_img
            else:
                source_img = enhanced_img
                
            if faces:
                # Sort by face size and take the largest
                faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
                bbox = faces[0].bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                
                # Add margin around face for better context
                margin = int((y2 - y1) * 0.15)  # 15% margin
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin) 
                x2 = min(source_img.shape[1], x2 + margin)
                y2 = min(source_img.shape[0], y2 + margin)
                
                if x2 <= x1 or y2 <= y1:
                    logger.debug("Invalid face bbox coordinates")
                    return None
                
                face_img = source_img[y1:y2, x1:x2]
                
                # Resize face to optimal size for DeepFace
                if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                    # Resize to standard size for consistent analysis
                    face_img = cv2.resize(face_img, (112, 112))  # Standard face recognition size
                    logger.debug(f"Face extracted and resized: {face_img.shape}")
                    return face_img
                else:
                    logger.debug(f"Extracted face invalid: {face_img.shape}")
                    
        except Exception as e:
            logger.warning(f"InsightFace face extraction failed: {e}")
            
        return None
    
    def _analyze_with_deepface(self, face_img: np.ndarray) -> Dict[str, Any]:
        """Analyze face with DeepFace"""
        try:
            # Ensure minimum face size for DeepFace
            h, w = face_img.shape[:2]
            if h < 48 or w < 48:
                # Resize to minimum required size
                face_img = cv2.resize(face_img, (48, 48))
            
            # Analyze with DeepFace
            results = DeepFace.analyze(
                img_path=face_img,
                actions=['age', 'gender', 'race', 'emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            
            # Process results
            if isinstance(results, list):
                result = results[0]
            else:
                result = results
            
            # Extract dominant attributes with better parsing
            age = result.get('age', 25)
            
            # Handle gender parsing more robustly
            gender = result.get('dominant_gender', 'unknown')
            if gender == 'unknown':
                # Try alternative gender field parsing
                gender_data = result.get('gender', {})
                if isinstance(gender_data, dict) and gender_data:
                    # Find the gender with highest confidence
                    gender = max(gender_data.keys(), key=lambda k: gender_data[k])
                elif isinstance(gender_data, str):
                    gender = gender_data
            
            race = result.get('dominant_race', 'unknown')
            emotion = result.get('dominant_emotion', 'neutral')
            
            # Calculate age group
            age_group = self._calculate_age_group(age)
            
            # Calculate confidence (use gender confidence as overall indicator)
            gender_scores = result.get('gender', {})
            if isinstance(gender_scores, dict) and gender_scores:
                confidence = max(gender_scores.values()) / 100.0 if max(gender_scores.values()) > 1 else max(gender_scores.values())
                confidence = min(1.0, max(0.1, confidence))  # Ensure 0.1-1.0 range
            else:
                confidence = 0.7
            
            return {
                'age': age,
                'age_group': age_group,
                'gender': gender.lower(),
                'race': race,
                'emotion': emotion,
                'confidence': confidence,
                'analysis_method': 'deepface_insightface'
            }
            
        except Exception as e:
            logger.warning(f"DeepFace analysis failed: {e}")
            return self._fallback_analysis(None)
    
    def _calculate_age_group(self, age: int) -> str:
        """Calculate age group from numeric age"""
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
    
    def _fallback_analysis(self, person_img: Optional[np.ndarray]) -> Dict[str, Any]:
        """Fallback to basic analysis when advanced models fail"""
        # Try basic OpenCV face detection as a last resort
        if person_img is not None:
            try:
                # Use simple heuristics for basic demographics
                gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
                
                # Load OpenCV face cascade
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # At least we detected a face with OpenCV
                    logger.debug(f"OpenCV detected {len(faces)} faces as fallback")
                    
                    # Try basic heuristics based on image properties
                    face_x, face_y, face_w, face_h = faces[0]
                    face_roi = gray[face_y:face_y+face_h, face_x:face_x+face_w]
                    
                    # Basic age estimation from face size/proportion
                    estimated_age = 25 + min(15, max(-5, int((face_h - 80) / 4)))  # Rough age estimation
                    age_group = self._calculate_age_group(estimated_age)
                    
                    return {
                        'age': estimated_age,
                        'age_group': age_group,
                        'gender': 'unknown',
                        'race': 'unknown',
                        'emotion': 'neutral',
                        'confidence': 0.4,  # Slightly higher confidence since we detected a face
                        'analysis_method': 'opencv_fallback'
                    }
                else:
                    logger.debug("No faces detected even with OpenCV fallback")
                    
            except Exception as e:
                logger.debug(f"OpenCV fallback also failed: {e}")
        
        # Ultimate fallback with very low confidence
        logger.debug("Using ultimate fallback demographics")
        return {
            'age': 30,
            'age_group': '25-34',  # Provide a reasonable default age group instead of unknown
            'gender': 'unknown',
            'race': 'unknown',
            'emotion': 'neutral',
            'confidence': 0.2,  # Slightly higher than before
            'analysis_method': 'ultimate_fallback'
        }


class FrameAnalysisService:
    """
    Enhanced Frame Analysis Service with Person Memory and Advanced Demographics
    
    This service maintains persistent tracking of people across frames
    and performs sophisticated demographic analysis using DeepFace and InsightFace.
    """
    
    def __init__(self):
        """Initialize the analysis service with enhanced tracking"""
        self.yolo_model = None
        self.demographic_analyzer = EnhancedDemographicAnalyzer()

        # Enhanced tracking and duplicate reduction
        self.person_tracker = None
        self.detection_memory = {
            'frame_count': 0,
            'last_frame_time': 0,
            'active_tracks': {},
            'people_database': {},
            'next_id': 1,
            'demographic_cache': {},  # Cache for demographic analysis
            'feature_database': {},   # Store visual features for re-identification
            'confidence_thresholds': {
                'detection': 0.5,
                'tracking': 0.3,
                'demographics': 0.7,
                'database_entry': 0.8  # Higher threshold for adding to database
            }
        }

        # Enhanced duplicate reduction settings
        self.duplicate_reduction = {
            'min_frames_for_db': 12,  # Minimum frames before adding to database
            'min_confidence_for_db': 0.75,  # Minimum confidence for database entry
            'demographic_consistency_threshold': 0.8,  # Consistency check for demographics
            'feature_similarity_threshold': 0.85,  # Threshold for feature-based duplicate detection
            'temporal_window': 30.0,  # Time window for duplicate detection (seconds)
            'spatial_threshold': 100   # Pixel distance threshold for spatial duplicate detection
        }

        # Initialize enhanced tracker
        self._initialize_enhanced_tracker()

    def _initialize_enhanced_tracker(self):
        """Initialize the enhanced person tracker with optimal settings"""
        tracker_config = {
            'max_age': 45,
            'min_hits': 8,  # Increased to reduce false positives
            'iou_threshold': 0.2,
            'distance_threshold': 120,
            'feature_weight': 0.4,
            'position_weight': 0.6,
            'min_track_confidence': 0.4,
            'confirmation_frames': 12,  # Frames needed before confirming person
            'reid_enabled': True,
            'reid_distance_threshold': 250,
            'reid_feature_threshold': 0.75,
            'reid_max_age': 120
        }

        # Import and initialize the enhanced tracker
        try:
            from ..detection.tracker import PersonTracker
            self.person_tracker = PersonTracker(tracker_config)
            logger.info("Enhanced person tracker initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import enhanced tracker: {e}")
            self.person_tracker = None

    def _check_demographic_consistency(self, person_id, new_demographics):
        """
        Check if new demographic analysis is consistent with previous results

        Args:
            person_id: ID of the person
            new_demographics: New demographic analysis results

        Returns:
            bool: True if consistent, False otherwise
        """
        if person_id not in self.detection_memory['people_database']:
            return True

        stored_demographics = self.detection_memory['people_database'][person_id].get('demographics', {})

        # Check gender consistency
        stored_gender = stored_demographics.get('gender', 'unknown')
        new_gender = new_demographics.get('gender', 'unknown')

        if stored_gender != 'unknown' and new_gender != 'unknown' and stored_gender != new_gender:
            # Allow gender change only if new confidence is significantly higher
            stored_confidence = stored_demographics.get('confidence', 0)
            new_confidence = new_demographics.get('confidence', 0)
            if new_confidence < stored_confidence + 0.3:
                return False

        # Check age group consistency (allow adjacent age groups)
        stored_age = stored_demographics.get('age_group', 'unknown')
        new_age = new_demographics.get('age_group', 'unknown')

        if stored_age != 'unknown' and new_age != 'unknown':
            age_groups = ['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
            try:
                stored_idx = age_groups.index(stored_age)
                new_idx = age_groups.index(new_age)
                # Allow change if within 1 age group or new confidence is much higher
                if abs(stored_idx - new_idx) > 1:
                    stored_confidence = stored_demographics.get('confidence', 0)
                    new_confidence = new_demographics.get('confidence', 0)
                    if new_confidence < stored_confidence + 0.4:
                        return False
            except ValueError:
                pass  # Unknown age group format

        return True

    def _detect_spatial_duplicates(self, new_person, existing_people):
        """
        Detect spatial duplicates based on position and features

        Args:
            new_person: New person detection
            existing_people: List of existing people in database

        Returns:
            ID of duplicate person if found, None otherwise
        """
        new_center = new_person.get('center', (0, 0))
        new_features = new_person.get('visual_features')

        for person_id, person_data in existing_people.items():
            # Skip if person was seen too long ago
            if time.time() - person_data.get('last_seen', 0) > self.duplicate_reduction['temporal_window']:
                continue

            # Check spatial proximity
            stored_center = person_data.get('last_center', (0, 0))
            distance = np.sqrt((new_center[0] - stored_center[0])**2 +
                             (new_center[1] - stored_center[1])**2)

            if distance < self.duplicate_reduction['spatial_threshold']:
                # Check feature similarity if available
                if new_features is not None and person_id in self.detection_memory['feature_database']:
                    stored_features = self.detection_memory['feature_database'][person_id]
                    similarity = self._calculate_feature_similarity(new_features, stored_features)

                    if similarity > self.duplicate_reduction['feature_similarity_threshold']:
                        logger.debug(f"Spatial duplicate detected: person {person_id} (similarity: {similarity:.3f})")
                        return person_id

        return None

    def _calculate_feature_similarity(self, features1, features2):
        """Calculate cosine similarity between two feature vectors"""
        if features1 is None or features2 is None:
            return 0.0
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity([features1], [features2])[0][0]
        except:
            return 0.0

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON serializable Python types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_json_serializable(item) for item in obj)
        else:
            return obj
        
    def _get_model(self) -> Optional[object]:
        """Get or load YOLO model"""
        if self.yolo_model is None:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO('yolov8n.pt')
                logger.info("YOLO model loaded successfully")
            except ImportError:
                logger.error("ultralytics package not installed")
                return None
            except Exception as e:
                logger.error(f"Error loading YOLO model: {e}")
                return None
        return self.yolo_model

    def decode_frame(self, image_data: str) -> Optional[np.ndarray]:
        """Decode base64 image data to OpenCV format"""
        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            img_bytes = base64.b64decode(image_data)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.error(f"Error decoding frame: {e}")
            return None

    def detect_people(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect people in frame using YOLO"""
        detections = []
        model = self._get_model()
        if model is None:
            return detections
        
        try:
            # Save frame temporarily for YOLO processing
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                cv2.imwrite(temp_file.name, frame)
                temp_path = temp_file.name
            
            # Run YOLO detection
            results = model(temp_path, conf=0.5, iou=0.7, verbose=False)
            
            # Process results
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls.item())
                        if cls == 0:  # Person class
                            confidence = float(box.conf.item())
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x, y = int(x1), int(y1)
                            w, h = int(x2 - x1), int(y2 - y1)
                            
                            # Extract person image for feature comparison
                            person_img = frame[y:y+h, x:x+w]
                            
                            detections.append({
                                'bbox': [int(x), int(y), int(w), int(h)],
                                'center': (float(x + w/2), float(y + h/2)),
                                'confidence': float(confidence),
                                'person_image': person_img,
                                'area': int(w * h)
                            })
            
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}")
        
        return detections

    def detect_and_analyze_face(self, person_img: np.ndarray) -> Dict[str, Any]:
        """Detect face and analyze demographics using enhanced models"""
        try:
            if person_img is None or person_img.size == 0:
                return {
                    'age_group': 'unknown',
                    'gender': 'unknown', 
                    'emotion': 'neutral',
                    'confidence': 0.0
                }
            
            # Use enhanced demographic analyzer
            demographics = self.demographic_analyzer.analyze_demographics(person_img)
            
            # Format for compatibility with existing code
            return {
                'age_group': demographics.get('age_group', 'unknown'),
                'gender': demographics.get('gender', 'unknown'),
                'emotion': demographics.get('emotion', 'neutral'),
                'confidence': demographics.get('confidence', 0.1),
                'age': demographics.get('age'),
                'race': demographics.get('race'),
                'analysis_method': demographics.get('analysis_method', 'enhanced')
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced face detection and analysis: {e}")
            return {
                'age_group': 'unknown',
                'gender': 'unknown', 
                'emotion': 'neutral',
                'confidence': 0.1
            }

    def track_people(self, new_detections: List[Dict], frame_shape: Tuple[int, int]) -> List[Dict]:
        """
        Enhanced track people across frames with duplicate reduction and re-identification

        Args:
            new_detections: List of new person detections with person_image
            frame_shape: Shape of the current frame (height, width)
            
        Returns:
            List of tracked people with persistent IDs and demographics
        """
        frame_time = datetime.now().timestamp()

        # Use enhanced tracker if available
        if self.person_tracker is not None:
            return self._enhanced_track_people(new_detections, frame_shape, frame_time)
        else:
            # Fallback to legacy tracking
            return self._legacy_track_people(new_detections, frame_shape, frame_time)

    def _enhanced_track_people(self, new_detections: List[Dict], frame_shape: Tuple[int, int], frame_time: float) -> List[Dict]:
        """Enhanced tracking using the new PersonTracker with visual features"""

        # Extract frame for feature extraction (create dummy frame if needed)
        if len(new_detections) > 0 and 'person_image' in new_detections[0]:
            # Create a dummy frame from detections for feature extraction
            frame = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)

            # Place person images in their bounding boxes for feature extraction
            for detection in new_detections:
                if 'person_image' in detection and detection['person_image'] is not None:
                    x, y, w, h = detection['bbox']
                    person_img = detection['person_image']

                    # Resize person image to fit bounding box
                    if person_img.shape[0] > 0 and person_img.shape[1] > 0:
                        resized = cv2.resize(person_img, (w, h))
                        frame[y:y+h, x:x+w] = resized
        else:
            frame = None

        # Convert detections to tracker format
        tracker_detections = []
        for detection in new_detections:
            x, y, w, h = detection['bbox']
            tracker_detections.append({
                'bbox': (x, y, x+w, y+h),  # Convert to x1,y1,x2,y2 format
                'confidence': detection['confidence'],
                'center': detection['center']
            })

        # Update tracker
        active_tracks = self.person_tracker.update(tracker_detections, frame)

        # Process tracks and analyze demographics
        tracked_people = []
        people_db = self.detection_memory['people_database']

        for track in active_tracks:
            track_id = track['id']

            # Check if person needs demographic analysis
            if (track['is_confirmed'] and
                track['frames_tracked'] >= self.duplicate_reduction['min_frames_for_db'] and
                track['confidence'] >= self.duplicate_reduction['min_confidence_for_db']):

                # Find matching detection for this track
                matching_detection = None
                for detection in new_detections:
                    if self._bbox_overlap(track['bbox'], detection['bbox']) > 0.5:
                        matching_detection = detection
                        break

                # Analyze demographics if we have a matching detection
                if matching_detection and 'person_image' in matching_detection:
                    if track_id not in people_db:
                        # New person - analyze demographics
                        demographics = self.detect_and_analyze_face(matching_detection['person_image'])

                        # Only add if demographic analysis has sufficient confidence
                        if demographics['confidence'] >= self.detection_memory['confidence_thresholds']['demographics']:
                            people_db[track_id] = {
                                'demographics': demographics,
                                'first_seen': frame_time,
                                'last_seen': frame_time,
                                'total_appearances': 1,
                                'average_confidence': track['confidence'],
                                'frames_tracked': track['frames_tracked']
                            }

                            # Store visual features for re-identification
                            if frame is not None:
                                features = self.person_tracker._extract_visual_features(frame, track['bbox'])
                                if features is not None:
                                    self.detection_memory['feature_database'][track_id] = features

                            # Save to database
                            self._save_person_to_database(track_id, demographics, matching_detection, frame_time)

                            logger.info(f"Added person {track_id} to database: {demographics.get('gender', 'unknown')} {demographics.get('age_group', 'unknown')} conf: {demographics.get('confidence', 0):.2f}")

                    else:
                        # Existing person - update info and potentially re-analyze
                        people_db[track_id]['last_seen'] = frame_time
                        people_db[track_id]['total_appearances'] += 1
                        people_db[track_id]['average_confidence'] = (
                            people_db[track_id]['average_confidence'] * 0.8 + track['confidence'] * 0.2
                        )

                        # Re-analyze if we have significantly better confidence
                        current_demo_conf = people_db[track_id]['demographics'].get('confidence', 0)
                        if track['confidence'] > current_demo_conf + 0.2:
                            new_demographics = self.detect_and_analyze_face(matching_detection['person_image'])

                            # Check consistency before updating
                            if (new_demographics['confidence'] > current_demo_conf and
                                self._check_demographic_consistency(track_id, new_demographics)):
                                people_db[track_id]['demographics'] = new_demographics

                                # Update database with new demographics
                                self._save_person_to_database(track_id, new_demographics, matching_detection, frame_time)

                                logger.info(f"Updated demographics for person {track_id}: {new_demographics.get('gender', 'unknown')} {new_demographics.get('age_group', 'unknown')} conf: {new_demographics.get('confidence', 0):.2f}")

                # Store detection data to database for all confirmed tracks
                if matching_detection:
                    self._save_detection_to_database(track_id, track['bbox'], track['confidence'], frame_time)

            # Create tracked person object
            demographics = people_db.get(track_id, {}).get('demographics', {
                'age_group': 'analyzing...',
                'gender': 'analyzing...',
                'emotion': 'neutral',
                'confidence': 0.1
            })

            tracked_person = {
                'id': track_id,
                'bbox': track['bbox'],
                'center': ((track['bbox'][0] + track['bbox'][2]) / 2, (track['bbox'][1] + track['bbox'][3]) / 2),
                'confidence': track['confidence'],
                'frames_tracked': track['frames_tracked'],
                'demographics': demographics,
                'dwell_time': max(0, track['frames_tracked'] * 0.04),  # Approximate dwell time
                'is_confirmed': track['is_confirmed']
            }

            tracked_people.append(tracked_person)

        return tracked_people

    def _bbox_overlap(self, bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes"""
        try:
            # Convert bbox formats if needed
            if len(bbox1) == 4 and len(bbox2) == 4:
                x1_1, y1_1, x2_1, y2_1 = bbox1
                x1_2, y1_2, w2, h2 = bbox2
                x2_2, y2_2 = x1_2 + w2, y1_2 + h2

                # Calculate intersection
                x1_i = max(x1_1, x1_2)
                y1_i = max(y1_1, y1_2)
                x2_i = min(x2_1, x2_2)
                y2_i = min(y2_1, y2_2)

                if x2_i < x1_i or y2_i < y1_i:
                    return 0.0

                intersection = (x2_i - x1_i) * (y2_i - y1_i)
                area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                area2 = w2 * h2
                union = area1 + area2 - intersection

                return intersection / union if union > 0 else 0.0
        except:
            return 0.0

        return 0.0

    def _legacy_track_people(self, new_detections: List[Dict], frame_shape: Tuple[int, int], frame_time: float) -> List[Dict]:
        """Legacy tracking method with enhanced duplicate reduction"""
        active_tracks = self.detection_memory['active_tracks']
        people_db = self.detection_memory['people_database']
        
        # Remove old tracks (not seen for more than 3 seconds)
        tracks_to_remove = []
        for track_id in list(active_tracks.keys()):
            track_info = active_tracks[track_id]
            if frame_time - track_info['last_seen'] > 3.0:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            if track_id in active_tracks:
                del active_tracks[track_id]
        
        # Track current detections with enhanced matching
        tracked_people = []
        used_detections = set()
        
        # Enhanced matching with spatial and feature-based duplicate detection
        for track_id, track_info in list(active_tracks.items()):
            best_match = None
            best_score = 0.0
            best_detection_idx = -1
            
            # Find best matching detection using multiple criteria
            for i, detection in enumerate(new_detections):
                if i in used_detections:
                    continue
                
                # Calculate spatial distance
                track_center = track_info['last_center']
                detection_center = detection['center']
                distance = np.sqrt((track_center[0] - detection_center[0])**2 + 
                                 (track_center[1] - detection_center[1])**2)
                
                # Calculate size similarity
                track_area = track_info['last_area']
                detection_area = detection['area']
                size_ratio = min(track_area, detection_area) / max(track_area, detection_area)

                # Combined score
                distance_score = max(0, 1 - distance / 120)  # Normalize distance
                size_score = size_ratio
                combined_score = 0.7 * distance_score + 0.3 * size_score

                # Check if this is the best match so far
                if combined_score > best_score and distance < 120:
                    best_score = combined_score
                    best_match = detection
                    best_detection_idx = i
            
            if best_match and best_score > 0.5:  # Minimum threshold for matching
                # Update existing track
                used_detections.add(best_detection_idx)
                
                # Update track info
                track_info['last_center'] = best_match['center']
                track_info['last_area'] = best_match['area']
                track_info['frames_tracked'] += 1
                track_info['last_seen'] = frame_time
                track_info['confidence_history'].append(best_match['confidence'])
                
                # Enhanced demographic analysis with consistency checking
                if (track_info['frames_tracked'] >= self.duplicate_reduction['min_frames_for_db'] and
                    np.mean(track_info['confidence_history'][-5:]) >= self.duplicate_reduction['min_confidence_for_db']):

                    if track_id not in people_db:
                        # New person - analyze demographics
                        demographics = self.detect_and_analyze_face(best_match['person_image'])

                        # Check for spatial duplicates before adding
                        duplicate_id = self._detect_spatial_duplicates(best_match, people_db)
                        if duplicate_id is None:
                            people_db[track_id] = {
                                'demographics': demographics,
                                'first_seen': track_info['first_seen'],
                                'last_seen': frame_time,
                                'total_appearances': 1,
                                'average_confidence': np.mean(track_info['confidence_history'][-10:]),
                                'frames_tracked': track_info['frames_tracked']
                            }
                            logger.info(f"Added person {track_id} to database: {demographics.get('gender', 'unknown')} {demographics.get('age_group', 'unknown')} conf: {demographics.get('confidence', 0):.2f}")
                        else:
                            logger.debug(f"Skipped duplicate person {track_id} (similar to {duplicate_id})")

                    else:
                        # Update existing person
                        people_db[track_id]['last_seen'] = frame_time
                        people_db[track_id]['total_appearances'] += 1
                        current_avg_confidence = np.mean(track_info['confidence_history'][-10:])
                        people_db[track_id]['average_confidence'] = current_avg_confidence
                        
                        # Re-analyze if we have significantly better confidence
                        if (current_avg_confidence > people_db[track_id]['demographics'].get('confidence', 0) + 0.2 and
                            best_match['person_image'] is not None):
                            new_demographics = self.detect_and_analyze_face(best_match['person_image'])

                            # Check consistency before updating
                            if (new_demographics['confidence'] > people_db[track_id]['demographics']['confidence'] and
                                self._check_demographic_consistency(track_id, new_demographics)):
                                people_db[track_id]['demographics'] = new_demographics
                                logger.info(f"Updated demographics for person {track_id}: {new_demographics.get('gender', 'unknown')} {new_demographics.get('age_group', 'unknown')} conf: {new_demographics.get('confidence', 0):.2f}")

                # Create tracked person object
                tracked_person = {
                    'id': track_id,
                    'bbox': best_match['bbox'],
                    'center': best_match['center'],
                    'confidence': best_match['confidence'],
                    'frames_tracked': track_info['frames_tracked'],
                    'demographics': people_db.get(track_id, {}).get('demographics', {
                        'age_group': 'analyzing...',
                        'gender': 'analyzing...',
                        'emotion': 'neutral',
                        'confidence': 0.1
                    }),
                    'dwell_time': frame_time - track_info['first_seen'],
                    'is_confirmed': track_info['frames_tracked'] >= self.duplicate_reduction['min_frames_for_db']
                }
                
                tracked_people.append(tracked_person)
        
        # Create new tracks for unmatched detections with duplicate checking
        for i, detection in enumerate(new_detections):
            if i not in used_detections:
                # Check for spatial duplicates before creating new track
                duplicate_id = self._detect_spatial_duplicates(detection, people_db)
                if duplicate_id is None:
                    # Create new track
                    new_id = self.detection_memory['next_id']
                    self.detection_memory['next_id'] += 1

                    active_tracks[new_id] = {
                        'last_center': detection['center'],
                        'last_area': detection['area'],
                        'frames_tracked': 1,
                        'first_seen': frame_time,
                        'last_seen': frame_time,
                        'confidence_history': [detection['confidence']]
                    }

                    # Create tracked person for new detection
                    tracked_person = {
                        'id': new_id,
                        'bbox': detection['bbox'],
                        'center': detection['center'],
                        'confidence': detection['confidence'],
                        'frames_tracked': 1,
                        'demographics': {
                            'age_group': 'analyzing...',
                            'gender': 'analyzing...',
                            'emotion': 'neutral',
                            'confidence': 0.1
                        },
                        'dwell_time': 0,
                        'is_confirmed': False
                    }

                    tracked_people.append(tracked_person)
                else:
                    logger.debug(f"Skipped potential duplicate detection (similar to person {duplicate_id})")

        return tracked_people

    def analyze_frame(self, image_data: str) -> Dict[str, Any]:
        """Enhanced frame analysis with persistent tracking and demographics"""
        try:
            # Decode frame
            frame = self.decode_frame(image_data)
            if frame is None:
                return {"error": "Failed to decode image"}
            
            # Update frame count
            self.detection_memory['frame_count'] += 1
            frame_time = datetime.now().timestamp()
            
            # Detect people in current frame
            try:
                detections = self.detect_people(frame)
            except Exception as e:
                logger.error(f"Error in detect_people: {e}")
                detections = []
            
            # Track people across frames with memory
            try:
                tracked_people = self.track_people(detections, frame.shape)
            except Exception as e:
                logger.error(f"Error in track_people: {e}")
                # Fallback: create simple tracked people
                tracked_people = []
                for i, detection in enumerate(detections):
                    tracked_people.append({
                        'id': f"temp_{i}",
                        'bbox': detection['bbox'],
                        'center': detection['center'],
                        'confidence': detection['confidence'],
                        'frames_tracked': 1,
                        'demographics': {
                            'age_group': 'analyzing...',
                            'gender': 'analyzing...',
                            'emotion': 'neutral',
                            'confidence': 0.1
                        },
                        'dwell_time': 0
                    })
            
            # Calculate demographics summary
            demographics_summary = {
                'male_count': 0,
                'female_count': 0,
                'age_groups': {},
                'total_analysed': 0
            }
            
            total_dwell = 0
            heatmap_points = []
            
            # Process each tracked person
            for person in tracked_people:
                # Count demographics
                gender = person.get('demographics', {}).get('gender', 'unknown')
                age_group = person.get('demographics', {}).get('age_group', 'unknown')
                
                if gender == 'male':
                    demographics_summary['male_count'] += 1
                elif gender == 'female':
                    demographics_summary['female_count'] += 1
                
                if age_group != 'unknown' and age_group != 'analyzing...':
                    demographics_summary['age_groups'][age_group] = demographics_summary['age_groups'].get(age_group, 0) + 1
                    demographics_summary['total_analysed'] += 1
                
                # Sum dwell time
                total_dwell += person.get('dwell_time', 0)
                
                # Create heatmap points from person's full bounding box
                bbox = person.get('bbox', [0, 0, 50, 50])
                center = person.get('center', (25, 25))
                confidence = person.get('confidence', 0.5)
                
                # Add multiple points across the bounding box for better heat distribution
                x, y, w, h = bbox
                for dx in [0.2, 0.5, 0.8]:  # Left, center, right
                    for dy in [0.3, 0.7]:   # Upper, lower
                        heat_x = x + w * dx
                        heat_y = y + h * dy
                        heatmap_points.append({
                            "x": heat_x,
                            "y": heat_y,
                            "value": confidence * 0.8,  # Slightly reduced for visual appeal
                            "id": person.get('id', 'unknown')
                        })
                
                # Add center point with higher intensity
                heatmap_points.append({
                    "x": center[0],
                    "y": center[1],
                    "value": confidence,
                    "id": person.get('id', 'unknown')
                })
            
            # Update memory
            self.detection_memory['last_frame_time'] = frame_time
            
            # Prepare comprehensive response
            response = {
                "success": True,
                "detections": {
                    "people": tracked_people
                },
                "heatmap": {
                    "points": heatmap_points
                },
                "frame_info": {
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "timestamp": frame_time,
                    "detection_count": len(tracked_people),
                    "total_tracks": len(self.detection_memory['active_tracks'])
                },
                "analytics": {
                    "total_people": len(tracked_people),
                    "male_count": demographics_summary['male_count'],
                    "female_count": demographics_summary['female_count'],
                    "average_dwell_time": total_dwell / max(1, len(tracked_people)),
                    "age_groups": demographics_summary['age_groups'],
                    "total_analysed": demographics_summary['total_analysed'],
                    "active_tracks": len(self.detection_memory['active_tracks']),
                    "people_database_size": len(self.detection_memory['people_database'])
                },

            }
            
            # Log tracking status for debugging
            if self.detection_memory['frame_count'] % 30 == 0:  # Every 30 frames
                active_count = len(self.detection_memory['active_tracks'])
                db_count = len(self.detection_memory['people_database'])
                logger.info(f"Frame {self.detection_memory['frame_count']}: {len(tracked_people)} current, {active_count} active tracks, {db_count} in database")
            
            # Convert response to JSON serializable format
            response = self._convert_to_json_serializable(response)
            return response
            
        except Exception as e:
            logger.error(f"Unexpected error in analyze_frame: {e}", exc_info=True)
            return {"error": f"Analysis failed: {str(e)}"}

    def _save_person_to_database(self, person_id, demographics, detection, timestamp):
        """
        Save person data to database for persistence

        Args:
            person_id: Track ID of the person
            demographics: Demographic analysis results
            detection: Detection data
            timestamp: Timestamp of the detection
        """
        try:
            # Get database instance from current app context
            from flask import current_app
            if hasattr(current_app, 'db'):
                db = current_app.db

                # Store demographics data
                db.store_demographics(
                    person_id=person_id,
                    demographics=demographics,
                    timestamp=datetime.fromtimestamp(timestamp),
                    analysis_model=demographics.get('analysis_method', 'enhanced')
                )

                logger.debug(f"Saved person {person_id} demographics to database")
        except Exception as e:
            logger.error(f"Error saving person {person_id} to database: {e}")

    def _save_detection_to_database(self, person_id, bbox, confidence, timestamp):
        """
        Save detection data to database

        Args:
            person_id: Track ID of the person
            bbox: Bounding box coordinates
            confidence: Detection confidence
            timestamp: Timestamp of the detection
        """
        try:
            # Get database instance from current app context
            from flask import current_app
            if hasattr(current_app, 'db'):
                db = current_app.db

                # Convert bbox format if needed
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    bbox_tuple = (x1, y1, x2, y2)
                else:
                    bbox_tuple = bbox

                # Store detection data
                db.store_detection(
                    person_id=person_id,
                    bbox=bbox_tuple,
                    confidence=confidence,
                    timestamp=datetime.fromtimestamp(timestamp),
                    camera_id='main'
                )

                logger.debug(f"Saved detection for person {person_id} to database")
        except Exception as e:
            logger.error(f"Error saving detection for person {person_id} to database: {e}")


# Global service instance
analysis_service = FrameAnalysisService()
