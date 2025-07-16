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
    
    def _extract_face_insightface(self, person_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract face using InsightFace detection"""
        try:
            if self.face_app is None:
                logger.debug("InsightFace face_app not available")
                return None
            
            # Ensure minimum image size
            if person_img.shape[0] < 32 or person_img.shape[1] < 32:
                logger.debug(f"Person image too small for face detection: {person_img.shape}")
                return None
                
            faces = self.face_app.get(person_img)
            if faces:
                # Sort by face size and take the largest
                faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
                bbox = faces[0].bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                
                # Ensure bbox is within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1) 
                x2 = min(person_img.shape[1], x2)
                y2 = min(person_img.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    logger.debug("Invalid face bbox coordinates")
                    return None
                
                face_img = person_img[y1:y2, x1:x2]
                
                # Ensure face is large enough
                if face_img.shape[0] > 30 and face_img.shape[1] > 30:
                    logger.debug(f"Face extracted successfully: {face_img.shape}")
                    return face_img
                else:
                    logger.debug(f"Extracted face too small: {face_img.shape}")
                    
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
            
            # Extract dominant attributes
            age = result.get('age', 25)
            gender = result.get('dominant_gender', 'unknown')
            race = result.get('dominant_race', 'unknown')
            emotion = result.get('dominant_emotion', 'neutral')
            
            # Calculate age group
            age_group = self._calculate_age_group(age)
            
            # Calculate confidence (use gender confidence as overall indicator)
            gender_scores = result.get('gender', {})
            confidence = max(gender_scores.values()) if gender_scores else 0.7
            
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
                    return {
                        'age': 30,
                        'age_group': '25-34',
                        'gender': 'unknown',
                        'race': 'unknown',
                        'emotion': 'neutral',
                        'confidence': 0.3,  # Higher confidence since we detected a face
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
            'age_group': 'unknown',
            'gender': 'unknown',
            'race': 'unknown',
            'emotion': 'neutral',
            'confidence': 0.1,
            'analysis_method': 'ultimate_fallback'
        }


class FrameAnalysisService:
    """
    Enhanced Frame Analysis Service with Person Memory and Advanced Demographics
    
    This service maintains persistent tracking of people across frames
    and performs sophisticated demographic analysis using DeepFace and InsightFace.
    """
    
    def __init__(self):
        self.yolo_model = None
        self.demographic_analyzer = EnhancedDemographicAnalyzer()
        self.detection_memory = {
            'frame_count': 0,
            'people_database': {},  # Stores persistent person information
            'active_tracks': {},    # Currently active person tracks
            'next_id': 1,
            'last_frame_time': 0
        }
        
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
                                'bbox': [x, y, w, h],
                                'center': (x + w/2, y + h/2),
                                'confidence': confidence,
                                'person_image': person_img,
                                'area': w * h
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
        Track people across frames with enhanced demographic memory
        
        Args:
            new_detections: List of new person detections
            frame_shape: Shape of the current frame (height, width)
            
        Returns:
            List of tracked people with persistent IDs and demographics
        """
        frame_time = datetime.now().timestamp()
        active_tracks = self.detection_memory['active_tracks']
        people_db = self.detection_memory['people_database']
        
        # Remove old tracks (not seen for more than 2 seconds)
        tracks_to_remove = []
        for track_id, track_info in active_tracks.items():
            if frame_time - track_info['last_seen'] > 2.0:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del active_tracks[track_id]
        
        # Track current detections
        tracked_people = []
        used_detections = set()
        
        # Match new detections to existing tracks
        for track_id, track_info in active_tracks.items():
            best_match = None
            best_distance = float('inf')
            best_detection_idx = -1
            
            # Find best matching detection
            for i, detection in enumerate(new_detections):
                if i in used_detections:
                    continue
                
                # Calculate distance between track center and detection center
                track_center = track_info['last_center']
                detection_center = detection['center']
                distance = np.sqrt((track_center[0] - detection_center[0])**2 + 
                                 (track_center[1] - detection_center[1])**2)
                
                # Check if this is the best match so far
                if distance < best_distance and distance < 100:  # 100 pixel threshold
                    best_distance = distance
                    best_match = detection
                    best_detection_idx = i
            
            if best_match:
                # Update existing track
                used_detections.add(best_detection_idx)
                
                # Update track info
                track_info['last_center'] = best_match['center']
                track_info['last_area'] = best_match['area']
                track_info['frames_tracked'] += 1
                track_info['last_seen'] = frame_time
                track_info['confidence_history'].append(best_match['confidence'])
                
                # Update person database if this is a well-tracked person
                if track_info['frames_tracked'] > 3:  # Analyze earlier for better demographics
                    if track_id not in people_db:
                        # Analyze demographics from face detection for new person
                        demographics = self.detect_and_analyze_face(best_match['person_image'])
                        people_db[track_id] = {
                            'demographics': demographics,
                            'first_seen': track_info['first_seen'],
                            'total_appearances': 1,
                            'average_confidence': np.mean(track_info['confidence_history'][-10:]),
                            'face_analysis_done': True
                        }
                        logger.info(f"Enhanced demographic analysis for person {track_id}: {demographics.get('gender', 'unknown')} {demographics.get('age_group', 'unknown')} confidence: {demographics.get('confidence', 0):.2f}")
                    else:
                        # Update existing person info and re-analyze if confidence improved
                        people_db[track_id]['total_appearances'] += 1
                        current_avg_confidence = np.mean(track_info['confidence_history'][-10:])
                        people_db[track_id]['average_confidence'] = current_avg_confidence
                        
                        # Re-analyze if we have better quality image
                        if (current_avg_confidence > people_db[track_id].get('demographics', {}).get('confidence', 0) + 0.15 and
                            best_match['person_image'] is not None):
                            updated_demographics = self.detect_and_analyze_face(best_match['person_image'])
                            if updated_demographics['confidence'] > people_db[track_id]['demographics']['confidence']:
                                people_db[track_id]['demographics'] = updated_demographics
                                logger.info(f"Updated demographics for person {track_id}: {updated_demographics.get('gender', 'unknown')} {updated_demographics.get('age_group', 'unknown')} confidence: {updated_demographics.get('confidence', 0):.2f}")
                
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
                    'dwell_time': frame_time - track_info['first_seen']
                }
                
                tracked_people.append(tracked_person)
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(new_detections):
            if i not in used_detections:
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
                    'dwell_time': 0
                }
                
                tracked_people.append(tracked_person)
        
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
            
            return response
            
        except Exception as e:
            logger.error(f"Unexpected error in analyze_frame: {e}", exc_info=True)
            return {"error": f"Analysis failed: {str(e)}"}


# Global service instance
analysis_service = FrameAnalysisService() 