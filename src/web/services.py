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

logger = logging.getLogger(__name__)


class FrameAnalysisService:
    """
    Enhanced Frame Analysis Service with Person Memory and Demographics
    
    This service maintains persistent tracking of people across frames,
    performs demographic analysis, and builds movement paths.
    """
    
    def __init__(self):
        self.yolo_model = None
        self.detection_memory = {
            'frame_count': 0,
            'people_database': {},  # Stores persistent person information
            'active_tracks': {},    # Currently active person tracks
            'next_id': 1,
            'movement_paths': {},   # Person movement paths
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
        """Detect face and analyze demographics from person image"""
        try:
            if person_img is None or person_img.size == 0:
                return {
                    'age_group': 'unknown',
                    'gender': 'unknown', 
                    'emotion': 'neutral',
                    'confidence': 0.0
                }
            
            # Initialize face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            
            if len(faces) == 0:
                # No face detected - analyze based on full body
                return self.analyze_body_features(person_img)
            
            # Use the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Extract face region
            face_img = person_img[y:y+h, x:x+w]
            
            # Analyze face features
            return self.analyze_face_features(face_img, gray[y:y+h, x:x+w])
            
        except Exception as e:
            logger.error(f"Error in face detection and analysis: {e}")
            return {
                'age_group': 'unknown',
                'gender': 'unknown', 
                'emotion': 'neutral',
                'confidence': 0.1
            }
    
    def analyze_face_features(self, face_img: np.ndarray, face_gray: np.ndarray) -> Dict[str, Any]:
        """Analyze demographics from detected face"""
        try:
            height, width = face_img.shape[:2]
            
            # Age estimation based on facial features
            age_group = self.estimate_age_from_face(face_img, face_gray)
            
            # Gender estimation based on facial structure
            gender = self.estimate_gender_from_face(face_img, face_gray)
            
            # Emotion detection from facial expression
            emotion = self.detect_emotion_from_face(face_img, face_gray)
            
            # Calculate confidence based on face quality
            confidence = self.calculate_face_confidence(face_img)
            
            return {
                'age_group': age_group,
                'gender': gender,
                'emotion': emotion,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error analyzing face features: {e}")
            return {
                'age_group': 'unknown',
                'gender': 'unknown',
                'emotion': 'neutral', 
                'confidence': 0.1
            }
    
    def estimate_age_from_face(self, face_img: np.ndarray, face_gray: np.ndarray) -> str:
        """Estimate age group from facial features"""
        try:
            # Analyze facial texture and structure for age estimation
            height, width = face_img.shape[:2]
            
            # Calculate facial texture variance (wrinkles, smoothness)
            laplacian_var = cv2.Laplacian(face_gray, cv2.CV_64F).var()
            
            # Calculate facial proportions
            face_ratio = height / width if width > 0 else 1.0
            
            # Analyze color properties
            avg_color = np.mean(face_img, axis=(0, 1))
            brightness = np.mean(avg_color)
            
            # Age estimation heuristics based on facial analysis
            if laplacian_var < 100:  # Very smooth face
                if brightness > 150:  # Bright, smooth face suggests youth
                    return "18-25"
                else:
                    return "25-35"
            elif laplacian_var < 200:  # Moderate texture
                if face_ratio > 1.3:  # Longer face suggests maturity
                    return "35-45"
                else:
                    return "25-35"
            else:  # Higher texture (possible wrinkles/lines)
                if brightness < 120:  # Darker, more textured suggests older
                    return "45-55"
                else:
                    return "35-45"
                    
        except Exception as e:
            logger.error(f"Error in age estimation: {e}")
            return "unknown"
    
    def estimate_gender_from_face(self, face_img: np.ndarray, face_gray: np.ndarray) -> str:
        """Estimate gender from facial structure"""
        try:
            height, width = face_img.shape[:2]
            
            # Analyze facial structure
            face_ratio = height / width if width > 0 else 1.0
            
            # Analyze jawline strength using edge detection
            edges = cv2.Canny(face_gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            
            # Analyze facial features in different regions
            upper_face = face_img[:height//3, :]  # Forehead area
            lower_face = face_img[2*height//3:, :]  # Chin/jaw area
            
            # Calculate feature prominence
            upper_brightness = np.mean(upper_face) if upper_face.size > 0 else 128
            lower_brightness = np.mean(lower_face) if lower_face.size > 0 else 128
            
            # Gender estimation heuristics
            masculine_score = 0
            feminine_score = 0
            
            # Facial ratio analysis
            if face_ratio < 1.2:  # Wider face
                masculine_score += 1
            else:  # Longer face
                feminine_score += 1
            
            # Jawline analysis
            if edge_density > 0.1:  # Strong jawline
                masculine_score += 1
            else:  # Softer features
                feminine_score += 1
            
            # Feature analysis
            if lower_brightness < upper_brightness - 10:  # Prominent jaw/chin
                masculine_score += 1
            else:
                feminine_score += 1
            
            # Make determination with confidence threshold
            if masculine_score > feminine_score:
                return "male"
            elif feminine_score > masculine_score:
                return "female"
            else:
                return "unknown"
                
        except Exception as e:
            logger.error(f"Error in gender estimation: {e}")
            return "unknown"
    
    def detect_emotion_from_face(self, face_img: np.ndarray, face_gray: np.ndarray) -> str:
        """Detect emotion from facial expression"""
        try:
            height, width = face_img.shape[:2]
            
            # Analyze facial regions for emotion detection
            mouth_region = face_gray[2*height//3:, width//4:3*width//4]  # Mouth area
            eye_region = face_gray[height//4:height//2, :]  # Eye area
            
            if mouth_region.size == 0 or eye_region.size == 0:
                return "neutral"
            
            # Detect facial contours for expression analysis
            mouth_edges = cv2.Canny(mouth_region, 50, 150)
            eye_edges = cv2.Canny(eye_region, 50, 150)
            
            mouth_activity = np.sum(mouth_edges > 0) / mouth_region.size
            eye_activity = np.sum(eye_edges > 0) / eye_region.size
            
            # Simple emotion classification based on facial activity
            if mouth_activity > 0.05:  # High mouth activity suggests smile
                return "happy"
            elif eye_activity > 0.08:  # High eye activity suggests alertness
                return "focused"
            elif mouth_activity < 0.02 and eye_activity < 0.04:  # Low activity
                return "calm"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error in emotion detection: {e}")
            return "neutral"
    
    def calculate_face_confidence(self, face_img: np.ndarray) -> float:
        """Calculate confidence score based on face image quality"""
        try:
            height, width = face_img.shape[:2]
            
            # Face size quality (larger faces = higher confidence)
            size_score = min(1.0, (height * width) / (100 * 100))
            
            # Image sharpness (higher sharpness = higher confidence)
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, sharpness / 500)
            
            # Brightness quality (avoid too dark or too bright)
            brightness = np.mean(face_img)
            brightness_score = 1.0 - abs(brightness - 128) / 128
            
            # Combined confidence score
            confidence = (size_score * 0.4 + sharpness_score * 0.4 + brightness_score * 0.2)
            
            return max(0.1, min(0.95, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating face confidence: {e}")
            return 0.1
    
    def analyze_body_features(self, person_img: np.ndarray) -> Dict[str, Any]:
        """Fallback analysis when no face is detected"""
        try:
            height, width = person_img.shape[:2]
            
            # Basic body analysis when face is not visible
            aspect_ratio = height / width if width > 0 else 1.0
            
            # Very basic heuristics for body-based analysis
            if aspect_ratio > 2.5:  # Very tall silhouette
                age_group = "25-45"  # Adult proportions
                gender = "unknown"  # Cannot determine from body alone
            elif aspect_ratio > 1.8:  # Normal adult proportions
                age_group = "18-35"
                gender = "unknown"
            else:  # Shorter/wider silhouette
                age_group = "unknown"
                gender = "unknown"
            
            return {
                'age_group': age_group,
                'gender': gender,
                'emotion': 'neutral',
                'confidence': 0.2  # Low confidence without face
            }
            
        except Exception as e:
            logger.error(f"Error in body analysis: {e}")
            return {
                'age_group': 'unknown',
                'gender': 'unknown',
                'emotion': 'neutral',
                'confidence': 0.1
            }

    def track_people(self, new_detections: List[Dict], frame_shape: Tuple[int, int]) -> List[Dict]:
        """Advanced people tracking with memory and path building"""
        height, width = frame_shape[:2]
        max_distance = 0.3 * ((width + height) / 2)  # Increased for better tracking
        
        active_tracks = self.detection_memory['active_tracks']
        people_db = self.detection_memory['people_database']
        movement_paths = self.detection_memory['movement_paths']
        frame_time = datetime.now().timestamp()
        
        tracked_people = []
        used_detections = set()
        
        # Try to match new detections with existing tracks
        for track_id, track_info in list(active_tracks.items()):
            best_match = None
            best_score = 0
            best_detection_idx = -1
            
            for i, detection in enumerate(new_detections):
                if i in used_detections:
                    continue
                
                # Calculate distance score
                last_center = track_info['last_center']
                current_center = detection['center']
                distance = np.sqrt(
                    (current_center[0] - last_center[0])**2 +
                    (current_center[1] - last_center[1])**2
                )
                
                if distance > max_distance:
                    continue
                
                # Calculate size similarity score
                last_area = track_info.get('last_area', detection['area'])
                area_ratio = min(detection['area'], last_area) / max(detection['area'], last_area)
                
                # Combined matching score
                distance_score = 1.0 - (distance / max_distance)
                size_score = area_ratio
                total_score = (distance_score * 0.7) + (size_score * 0.3)
                
                if total_score > best_score and total_score > 0.5:
                    best_score = total_score
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
                
                # Update movement path
                if track_id not in movement_paths:
                    movement_paths[track_id] = []
                
                movement_paths[track_id].append({
                    'x': best_match['center'][0],
                    'y': best_match['center'][1],
                    'timestamp': frame_time,
                    'confidence': best_match['confidence']
                })
                
                # Limit path history
                if len(movement_paths[track_id]) > 50:
                    movement_paths[track_id] = movement_paths[track_id][-50:]
                
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
                        logger.info(f"Analyzed demographics for person {track_id}: {demographics}")
                    else:
                        # Update existing person info and re-analyze if confidence improved
                        people_db[track_id]['total_appearances'] += 1
                        current_avg_confidence = np.mean(track_info['confidence_history'][-10:])
                        people_db[track_id]['average_confidence'] = current_avg_confidence
                        
                        # Re-analyze if we have better quality image
                        if (current_avg_confidence > people_db[track_id].get('demographics', {}).get('confidence', 0) + 0.1 and
                            best_match['person_image'] is not None):
                            updated_demographics = self.detect_and_analyze_face(best_match['person_image'])
                            if updated_demographics['confidence'] > people_db[track_id]['demographics']['confidence']:
                                people_db[track_id]['demographics'] = updated_demographics
                                logger.info(f"Updated demographics for person {track_id}: {updated_demographics}")
                
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
                    'path': movement_paths.get(track_id, [])[-10:]  # Last 10 path points
                }
                
                tracked_people.append(tracked_person)
            else:
                # Track lost - mark for potential removal
                track_info['frames_lost'] = track_info.get('frames_lost', 0) + 1
                if track_info['frames_lost'] > 30:  # Remove after 30 frames lost
                    del active_tracks[track_id]
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(new_detections):
            if i not in used_detections:
                # Create new track
                new_id = self.detection_memory['next_id']
                self.detection_memory['next_id'] += 1
                
                active_tracks[new_id] = {
                    'first_seen': frame_time,
                    'last_seen': frame_time,
                    'last_center': detection['center'],
                    'last_area': detection['area'],
                    'frames_tracked': 1,
                    'frames_lost': 0,
                    'confidence_history': [detection['confidence']]
                }
                
                # Initialize movement path
                movement_paths[new_id] = [{
                    'x': detection['center'][0],
                    'y': detection['center'][1],
                    'timestamp': frame_time,
                    'confidence': detection['confidence']
                }]
                
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
                    'path': []
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
                        'dwell_time': 0,
                        'path': []
                    })
            
            # Calculate demographics summary
            demographics_summary = {
                'male_count': 0,
                'female_count': 0,
                'age_groups': {},
                'total_analyzed': 0
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
                    demographics_summary['total_analyzed'] += 1
                
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
                    "total_analyzed": demographics_summary['total_analyzed'],
                    "active_tracks": len(self.detection_memory['active_tracks']),
                    "people_database_size": len(self.detection_memory['people_database'])
                },
                "paths": {
                    person['id']: person.get('path', [])
                    for person in tracked_people
                    if person.get('path') and len(person['path']) > 1
                }
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