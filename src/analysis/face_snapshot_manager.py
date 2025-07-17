"""
Face Snapshot Manager for Retail Analytics System

This module manages face snapshots storage, quality assessment, and file system operations
to maintain the best quality face image for each person.
"""

import os
import cv2
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import base64

from .face_quality import FaceQualityScorer

logger = logging.getLogger(__name__)

class FaceSnapshotManager:
    """
    Manages face snapshots for each person, ensuring only high-quality faces are stored
    and used for demographic analysis
    """
    
    def __init__(self, config=None, database=None):
        """
        Initialize the face snapshot manager
        
        Args:
            config (dict): Configuration dictionary
            database: Database connection for metadata storage
        """
        self.config = config or {}
        self.database = database
        
        # Storage configuration
        self.storage_dir = Path(self.config.get('storage_directory', 'data/faces'))
        self.storage_format = self.config.get('storage_format', 'jpg')
        self.storage_quality = self.config.get('storage_quality', 95)
        self.max_faces_per_person = self.config.get('max_faces_per_person', 3)
        self.enable_file_storage = self.config.get('enable_file_storage', True)
        self.enable_db_storage = self.config.get('enable_db_storage', True)
        
        # Quality thresholds
        self.min_quality_threshold = self.config.get('min_quality_threshold', 0.5)
        self.improvement_threshold = self.config.get('improvement_threshold', 0.1)
        self.max_face_size = self.config.get('max_face_size', (224, 224))
        
        # Initialize quality scorer
        quality_config = self.config.get('quality_scorer', {})
        self.quality_scorer = FaceQualityScorer(quality_config)
        
        # Ensure storage directory exists
        if self.enable_file_storage:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Face snapshot manager initialized (storage: {self.storage_dir})")
    
    def process_face_snapshot(self, person_id: int, face_img: np.ndarray, bbox: Tuple[int, int, int, int],
                             detection_confidence: float, landmarks=None, timestamp=None) -> Dict[str, Any]:
        """
        Process a new face snapshot, determining if it should be stored
        
        Args:
            person_id (int): Person track ID
            face_img (np.ndarray): Face image array
            bbox (tuple): Face bounding box (x1, y1, x2, y2)
            detection_confidence (float): Detection confidence score
            landmarks: Optional facial landmarks
            timestamp (datetime): When snapshot was taken
            
        Returns:
            dict: Processing result with quality score and storage info
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Validate inputs
            if not isinstance(face_img, np.ndarray):
                logger.error(f"Expected numpy array for face_img, got {type(face_img)}")
                return {
                    'person_id': person_id,
                    'quality_score': 0.0,
                    'quality_grade': 'F',
                    'timestamp': timestamp,
                    'stored': False,
                    'is_primary': False,
                    'reason': f'Invalid face image type: {type(face_img)}'
                }
            
            if len(face_img.shape) < 2:
                logger.error(f"Invalid face image shape: {face_img.shape}")
                return {
                    'person_id': person_id,
                    'quality_score': 0.0,
                    'quality_grade': 'F',
                    'timestamp': timestamp,
                    'stored': False,
                    'is_primary': False,
                    'reason': f'Invalid face image shape: {face_img.shape}'
                }
            
            # Calculate quality score
            quality_score = self.quality_scorer.calculate_quality_score(
                face_img, bbox, detection_confidence, landmarks
            )
            
            result = {
                'person_id': person_id,
                'quality_score': quality_score,
                'quality_grade': self.quality_scorer.get_quality_grade(quality_score),
                'timestamp': timestamp,
                'stored': False,
                'is_primary': False,
                'reason': ''
            }
            
            # Check minimum quality threshold
            if quality_score < self.min_quality_threshold:
                result['reason'] = f'Quality too low ({quality_score:.3f} < {self.min_quality_threshold})'
                return result
            
            # Check if we should store this face
            should_store, is_primary, reason = self._should_store_face(person_id, quality_score)
            
            result['reason'] = reason
            
            if should_store:
                # Store the face snapshot
                storage_result = self._store_face_snapshot(
                    person_id, face_img, bbox, quality_score, 
                    detection_confidence, landmarks, timestamp, is_primary
                )
                
                if storage_result:
                    result['stored'] = True
                    result['is_primary'] = is_primary
                    result['snapshot_id'] = storage_result.get('snapshot_id')
                    result['file_path'] = storage_result.get('file_path')
                    
                    # Cleanup old snapshots if needed
                    if self.database:
                        self.database.cleanup_old_face_snapshots(person_id, self.max_faces_per_person)
                else:
                    result['reason'] = 'Storage failed'
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing face snapshot for person {person_id}: {e}")
            return {
                'person_id': person_id,
                'quality_score': 0.0,
                'stored': False,
                'reason': f'Processing error: {e}'
            }
    
    def _should_store_face(self, person_id: int, new_quality: float) -> Tuple[bool, bool, str]:
        """
        Determine if a face should be stored and if it should be primary
        
        Args:
            person_id (int): Person track ID
            new_quality (float): Quality score of new face
            
        Returns:
            tuple: (should_store, is_primary, reason)
        """
        try:
            if not self.database:
                return True, True, "No database - storing as primary"
            
            # Get current primary face
            primary_face = self.database.get_primary_face_snapshot(person_id)
            
            if not primary_face:
                # No existing face - store as primary
                return True, True, "First face for person"
            
            current_quality = primary_face.get('quality_score', 0.0)
            
            # Check if new face is significantly better
            if self.quality_scorer.should_update_face(current_quality, new_quality, self.improvement_threshold):
                return True, True, f"Quality improvement ({current_quality:.3f} -> {new_quality:.3f})"
            
            # Check if we have room for additional non-primary faces
            # This could be useful for backup faces or tracking quality over time
            return False, False, f"Quality not significantly better ({new_quality:.3f} vs {current_quality:.3f})"
            
        except Exception as e:
            logger.error(f"Error checking if face should be stored: {e}")
            return True, True, "Error in check - storing as fallback"
    
    def _store_face_snapshot(self, person_id: int, face_img: np.ndarray, bbox: Tuple[int, int, int, int],
                           quality_score: float, detection_confidence: float, landmarks=None,
                           timestamp=None, is_primary=False) -> Optional[Dict[str, Any]]:
        """
        Store face snapshot to file system and/or database
        
        Args:
            person_id (int): Person track ID
            face_img (np.ndarray): Face image
            bbox (tuple): Bounding box
            quality_score (float): Quality score
            detection_confidence (float): Detection confidence
            landmarks: Facial landmarks
            timestamp (datetime): Timestamp
            is_primary (bool): Whether this is the primary face
            
        Returns:
            dict: Storage result with paths and IDs
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            result = {}
            
            # Prepare face image for storage
            processed_face = self._prepare_face_for_storage(face_img)
            
            # Store to file system if enabled
            file_path = None
            if self.enable_file_storage:
                file_path = self._save_face_to_file(person_id, processed_face, timestamp, is_primary)
                result['file_path'] = file_path
            
            # Store to database if enabled
            snapshot_id = None
            if self.enable_db_storage and self.database:
                # Prepare data for database storage
                face_data = None
                embedding_data = None
                
                if self.config.get('store_image_in_db', False):
                    # Encode image as binary data
                    _, buffer = cv2.imencode(f'.{self.storage_format}', processed_face)
                    face_data = buffer.tobytes()
                
                if landmarks is not None:
                    # Serialize landmarks
                    embedding_data = pickle.dumps(landmarks)
                
                snapshot_id = self.database.store_face_snapshot(
                    person_id=person_id,
                    face_image_data=face_data,
                    face_image_path=file_path,
                    quality_score=quality_score,
                    confidence=detection_confidence,
                    bbox=bbox,
                    embedding_vector=embedding_data,
                    analysis_method='face_quality_scorer',
                    timestamp=timestamp,
                    is_primary=is_primary
                )
                
                result['snapshot_id'] = snapshot_id
            
            logger.info(f"Stored face snapshot for person {person_id} "
                       f"(quality: {quality_score:.3f}, primary: {is_primary})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error storing face snapshot: {e}")
            return None
    
    def _prepare_face_for_storage(self, face_img: np.ndarray) -> np.ndarray:
        """
        Prepare face image for storage (resize, enhance, etc.)
        
        Args:
            face_img (np.ndarray): Original face image
            
        Returns:
            np.ndarray: Processed face image
        """
        try:
            # Validate input
            if not isinstance(face_img, np.ndarray):
                logger.error(f"Expected numpy array, got {type(face_img)}")
                raise ValueError(f"face_img must be numpy array, got {type(face_img)}")
            
            if len(face_img.shape) < 2:
                logger.error(f"Invalid face image shape: {face_img.shape}")
                raise ValueError(f"Invalid face image shape: {face_img.shape}")
            
            # Resize to max dimensions if needed
            h, w = face_img.shape[:2]
            max_h, max_w = self.max_face_size
            
            if h > max_h or w > max_w:
                scale = min(max_h / h, max_w / w)
                new_h, new_w = int(h * scale), int(w * scale)
                face_img = cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Ensure minimum size
            h, w = face_img.shape[:2]
            if h < 48 or w < 48:
                face_img = cv2.resize(face_img, (max(48, w), max(48, h)), interpolation=cv2.INTER_CUBIC)
            
            return face_img
            
        except Exception as e:
            logger.error(f"Error preparing face for storage: {e}")
            return face_img
    
    def _save_face_to_file(self, person_id: int, face_img: np.ndarray, 
                          timestamp: datetime, is_primary: bool) -> Optional[str]:
        """
        Save face image to file system
        
        Args:
            person_id (int): Person track ID
            face_img (np.ndarray): Face image
            timestamp (datetime): Timestamp
            is_primary (bool): Whether this is primary face
            
        Returns:
            str: File path or None if failed
        """
        try:
            # Create person directory
            person_dir = self.storage_dir / str(person_id)
            person_dir.mkdir(exist_ok=True)
            
            # Generate filename
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
            prefix = "primary_" if is_primary else "face_"
            filename = f"{prefix}{timestamp_str}.{self.storage_format}"
            file_path = person_dir / filename
            
            # Save image
            if self.storage_format.lower() in ['jpg', 'jpeg']:
                cv2.imwrite(str(file_path), face_img, [cv2.IMWRITE_JPEG_QUALITY, self.storage_quality])
            elif self.storage_format.lower() == 'png':
                cv2.imwrite(str(file_path), face_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            else:
                cv2.imwrite(str(file_path), face_img)
            
            # Return relative path for database storage
            relative_path = str(file_path.relative_to(self.storage_dir.parent))
            return relative_path
            
        except Exception as e:
            logger.error(f"Error saving face to file: {e}")
            return None
    
    def get_primary_face_image(self, person_id: int) -> Optional[np.ndarray]:
        """
        Get the primary face image for a person
        
        Args:
            person_id (int): Person track ID
            
        Returns:
            np.ndarray: Face image or None if not found
        """
        try:
            if not self.database:
                return None
            
            # Get primary face snapshot record
            snapshot = self.database.get_primary_face_snapshot(person_id)
            if not snapshot:
                return None
            
            # Try to load from file first
            if snapshot.get('face_image_path'):
                file_path = Path(self.storage_dir.parent) / snapshot['face_image_path']
                if file_path.exists():
                    return cv2.imread(str(file_path))
            
            # Try to load from database blob
            if snapshot.get('face_image_data'):
                nparr = np.frombuffer(snapshot['face_image_data'], np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting primary face image: {e}")
            return None
    
    def get_face_thumbnail_base64(self, person_id: int, size: Tuple[int, int] = (64, 64)) -> Optional[str]:
        """
        Get a base64-encoded thumbnail of the primary face for web display
        
        Args:
            person_id (int): Person track ID
            size (tuple): Thumbnail size (width, height)
            
        Returns:
            str: Base64-encoded image or None
        """
        try:
            face_img = self.get_primary_face_image(person_id)
            if face_img is None:
                return None
            
            # Resize to thumbnail size
            thumbnail = cv2.resize(face_img, size, interpolation=cv2.INTER_AREA)
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # Convert to base64
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Error creating face thumbnail: {e}")
            return None
    
    def cleanup_person_faces(self, person_id: int) -> int:
        """
        Clean up all face snapshots for a person
        
        Args:
            person_id (int): Person track ID
            
        Returns:
            int: Number of files removed
        """
        try:
            removed_count = 0
            
            # Remove from file system
            person_dir = self.storage_dir / str(person_id)
            if person_dir.exists():
                for file_path in person_dir.glob(f"*.{self.storage_format}"):
                    file_path.unlink()
                    removed_count += 1
                
                # Remove directory if empty
                if not any(person_dir.iterdir()):
                    person_dir.rmdir()
            
            # Database cleanup is handled by foreign key constraints
            logger.info(f"Cleaned up {removed_count} face files for person {person_id}")
            return removed_count
            
        except Exception as e:
            logger.error(f"Error cleaning up faces for person {person_id}: {e}")
            return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics
        
        Returns:
            dict: Storage statistics
        """
        try:
            stats = {
                'total_persons': 0,
                'total_faces': 0,
                'total_size_mb': 0.0,
                'primary_faces': 0
            }
            
            if not self.storage_dir.exists():
                return stats
            
            for person_dir in self.storage_dir.iterdir():
                if person_dir.is_dir():
                    stats['total_persons'] += 1
                    
                    for face_file in person_dir.glob(f"*.{self.storage_format}"):
                        stats['total_faces'] += 1
                        stats['total_size_mb'] += face_file.stat().st_size / (1024 * 1024)
                        
                        if face_file.name.startswith('primary_'):
                            stats['primary_faces'] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {'error': str(e)} 