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
import threading
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

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
# Set up logger first
logger = logging.getLogger(__name__)

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("DeepFace loaded successfully")
except ImportError as e:
    DEEPFACE_AVAILABLE = False
    logger.warning(f"DeepFace not available: {e}")

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
    logger.info("InsightFace loaded successfully")
except ImportError as e:
    INSIGHTFACE_AVAILABLE = False
    logger.warning(f"InsightFace not available: {e}")

# Both required for advanced models
ADVANCED_MODELS_AVAILABLE = DEEPFACE_AVAILABLE and INSIGHTFACE_AVAILABLE
if ADVANCED_MODELS_AVAILABLE:
    logger.info("Advanced demographic models (DeepFace + InsightFace) available")
else:
    logger.warning(f"Advanced demographic models not available (DeepFace: {DEEPFACE_AVAILABLE}, InsightFace: {INSIGHTFACE_AVAILABLE})")


# Note: EnhancedDemographicAnalyzer class removed - now using simplified DemographicAnalyzer from analysis module


class FrameAnalysisService:
    """
    Enhanced Frame Analysis Service with Face Detection and Advanced Demographics
    
    This service maintains persistent tracking of faces across frames
    and performs sophisticated demographic analysis using simplified InsightFace buffalo_l model.
    """
    
    def __init__(self):
        """Initialize the analysis service with face detection"""
        # Import the new detection pipeline
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        from detection import get_detector, get_tracker
        from analysis import DemographicAnalyzer, FaceSnapshotManager
        import yaml
        from pathlib import Path
        
        # Load configuration to determine detection mode
        try:
            config_path = Path(__file__).parent.parent / 'config.yaml'
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Set up detection pipeline based on config mode
            detection_mode = config.get('mode', 'face')
            detection_config = config['detection'].copy()
            tracking_config = config['tracking'].copy()
            
            detection_config['mode'] = detection_mode
            tracking_config['mode'] = detection_mode
            
            if detection_mode == 'face':
                # Set model directory for face detection
                models_dir = Path(__file__).parent.parent.parent / 'models'
                detection_config['model_dir'] = str(models_dir)
            
            # Initialize new detection pipeline
            self.detector = get_detector(detection_config)
            self.tracker = get_tracker(tracking_config)
            self.detection_mode = detection_mode
            
            # Use simplified demographic analyzer with config
            self.demographic_analyzer = DemographicAnalyzer(config['analysis']['demographics'])
            
            # Initialize face snapshot manager
            face_storage_config = config.get('face_storage', {})
            face_storage_config.update(config['analysis'].get('demographics', {}))
            self.face_snapshot_manager = FaceSnapshotManager(face_storage_config, database=None)  # Database will be set later
            
            logger.info(f"FrameAnalysisService initialized with {detection_mode} detection mode and simplified demographic analyzer")
            
        except Exception as e:
            logger.error(f"Error initializing detection pipeline: {e}")
            # Fallback to legacy mode with simplified analyzer
            self.detector = None
            self.tracker = None
            self.detection_mode = 'person'
            self.yolo_model = None
            # Use simplified demographic analyzer as fallback
            fallback_config = {
                'model_dir': './models',
                'confidence_threshold': 0.6,
                'min_face_size': 48,
                'detection_interval': 15,
                'one_time_demographics': True
            }
            self.demographic_analyzer = DemographicAnalyzer(fallback_config)

        # Database reference (set later)
        self.database = None

        # Thread pools for async operations
        self.db_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='db_worker')
        self.analysis_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='analysis_worker')
        self.demographics_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='demographics_worker')
        
        # Rate limiting for frame analysis - optimized for RTSP stability
        self.analysis_lock = threading.Lock()
        self.last_analysis_time = 0
        self.min_analysis_interval = 0.1  # Increased to 100ms to prevent RTSP overload
        self.analysis_in_progress = False
        
        # Cache for recent analysis results - enhanced for better performance
        self.result_cache = {
            'last_result': None,
            'cache_time': 0,
            'cache_duration': 1.0  # Increased cache duration to reduce processing load
        }
        
        # Demographic analysis cache and queue
        self.demographic_cache = {
            'person_images': {},  # Cache person images to avoid re-analysis
            'processing_queue': {},  # Track what's being processed
            'results_cache': {},  # Cache completed demographic results
            'feature_cache': {},  # Cache visual features
            'last_cache_cleanup': time.time()
        }
        
        # Frame processing optimization settings - optimized to prevent RTSP overload
        self.processing_config = {
            'analysis_resolution': (960, 540),  # Reduced resolution for better performance
            'frame_skip_factor': 3,  # Process every 3rd frame for reduced load
            'demographic_skip_factor': 5,  # Reduced for more frequent analysis with simplified model
            'motion_threshold': 1000,  # Higher threshold for better stability
            'max_processing_time': 2.0,  # Increased for more stability
            'enable_motion_detection': True,
            'enable_frame_skipping': True,
            'enable_fast_yolo': True,
            'enable_async_demographics': False,  # Disabled for simplified synchronous processing
            'enable_demographic_caching': True,  # Keep caching enabled
            'yolo_confidence': 0.4,  # Higher for fewer false positives
            'yolo_iou': 0.5,  # Higher for better tracking
            'yolo_imgsz': 416  # Smaller input size for faster processing
        }
        
        # Motion detection state
        self.motion_detector = {
            'last_frame': None,
            'frame_count': 0,
            'motion_detected': True  # Start with motion to ensure first analysis
        }
        
        # Performance monitoring
        self.performance_stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'avg_processing_time': 0,
            'last_processing_times': [],
            'max_processing_time_samples': 10,
            'demographic_analysis_count': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        # Enhanced tracking and duplicate reduction
        self.detection_memory = {
            'active_tracks': {},
            'track_history': {},
            'last_frame_time': time.time(),
            'people_database': {},  # Persistent person database
            'reid_features': {},  # Re-identification features
            'track_id_counter': 1,
            'max_track_history': 100,
            'track_timeout': 30.0,  # 30 seconds
            'similarity_threshold': 0.8,
            'confirmation_frames': 3,
            'max_distance_threshold': 100,
            'frame_count': 0  # Add missing frame_count
        }

        # Initialize zone manager and zone-based dwell analyzer
        try:
            from analysis.zone_manager import ZoneManager
            from analysis.zone_dwell import ZoneDwellTimeAnalyzer
            from database.sqlite_db import SQLiteDatabase
            
            # For now, create a temporary database connection for zones
            # This will be properly connected when set_database is called
            self.zone_manager = None
            self.zone_dwell_analyzer = None
            
            logger.info("Zone management components imported successfully")
        except ImportError as e:
            logger.warning(f"Zone management not available: {e}")
            self.zone_manager = None
            self.zone_dwell_analyzer = None

        logger.info("FrameAnalysisService initialized with enhanced tracking and zone support")

        # Enhanced duplicate reduction settings
        self.duplicate_reduction = {
            'min_frames_for_db': 8,  # Reduced minimum frames for faster addition
            'min_confidence_for_db': 0.65,  # Reduced for faster processing
            'demographic_consistency_threshold': 0.75,  # Slightly lowered
            'feature_similarity_threshold': 0.8,  # Slightly lowered for better matching
            'temporal_window': 30.0,  # Time window for duplicate detection (seconds)
            'spatial_threshold': 100   # Pixel distance threshold for spatial duplicate detection
        }

        # Enhanced re-identification settings for maintaining person identity
        self.reid_settings = {
            'enable_reid': True,
            'reid_similarity_threshold': 0.7,  # Slightly lowered for better matching
            'reid_temporal_window': 300.0,  # 5 minute window for re-identification
            'reid_spatial_threshold': 200,  # Larger spatial threshold for re-ID
            'reid_confidence_boost': 0.1,  # Confidence boost for re-identified persons
            'reid_demographic_weight': 0.25,  # Reduced weight for faster processing
            'enable_face_embedding_reid': True,  # Use face embeddings for re-ID
            'face_embedding_threshold': 0.65,  # Lowered threshold for better matching
            'enable_progressive_reid': True,  # New: progressive re-identification
            'enable_cached_reid': True  # New: use cached results for re-identification
        }

        # Person re-identification database
        self.reid_database = {
            'face_embeddings': {},  # Store face embeddings for each person
            'demographic_signatures': {},  # Store demographic signatures
            'last_seen_locations': {},  # Store last known locations
            'person_id_mapping': {},  # Map old IDs to current IDs for continuity
            'visual_features': {},  # Store enhanced visual features
            'last_cleanup': time.time()  # Track cleanup timing
        }

        # Initialize enhanced tracker
        self._initialize_enhanced_tracker()
        
        # Start background cleanup task
        self._schedule_cache_cleanup()
    
    def set_database(self, database):
        """
        Set the database reference for the service and related components
        
        Args:
            database: Database connection object
        """
        self.database = database
        if hasattr(self, 'face_snapshot_manager') and self.face_snapshot_manager:
            self.face_snapshot_manager.database = database
            logger.info("Database reference set for FrameAnalysisService and FaceSnapshotManager")
        else:
            logger.warning("FaceSnapshotManager not available for database reference setting")
            
        # Initialize zone manager and zone-based dwell analyzer now that we have database
        try:
            if self.zone_manager is None:
                from analysis.zone_manager import ZoneManager
                self.zone_manager = ZoneManager(database)
                logger.info("ZoneManager initialized with database")
                
            if self.zone_dwell_analyzer is None:
                from analysis.zone_dwell import ZoneDwellTimeAnalyzer
                # Use dwell time config from main config
                dwell_config = {
                    'min_dwell_time': 0.5,  # Reduced from 2.0 to 0.5 seconds for more responsive feedback
                    'max_inactive_time': 10.0
                }
                self.zone_dwell_analyzer = ZoneDwellTimeAnalyzer(
                    dwell_config, 
                    self.zone_manager, 
                    database=database, 
                    camera_id='default'
                )
                logger.info(f"ZoneDwellTimeAnalyzer initialized with config: {dwell_config}")
                
                # Test zone loading
                zones = self.zone_manager.get_zones('default')
                logger.info(f"Loaded {len(zones)} zones for dwell analysis")
                
        except Exception as e:
            logger.error(f"Failed to initialize zone components with database: {e}", exc_info=True)
            self.zone_manager = None
            self.zone_dwell_analyzer = None
    
    def _schedule_cache_cleanup(self):
        """Schedule periodic cache cleanup to prevent memory leaks"""
        def cleanup_task():
            while True:
                time.sleep(60)  # Run every minute
                try:
                    self._cleanup_caches()
                except Exception as e:
                    logger.error(f"Error in cache cleanup: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_caches(self):
        """Clean up old cache entries to prevent memory leaks"""
        current_time = time.time()
        
        # Clean demographic cache (keep entries for 5 minutes)
        cache_timeout = 300.0
        for cache_name in ['person_images', 'results_cache', 'feature_cache']:
            cache = self.demographic_cache.get(cache_name, {})
            expired_keys = []
            for key, value in cache.items():
                if isinstance(value, dict) and 'timestamp' in value:
                    if current_time - value['timestamp'] > cache_timeout:
                        expired_keys.append(key)
            
            for key in expired_keys:
                cache.pop(key, None)
        
        # Clean reid database (keep entries for 10 minutes)
        reid_timeout = 600.0
        for db_name in ['face_embeddings', 'demographic_signatures', 'visual_features']:
            db = self.reid_database.get(db_name, {})
            expired_keys = []
            for key, value in db.items():
                if isinstance(value, dict) and 'timestamp' in value:
                    if current_time - value['timestamp'] > reid_timeout:
                        expired_keys.append(key)
                elif key not in self.detection_memory['people_database']:
                    # Remove entries for people no longer tracked
                    expired_keys.append(key)
            
            for key in expired_keys:
                db.pop(key, None)
        
        self.demographic_cache['last_cache_cleanup'] = current_time
        self.reid_database['last_cleanup'] = current_time
        
        logger.debug(f"Cache cleanup completed. Cache sizes: "
                    f"person_images={len(self.demographic_cache.get('person_images', {}))}, "
                    f"results_cache={len(self.demographic_cache.get('results_cache', {}))}, "
                    f"reid_db={len(self.reid_database.get('face_embeddings', {}))}")

    def _get_person_image_hash(self, person_img: np.ndarray) -> str:
        """Generate a hash for person image to use as cache key"""
        try:
            if person_img is None or person_img.size == 0:
                return "empty"
            
            # Use image properties to create a quick hash
            h, w = person_img.shape[:2]
            # Sample a few pixels for quick comparison
            sample_pixels = person_img[::h//4, ::w//4].flatten()[:50]
            hash_str = f"{h}x{w}_{hash(tuple(sample_pixels.astype(int)))}"
            return hash_str
        except Exception:
            return f"fallback_{time.time()}"
    
    def _get_cached_demographics(self, person_img: np.ndarray, person_id: int = None) -> Optional[Dict[str, Any]]:
        """Get cached demographic analysis if available"""
        if not self.processing_config.get('enable_demographic_caching', True):
            return None
        
        try:
            img_hash = self._get_person_image_hash(person_img)
            
            # Check results cache first
            if img_hash in self.demographic_cache['results_cache']:
                cached_result = self.demographic_cache['results_cache'][img_hash]
                if time.time() - cached_result['timestamp'] < 300:  # 5 minute cache
                    self.performance_stats['cache_hits'] += 1
                    logger.debug(f"Cache hit for demographic analysis: {img_hash}")
                    return cached_result['demographics']
            
            # Check person-specific cache
            if person_id and person_id in self.detection_memory['people_database']:
                person_data = self.detection_memory['people_database'][person_id]
                demographics = person_data.get('demographics', {})
                if demographics.get('confidence', 0) > 0.5:
                    # Use existing demographics if confident enough
                    self.performance_stats['cache_hits'] += 1
                    return demographics
            
            self.performance_stats['cache_misses'] += 1
            return None
            
        except Exception as e:
            logger.debug(f"Error checking demographic cache: {e}")
            return None
    
    def _cache_demographics(self, person_img: np.ndarray, demographics: Dict[str, Any], person_id: int = None):
        """Cache demographic analysis results"""
        try:
            if not self.processing_config.get('enable_demographic_caching', True):
                return
            
            img_hash = self._get_person_image_hash(person_img)
            
            # Cache in results cache
            self.demographic_cache['results_cache'][img_hash] = {
                'demographics': demographics,
                'timestamp': time.time(),
                'person_id': person_id
            }
            
            # Cache person image for similarity checks
            self.demographic_cache['person_images'][img_hash] = {
                'image': person_img.copy(),
                'timestamp': time.time(),
                'person_id': person_id
            }
            
            logger.debug(f"Cached demographic analysis: {img_hash} -> {demographics.get('gender', 'unknown')} {demographics.get('age_group', 'unknown')}")
            
        except Exception as e:
            logger.debug(f"Error caching demographics: {e}")
    
    def detect_and_analyze_face_async(self, person_img: np.ndarray, person_id: int, callback=None) -> Dict[str, Any]:
        """
        Asynchronous face detection and demographic analysis with caching
        
        Returns immediate result if cached, otherwise returns basic result and processes async
        """
        try:
            if person_img is None or person_img.size == 0:
                return self._get_fallback_demographics()
            
            # Check cache first
            cached_demographics = self._get_cached_demographics(person_img, person_id)
            if cached_demographics:
                return cached_demographics
            
            # Check if already processing this person
            img_hash = self._get_person_image_hash(person_img)
            if img_hash in self.demographic_cache['processing_queue']:
                # Return basic demographics while processing
                return self._get_basic_demographics()
            
            # Mark as processing
            self.demographic_cache['processing_queue'][img_hash] = {
                'person_id': person_id,
                'start_time': time.time()
            }
            
            # Return basic demographics immediately
            basic_demographics = self._get_basic_demographics()
            
            # Start async demographic analysis
            if self.processing_config.get('enable_async_demographics', True):
                future = self.demographics_executor.submit(
                    self._analyze_demographics_background, 
                    person_img.copy(), 
                    person_id, 
                    img_hash,
                    callback
                )
                # Store future for potential cancellation
                self.demographic_cache['processing_queue'][img_hash]['future'] = future
            
            return basic_demographics
            
        except Exception as e:
            logger.error(f"Error in async face detection and analysis: {e}")
            return self._get_fallback_demographics()
    
    def _analyze_demographics_background(self, person_img: np.ndarray, person_id: int, img_hash: str, callback=None):
        """Simplified demographic analysis task using single InsightFace model"""
        try:
            start_time = time.time()
            
            # Perform the actual demographic analysis using simplified analyzer
            demographics = self.demographic_analyzer.analyze(person_img, person_id)
            
            # Cache the results
            self._cache_demographics(person_img, demographics, person_id)
            
            # Update person database
            if person_id in self.detection_memory['people_database']:
                person_data = self.detection_memory['people_database'][person_id]
                
                # Check if this is a better result
                current_conf = person_data['demographics'].get('confidence', 0)
                new_conf = demographics.get('confidence', 0)
                
                if new_conf > current_conf or person_data.get('analyzing', False):
                    # Update with better demographics
                    person_data['demographics'] = demographics
                    person_data['analyzing'] = False
                    
                    logger.info(f"Completed demographic analysis for person {person_id}: {demographics.get('gender', 'unknown')} {demographics.get('age_group', 'unknown')} conf: {demographics.get('confidence', 0):.2f}")
                
                # Update reid database - creating a mock detection dict since we don't have the full detection data
                mock_detection = {
                    'center': (0, 0),
                    'person_image': person_img if 'person_img' in locals() else None
                }
                self._update_reid_database(person_id, mock_detection, demographics)
            
            # Remove from processing queue
            self.demographic_cache['processing_queue'].pop(img_hash, None)
            
            processing_time = time.time() - start_time
            self.performance_stats['demographic_analysis_count'] += 1
            
            logger.debug(f"Simplified demographic analysis completed for person {person_id} in {processing_time:.3f}s")
            
            # Call callback if provided
            if callback:
                try:
                    callback(person_id, demographics)
                except Exception as e:
                    logger.error(f"Error in demographic analysis callback: {e}")
            
            return demographics
            
        except Exception as e:
            logger.error(f"Error in simplified demographic analysis: {e}")
            # Remove from processing queue on error
            self.demographic_cache['processing_queue'].pop(img_hash, None)
            return self._get_fallback_demographics()
    
    def _get_basic_demographics(self) -> Dict[str, Any]:
        """Return basic demographics while full analysis is processing"""
        return {
            'age_group': 'analyzing...',
            'gender': 'analyzing...',
            'emotion': 'neutral',
            'confidence': 0.3,
            'age': 25,
            'race': 'unknown',
            'analysis_method': 'processing'
        }
    
    def _get_fallback_demographics(self) -> Dict[str, Any]:
        """Return fallback demographics when analysis fails"""
        return {
            'age_group': 'unknown',
            'gender': 'unknown',
            'emotion': 'neutral',
            'confidence': 0.1,
            'age': 0,
            'race': 'unknown',
            'analysis_method': 'fallback'
        }
    
    def detect_and_analyze_face(self, person_img: np.ndarray) -> Dict[str, Any]:
        """
        Enhanced face detection and analysis with intelligent caching
        
        This method now uses caching and async processing for better performance
        """
        try:
            if person_img is None or person_img.size == 0:
                return self._get_fallback_demographics()
            
            # Try to get cached result first
            cached_demographics = self._get_cached_demographics(person_img)
            if cached_demographics:
                return cached_demographics
            
            # For immediate response, try quick demographic estimation
            if self.processing_config.get('enable_async_demographics', True):
                # Return basic result for immediate response
                return self._get_basic_demographics()
            else:
                # Fallback to synchronous analysis (original behavior)
                demographics = self.demographic_analyzer.analyze(person_img)
                self._cache_demographics(person_img, demographics)
                
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
            return self._get_fallback_demographics()

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

    def _detect_motion(self, frame: np.ndarray) -> bool:
        """
        Detect motion between frames to optimize processing
        
        Args:
            frame: Current frame to analyze
            
        Returns:
            bool: True if significant motion detected
        """
        if not self.processing_config['enable_motion_detection']:
            return True
            
        try:
            # Convert to grayscale for motion detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
            
            # If this is the first frame, store it and return True
            if self.motion_detector['last_frame'] is None:
                self.motion_detector['last_frame'] = gray_frame
                return True
            
            # Ensure frames have the same size before comparison
            last_frame = self.motion_detector['last_frame']
            if last_frame.shape != gray_frame.shape:
                # Resize last frame to match current frame size
                last_frame = cv2.resize(last_frame, (gray_frame.shape[1], gray_frame.shape[0]))
                logger.debug(f"Resized last frame from {self.motion_detector['last_frame'].shape} to {gray_frame.shape}")
            
            # Calculate difference between frames
            frame_diff = cv2.absdiff(last_frame, gray_frame)
            thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Count non-zero pixels
            motion_pixels = cv2.countNonZero(thresh)
            
            # Update last frame
            self.motion_detector['last_frame'] = gray_frame
            
            # Check if motion exceeds threshold
            motion_detected = motion_pixels > self.processing_config['motion_threshold']
            
            if motion_detected:
                logger.debug(f"Motion detected: {motion_pixels} pixels changed")
            
            return motion_detected
            
        except Exception as e:
            logger.error(f"Error in motion detection: {e}")
            return True  # Default to processing if motion detection fails

    def _optimize_frame_for_analysis(self, frame: np.ndarray) -> np.ndarray:
        """
        Optimize frame for faster analysis
        
        Args:
            frame: Original frame
            
        Returns:
            np.ndarray: Optimized frame for analysis
        """
        try:
            # Get target resolution for analysis
            target_width, target_height = self.processing_config['analysis_resolution']
            
            # Calculate scaling factors
            height, width = frame.shape[:2]
            scale_x = target_width / width
            scale_y = target_height / height
            
            # Use smaller scale to maintain aspect ratio
            scale = min(scale_x, scale_y)
            
            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize frame
            optimized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            logger.debug(f"Frame optimized from {width}x{height} to {new_width}x{new_height}")
            
            return optimized_frame
            
        except Exception as e:
            logger.error(f"Error optimizing frame: {e}")
            return frame

    def _should_skip_frame(self) -> bool:
        """
        Determine if current frame should be skipped based on frame skipping factor
        
        Returns:
            bool: True if frame should be skipped
        """
        if not self.processing_config['enable_frame_skipping']:
            return False
            
        self.motion_detector['frame_count'] += 1
        skip_factor = self.processing_config['frame_skip_factor']
        
        # Process every skip_factor-th frame
        should_process = (self.motion_detector['frame_count'] % skip_factor) == 0
        
        if not should_process:
            logger.debug(f"Skipping frame {self.motion_detector['frame_count']} (factor: {skip_factor})")
        
        return not should_process

    def _update_performance_stats(self, processing_time: float):
        """
        Update performance statistics
        
        Args:
            processing_time: Time taken for processing in seconds
        """
        self.performance_stats['processed_frames'] += 1
        self.performance_stats['last_processing_times'].append(processing_time)
        
        # Keep only recent processing times
        max_samples = self.performance_stats['max_processing_time_samples']
        if len(self.performance_stats['last_processing_times']) > max_samples:
            self.performance_stats['last_processing_times'] = \
                self.performance_stats['last_processing_times'][-max_samples:]
        
        # Calculate average processing time
        self.performance_stats['avg_processing_time'] = \
            sum(self.performance_stats['last_processing_times']) / \
            len(self.performance_stats['last_processing_times'])
        
        # Log performance if processing is slow
        if processing_time > self.processing_config['max_processing_time']:
            logger.warning(f"Slow processing detected: {processing_time:.2f}s (avg: {self.performance_stats['avg_processing_time']:.2f}s)")

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
        """Detect people/faces in frame using the configured detection pipeline"""
        detections = []
        
        try:
            # Use new detection pipeline if available
            if self.detector and self.detection_mode == 'face':
                # Use face detection
                face_detections = self.detector.detect(frame)
                
                for face_detection in face_detections:
                    x1, y1, x2, y2 = face_detection['bbox']
                    w, h = x2 - x1, y2 - y1
                    
                    # Extract face image
                    try:
                        face_img = frame[y1:y2, x1:x2]
                        if face_img.size == 0:
                            continue
                    except Exception as e:
                        logger.debug(f"Error extracting face image: {e}")
                        continue
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],  # Keep original x1,y1,x2,y2 format
                        'center': (float(x1 + w/2), float(y1 + h/2)),
                        'confidence': float(face_detection['confidence']),
                        'person_image': face_img,  # This is actually face image now
                        'area': int(w * h),
                        'face_data': face_detection,  # Store additional face data
                        'quality_score': face_detection.get('quality_score', 1.0),
                        'embedding': face_detection.get('embedding')
                    })
                
                logger.debug(f"Face detection found {len(detections)} faces")
                return detections
                
            elif self.detector and self.detection_mode == 'person':
                # Use person detection (for backward compatibility)
                # This would need implementation if needed
                logger.warning("Person detection mode not fully implemented in new pipeline")
                return self._detect_people_legacy(frame)
            else:
                # Fallback to legacy YOLO detection
                return self._detect_people_legacy(frame)
                
        except Exception as e:
            logger.error(f"Error in detection pipeline: {e}")
            # Fallback to legacy detection
            return self._detect_people_legacy(frame)

    def _detect_people_legacy(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Legacy YOLO person detection for fallback"""
        detections = []
        model = self._get_model()
        if model is None:
            return detections
        
        try:
            # Optimize frame for YOLO if enabled
            yolo_frame = frame
            if self.processing_config['enable_fast_yolo']:
                # Use the optimized frame for analysis (640x360)
                yolo_frame = self._optimize_frame_for_analysis(frame)
                
                # Calculate scale factor for bbox conversion back to original frame
                original_height, original_width = frame.shape[:2]
                yolo_height, yolo_width = yolo_frame.shape[:2]
                scale_x = original_width / yolo_width
                scale_y = original_height / yolo_height
            else:
                scale_x = scale_y = 1.0
            
            # Run YOLO detection with optimized parameters
            yolo_conf = self.processing_config.get('yolo_confidence', 0.3)
            yolo_iou = self.processing_config.get('yolo_iou', 0.4)
            yolo_imgsz = self.processing_config.get('yolo_imgsz', 416)
            
            results = model(yolo_frame, conf=yolo_conf, iou=yolo_iou, verbose=False, imgsz=yolo_imgsz)
            
            # Process results
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls.item())
                        if cls == 0:  # Person class
                            confidence = float(box.conf.item())
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            # Scale coordinates back to original frame if needed
                            if scale_x != 1.0 or scale_y != 1.0:
                                x1 = int(x1 * scale_x)
                                y1 = int(y1 * scale_y)
                                x2 = int(x2 * scale_x)
                                y2 = int(y2 * scale_y)
                            
                            x, y = int(x1), int(y1)
                            w, h = int(x2 - x1), int(y2 - y1)
                            
                            # Ensure bounding box is within frame bounds
                            x = max(0, min(x, frame.shape[1] - 1))
                            y = max(0, min(y, frame.shape[0] - 1))
                            w = max(1, min(w, frame.shape[1] - x))
                            h = max(1, min(h, frame.shape[0] - y))
                            
                            # Extract person image for feature comparison from original frame
                            try:
                                person_img = frame[y:y+h, x:x+w]
                                if person_img.size == 0:
                                    continue
                            except Exception as e:
                                logger.debug(f"Error extracting person image: {e}")
                                continue
                            
                            detections.append({
                                'bbox': [int(x), int(y), int(w), int(h)],
                                'center': (float(x + w/2), float(y + h/2)),
                                'confidence': float(confidence),
                                'person_image': person_img,
                                'area': int(w * h)
                            })
                
        except Exception as e:
            logger.error(f"Error in legacy YOLO detection: {e}")
        
        return detections

    def detect_and_analyze_face(self, person_img: np.ndarray) -> Dict[str, Any]:
        """Enhanced face detection and analysis using optimized DemographicAnalyzer with face extraction"""
        try:
            if person_img is None or person_img.size == 0:
                logger.debug("No person image provided for face analysis")
                return {
                    'age_group': 'unknown',
                    'gender': 'unknown', 
                    'emotion': 'neutral',
                    'confidence': 0.0,
                    'analysis_method': 'no_image'
                }
            
            logger.debug(f"Starting enhanced face analysis on person image: {person_img.shape}")
            
            # Use enhanced demographic analyzer with face extraction
            demographics = self.demographic_analyzer.analyze(person_img)
            
            # Log successful analysis with detailed information
            method = demographics.get('analysis_method', 'unknown')
            confidence = demographics.get('confidence', 0)
            gender = demographics.get('gender', 'unknown')
            age_group = demographics.get('age_group', 'unknown')
            age = demographics.get('age', 'unknown')
            
            if confidence > 0.5:
                logger.info(f"Enhanced face analysis successful: {gender} {age_group} (age: {age}, conf: {confidence:.2f}, method: {method})")
            elif confidence > 0.3:
                logger.info(f"Moderate face analysis: {gender} {age_group} (conf: {confidence:.2f}, method: {method})")
            else:
                logger.warning(f"Low confidence face analysis: {gender} {age_group} (conf: {confidence:.2f}, method: {method})")
            
            # Format for compatibility with existing code
            return {
                'age_group': age_group,
                'gender': gender,
                'emotion': demographics.get('emotion', 'neutral'),
                'confidence': confidence,
                'age': demographics.get('age'),
                'race': demographics.get('race'),
                'analysis_method': method,
                'face_area': demographics.get('face_area', 0),
                'det_score': demographics.get('det_score', 0),
                'is_extracted_face': demographics.get('is_extracted_face', False)
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced face detection and analysis: {e}", exc_info=True)
            return {
                'age_group': 'unknown',
                'gender': 'unknown', 
                'emotion': 'neutral',
                'confidence': 0.1,
                'analysis_method': 'error'
            }

    def track_people(self, new_detections: List[Dict], frame_shape: Tuple[int, int]) -> List[Dict]:
        """
        Enhanced track people/faces across frames with duplicate reduction and re-identification

        Args:
            new_detections: List of new detections (faces or persons) with images
            frame_shape: Shape of the current frame (height, width)
            
        Returns:
            List of tracked people with persistent IDs and demographics
        """
        frame_time = datetime.now().timestamp()

        # Use new tracker if available
        if self.tracker is not None:
            return self._enhanced_track_people_new(new_detections, frame_shape, frame_time)
        elif self.person_tracker is not None:
            # Use legacy enhanced tracker
            return self._enhanced_track_people(new_detections, frame_shape, frame_time)
        else:
            # Fallback to legacy tracking
            return self._legacy_track_people(new_detections, frame_shape, frame_time)

    def _enhanced_track_people_new(self, new_detections: List[Dict], frame_shape: Tuple[int, int], frame_time: float) -> List[Dict]:
        """Enhanced tracking using the new FaceTracker/PersonTracker with improved demographic analysis"""
        
        try:
            # Convert detections to tracker format
            tracker_detections = []
            for detection in new_detections:
                bbox = detection['bbox']
                if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:  # Ensure valid bbox
                    if self.detection_mode == 'face':
                        # Face detection uses x1,y1,x2,y2 format
                        x1, y1, x2, y2 = bbox
                        w, h = x2 - x1, y2 - y1
                        tracker_detections.append({
                            'bbox': (x1, y1, x2, y2),  # Already in correct format
                            'confidence': detection['confidence'],
                            'quality_score': detection.get('quality_score', 1.0),
                            'embedding': detection.get('embedding'),
                            'face_size': detection.get('area', w*h)
                        })
                    else:
                        # Person detection might use x,y,w,h format  
                        x, y, w, h = bbox
                        tracker_detections.append({
                            'bbox': (x, y, x+w, y+h),  # Convert to x1,y1,x2,y2 format
                            'confidence': detection['confidence'],
                            'quality_score': detection.get('quality_score', 1.0),
                            'embedding': detection.get('embedding'),
                            'face_size': None
                        })
            
            # Create dummy frame for tracking if needed
            frame = None
            if self.detection_mode == 'person' and len(new_detections) > 0:
                # Create dummy frame for person tracking feature extraction
                frame = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
                for detection in new_detections:
                    if 'person_image' in detection and detection['person_image'] is not None:
                        x, y, w, h = detection['bbox']
                        person_img = detection['person_image']
                        if person_img.shape[0] > 0 and person_img.shape[1] > 0:
                            resized = cv2.resize(person_img, (w, h))
                            frame[y:y+h, x:x+w] = resized
            
            # Update tracker
            active_tracks = self.tracker.update(tracker_detections, frame)
            
            # Process tracks and analyze demographics
            tracked_people = []
            people_db = self.detection_memory['people_database']
            
            for track in active_tracks:
                track_id = track['id']
                
                # Find matching detection with person image
                matching_detection = None
                for detection in new_detections:
                    det_center = detection['center']
                    track_center = ((track['bbox'][0] + track['bbox'][2]) / 2, (track['bbox'][1] + track['bbox'][3]) / 2)
                    
                    # Calculate distance between detection and track
                    distance = np.sqrt((det_center[0] - track_center[0])**2 + (det_center[1] - track_center[1])**2)
                    
                    if distance < 50:  # Reasonable matching threshold
                        matching_detection = detection
                        break
                
                # Analyze demographics if we have a matching detection with image
                if matching_detection and 'person_image' in matching_detection:
                    person_image = matching_detection['person_image']
                    
                    # Validate person image quality
                    if person_image is None or person_image.size == 0:
                        logger.warning(f"Invalid person image for track {track_id}")
                        continue
                    
                    # Check if this person is already in our database
                    if track_id not in people_db:
                        # New person - use enhanced demographic analysis 
                        try:
                            logger.debug(f"Analyzing new person {track_id} with image shape: {person_image.shape}")
                            
                            # Set database reference for face snapshot manager
                            if hasattr(self, 'database') and self.database:
                                self.face_snapshot_manager.database = self.database
                            
                            # Simplified demographic analysis using InsightFace primary + DeepFace fallback
                            demo_data = self.demographic_analyzer.analyze(
                                person_img=person_image,
                                person_id=track_id
                            )
                            
                            # Process face snapshot for storage and reidentification
                            face_snapshot_result = None
                            if demo_data.get('confidence', 0) >= 0.3:  # Only save faces with reasonable confidence
                                try:
                                    # Extract face bounding box from track
                                    face_bbox = track['bbox']
                                    detection_confidence = track.get('confidence', 0.7)
                                    
                                    # Process face snapshot (includes quality assessment and storage)
                                    face_snapshot_result = self.face_snapshot_manager.process_face_snapshot(
                                        person_id=track_id,
                                        face_img=person_image,
                                        bbox=face_bbox,
                                        detection_confidence=detection_confidence,
                                        landmarks=None,  # Could be enhanced later
                                        timestamp=datetime.fromtimestamp(frame_time)
                                    )
                                    
                                    if face_snapshot_result and face_snapshot_result.get('stored'):
                                        logger.info(f"Saved face snapshot for person {track_id} "
                                                  f"(quality: {face_snapshot_result.get('quality_score', 0):.3f}, "
                                                  f"primary: {face_snapshot_result.get('is_primary', False)})")
                                    
                                except Exception as e:
                                    logger.warning(f"Failed to process face snapshot for person {track_id}: {e}")
                            
                            # Validate demographic results
                            if demo_data.get('confidence', 0) < 0.2:
                                logger.warning(f"Low confidence demographic analysis for person {track_id}: {demo_data.get('confidence', 0):.2f}")
                            
                            # Store in people database
                            people_db[track_id] = {
                                'id': track_id,
                                'demographics': demo_data,
                                'first_seen': frame_time,
                                'last_seen': frame_time,
                                'total_detections': 1,
                                'bbox_history': [track['bbox']],
                                'confidence_history': [track['confidence']],
                                'saved_to_db': False,
                                'analysis_method': demo_data.get('analysis_method', 'enhanced'),
                                'quality_scores': [demo_data.get('confidence', 0.1)]
                            }
                            
                            # Log successful demographic analysis
                            confidence = demo_data.get('confidence', 0)
                            gender = demo_data.get('gender', 'unknown')
                            age_group = demo_data.get('age_group', 'unknown')
                            method = demo_data.get('analysis_method', 'unknown')
                            
                            if confidence > 0.5:
                                logger.info(f"✓ High-quality demographics for person {track_id}: {gender} {age_group} (conf: {confidence:.2f}, method: {method})")
                            elif confidence > 0.3:
                                logger.info(f"○ Moderate demographics for person {track_id}: {gender} {age_group} (conf: {confidence:.2f}, method: {method})")
                            else:
                                logger.warning(f"△ Low-quality demographics for person {track_id}: {gender} {age_group} (conf: {confidence:.2f}, method: {method})")
                                
                            # Save person demographics to database
                            try:
                                # Mark as not saved initially  
                                people_db[track_id]['saved_to_db'] = False
                                
                                # Save detection first to get detection_id
                                detection_id = None
                                try:
                                    detection_id = self._save_detection_to_database_sync(track_id, track['bbox'], track['confidence'], frame_time)
                                    logger.debug(f"📄 Saved detection for person {track_id} to database, got ID: {detection_id}")
                                except Exception as e:
                                    logger.error(f"Failed to save detection for person {track_id}: {e}")
                                
                                # Save demographics with detection_id link
                                if confidence >= 0.3:  # Save demographics with reasonable confidence
                                    result = self._save_person_to_database_sync(track_id, demo_data, matching_detection, frame_time, detection_id)
                                    if result:
                                        people_db[track_id]['saved_to_db'] = True
                                        logger.info(f"📄 Saved person {track_id} demographics to database with detection_id: {detection_id}")
                                    else:
                                        logger.error(f"Failed to save person {track_id} demographics to database")
                                else:
                                    logger.debug(f"📄 Skipped database save for person {track_id} due to low confidence ({confidence:.2f})")
                                    
                            except Exception as e:
                                logger.error(f"Failed to save person {track_id} to database: {e}")
                                
                            # Save detection data to database
                            try:
                                self._save_detection_to_database_sync(track_id, track['bbox'], track['confidence'], frame_time)
                                logger.debug(f"📄 Saved detection for person {track_id} to database")
                            except Exception as e:
                                logger.error(f"Failed to save detection for person {track_id}: {e}")
                                
                        except Exception as e:
                            logger.error(f"Error in enhanced demographic analysis for person {track_id}: {e}")
                            # Fallback to basic demographics with better defaults
                            demo_data = {
                                'age': 30,
                                'age_group': '25-34',
                                'gender': 'unknown',
                                'race': 'unknown',
                                'emotion': 'neutral',
                                'confidence': 0.2,  # Low but not zero confidence
                                'analysis_method': 'fallback_enhanced'
                            }
                            
                            people_db[track_id] = {
                                'id': track_id,
                                'demographics': demo_data,
                                'first_seen': frame_time,
                                'last_seen': frame_time,
                                'total_detections': 1,
                                'bbox_history': [track['bbox']],
                                'confidence_history': [track['confidence']],
                                'saved_to_db': False,
                                'analysis_method': 'fallback_enhanced'
                            }
                            
                            logger.info(f"Used enhanced fallback demographics for person {track_id}")
                            
                            # Save fallback demographics to database
                            try:
                                self._save_person_to_database_sync(track_id, demo_data, matching_detection, frame_time)
                                people_db[track_id]['saved_to_db'] = True
                                logger.info(f"📄 Saved fallback person {track_id} demographics to database")
                                
                                # Save detection data to database
                                self._save_detection_to_database_sync(track_id, track['bbox'], track['confidence'], frame_time)
                                logger.debug(f"📄 Saved detection for fallback person {track_id} to database")
                            except Exception as e:
                                logger.error(f"Failed to save fallback person {track_id} to database: {e}")
                    else:
                        # Update existing person with potential improvement
                        people_db[track_id]['last_seen'] = frame_time
                        people_db[track_id]['total_detections'] += 1
                        people_db[track_id]['bbox_history'].append(track['bbox'])
                        people_db[track_id]['confidence_history'].append(track['confidence'])
                        
                        # Try to improve demographics if current confidence is low
                        current_confidence = people_db[track_id]['demographics'].get('confidence', 0)
                        if current_confidence < 0.7:  # Try to improve if confidence is not high
                            try:
                                new_demo_data = self.demographic_analyzer.analyze(
                                    person_img=person_image,
                                    person_id=track_id
                                )
                                
                                new_confidence = new_demo_data.get('confidence', 0)
                                if new_confidence > current_confidence + 0.1:  # Significant improvement
                                    people_db[track_id]['demographics'] = new_demo_data
                                    people_db[track_id]['analysis_method'] = new_demo_data.get('analysis_method', 'enhanced')
                                    logger.debug(f"Improved demographics for person {track_id}: {current_confidence:.2f} → {new_confidence:.2f}")
                                    
                                    # Save improved demographics to database
                                    try:
                                        if new_confidence >= 0.3:  # Save improved demographics with good confidence
                                            self._save_person_to_database_sync(track_id, new_demo_data, matching_detection, frame_time)
                                            people_db[track_id]['saved_to_db'] = True
                                            logger.info(f"📄 Updated person {track_id} demographics in database")
                                    except Exception as e:
                                        logger.error(f"Failed to update person {track_id} in database: {e}")
                                    
                                    # Process improved face snapshot if confidence is good enough
                                    if new_confidence >= 0.5:  # Higher threshold for updates
                                        try:
                                            face_bbox = track['bbox']
                                            detection_confidence = track.get('confidence', 0.7)
                                            
                                            face_snapshot_result = self.face_snapshot_manager.process_face_snapshot(
                                                person_id=track_id,
                                                face_img=person_image,
                                                bbox=face_bbox,
                                                detection_confidence=detection_confidence,
                                                landmarks=None,
                                                timestamp=datetime.fromtimestamp(frame_time)
                                            )
                                            
                                            if face_snapshot_result and face_snapshot_result.get('stored'):
                                                logger.info(f"Updated face snapshot for person {track_id} with improved quality")
                                        except Exception as e:
                                            logger.warning(f"Failed to update face snapshot for person {track_id}: {e}")
                                    
                            except Exception as e:
                                logger.debug(f"Failed to improve demographics for person {track_id}: {e}")
                        
                        # Save detection data for existing person (always save detections)
                        try:
                            self._save_detection_to_database_sync(track_id, track['bbox'], track['confidence'], frame_time)
                            logger.debug(f"📄 Saved detection for existing person {track_id} to database")
                        except Exception as e:
                            logger.error(f"Failed to save detection for existing person {track_id}: {e}")
                
                # Add to tracked people list
                person_data = people_db.get(track_id, {})
                demographics = person_data.get('demographics', {})
                
                # Convert bbox to frontend format [x, y, width, height]
                if self.detection_mode == 'face':
                    # Convert from (x1, y1, x2, y2) to [x, y, w, h]
                    x1, y1, x2, y2 = track['bbox']
                    bbox_frontend = [x1, y1, x2 - x1, y2 - y1]
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                else:
                    # Person detection bbox is already in correct format
                    bbox_frontend = track['bbox']
                    center = ((track['bbox'][0] + track['bbox'][2]) / 2, (track['bbox'][1] + track['bbox'][3]) / 2)
                
                tracked_person = {
                    'id': track_id,
                    'bbox': bbox_frontend,
                    'center': center,
                    'confidence': track['confidence'],
                    'demographics': demographics,
                    'frames_tracked': track.get('frames_tracked', 1),
                    'quality_score': track.get('quality_score', 1.0),
                    'first_seen': person_data.get('first_seen', frame_time),
                    'last_seen': person_data.get('last_seen', frame_time),
                    'total_detections': person_data.get('total_detections', 1)
                }
                
                # Add zone information if zone manager is available
                if self.zone_manager:
                    try:
                        # Get center point for zone calculation
                        cx, cy = center
                        zone_id = self.zone_manager.point_to_zone('default', int(cx), int(cy))
                        zones = self.zone_manager.get_zones('default')
                        zone_name = 'Outside'
                        
                        # Find zone name
                        for zone in zones:
                            if zone['id'] == zone_id:
                                zone_name = zone['name']
                                break
                        
                        tracked_person['zone_id'] = zone_id
                        tracked_person['zone_name'] = zone_name
                        
                    except Exception as e:
                        logger.debug(f"Error adding zone info for person {track_id}: {e}")
                        tracked_person['zone_id'] = 0
                        tracked_person['zone_name'] = 'Outside'

                tracked_people.append(tracked_person)
            
            return tracked_people
            
        except Exception as e:
            logger.error(f"Error in enhanced people tracking: {e}")
            # Fallback to legacy tracking
            if self.person_tracker is not None:
                return self._enhanced_track_people(new_detections, frame_shape, frame_time)
            else:
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
                    if self.detection_mode == 'face':
                        # Face detection uses x1,y1,x2,y2 format
                        x1, y1, x2, y2 = detection['bbox']
                        w, h = x2 - x1, y2 - y1
                        x, y = x1, y1
                    else:
                        # Person detection uses x,y,w,h format
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
            if self.detection_mode == 'face':
                # Face detection uses x1,y1,x2,y2 format
                x1, y1, x2, y2 = detection['bbox']
                tracker_detections.append({
                    'bbox': (x1, y1, x2, y2),  # Already in correct format
                    'confidence': detection['confidence'],
                    'center': detection['center']
                })
            else:
                # Person detection uses x,y,w,h format
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
                        # New person - use async demographic analysis for immediate response
                        demographics = self.detect_and_analyze_face_async(
                            matching_detection['person_image'], 
                            track_id,
                            callback=lambda pid, demo: self._on_demographic_analysis_complete(pid, demo, matching_detection, frame_time)
                        )

                        # Add person immediately with basic demographics
                        people_db[track_id] = {
                            'demographics': demographics,
                            'first_seen': frame_time,
                            'last_seen': frame_time,
                            'total_appearances': 1,
                            'average_confidence': track['confidence'],
                            'frames_tracked': track['frames_tracked'],
                            'analyzing': demographics.get('analysis_method') == 'processing'
                        }

                        # Store visual features for re-identification
                        if frame is not None:
                            features = self.person_tracker._extract_visual_features(frame, track['bbox'])
                            if features is not None:
                                self.detection_memory['feature_database'][track_id] = features

                        # Save basic info to database immediately (demographics will be updated async)
                        self._save_person_to_database_sync(track_id, demographics, matching_detection, frame_time)

                        logger.info(f"Added person {track_id} to database: {demographics.get('gender', 'unknown')} {demographics.get('age_group', 'unknown')} conf: {demographics.get('confidence', 0):.2f}")

                    else:
                        # Existing person - update info and potentially re-analyze
                        people_db[track_id]['last_seen'] = frame_time
                        people_db[track_id]['total_appearances'] += 1
                        people_db[track_id]['average_confidence'] = (
                            people_db[track_id]['average_confidence'] * 0.8 + track['confidence'] * 0.2
                        )

                        # Check if we should trigger re-analysis for better demographics
                        current_demo = people_db[track_id]['demographics']
                        current_demo_conf = current_demo.get('confidence', 0)
                        
                        # Re-analyze if we have significantly better confidence or if still analyzing
                        should_reanalyze = (
                            track['confidence'] > current_demo_conf + 0.15 or
                            current_demo.get('analysis_method') == 'processing' or
                            current_demo_conf < 0.6
                        )
                        
                        if should_reanalyze:
                            # Check if not already processing
                            img_hash = self._get_person_image_hash(matching_detection['person_image'])
                            if img_hash not in self.demographic_cache['processing_queue']:
                                new_demographics = self.detect_and_analyze_face_async(
                                    matching_detection['person_image'], 
                                    track_id,
                                    callback=lambda pid, demo: self._on_demographic_analysis_complete(pid, demo, matching_detection, frame_time)
                                )

                                # Update immediately if we got better results
                                if new_demographics.get('confidence', 0) > current_demo_conf:
                                    people_db[track_id]['demographics'] = new_demographics

                # Store detection data to database for all confirmed tracks
                if matching_detection:
                    self._save_detection_to_database_sync(track_id, track['bbox'], track['confidence'], frame_time)

            # Create tracked person object
            demographics = people_db.get(track_id, {}).get('demographics', {
                'age_group': 'analyzing...',
                'gender': 'analyzing...',
                'emotion': 'neutral',
                'confidence': 0.1
            })

            # Convert bbox to frontend format [x, y, width, height]
            if self.detection_mode == 'face':
                # Convert from (x1, y1, x2, y2) to [x, y, w, h]
                x1, y1, x2, y2 = track['bbox']
                bbox_frontend = [x1, y1, x2 - x1, y2 - y1]
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
            else:
                # Person detection bbox is already in correct format
                bbox_frontend = track['bbox']
                center = ((track['bbox'][0] + track['bbox'][2]) / 2, (track['bbox'][1] + track['bbox'][3]) / 2)

            tracked_person = {
                'id': track_id,
                'bbox': bbox_frontend,
                'center': center,
                'confidence': track['confidence'],
                'frames_tracked': track['frames_tracked'],
                'demographics': demographics,
                'dwell_time': self._calculate_person_dwell_time(track_id, track),  # Use proper dwell time calculation
                'is_confirmed': track['is_confirmed']
            }
            
            # Add zone information if zone manager is available
            if self.zone_manager:
                try:
                    # Get center point for zone calculation
                    cx, cy = center
                    zone_id = self.zone_manager.point_to_zone('default', int(cx), int(cy))
                    zones = self.zone_manager.get_zones('default')
                    zone_name = 'Outside'
                    
                    # Find zone name
                    for zone in zones:
                        if zone['id'] == zone_id:
                            zone_name = zone['name']
                            break
                    
                    tracked_person['zone_id'] = zone_id
                    tracked_person['zone_name'] = zone_name
                    
                except Exception as e:
                    logger.debug(f"Error adding zone info for person {track_id}: {e}")
                    tracked_person['zone_id'] = 0
                    tracked_person['zone_name'] = 'Outside'

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
                    'dwell_time': self._calculate_person_dwell_time(track_id, {'frames_tracked': track_info['frames_tracked']}),
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
        """Enhanced frame analysis with comprehensive optimization to prevent RTSP stream overload"""
        analysis_start_time = time.time()
        current_time = analysis_start_time
        
        # Update performance stats
        self.performance_stats['total_frames'] += 1
        
        # Check rate limiting and return cached result if called too frequently
        with self.analysis_lock:
            if self.analysis_in_progress:
                # Return cached result if analysis is already in progress
                if (self.result_cache['last_result'] and 
                    current_time - self.result_cache['cache_time'] < self.result_cache['cache_duration']):
                    logger.debug("Analysis in progress, returning cached result")
                    return self.result_cache['last_result']
                else:
                    return {"error": "Analysis in progress, please wait"}
            
            # Check minimum interval between analyses
            if current_time - self.last_analysis_time < self.min_analysis_interval:
                # Return cached result if we have recent analysis
                if (self.result_cache['last_result'] and 
                    current_time - self.result_cache['cache_time'] < self.result_cache['cache_duration']):
                    logger.debug(f"Rate limited, returning cached result (last analysis: {current_time - self.last_analysis_time:.3f}s ago)")
                    return self.result_cache['last_result']
                else:
                    return {"error": "Rate limited, analysis too frequent"}
            
            # Set analysis in progress
            self.analysis_in_progress = True
            self.last_analysis_time = current_time
        
        try:
            # Decode frame
            frame = self.decode_frame(image_data)
            if frame is None:
                return {"error": "Failed to decode image"}
            
            # Frame skipping optimization
            if self._should_skip_frame():
                # Return cached result if skipping frames
                if self.result_cache['last_result']:
                    return self.result_cache['last_result']
                else:
                    return {"error": "Frame skipped for optimization"}
            
            # Motion detection optimization
            motion_detected = self._detect_motion(frame)
            if not motion_detected:
                logger.debug("No significant motion detected, returning cached result")
                if self.result_cache['last_result']:
                    return self.result_cache['last_result']
                else:
                    # Create minimal response for no motion
                    return {
                        "success": True,
                        "detections": {"people": []},
                        "heatmap": {"points": []},
                        "frame_info": {
                            "width": frame.shape[1],
                            "height": frame.shape[0],
                            "timestamp": time.time(),
                            "detection_count": 0,
                            "total_tracks": len(self.detection_memory['active_tracks'])
                        },
                        "analytics": {
                            "total_people": 0,
                            "male_count": 0,
                            "female_count": 0,
                            "average_dwell_time": 0,
                            "age_groups": {},
                            "total_analysed": 0,
                            "active_tracks": len(self.detection_memory['active_tracks']),
                            "people_database_size": self._get_actual_database_count()
                        }
                    }
            
            # Optimize frame for analysis
            optimized_frame = self._optimize_frame_for_analysis(frame)
            
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
            
            # Process zone-based dwell time if zone analyzer is available
            if self.zone_dwell_analyzer and tracked_people:
                try:
                    # Convert tracked people to format expected by zone dwell analyzer
                    tracks_for_dwell = []
                    for person in tracked_people:
                        # Convert bbox format for zone analyzer (needs x1,y1,x2,y2)
                        bbox = person['bbox']
                        if len(bbox) == 4:
                            # If bbox is [x, y, w, h], convert to [x1, y1, x2, y2]
                            if isinstance(bbox[2], (int, float)) and bbox[2] > 0 and bbox[2] < 2000:  # width/height format
                                bbox_zone = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                            else:  # already x1,y1,x2,y2 format
                                bbox_zone = bbox
                        else:
                            bbox_zone = bbox
                            
                        tracks_for_dwell.append({
                            'id': person['id'],
                            'bbox': bbox_zone
                        })
                    
                    # Update zone-based dwell times
                    zone_dwell_records = self.zone_dwell_analyzer.update(tracks_for_dwell)
                    
                    # Update tracked people with zone-based dwell times
                    for person in tracked_people:
                        person_id = person['id']
                        
                        # Check if person has active zone dwell
                        if person_id in self.zone_dwell_analyzer.active:
                            zone_dwell_info = self.zone_dwell_analyzer.active[person_id]
                            current_time = time.time()
                            zone_dwell_time = current_time - zone_dwell_info['start_time']
                            person['zone_dwell_time'] = zone_dwell_time
                            person['zone_dwell_zone_id'] = zone_dwell_info['zone_id']
                            
                            # Debug logging
                            if zone_dwell_time > 1.0:  # Only log after 1 second
                                logger.info(f"Person {person_id} zone dwell: {zone_dwell_time:.1f}s in zone {zone_dwell_info['zone_id']}")
                        else:
                            person['zone_dwell_time'] = 0
                            person['zone_dwell_zone_id'] = person.get('zone_id', 0)
                            
                except Exception as e:
                    logger.error(f"Error processing zone-based dwell time: {e}")
                    # Add default zone dwell info to prevent errors
                    for person in tracked_people:
                        person['zone_dwell_time'] = person.get('dwell_time', 0)
                        person['zone_dwell_zone_id'] = person.get('zone_id', 0)
            else:
                # Zone analyzer not available, log this
                if tracked_people:
                    logger.debug(f"Zone dwell analyzer not available (analyzer: {self.zone_dwell_analyzer is not None}, people: {len(tracked_people)})")
                # Add default zone dwell info to prevent errors
                for person in tracked_people:
                    person['zone_dwell_time'] = person.get('dwell_time', 0)
                    person['zone_dwell_zone_id'] = person.get('zone_id', 0)

            # Calculate demographics summary
            demographics_summary = {
                'male_count': 0,
                'female_count': 0,
                'age_groups': {},
                "total_analysed": 0
            }
            
            total_dwell = 0
            heatmap_points = []
            
            # Process each tracked person with enhanced individual demographic rendering
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
                
                # Enhanced individual demographic rendering for overlay
                demographics = person.get('demographics', {})
                person_id = person.get('id', 'unknown')
                
                # Add individual demographic overlay data for each person
                person['individual_demographics'] = {
                    'id': person_id,
                    'age': demographics.get('age', 0),
                    'age_group': demographics.get('age_group', 'analyzing...'),
                    'gender': demographics.get('gender', 'analyzing...'),
                    'race': demographics.get('race', 'unknown'),
                    'emotion': demographics.get('emotion', 'neutral'),
                    'confidence': demographics.get('confidence', 0.1),
                    'analysis_method': demographics.get('analysis_method', 'enhanced'),
                    'frames_tracked': person.get('frames_tracked', 0),
                    'dwell_time': person.get('dwell_time', 0),
                    'is_confirmed': person.get('is_confirmed', False),
                    'display_label': self._create_person_display_label(person_id, demographics)
                }
                
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
                    "people_database_size": self._get_actual_database_count(),
                    "zone_analytics": self._get_zone_analytics(tracked_people)
                },

            }
            
            # Log tracking status for debugging
            if self.detection_memory['frame_count'] % 30 == 0:  # Every 30 frames
                active_count = len(self.detection_memory['active_tracks'])
                memory_db_count = len(self.detection_memory['people_database'])
                actual_db_count = self._get_actual_database_count()
                logger.info(f"Frame {self.detection_memory['frame_count']}: {len(tracked_people)} current, {active_count} active tracks, {memory_db_count} in memory, {actual_db_count} in database")
            
            # Convert response to JSON serializable format
            response = self._convert_to_json_serializable(response)
            
            # Update performance statistics
            processing_time = time.time() - analysis_start_time
            self._update_performance_stats(processing_time)
            
            # Add performance info to response for debugging
            response['performance'] = {
                'processing_time': processing_time,
                'avg_processing_time': self.performance_stats['avg_processing_time'],
                'total_frames': self.performance_stats['total_frames'],
                'processed_frames': self.performance_stats['processed_frames'],
                'demographic_analysis_count': self.performance_stats['demographic_analysis_count'],
                'cache_hits': self.performance_stats['cache_hits'],
                'cache_misses': self.performance_stats['cache_misses'],
                'cache_hit_rate': self.performance_stats['cache_hits'] / max(1, self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']),
                'active_demographic_queue': len(self.demographic_cache.get('processing_queue', {})),
                'optimization_enabled': {
                    'motion_detection': self.processing_config['enable_motion_detection'],
                    'frame_skipping': self.processing_config['enable_frame_skipping'],
                    'fast_yolo': self.processing_config['enable_fast_yolo'],
                    'async_demographics': self.processing_config['enable_async_demographics'],
                    'demographic_caching': self.processing_config['enable_demographic_caching']
                }
            }
            
            # Cache the result and release the analysis lock
            with self.analysis_lock:
                self.result_cache['last_result'] = response
                self.result_cache['cache_time'] = current_time
                self.analysis_in_progress = False
            
            return response
            
        except Exception as e:
            logger.error(f"Unexpected error in analyze_frame: {e}", exc_info=True)
            
            # Update performance statistics even on error
            processing_time = time.time() - analysis_start_time
            self._update_performance_stats(processing_time)
            
            error_response = {
                "error": f"Analysis failed: {str(e)}",
                "performance": {
                    "processing_time": processing_time,
                    "avg_processing_time": self.performance_stats['avg_processing_time']
                }
            }
            
            # Release the analysis lock even on error
            with self.analysis_lock:
                self.analysis_in_progress = False
            
            return error_response
        finally:
            # Ensure analysis lock is always released
            with self.analysis_lock:
                self.analysis_in_progress = False

    def _save_person_to_database(self, person_id, demographics, detection, timestamp):
        """
        Save person data to database with proper Flask context handling

        Args:
            person_id: Track ID of the person
            demographics: Demographic analysis results
            detection: Detection data
            timestamp: Timestamp of the detection
        """
        def _async_save_person():
            try:
                # First try direct database reference
                if hasattr(self, 'database') and self.database:
                    db = self.database
                else:
                    # Fallback to Flask context
                    from flask import current_app
                    
                    if hasattr(current_app, 'db'):
                        db = current_app.db
                    else:
                        logger.error(f"No database available to save person {person_id}")
                        return
                
                # Store demographics data with proper error handling
                result = db.store_demographics(
                    person_id=person_id,
                    demographics=demographics,
                    timestamp=datetime.fromtimestamp(timestamp),
                    analysis_model=demographics.get('analysis_method', 'enhanced')
                )
                
                if result:
                    logger.info(f"Successfully saved person {person_id} demographics to database")
                else:
                    logger.error(f"Failed to save person {person_id} demographics to database - store_demographics returned None")
                        
            except Exception as e:
                logger.error(f"Error saving person {person_id} to database: {e}", exc_info=True)

        # Submit to thread pool for async execution
        try:
            self.db_executor.submit(_async_save_person)
        except Exception as e:
            logger.error(f"Failed to submit async database save for person {person_id}: {e}")

    def _save_detection_to_database(self, person_id, bbox, confidence, timestamp):
        """
        Save detection data to database with proper Flask context handling

        Args:
            person_id: Track ID of the person
            bbox: Bounding box coordinates
            confidence: Detection confidence
            timestamp: Timestamp of the detection
        """
        def _async_save_detection():
            try:
                # First try direct database reference
                if hasattr(self, 'database') and self.database:
                    db = self.database
                else:
                    # Fallback to Flask context
                    from flask import current_app
                    
                    if hasattr(current_app, 'db'):
                        db = current_app.db
                    else:
                        logger.error(f"No database available to save detection for person {person_id}")
                        return

                # Convert bbox format if needed
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    bbox_tuple = (x1, y1, x2, y2)
                else:
                    bbox_tuple = bbox

                # Store detection data with proper error handling
                result = db.store_detection(
                    person_id=person_id,
                    bbox=bbox_tuple,
                    confidence=confidence,
                    timestamp=datetime.fromtimestamp(timestamp),
                    camera_id='main'
                )

                if result:
                    logger.info(f"Successfully saved detection for person {person_id} to database")
                else:
                    logger.error(f"Failed to save detection for person {person_id} to database - store_detection returned None")
                        
            except Exception as e:
                logger.error(f"Error saving detection for person {person_id} to database: {e}", exc_info=True)

        # Submit to thread pool for async execution
        try:
            self.db_executor.submit(_async_save_detection)
        except Exception as e:
            logger.error(f"Failed to submit async database save for detection {person_id}: {e}")

    def set_database(self, database):
        """Set the database reference for direct access"""
        self.database = database

    def get_analytics_summary(self, hours=24):
        """
        Get analytics summary data
        
        Args:
            hours (int): Number of hours to look back
            
        Returns:
            dict: Analytics summary data
        """
        try:
            # First try direct database reference
            if hasattr(self, 'database') and self.database:
                return self.database.get_analytics_summary(hours=hours)
            
            # Fallback to Flask context
            from flask import current_app
            if hasattr(current_app, 'db'):
                db = current_app.db
                return db.get_analytics_summary(hours=hours)
            else:
                logger.warning("Database not available, returning empty analytics summary")
                return self._get_empty_analytics_summary()
        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}")
            return self._get_empty_analytics_summary()

    def get_traffic_data(self, hours=24):
        """
        Get traffic data for charts
        
        Args:
            hours (int): Number of hours to look back
            
        Returns:
            dict: Traffic data for charts
        """
        try:
            # First try direct database reference
            if hasattr(self, 'database') and self.database:
                return self.database.get_hourly_traffic(hours=hours)
            
            # Fallback to Flask context
            from flask import current_app
            if hasattr(current_app, 'db'):
                db = current_app.db
                return db.get_hourly_traffic(hours=hours)
            else:
                logger.warning("Database not available, returning empty traffic data")
                return self._get_empty_traffic_data()
        except Exception as e:
            logger.error(f"Error getting traffic data: {e}")
            return self._get_empty_traffic_data()

    def _get_actual_database_count(self):
        """
        Get actual count of persons in database
        
        Returns:
            int: Actual database count
        """
        try:
            from flask import current_app
            
            # Try to get database from current Flask context
            if hasattr(current_app, 'db'):
                db = current_app.db
                return db.get_total_persons_count()
            else:
                logger.debug("No current Flask app context, falling back to memory count")
                return len(self.detection_memory.get('people_database', {}))
        except Exception as e:
            logger.error(f"Error getting actual database count: {e}")
            # Fallback to memory count in case of database error
            return len(self.detection_memory.get('people_database', {}))

    def get_demographics_data(self, page=1, per_page=10, search='', sort_by='timestamp', sort_order='desc', hours=24):
        """
        Get detailed demographics data for table
        
        Args:
            page (int): Page number
            per_page (int): Items per page
            search (str): Search query
            sort_by (str): Sort field
            sort_order (str): Sort order
            hours (int): Number of hours to look back
            
        Returns:
            dict: Demographics data for table
        """
        try:
            from flask import current_app
            if hasattr(current_app, 'db'):
                db = current_app.db
                return db.get_demographics_data(
                    page=page,
                    per_page=per_page,
                    search=search,
                    sort_by=sort_by,
                    sort_order=sort_order,
                    hours=hours
                )
            else:
                logger.warning("Database not available, returning empty demographics data")
                return self._get_empty_demographics_data()
        except Exception as e:
            logger.error(f"Error getting demographics data: {e}")
            return self._get_empty_demographics_data()

    def get_demographic_trends(self, hours=24):
        """
        Get demographic trends over time
        
        Args:
            hours (int): Number of hours to look back
            
        Returns:
            dict: Demographic trends data
        """
        try:
            from flask import current_app
            if hasattr(current_app, 'db'):
                db = current_app.db
                return db.get_demographic_trends(hours=hours)
            else:
                logger.warning("Database not available, returning empty demographic trends")
                return self._get_empty_demographic_trends()
        except Exception as e:
            logger.error(f"Error getting demographic trends: {e}")
            return self._get_empty_demographic_trends()

    def _get_empty_analytics_summary(self):
        """Return empty analytics summary structure"""
        return {
            "total_visitors": 0,
            "avg_dwell_time": 0,
            "conversion_rate": 0,
            "peak_hour": "--:--",
            "gender_distribution": {},
            "age_groups": {},
            "emotions": {},
            "races": {},
            "avg_age": 0,
            "start_time": "",
            "end_time": ""
        }

    def _get_empty_traffic_data(self):
        """Return empty traffic data structure"""
        return {
            "labels": [],
            "data": []
        }

    def _get_empty_demographics_data(self):
        """Return empty demographics data structure"""
        return {
            "data": [],
            "pagination": {
                "page": 1,
                "per_page": 10,
                "total": 0,
                "pages": 0
            }
        }

    def _get_empty_demographic_trends(self):
        """Return empty demographic trends structure"""
        return {
            "success": False,
            "time_periods": [],
            "time_label": "Time",
            "gender_trends": {},
            "age_trends": {},
            "emotion_trends": {}
        }

    def _create_person_display_label(self, person_id, demographics):
        """
        Create a display label for individual person demographic overlay
        
        Args:
            person_id: ID of the person
            demographics: Demographic data dictionary
            
        Returns:
            str: Formatted display label for the person
        """
        try:
            # Extract demographic information with type validation
            age = demographics.get('age', 0)
            age_group = demographics.get('age_group', 'analyzing...')
            gender = demographics.get('gender', 'analyzing...')
            race = demographics.get('race', 'unknown')
            emotion = demographics.get('emotion', 'neutral')
            confidence = demographics.get('confidence', 0.1)
            
            # Ensure all string fields are actually strings
            if not isinstance(gender, str):
                gender = str(gender) if gender is not None else 'unknown'
            if not isinstance(emotion, str):
                emotion = str(emotion) if emotion is not None else 'neutral'
            if not isinstance(age_group, str):
                age_group = str(age_group) if age_group is not None else 'unknown'
            if not isinstance(race, str):
                race = str(race) if race is not None else 'unknown'
                
            # Handle cases where demographic data might be dictionaries
            if isinstance(gender, dict):
                # Extract dominant gender if it's a dictionary of scores
                gender = max(gender.keys(), key=lambda k: gender[k]) if gender else 'unknown'
            if isinstance(emotion, dict):
                # Extract dominant emotion if it's a dictionary of scores
                emotion = max(emotion.keys(), key=lambda k: emotion[k]) if emotion else 'neutral'
            
            # Ensure numeric age
            try:
                age = float(age) if age is not None else 0
            except (ValueError, TypeError):
                age = 0
            
            # Create a concise but informative label
            if confidence > 0.7 and gender != 'analyzing...' and age_group != 'analyzing...':
                # High confidence - show detailed info
                if age > 0:
                    label = f"ID:{person_id} | {gender.title()} {int(age)}y | {emotion.title()}"
                else:
                    label = f"ID:{person_id} | {gender.title()} {age_group} | {emotion.title()}"
            elif confidence > 0.4 and gender != 'analyzing...':
                # Medium confidence - show basic info
                label = f"ID:{person_id} | {gender.title()} | {emotion.title()}"
            else:
                # Low confidence or still analyzing
                label = f"ID:{person_id} | Analyzing..."
            
            # Add confidence indicator if it's notably low
            if confidence < 0.6 and gender != 'analyzing...':
                label += f" ({int(confidence*100)}%)"
            
            return label
            
        except Exception as e:
            logger.error(f"Error creating person display label: {e}")
            return f"ID:{person_id} | Unknown"

    def _extract_face_embedding(self, person_img):
        """
        Extract face embedding for re-identification with caching
        
        Args:
            person_img: Person image array
            
        Returns:
            np.ndarray or None: Face embedding vector
        """
        try:
            if not ADVANCED_MODELS_AVAILABLE or not self.demographic_analyzer.models_loaded:
                return None
            
            if self.demographic_analyzer.face_app is None:
                return None
            
            # Check cache first
            if self.reid_settings.get('enable_cached_reid', True):
                img_hash = self._get_person_image_hash(person_img)
                if img_hash in self.demographic_cache['feature_cache']:
                    cached_features = self.demographic_cache['feature_cache'][img_hash]
                    if time.time() - cached_features['timestamp'] < 300:  # 5 minute cache
                        logger.debug(f"Cache hit for face embedding: {img_hash}")
                        return cached_features['embedding']
            
            # Get face embedding using InsightFace
            faces = self.demographic_analyzer.face_app.get(person_img)
            if faces:
                # Use the largest face
                largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                embedding = largest_face.embedding
                
                # Cache the result
                if self.reid_settings.get('enable_cached_reid', True):
                    img_hash = self._get_person_image_hash(person_img)
                    self.demographic_cache['feature_cache'][img_hash] = {
                        'embedding': embedding,
                        'timestamp': time.time()
                    }
                
                return embedding
            
            return None
            
        except Exception as e:
            logger.debug(f"Face embedding extraction failed: {e}")
            return None

    def _calculate_enhanced_similarity(self, person1_data: Dict, person2_data: Dict, spatial_distance: float) -> float:
        """
        Calculate enhanced similarity score for re-identification
        
        Args:
            person1_data: First person's data (demographics, features, etc.)
            person2_data: Second person's data
            spatial_distance: Spatial distance between detections
            
        Returns:
            float: Combined similarity score (0-1)
        """
        try:
            total_score = 0.0
            total_weight = 0.0
            
            # Spatial component (inverse distance)
            if spatial_distance <= self.reid_settings['reid_spatial_threshold']:
                spatial_score = 1.0 - (spatial_distance / self.reid_settings['reid_spatial_threshold'])
                total_score += 0.3 * spatial_score
                total_weight += 0.3
            
            # Demographic similarity
            demo1 = person1_data.get('demographics', {})
            demo2 = person2_data.get('demographics', {})
            if demo1 and demo2:
                demo_similarity = self._calculate_demographic_similarity(demo1, demo2)
                total_score += self.reid_settings['reid_demographic_weight'] * demo_similarity
                total_weight += self.reid_settings['reid_demographic_weight']
            
            # Face embedding similarity (if available)
            if (self.reid_settings.get('enable_face_embedding_reid', True) and
                'face_embedding' in person1_data and 'face_embedding' in person2_data):
                
                embedding1 = person1_data['face_embedding']
                embedding2 = person2_data['face_embedding']
                
                if embedding1 is not None and embedding2 is not None:
                    face_similarity = self._calculate_feature_similarity(embedding1, embedding2)
                    total_score += 0.4 * face_similarity
                    total_weight += 0.4
            
            # Visual features similarity (fallback)
            elif ('visual_features' in person1_data and 'visual_features' in person2_data):
                features1 = person1_data['visual_features']
                features2 = person2_data['visual_features']
                
                if features1 is not None and features2 is not None:
                    visual_similarity = self._calculate_feature_similarity(features1, features2)
                    total_score += 0.3 * visual_similarity
                    total_weight += 0.3
            
            # Normalize score
            if total_weight > 0:
                normalized_score = total_score / total_weight
                logger.debug(f"Enhanced similarity calculated: {normalized_score:.3f} (spatial: {spatial_distance:.1f}px)")
                return normalized_score
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating enhanced similarity: {e}")
            return 0.0

    def _calculate_demographic_similarity(self, demo1, demo2):
        """
        Calculate similarity between two demographic profiles
        
        Args:
            demo1: First demographic profile
            demo2: Second demographic profile
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            if not demo1 or not demo2:
                return 0.0
            
            similarity_score = 0.0
            total_weight = 0.0
            
            # Gender similarity (high weight)
            gender1 = demo1.get('gender', 'unknown')
            gender2 = demo2.get('gender', 'unknown')
            if gender1 != 'unknown' and gender2 != 'unknown':
                if gender1 == gender2:
                    similarity_score += 0.4
                total_weight += 0.4
            
            # Age similarity (medium weight)
            age1 = demo1.get('age', 0)
            age2 = demo2.get('age', 0)
            if age1 > 0 and age2 > 0:
                age_diff = abs(age1 - age2)
                age_similarity = max(0, 1 - age_diff / 20)  # Allow 20 year difference
                similarity_score += 0.3 * age_similarity
                total_weight += 0.3
            
            # Race similarity (medium weight)
            race1 = demo1.get('race', 'unknown')
            race2 = demo2.get('race', 'unknown')
            if race1 != 'unknown' and race2 != 'unknown':
                if race1 == race2:
                    similarity_score += 0.2
                total_weight += 0.2
            
            # Emotion similarity (low weight)
            emotion1 = demo1.get('emotion', 'neutral')
            emotion2 = demo2.get('emotion', 'neutral')
            if emotion1 == emotion2:
                similarity_score += 0.1
            total_weight += 0.1
            
            # Normalize score
            if total_weight > 0:
                return similarity_score / total_weight
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating demographic similarity: {e}")
            return 0.0

    def _attempt_person_reidentification(self, detection, current_time):
        """
        Enhanced person re-identification with improved caching and feature matching
        
        Args:
            detection: Current person detection with person_image
            current_time: Current timestamp
            
        Returns:
            int or None: Existing person ID if re-identified, None otherwise
        """
        if not self.reid_settings['enable_reid']:
            return None
            
        try:
            if 'person_image' not in detection or detection['person_image'] is None:
                return None
            
            person_img = detection['person_image']
            detection_center = detection['center']
            people_db = self.detection_memory['people_database']
            
            best_match_id = None
            best_match_score = 0.0
            
            # Extract features for current detection (with caching)
            current_face_embedding = self._extract_face_embedding(person_img)
            current_visual_features = None
            
            # Prepare current person data for comparison
            current_person_data = {
                'demographics': {},  # Will be filled if needed
                'face_embedding': current_face_embedding,
                'visual_features': current_visual_features,
                'location': detection_center
            }
            
            # Check against known people within the re-identification window
            for person_id, person_data in people_db.items():
                # Skip if person was seen too long ago
                last_seen = person_data.get('last_seen', 0)
                if current_time - last_seen > self.reid_settings['reid_temporal_window']:
                    continue
                
                # Calculate spatial distance
                last_location = self.reid_database['last_seen_locations'].get(person_id, (0, 0))
                spatial_distance = np.sqrt((detection_center[0] - last_location[0])**2 + 
                                         (detection_center[1] - last_location[1])**2)
                
                if spatial_distance > self.reid_settings['reid_spatial_threshold']:
                    continue
                
                # Prepare stored person data
                stored_person_data = {
                    'demographics': person_data.get('demographics', {}),
                    'face_embedding': self.reid_database['face_embeddings'].get(person_id),
                    'visual_features': self.reid_database['visual_features'].get(person_id),
                    'location': last_location
                }
                
                # Calculate enhanced similarity
                similarity_score = self._calculate_enhanced_similarity(
                    current_person_data, stored_person_data, spatial_distance
                )
                
                # Check if this is the best match
                if (similarity_score > best_match_score and
                    similarity_score > self.reid_settings['reid_similarity_threshold']):
                    best_match_score = similarity_score
                    best_match_id = person_id
            
            if best_match_id:
                logger.info(f"Re-identified person as ID {best_match_id} (confidence: {best_match_score:.3f})")
                
                # Update re-identification database
                self.reid_database['last_seen_locations'][best_match_id] = detection_center
                
                # Update face embedding if we have a better one
                if current_face_embedding is not None:
                    self.reid_database['face_embeddings'][best_match_id] = current_face_embedding
                
                return best_match_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error in enhanced person re-identification: {e}")
            return None

    def _update_reid_database(self, person_id, detection, demographics):
        """
        Enhanced re-identification database update with better caching
        
        Args:
            person_id: Person ID
            detection: Detection data
            demographics: Demographic analysis results
        """
        try:
            current_time = time.time()
            
            # Update last seen location
            self.reid_database['last_seen_locations'][person_id] = detection['center']
            
            # Update demographic signature with timestamp
            self.reid_database['demographic_signatures'][person_id] = {
                'demographics': demographics,
                'timestamp': current_time
            }
            
            # Update face embedding if available
            if (self.reid_settings['enable_face_embedding_reid'] and 
                'person_image' in detection):
                
                face_embedding = self._extract_face_embedding(detection['person_image'])
                if face_embedding is not None:
                    self.reid_database['face_embeddings'][person_id] = {
                        'embedding': face_embedding,
                        'timestamp': current_time
                    }
            
            # Update visual features if available
            if hasattr(self, 'person_tracker') and self.person_tracker:
                # Create dummy frame for feature extraction
                h, w = detection['person_image'].shape[:2]
                frame = np.zeros((h*2, w*2, 3), dtype=np.uint8)
                frame[:h, :w] = detection['person_image']
                
                visual_features = self.person_tracker._extract_visual_features(
                    frame, [0, 0, w, h]
                )
                if visual_features is not None:
                    self.reid_database['visual_features'][person_id] = {
                        'features': visual_features,
                        'timestamp': current_time
                    }
            
        except Exception as e:
            logger.error(f"Error updating enhanced re-identification database: {e}")

    def _on_demographic_analysis_complete(self, person_id: int, demographics: Dict[str, Any], detection: Dict, timestamp: float):
        """
        Callback when background demographic analysis completes
        
        Args:
            person_id: Person ID
            demographics: Completed demographic analysis
            detection: Original detection data
            timestamp: Analysis timestamp
        """
        try:
            # Update person database with completed analysis
            if person_id in self.detection_memory['people_database']:
                person_data = self.detection_memory['people_database'][person_id]
                
                # Check if this is a better result
                current_conf = person_data['demographics'].get('confidence', 0)
                new_conf = demographics.get('confidence', 0)
                
                if new_conf > current_conf or person_data.get('analyzing', False):
                    # Update with better demographics
                    person_data['demographics'] = demographics
                    person_data['analyzing'] = False
                    
                    # Update database with full demographic info
                    self._save_person_to_database(person_id, demographics, detection, timestamp)
                    
                    logger.info(f"Completed demographic analysis for person {person_id}: {demographics.get('gender', 'unknown')} {demographics.get('age_group', 'unknown')} conf: {demographics.get('confidence', 0):.2f}")
                
                # Update reid database
                self._update_reid_database(person_id, detection, demographics)
            
        except Exception as e:
            logger.error(f"Error in demographic analysis completion callback: {e}")

    def _save_person_to_database_minimal(self, person_id, demographics, detection, timestamp):
        """
        Save minimal person data to database immediately with proper Flask context handling
        
        Args:
            person_id: Track ID of the person
            demographics: Basic demographic information
            detection: Detection data
            timestamp: Timestamp of the detection
        """
        def _async_save_minimal():
            try:
                # Try to get database from current Flask context
                from flask import current_app
                
                if hasattr(current_app, 'db'):
                    db = current_app.db
                    
                    # Create minimal demographic entry
                    minimal_demographics = {
                        'age': demographics.get('age', 25),
                        'gender': demographics.get('gender', 'unknown'),
                        'race': demographics.get('race', 'unknown'),
                        'emotion': demographics.get('emotion', 'neutral'),
                        'confidence': demographics.get('confidence', 0.3)
                    }
                    
                    # Store minimal demographics data quickly with proper error handling
                    result = db.store_demographics(
                        person_id=person_id,
                        demographics=minimal_demographics,
                        timestamp=datetime.fromtimestamp(timestamp),
                        analysis_model=demographics.get('analysis_method', 'minimal')
                    )
                    
                    if result:
                        logger.info(f"Successfully saved minimal person {person_id} demographics to database")
                    else:
                        logger.error(f"Failed to save minimal person {person_id} demographics to database - store_demographics returned None")
                else:
                    logger.error(f"No database available to save minimal person {person_id}")
                        
            except ImportError as e:
                logger.error(f"Cannot import Flask app for minimal save of person {person_id}: {e}")
            except Exception as e:
                logger.error(f"Error saving minimal person {person_id} to database: {e}")
        
        # Submit to thread pool for async execution
        try:
            self.db_executor.submit(_async_save_minimal)
        except Exception as e:
            logger.error(f"Failed to submit async minimal database save for person {person_id}: {e}")

    def _save_person_to_database_sync(self, person_id, demographics, detection, timestamp, detection_id=None):
        """
        Save person data to database synchronously with proper Flask context handling
        
        Args:
            person_id: Track ID of the person
            demographics: Demographic analysis results
            detection: Detection data
            timestamp: Timestamp of the detection
            detection_id: Optional detection ID to link demographics to specific detection
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # First try direct database reference
            if hasattr(self, 'database') and self.database:
                db = self.database
            else:
                # Fallback to Flask context
                from flask import current_app
                
                if hasattr(current_app, 'db'):
                    db = current_app.db
                else:
                    logger.error(f"❌ No database available to save person {person_id}")
                    return False
            
            # Store demographics data with proper error handling
            result = db.store_demographics(
                person_id=person_id,
                demographics=demographics,
                timestamp=datetime.fromtimestamp(timestamp),
                detection_id=detection_id,
                analysis_model=demographics.get('analysis_method', 'enhanced')
            )
            
            if result:
                logger.info(f"✅ Successfully saved person {person_id} to database (sync)")
                return True
            else:
                logger.error(f"❌ Failed to save person {person_id} to database - store_demographics returned {result}")
                return False
                    
        except Exception as e:
            logger.error(f"❌ Error saving person {person_id} to database (sync): {e}", exc_info=True)
            return False

    def _save_detection_to_database_sync(self, person_id, bbox, confidence, timestamp):
        """
        Save detection data to database synchronously
        
        Args:
            person_id: Track ID of the person
            bbox: Bounding box coordinates
            confidence: Detection confidence
            timestamp: Timestamp of the detection
            
        Returns:
            int: Detection ID if successful, None otherwise
        """
        try:
            # First try direct database reference
            if hasattr(self, 'database') and self.database:
                db = self.database
            else:
                # Fallback to Flask context
                from flask import current_app
                
                if hasattr(current_app, 'db'):
                    db = current_app.db
                else:
                    logger.error(f"❌ No database available to save detection for person {person_id}")
                    return None

            # Convert bbox format if needed
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                bbox_tuple = (x1, y1, x2, y2)
            else:
                bbox_tuple = bbox

            # Store detection data with proper error handling
            result = db.store_detection(
                person_id=person_id,
                bbox=bbox_tuple,
                confidence=confidence,
                timestamp=datetime.fromtimestamp(timestamp),
                camera_id='main'
            )

            if result:
                logger.info(f"✅ Successfully saved detection for person {person_id} to database (sync)")
                return result  # Return the detection_id
            else:
                logger.error(f"❌ Failed to save detection for person {person_id} to database - store_detection returned {result}")
                return None
                        
        except Exception as e:
            logger.error(f"❌ Error saving detection for person {person_id} to database (sync): {e}", exc_info=True)
            return None

    def get_demographics_data(self, page=1, per_page=10, search='', sort_by='timestamp', sort_order='desc', hours=24):
        """
        Get demographics data for table display
        
        Args:
            page (int): Page number
            per_page (int): Items per page
            search (str): Search query
            sort_by (str): Sort field
            sort_order (str): Sort order
            hours (int): Number of hours to look back
            
        Returns:
            dict: Demographics data for table
        """
        try:
            # First try direct database reference
            if hasattr(self, 'database') and self.database:
                return self.database.get_demographics_data(page, per_page, search, sort_by, sort_order, hours)
            
            # Fallback to Flask context
            from flask import current_app
            if hasattr(current_app, 'db'):
                db = current_app.db
                return db.get_demographics_data(page, per_page, search, sort_by, sort_order, hours)
            else:
                logger.warning("Database not available, returning empty demographics data")
                return {'data': [], 'pagination': {'page': 1, 'per_page': per_page, 'total_records': 0, 'total_pages': 0, 'has_prev': False, 'has_next': False, 'prev_num': None, 'next_num': None}}
        except Exception as e:
            logger.error(f"Error getting demographics data: {e}")
            return {'data': [], 'pagination': {'page': 1, 'per_page': per_page, 'total_records': 0, 'total_pages': 0, 'has_prev': False, 'has_next': False, 'prev_num': None, 'next_num': None}}

    def get_demographic_trends(self, hours=24):
        """
        Get demographic trends data
        
        Args:
            hours (int): Number of hours to look back
            
        Returns:
            dict: Demographic trends data
        """
        try:
            # First try direct database reference
            if hasattr(self, 'database') and self.database:
                return self.database.get_demographic_trends(hours=hours)
            
            # Fallback to Flask context
            from flask import current_app
            if hasattr(current_app, 'db'):
                db = current_app.db
                return db.get_demographic_trends(hours=hours)
            else:
                logger.warning("Database not available, returning empty demographic trends")
                return {"success": False, "message": "Database not available"}
        except Exception as e:
            logger.error(f"Error getting demographic trends: {e}")
            return {"success": False, "message": f"Error: {e}"}

    def _bbox_overlap_coords(self, bbox1, bbox2):
        """
        Calculate overlap between two bounding boxes in coordinate format
        
        Args:
            bbox1: (x1, y1, x2, y2)
            bbox2: (x1, y1, x2, y2)
            
        Returns:
            float: Overlap ratio (0-1)
        """
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Calculate intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i < x1_i or y2_i < y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            
            # Calculate union
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            if union <= 0:
                return 0.0
                
            return intersection / union
            
        except Exception as e:
            logger.debug(f"Error calculating bbox overlap: {e}")
            return 0.0

    def _calculate_person_dwell_time(self, track_id, track):
        """
        Calculate the dwell time for a tracked person using timestamps
        
        Args:
            track_id: ID of the tracked person
            track: Track information
            
        Returns:
            float: Dwell time in seconds
        """
        try:
            current_time = time.time()
            people_db = self.detection_memory['people_database']
            
            # Check if we have the person in our database
            if track_id in people_db:
                person_data = people_db[track_id]
                first_seen = person_data.get('first_seen', current_time)
                dwell_time = current_time - first_seen
                
                # Debug logging for dwell time
                if dwell_time > 1.0:  # Only log after 1 second
                    logger.debug(f"Person {track_id} general dwell: {dwell_time:.1f}s (first_seen: {first_seen:.1f}, current: {current_time:.1f})")
                
                # Ensure minimum dwell time threshold
                min_dwell = 0.5  # 0.5 seconds minimum
                return max(min_dwell, dwell_time) if dwell_time > 0 else 0.0
            else:
                # New person, minimal dwell time
                logger.debug(f"Person {track_id} not in database, returning 0 dwell time")
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating dwell time for person {track_id}: {e}")
            # Fallback to frame-based calculation
            return max(0.0, track['frames_tracked'] * 0.033)  # ~30fps assumption

    def _get_zone_analytics(self, tracked_people):
        """
        Calculate zone-based analytics from tracked people data
        
        Args:
            tracked_people: List of tracked person objects with zone information
            
        Returns:
            dict: Zone analytics including per-zone dwell times and occupancy
        """
        try:
            if not self.zone_manager:
                return {"zones_available": False}
            
            zones = self.zone_manager.get_zones('default')
            zone_analytics = {
                "zones_available": True,
                "total_zones": len(zones),
                "zone_occupancy": {},
                "zone_dwell_times": {},
                "zone_demographics": {}
            }
            
            # Initialize zone data
            for zone in zones:
                zone_id = zone['id']
                zone_name = zone['name']
                zone_analytics["zone_occupancy"][zone_name] = {
                    "zone_id": zone_id,
                    "current_count": 0,
                    "people_ids": []
                }
                zone_analytics["zone_dwell_times"][zone_name] = {
                    "zone_id": zone_id,
                    "active_dwell_times": [],
                    "average_dwell_time": 0
                }
                zone_analytics["zone_demographics"][zone_name] = {
                    "zone_id": zone_id,
                    "male_count": 0,
                    "female_count": 0,
                    "age_groups": {}
                }
            
            # Add "Outside" zone for people not in any defined zone
            zone_analytics["zone_occupancy"]["Outside"] = {
                "zone_id": 0,
                "current_count": 0,
                "people_ids": []
            }
            zone_analytics["zone_dwell_times"]["Outside"] = {
                "zone_id": 0,
                "active_dwell_times": [],
                "average_dwell_time": 0
            }
            zone_analytics["zone_demographics"]["Outside"] = {
                "zone_id": 0,
                "male_count": 0,
                "female_count": 0,
                "age_groups": {}
            }
            
            # Process each tracked person
            for person in tracked_people:
                zone_name = person.get('zone_name', 'Outside')
                zone_dwell_time = person.get('zone_dwell_time', person.get('dwell_time', 0))
                person_id = person.get('id', 'unknown')
                demographics = person.get('demographics', {})
                
                # Update occupancy
                if zone_name in zone_analytics["zone_occupancy"]:
                    zone_analytics["zone_occupancy"][zone_name]["current_count"] += 1
                    zone_analytics["zone_occupancy"][zone_name]["people_ids"].append(person_id)
                
                # Update dwell times
                if zone_name in zone_analytics["zone_dwell_times"] and zone_dwell_time > 0:
                    zone_analytics["zone_dwell_times"][zone_name]["active_dwell_times"].append(zone_dwell_time)
                
                # Update demographics
                if zone_name in zone_analytics["zone_demographics"]:
                    gender = demographics.get('gender', 'unknown')
                    age_group = demographics.get('age_group', 'unknown')
                    
                    if gender == 'male':
                        zone_analytics["zone_demographics"][zone_name]["male_count"] += 1
                    elif gender == 'female':
                        zone_analytics["zone_demographics"][zone_name]["female_count"] += 1
                    
                    if age_group != 'unknown' and age_group != 'analyzing...':
                        if age_group not in zone_analytics["zone_demographics"][zone_name]["age_groups"]:
                            zone_analytics["zone_demographics"][zone_name]["age_groups"][age_group] = 0
                        zone_analytics["zone_demographics"][zone_name]["age_groups"][age_group] += 1
            
            # Calculate average dwell times
            for zone_name in zone_analytics["zone_dwell_times"]:
                dwell_times = zone_analytics["zone_dwell_times"][zone_name]["active_dwell_times"]
                if dwell_times:
                    zone_analytics["zone_dwell_times"][zone_name]["average_dwell_time"] = sum(dwell_times) / len(dwell_times)
            
            return zone_analytics
            
        except Exception as e:
            logger.error(f"Error calculating zone analytics: {e}")
            return {"zones_available": False, "error": str(e)}

# Global service instance
analysis_service = FrameAnalysisService()
