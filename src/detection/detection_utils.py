"""
Detection utilities for choosing between person and face detection modes
"""

import logging
from .detector import PersonDetector
from .tracker import PersonTracker
from .face_detector import FaceDetector
from .face_tracker import FaceTracker

logger = logging.getLogger(__name__)

def get_detector(config):
    """
    Get the appropriate detector based on configuration
    
    Args:
        config (dict): Detection configuration
        
    Returns:
        Detector instance (PersonDetector or FaceDetector)
    """
    detection_mode = config.get('mode', 'face')  # Default to face detection
    
    # Get common settings
    common_config = {
        'confidence_threshold': config.get('confidence_threshold', 0.5),
        'detection_interval': config.get('detection_interval', 1)
    }
    
    if detection_mode.lower() == 'face':
        logger.info("Using face detection mode")
        # Merge common config with face-specific config
        face_config = config.get('face', {})
        face_config.update(common_config)
        return FaceDetector(face_config)
    elif detection_mode.lower() == 'person':
        logger.info("Using person detection mode")
        # Merge common config with person-specific config
        person_config = config.get('person', {})
        person_config.update(common_config)
        return PersonDetector(person_config)
    else:
        logger.warning(f"Unknown detection mode '{detection_mode}', defaulting to face detection")
        face_config = config.get('face', {})
        face_config.update(common_config)
        return FaceDetector(face_config)

def get_tracker(config):
    """
    Get the appropriate tracker based on configuration
    
    Args:
        config (dict): Tracking configuration
        
    Returns:
        Tracker instance (PersonTracker or FaceTracker)
    """
    detection_mode = config.get('mode', 'face')  # Default to face tracking
    
    # Get common settings
    common_config = {
        'reid_enabled': config.get('reid_enabled', True)
    }
    
    if detection_mode.lower() == 'face':
        logger.info("Using face tracking mode")
        # Merge common config with face-specific config
        face_config = config.get('face', {})
        face_config.update(common_config)
        return FaceTracker(face_config)
    elif detection_mode.lower() == 'person':
        logger.info("Using person tracking mode")
        # Merge common config with person-specific config
        person_config = config.get('person', {})
        person_config.update(common_config)
        return PersonTracker(person_config)
    else:
        logger.warning(f"Unknown tracking mode '{detection_mode}', defaulting to face tracking")
        face_config = config.get('face', {})
        face_config.update(common_config)
        return FaceTracker(face_config)

def get_detection_pipeline(config):
    """
    Get a complete detection pipeline (detector + tracker)
    
    Args:
        config (dict): Configuration with 'detection' and 'tracking' sections
        
    Returns:
        tuple: (detector, tracker)
    """
    detection_config = config.get('detection', {})
    tracking_config = config.get('tracking', {})
    
    # Ensure both configs have the same mode
    mode = config.get('mode', 'face')
    detection_config['mode'] = mode
    tracking_config['mode'] = mode
    
    detector = get_detector(detection_config)
    tracker = get_tracker(tracking_config)
    
    return detector, tracker

def count_people_from_detections(detections, mode='face'):
    """
    Count people from detections based on the detection mode
    
    Args:
        detections (list): List of detections
        mode (str): Detection mode ('face' or 'person')
        
    Returns:
        int: Number of people detected
    """
    if not detections:
        return 0
    
    if mode.lower() == 'face':
        # For face detection: count faces = count people
        # Filter by quality if available
        high_quality_faces = [
            det for det in detections 
            if det.get('quality_score', 1.0) > 0.4
        ]
        return len(high_quality_faces)
    elif mode.lower() == 'person':
        # For person detection: count persons directly
        return len(detections)
    else:
        return len(detections) 