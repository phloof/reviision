"""
Detection module for Retail Analytics System
Provides person and face detection and tracking functionality
"""

from .detector import PersonDetector
from .tracker import PersonTracker
from .face_detector import FaceDetector
from .face_tracker import FaceTracker
from .detection_utils import get_detector, get_tracker, get_detection_pipeline, count_people_from_detections

__all__ = [
    'PersonDetector', 'PersonTracker', 
    'FaceDetector', 'FaceTracker',
    'get_detector', 'get_tracker', 'get_detection_pipeline', 'count_people_from_detections'
] 