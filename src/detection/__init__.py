"""
Detection module for Retail Analytics System
Provides person detection and tracking functionality
"""

from .detector import PersonDetector
from .tracker import PersonTracker

__all__ = ['PersonDetector', 'PersonTracker'] 