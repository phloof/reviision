"""
Base Camera class for Retail Analytics System
Provides a common interface for all camera types
"""

import time
import threading
import logging
import cv2
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseCamera(ABC):
    """
    Base camera class that defines the interface for all camera implementations
    
    This is an abstract base class that should be extended by specific camera implementations.
    It provides a thread-safe frame buffer and common processing methods.
    """
    
    def __init__(self, config):
        """
        Initialize the camera with the provided configuration
        
        Args:
            config (dict): Camera configuration dictionary
        """
        self.config = config
        self.resolution = config.get('resolution', [640, 480])
        self.fps = config.get('fps', 30)
        self.frame_buffer = None
        self.last_frame_time = 0
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Pre-processing settings
        self.resize_factor = config.get('resize_factor', 1.0)
        self.frame_skip = config.get('frame_skip', 0)
        self.frame_count = 0
        
        logger.debug(f"Initialized camera with resolution {self.resolution}, FPS {self.fps}")
    
    def start(self):
        """Start the camera capture thread"""
        if self.is_running:
            logger.warning("Camera is already running")
            return
        
        logger.info("Starting camera capture thread")
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the camera capture thread"""
        logger.info("Stopping camera capture thread")
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=5.0)
    
    def get_frame(self):
        """
        Get the latest frame from the camera
        
        Returns:
            numpy.ndarray: Latest frame from the camera or None if no frame is available
        """
        with self.lock:
            if self.frame_buffer is None:
                return None
            return self.frame_buffer.copy()
    
    def _set_frame(self, frame):
        """
        Set the latest frame in the buffer (thread-safe)
        
        Args:
            frame (numpy.ndarray): Frame to store in the buffer
        """
        with self.lock:
            self.frame_buffer = frame
            self.last_frame_time = time.time()
    
    def _preprocess_frame(self, frame):
        """
        Preprocess the frame before storing in the buffer
        
        Args:
            frame (numpy.ndarray): Raw frame from the camera
            
        Returns:
            numpy.ndarray: Preprocessed frame
        """
        if frame is None:
            return None
        
        # Skip frames if configured
        self.frame_count += 1
        if self.frame_skip > 0 and (self.frame_count % (self.frame_skip + 1)) != 0:
            return None
        
        # Resize frame if needed
        if self.resize_factor != 1.0:
            height, width = frame.shape[:2]
            new_height = int(height * self.resize_factor)
            new_width = int(width * self.resize_factor)
            frame = cv2.resize(frame, (new_width, new_height))
        
        return frame
    
    @abstractmethod
    def _capture_loop(self):
        """
        Main capture loop to be implemented by subclasses
        
        This method should run in a separate thread and continuously update the frame buffer.
        """
        pass
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop() 