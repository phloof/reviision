"""
USB Camera implementation for Retail Analytics System
"""

import time
import logging
import cv2
from .base import BaseCamera

logger = logging.getLogger(__name__)

class USBCamera(BaseCamera):
    """
    USB Camera implementation for local camera devices
    
    This class handles video capture from USB webcams or integrated cameras
    via OpenCV's VideoCapture interface.
    """
    
    def __init__(self, config):
        """
        Initialize USB camera with the provided configuration
        
        Args:
            config (dict): Camera configuration dictionary
        """
        super().__init__(config)
        self.device = config.get('device', '/dev/video0')
        self.cap = None
        
        # Additional USB camera specific settings
        self.retry_interval = config.get('retry_interval', 5.0)  # seconds between reconnection attempts
        self.max_retries = config.get('max_retries', -1)  # -1 means infinite retries
    
    def _open_camera(self):
        """
        Open the USB camera
        
        Returns:
            bool: True if camera was opened successfully, False otherwise
        """
        try:
            # Try to open the camera using device path or index
            try:
                device_id = int(self.device)
            except ValueError:
                device_id = self.device
            
            self.cap = cv2.VideoCapture(device_id)
            
            # Set camera properties
            width, height = self.resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Check if camera is open
            if not self.cap.isOpened():
                logger.error(f"Failed to open USB camera: {self.device}")
                return False
            
            # Log actual camera properties
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"USB camera opened: {self.device}")
            logger.info(f"Requested resolution: {width}x{height}, Actual: {actual_width}x{actual_height}")
            logger.info(f"Requested FPS: {self.fps}, Actual: {actual_fps}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error opening USB camera {self.device}: {e}")
            return False
    
    def _close_camera(self):
        """Close the camera and release resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info(f"USB camera closed: {self.device}")
    
    def _capture_loop(self):
        """
        Main capture loop for USB camera
        
        This method continuously reads frames from the camera and updates the frame buffer.
        It handles reconnection attempts if the camera disconnects.
        """
        retries = 0
        
        while self.is_running:
            # Check if camera is open, try to open if not
            if self.cap is None or not self.cap.isOpened():
                if self.max_retries >= 0 and retries >= self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) reached, giving up on USB camera")
                    break
                
                logger.info(f"Trying to open USB camera (attempt {retries + 1})")
                if self._open_camera():
                    retries = 0  # Reset retry counter on successful connection
                else:
                    retries += 1
                    time.sleep(self.retry_interval)
                    continue
            
            # Read frame from camera
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    logger.warning("Failed to get frame from USB camera")
                    self._close_camera()  # Close and retry
                    time.sleep(0.1)
                    continue
                
                # Preprocess frame
                processed_frame = self._preprocess_frame(frame)
                if processed_frame is not None:
                    self._set_frame(processed_frame)
                
                # Control frame rate
                time.sleep(max(0, 1.0/self.fps - 0.01))
                
            except Exception as e:
                logger.error(f"Error capturing frame from USB camera: {e}")
                self._close_camera()
                time.sleep(0.1)
        
        # Clean up when loop exits
        self._close_camera()
    
    def stop(self):
        """Stop the camera capture thread and release resources"""
        super().stop()
        self._close_camera() 