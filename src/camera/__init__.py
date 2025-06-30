"""
Camera module for Retail Analytics System
Provides interfaces for different camera types (USB, RTSP, RTSPS, ONVIF, Video Files)
"""

import logging
from .usb_camera import USBCamera
from .rtsp_camera import RTSPCamera
from .rtsps_camera import RTSPSCamera
from .mp4_camera import VideoFileCamera
from .onvif_camera import ONVIFCameraController

logger = logging.getLogger(__name__)

# Global camera manager to prevent multiple instances
class CameraManager:
    """Global camera manager to ensure only one camera instance is active at a time"""
    
    def __init__(self):
        self._current_camera = None
        self._current_config = None
    
    def get_camera(self, config):
        """
        Get camera instance, reusing existing if config matches or creating new one
        
        Args:
            config (dict): Camera configuration dictionary
            
        Returns:
            Camera: Camera instance
        """
        # If we have a current camera and config matches, return it
        if (self._current_camera is not None and 
            self._current_config is not None and
            self._config_matches(config, self._current_config)):
            return self._current_camera
        
        # Stop current camera if it exists
        if self._current_camera is not None:
            logger.info("Stopping previous camera instance")
            try:
                self._current_camera.stop()
            except Exception as e:
                logger.error(f"Error stopping previous camera: {e}")
        
        # Create new camera instance
        self._current_camera = self._create_camera(config)
        self._current_config = config.copy()
        
        return self._current_camera
    
    def stop_current_camera(self):
        """Stop the current camera instance"""
        if self._current_camera is not None:
            logger.info("Stopping current camera instance")
            try:
                self._current_camera.stop()
            except Exception as e:
                logger.error(f"Error stopping camera: {e}")
            finally:
                self._current_camera = None
                self._current_config = None
    
    def _config_matches(self, config1, config2):
        """Check if two camera configurations are equivalent"""
        # Compare essential fields
        essential_fields = ['type', 'file_path', 'url', 'device', 'host', 'username', 'password']
        for field in essential_fields:
            if config1.get(field) != config2.get(field):
                return False
        return True
    
    def _create_camera(self, config):
        """Create camera instance based on configuration"""
        camera_type = config.get('type', '').lower()
        
        if camera_type == 'usb':
            logger.info(f"Initializing USB camera: {config.get('device', '/dev/video0')}")
            return USBCamera(config)
        elif camera_type == 'rtsp':
            logger.info(f"Initializing RTSP camera: {config.get('url')}")
            return RTSPCamera(config)
        elif camera_type == 'rtsps':
            logger.info(f"Initializing RTSPS (secure) camera: {config.get('url')}")
            return RTSPSCamera(config)
        elif camera_type == 'onvif':
            logger.info(f"Initializing ONVIF camera: {config.get('host')}:{config.get('port', 80)}")
            return ONVIFCameraController(config)
        elif camera_type in ['video_file', 'mp4', 'mpg', 'mpeg']:
            logger.info(f"Initializing Video File camera: {config.get('file_path')}")
            return VideoFileCamera(config)
        else:
            error_msg = f"Unsupported camera type: {camera_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

# Global camera manager instance
_camera_manager = CameraManager()

def get_camera(config):
    """
    Factory function to create camera instance based on configuration
    
    Args:
        config (dict): Camera configuration dictionary
        
    Returns:
        Camera: Camera instance based on the specified type
        
    Raises:
        ValueError: If camera type is not supported
    """
    return _camera_manager.get_camera(config)

def stop_camera():
    """Stop the current camera instance"""
    _camera_manager.stop_current_camera() 