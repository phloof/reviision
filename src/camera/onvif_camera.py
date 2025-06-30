"""
ONVIF Camera implementation for Retail Analytics System
Supports ONVIF-compliant cameras with PTZ (Pan Tilt Zoom) functionality
"""

import time
import logging
import cv2
import threading
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse

from onvif import ONVIFCamera
from zeep import wsse
import numpy as np

from .base import BaseCamera

logger = logging.getLogger(__name__)

class ONVIFCameraController(BaseCamera):
    """
    ONVIF Camera implementation with PTZ support
    
    Supports ONVIF-compliant cameras like the Tapo C220
    Features:
    - Live streaming via RTSP
    - Pan/Tilt/Zoom control
    - Preset positions
    - Camera configuration
    """
    
    def __init__(self, config):
        """
        Initialize ONVIF camera
        
        Args:
            config (dict): Camera configuration with ONVIF settings
                Required keys:
                - host: Camera IP address
                - port: ONVIF port (default: 80)
                - username: ONVIF username
                - password: ONVIF password
                Optional keys:
                - rtsp_port: RTSP port (default: 554)
                - profile_index: Media profile index (default: 0)
                - use_https: Use HTTPS for ONVIF (default: False)
                - ptz_timeout: PTZ operation timeout (default: 5.0)
        """
        super().__init__(config)
        
        # ONVIF Configuration
        self.host = config['host']
        self.port = config.get('port', 80)
        self.username = config['username']
        self.password = config['password']
        self.rtsp_port = config.get('rtsp_port', 554)
        self.profile_index = config.get('profile_index', 0)
        self.use_https = config.get('use_https', False)
        self.ptz_timeout = config.get('ptz_timeout', 5.0)
        
        # ONVIF Camera and Services
        self.onvif_camera = None
        self.media_service = None
        self.ptz_service = None
        self.imaging_service = None
        self.device_service = None
        
        # Media Profile and PTZ Node
        self.media_profile = None
        self.ptz_node = None
        self.ptz_configuration = None
        
        # RTSP Stream
        self.rtsp_url = None
        self.cap = None
        
        # PTZ Capabilities
        self.ptz_capabilities = {
            'pan': False,
            'tilt': False,
            'zoom': False,
            'absolute': False,
            'relative': False,
            'continuous': False,
            'presets': False
        }
        
        # PTZ Limits
        self.ptz_limits = {
            'pan': {'min': -1.0, 'max': 1.0},
            'tilt': {'min': -1.0, 'max': 1.0},
            'zoom': {'min': 0.0, 'max': 1.0}
        }
        
        # Current PTZ Position
        self.current_position = {'pan': 0.0, 'tilt': 0.0, 'zoom': 0.0}
        
        logger.info(f"Initializing ONVIF camera at {self.host}:{self.port}")
        
        # Initialize ONVIF connection
        self._connect_onvif()
    
    def _connect_onvif(self):
        """Establish ONVIF connection and initialize services"""
        try:
            # Create ONVIF camera instance
            self.onvif_camera = ONVIFCamera(
                self.host, 
                self.port, 
                self.username, 
                self.password
            )
            
            # Get device service for device info
            self.device_service = self.onvif_camera.create_devicemgmt_service()
            
            # Get device information
            device_info = self.device_service.GetDeviceInformation()
            logger.info(f"Connected to ONVIF device: {device_info.Manufacturer} {device_info.Model}")
            
            # Initialize media service
            self.media_service = self.onvif_camera.create_media_service()
            
            # Get media profiles
            profiles = self.media_service.GetProfiles()
            if not profiles:
                raise RuntimeError("No media profiles found")
            
            # Select media profile
            if self.profile_index >= len(profiles):
                logger.warning(f"Profile index {self.profile_index} not available, using profile 0")
                self.profile_index = 0
            
            self.media_profile = profiles[self.profile_index]
            logger.info(f"Using media profile: {self.media_profile.Name}")
            
            # Get RTSP stream URL
            self._get_rtsp_url()
            
            # Initialize PTZ if available
            self._initialize_ptz()
            
        except Exception as e:
            logger.error(f"Failed to connect to ONVIF camera: {e}")
            raise
    
    def _get_rtsp_url(self):
        """Get RTSP streaming URL from media profile"""
        try:
            # Create stream setup request
            stream_setup = self.media_service.create_type('GetStreamUri')
            stream_setup.ProfileToken = self.media_profile.token
            stream_setup.StreamSetup = {
                'Stream': 'RTP-Unicast',
                'Transport': {'Protocol': 'RTSP'}
            }
            
            # Get stream URI
            stream_uri = self.media_service.GetStreamUri(stream_setup)
            self.rtsp_url = stream_uri.Uri
            
            # Update RTSP URL with credentials if needed
            parsed_url = urlparse(self.rtsp_url)
            if not parsed_url.username:
                # Add credentials to RTSP URL
                scheme = parsed_url.scheme
                netloc = f"{self.username}:{self.password}@{parsed_url.hostname}"
                if parsed_url.port:
                    netloc += f":{parsed_url.port}"
                self.rtsp_url = f"{scheme}://{netloc}{parsed_url.path}"
                if parsed_url.query:
                    self.rtsp_url += f"?{parsed_url.query}"
            
            logger.info(f"RTSP URL obtained: {parsed_url.scheme}://{parsed_url.hostname}:{parsed_url.port}{parsed_url.path}")
            
        except Exception as e:
            logger.error(f"Failed to get RTSP URL: {e}")
            raise
    
    def _initialize_ptz(self):
        """Initialize PTZ service and capabilities"""
        try:
            # Check if PTZ is supported
            capabilities = self.device_service.GetCapabilities()
            if not hasattr(capabilities, 'PTZ') or not capabilities.PTZ:
                logger.info("PTZ not supported by this camera")
                return
            
            # Create PTZ service
            self.ptz_service = self.onvif_camera.create_ptz_service()
            
            # Get PTZ configuration
            ptz_configs = self.ptz_service.GetConfigurations()
            if not ptz_configs:
                logger.warning("No PTZ configurations found")
                return
            
            self.ptz_configuration = ptz_configs[0]
            
            # Get PTZ node
            if hasattr(self.ptz_configuration, 'NodeToken'):
                nodes = self.ptz_service.GetNodes()
                for node in nodes:
                    if node.token == self.ptz_configuration.NodeToken:
                        self.ptz_node = node
                        break
            
            if not self.ptz_node:
                logger.warning("PTZ node not found")
                return
            
            # Analyze PTZ capabilities
            self._analyze_ptz_capabilities()
            
            # Get current PTZ position
            self._update_current_position()
            
            logger.info("PTZ initialized successfully")
            
        except Exception as e:
            logger.warning(f"PTZ initialization failed: {e}")
    
    def _analyze_ptz_capabilities(self):
        """Analyze and store PTZ capabilities"""
        if not self.ptz_node:
            return
        
        try:
            # Check supported PTZ spaces
            if hasattr(self.ptz_node, 'SupportedPTZSpaces'):
                spaces = self.ptz_node.SupportedPTZSpaces
                
                # Check for pan/tilt support
                if hasattr(spaces, 'AbsolutePanTiltPositionSpace'):
                    self.ptz_capabilities['pan'] = True
                    self.ptz_capabilities['tilt'] = True
                    self.ptz_capabilities['absolute'] = True
                    
                    # Get pan/tilt limits
                    for space in spaces.AbsolutePanTiltPositionSpace:
                        if hasattr(space, 'XRange'):
                            self.ptz_limits['pan']['min'] = space.XRange.Min
                            self.ptz_limits['pan']['max'] = space.XRange.Max
                        if hasattr(space, 'YRange'):
                            self.ptz_limits['tilt']['min'] = space.YRange.Min
                            self.ptz_limits['tilt']['max'] = space.YRange.Max
                
                # Check for zoom support
                if hasattr(spaces, 'AbsoluteZoomPositionSpace'):
                    self.ptz_capabilities['zoom'] = True
                    
                    # Get zoom limits
                    for space in spaces.AbsoluteZoomPositionSpace:
                        if hasattr(space, 'XRange'):
                            self.ptz_limits['zoom']['min'] = space.XRange.Min
                            self.ptz_limits['zoom']['max'] = space.XRange.Max
                
                # Check for relative movement
                if hasattr(spaces, 'RelativePanTiltTranslationSpace'):
                    self.ptz_capabilities['relative'] = True
                
                # Check for continuous movement
                if hasattr(spaces, 'ContinuousPanTiltVelocitySpace'):
                    self.ptz_capabilities['continuous'] = True
            
            # Check for preset support
            try:
                presets = self.ptz_service.GetPresets({'ProfileToken': self.media_profile.token})
                self.ptz_capabilities['presets'] = True
            except:
                self.ptz_capabilities['presets'] = False
            
            logger.info(f"PTZ capabilities: {self.ptz_capabilities}")
            logger.info(f"PTZ limits: {self.ptz_limits}")
            
        except Exception as e:
            logger.warning(f"Failed to analyze PTZ capabilities: {e}")
    
    def _update_current_position(self):
        """Update current PTZ position"""
        if not self.ptz_service or not self.media_profile:
            return
        
        try:
            status = self.ptz_service.GetStatus({'ProfileToken': self.media_profile.token})
            if hasattr(status, 'Position'):
                position = status.Position
                if hasattr(position, 'PanTilt'):
                    self.current_position['pan'] = position.PanTilt.x
                    self.current_position['tilt'] = position.PanTilt.y
                if hasattr(position, 'Zoom'):
                    self.current_position['zoom'] = position.Zoom.x
        except Exception as e:
            logger.warning(f"Failed to update PTZ position: {e}")
    
    def _capture_loop(self):
        """Main capture loop implementation"""
        retry_count = 0
        max_retries = 5
        retry_delay = 2.0
        
        while self.is_running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    # Initialize video capture
                    logger.info(f"Connecting to RTSP stream: {self.rtsp_url}")
                    self.cap = cv2.VideoCapture(self.rtsp_url)
                    
                    # Set buffer size to reduce latency
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    if not self.cap.isOpened():
                        raise RuntimeError("Failed to open RTSP stream")
                    
                    retry_count = 0
                    logger.info("RTSP stream connected successfully")
                
                # Read frame
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    # Preprocess frame
                    processed_frame = self._preprocess_frame(frame)
                    if processed_frame is not None:
                        self._set_frame(processed_frame)
                    retry_count = 0
                else:
                    # Handle read failure
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error("Too many consecutive frame read failures, releasing capture")
                        if self.cap:
                            self.cap.release()
                            self.cap = None
                        retry_count = 0
                    
                    time.sleep(retry_delay)
                    continue
                
                # Control frame rate
                time.sleep(1.0 / self.fps)
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error("Maximum retries reached, stopping capture")
                    self.is_running = False
                else:
                    time.sleep(retry_delay * retry_count)
    
    def stop(self):
        """Stop camera capture and release resources"""
        logger.info("Stopping ONVIF camera")
        super().stop()
        
        if self.cap:
            self.cap.release()
            self.cap = None
    
    # ============================================================================
    # PTZ Control Methods
    # ============================================================================
    
    def move_absolute(self, pan: float, tilt: float, zoom: float) -> bool:
        """
        Move camera to absolute position
        
        Args:
            pan: Pan position (-1.0 to 1.0)
            tilt: Tilt position (-1.0 to 1.0)
            zoom: Zoom level (0.0 to 1.0)
            
        Returns:
            bool: True if successful
        """
        if not self.ptz_service or not self.ptz_capabilities['absolute']:
            logger.warning("Absolute PTZ movement not supported")
            return False
        
        try:
            # Clamp values to limits
            pan = max(self.ptz_limits['pan']['min'], min(self.ptz_limits['pan']['max'], pan))
            tilt = max(self.ptz_limits['tilt']['min'], min(self.ptz_limits['tilt']['max'], tilt))
            zoom = max(self.ptz_limits['zoom']['min'], min(self.ptz_limits['zoom']['max'], zoom))
            
            # Create absolute move request
            request = self.ptz_service.create_type('AbsoluteMove')
            request.ProfileToken = self.media_profile.token
            
            # Set position
            request.Position = {
                'PanTilt': {'x': pan, 'y': tilt},
                'Zoom': {'x': zoom}
            }
            
            # Execute move
            self.ptz_service.AbsoluteMove(request)
            
            # Update current position
            self.current_position.update({'pan': pan, 'tilt': tilt, 'zoom': zoom})
            
            logger.info(f"PTZ moved to absolute position: pan={pan:.2f}, tilt={tilt:.2f}, zoom={zoom:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Absolute PTZ move failed: {e}")
            return False
    
    def move_relative(self, pan_delta: float, tilt_delta: float, zoom_delta: float) -> bool:
        """
        Move camera by relative amounts
        
        Args:
            pan_delta: Pan movement delta
            tilt_delta: Tilt movement delta
            zoom_delta: Zoom movement delta
            
        Returns:
            bool: True if successful
        """
        if not self.ptz_service or not self.ptz_capabilities['relative']:
            logger.warning("Relative PTZ movement not supported")
            return False
        
        try:
            # Create relative move request
            request = self.ptz_service.create_type('RelativeMove')
            request.ProfileToken = self.media_profile.token
            
            # Set translation
            request.Translation = {
                'PanTilt': {'x': pan_delta, 'y': tilt_delta},
                'Zoom': {'x': zoom_delta}
            }
            
            # Execute move
            self.ptz_service.RelativeMove(request)
            
            logger.info(f"PTZ moved relatively: pan={pan_delta:.2f}, tilt={tilt_delta:.2f}, zoom={zoom_delta:.2f}")
            
            # Update current position after move
            time.sleep(0.5)  # Wait for movement to complete
            self._update_current_position()
            
            return True
            
        except Exception as e:
            logger.error(f"Relative PTZ move failed: {e}")
            return False
    
    def start_continuous_move(self, pan_speed: float, tilt_speed: float, zoom_speed: float) -> bool:
        """
        Start continuous PTZ movement
        
        Args:
            pan_speed: Pan speed (-1.0 to 1.0)
            tilt_speed: Tilt speed (-1.0 to 1.0)
            zoom_speed: Zoom speed (-1.0 to 1.0)
            
        Returns:
            bool: True if successful
        """
        if not self.ptz_service or not self.ptz_capabilities['continuous']:
            logger.warning("Continuous PTZ movement not supported")
            return False
        
        try:
            # Create continuous move request
            request = self.ptz_service.create_type('ContinuousMove')
            request.ProfileToken = self.media_profile.token
            
            # Set velocity
            request.Velocity = {
                'PanTilt': {'x': pan_speed, 'y': tilt_speed},
                'Zoom': {'x': zoom_speed}
            }
            
            # Execute continuous move
            self.ptz_service.ContinuousMove(request)
            
            logger.info(f"PTZ continuous move started: pan_speed={pan_speed:.2f}, tilt_speed={tilt_speed:.2f}, zoom_speed={zoom_speed:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Continuous PTZ move failed: {e}")
            return False
    
    def stop_ptz(self) -> bool:
        """
        Stop all PTZ movement
        
        Returns:
            bool: True if successful
        """
        if not self.ptz_service:
            return False
        
        try:
            request = self.ptz_service.create_type('Stop')
            request.ProfileToken = self.media_profile.token
            request.PanTilt = True
            request.Zoom = True
            
            self.ptz_service.Stop(request)
            
            logger.info("PTZ movement stopped")
            
            # Update current position
            time.sleep(0.2)
            self._update_current_position()
            
            return True
            
        except Exception as e:
            logger.error(f"PTZ stop failed: {e}")
            return False
    
    def get_presets(self) -> list:
        """
        Get available preset positions
        
        Returns:
            list: List of preset dictionaries
        """
        if not self.ptz_service or not self.ptz_capabilities['presets']:
            return []
        
        try:
            presets = self.ptz_service.GetPresets({'ProfileToken': self.media_profile.token})
            preset_list = []
            
            for preset in presets:
                preset_info = {
                    'token': preset.token,
                    'name': preset.Name if hasattr(preset, 'Name') else f"Preset {preset.token}"
                }
                preset_list.append(preset_info)
            
            return preset_list
            
        except Exception as e:
            logger.error(f"Failed to get presets: {e}")
            return []
    
    def goto_preset(self, preset_token: str) -> bool:
        """
        Move camera to a preset position
        
        Args:
            preset_token: Token of the preset to go to
            
        Returns:
            bool: True if successful
        """
        if not self.ptz_service or not self.ptz_capabilities['presets']:
            logger.warning("PTZ presets not supported")
            return False
        
        try:
            request = self.ptz_service.create_type('GotoPreset')
            request.ProfileToken = self.media_profile.token
            request.PresetToken = preset_token
            
            self.ptz_service.GotoPreset(request)
            
            logger.info(f"Moved to preset: {preset_token}")
            
            # Update current position after preset move
            time.sleep(1.0)  # Wait for movement to complete
            self._update_current_position()
            
            return True
            
        except Exception as e:
            logger.error(f"Goto preset failed: {e}")
            return False
    
    def set_preset(self, name: str) -> Optional[str]:
        """
        Set a new preset at current position
        
        Args:
            name: Name for the new preset
            
        Returns:
            str: Preset token if successful, None otherwise
        """
        if not self.ptz_service or not self.ptz_capabilities['presets']:
            logger.warning("PTZ presets not supported")
            return None
        
        try:
            request = self.ptz_service.create_type('SetPreset')
            request.ProfileToken = self.media_profile.token
            request.PresetName = name
            
            response = self.ptz_service.SetPreset(request)
            preset_token = response.PresetToken if hasattr(response, 'PresetToken') else None
            
            logger.info(f"Preset '{name}' set with token: {preset_token}")
            return preset_token
            
        except Exception as e:
            logger.error(f"Set preset failed: {e}")
            return None
    
    def remove_preset(self, preset_token: str) -> bool:
        """
        Remove a preset
        
        Args:
            preset_token: Token of the preset to remove
            
        Returns:
            bool: True if successful
        """
        if not self.ptz_service or not self.ptz_capabilities['presets']:
            logger.warning("PTZ presets not supported")
            return False
        
        try:
            request = self.ptz_service.create_type('RemovePreset')
            request.ProfileToken = self.media_profile.token
            request.PresetToken = preset_token
            
            self.ptz_service.RemovePreset(request)
            
            logger.info(f"Preset {preset_token} removed")
            return True
            
        except Exception as e:
            logger.error(f"Remove preset failed: {e}")
            return False
    
    # ============================================================================
    # Camera Information and Status
    # ============================================================================
    
    def get_camera_info(self) -> Dict[str, Any]:
        """
        Get camera device information
        
        Returns:
            dict: Camera information
        """
        info = {
            'host': self.host,
            'port': self.port,
            'rtsp_url': self.rtsp_url,
            'ptz_capabilities': self.ptz_capabilities,
            'ptz_limits': self.ptz_limits,
            'current_position': self.current_position.copy()
        }
        
        if self.device_service:
            try:
                device_info = self.device_service.GetDeviceInformation()
                info.update({
                    'manufacturer': device_info.Manufacturer,
                    'model': device_info.Model,
                    'firmware_version': device_info.FirmwareVersion,
                    'serial_number': device_info.SerialNumber,
                    'hardware_id': device_info.HardwareId
                })
            except Exception as e:
                logger.warning(f"Failed to get device information: {e}")
        
        return info
    
    def get_ptz_status(self) -> Dict[str, Any]:
        """
        Get current PTZ status and position
        
        Returns:
            dict: PTZ status information
        """
        self._update_current_position()
        
        return {
            'position': self.current_position.copy(),
            'capabilities': self.ptz_capabilities.copy(),
            'limits': self.ptz_limits.copy(),
            'presets': self.get_presets()
        }
