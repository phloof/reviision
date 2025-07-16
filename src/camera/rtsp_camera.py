"""
RTSP Camera implementation for Retail Analytics System
"""

import time
import logging
import cv2
import urllib.parse
from .base import BaseCamera

logger = logging.getLogger(__name__)

class RTSPCamera(BaseCamera):
    """
    RTSP Camera implementation for IP cameras
    
    This class handles video capture from IP cameras via RTSP protocol
    using OpenCV's VideoCapture interface with secure credential handling.
    """
    
    def __init__(self, config):
        """
        Initialize RTSP camera with the provided configuration
        
        Args:
            config (dict): Camera configuration dictionary
        """
        super().__init__(config)
        
        # Get URL directly or from credential reference
        self.url = config.get('url')
        self.credential_ref = config.get('credential_ref')
        
        # If no direct URL but credential reference exists, handle it separately
        # The actual credential lookup happens in _open_camera
        if not self.url and not self.credential_ref:
            raise ValueError("RTSP URL or credential_ref must be provided in the configuration")
        
        self.cap = None
        
        # Additional RTSP camera specific settings
        self.retry_interval = config.get('retry_interval', 5.0)  # seconds between reconnection attempts
        self.max_retries = config.get('max_retries', -1)  # -1 means infinite retries
        self.connection_timeout = config.get('connection_timeout', 10.0)  # seconds to wait for connection
        self.buffer_size = config.get('buffer_size', 1)  # OpenCV buffer size (1 = no buffering)
        
        # Set RTSP transport method
        self.rtsp_transport = config.get('rtsp_transport', 'tcp')  # tcp or udp
        
        # Base cap options (always available)
        self.cap_options = [
            cv2.CAP_PROP_BUFFERSIZE, self.buffer_size,
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264')
        ]
        
        # Add RTSP transport option if available (compatibility check)
        if hasattr(cv2, 'CAP_PROP_RTSP_TRANSPORT'):
            if self.rtsp_transport == 'tcp':
                self.cap_options.extend([cv2.CAP_PROP_RTSP_TRANSPORT, 0])  # Use TCP (0 = TCP, 1 = UDP)
            else:
                self.cap_options.extend([cv2.CAP_PROP_RTSP_TRANSPORT, 1])  # Use UDP (0 = TCP, 1 = UDP)
            logger.debug(f"RTSP transport set to: {self.rtsp_transport}")
        else:
            logger.warning("CAP_PROP_RTSP_TRANSPORT not available in this OpenCV version. "
                         "RTSP transport protocol will use OpenCV defaults.")
        
        # Additional cap options from config
        extra_options = config.get('cap_options', [])
        if extra_options and isinstance(extra_options, list) and len(extra_options) % 2 == 0:
            self.cap_options.extend(extra_options)
    
    def _get_url_with_auth(self):
        """
        Get the RTSP URL with authentication credentials if available
        
        This method checks for embedded credentials in the URL, credential reference,
        or explicit username/password in config, and constructs the appropriate URL.
        
        Returns:
            str: RTSP URL with authentication if available
        """
        # If credential reference is provided, try to get URL from credentials
        if self.credential_ref:
            # If credential_manager is available in config
            credential_manager = self.config.get('credential_manager')
            if credential_manager:
                try:
                    # Get credentials using the reference
                    creds = credential_manager.get_service_credentials(self.credential_ref)
                    
                    # If credentials contain URL, use it
                    if creds and 'url' in creds:
                        url = creds['url']
                        username = creds.get('username')
                        password = creds.get('password')
                        
                        # If URL already has credentials embedded, use as is
                        if '://' in url and '@' in url.split('://', 1)[1].split('/', 1)[0]:
                            return url
                        
                        # If credentials are provided separately, add them to the URL
                        if username and password:
                            # Parse URL
                            parsed = urllib.parse.urlparse(url)
                            
                            # Add credentials to netloc
                            netloc = f"{urllib.parse.quote(username)}:{urllib.parse.quote(password)}@{parsed.netloc}"
                            
                            # Reconstruct URL
                            return urllib.parse.urlunparse((
                                parsed.scheme,
                                netloc,
                                parsed.path,
                                parsed.params,
                                parsed.query,
                                parsed.fragment
                            ))
                        
                        return url
                except Exception as e:
                    logger.error(f"Error getting credentials for {self.credential_ref}: {e}")
        
        # If direct URL is provided
        if self.url:
            # Check if URL already has credentials embedded
            if '://' in self.url and '@' in self.url.split('://', 1)[1].split('/', 1)[0]:
                return self.url
            
            # Check for explicit username/password in config
            username = self.config.get('username')
            password = self.config.get('password')
            
            if username and password:
                # Parse URL
                parsed = urllib.parse.urlparse(self.url)
                
                # Add credentials to netloc
                netloc = f"{urllib.parse.quote(username)}:{urllib.parse.quote(password)}@{parsed.netloc}"
                
                # Reconstruct URL
                return urllib.parse.urlunparse((
                    parsed.scheme,
                    netloc,
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment
                ))
            
            return self.url
        
        # If neither credential_ref nor url is available
        raise ValueError("No RTSP URL available")
    
    def _open_camera(self):
        """
        Open the RTSP camera
        
        Returns:
            bool: True if camera was opened successfully, False otherwise
        """
        try:
            # Get URL with authentication
            url = self._get_url_with_auth()
            
            # Log URL without credentials for security
            safe_url = self._get_safe_url(url)
            logger.info(f"Connecting to RTSP stream: {safe_url}")
            
            # Create capture with specific options for RTSP
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            
            # Set capture options
            for i in range(0, len(self.cap_options), 2):
                self.cap.set(self.cap_options[i], self.cap_options[i+1])
            
            # Wait for the connection to establish
            start_time = time.time()
            while not self.cap.isOpened():
                if time.time() - start_time > self.connection_timeout:
                    logger.error(f"Connection timeout to RTSP stream: {safe_url}")
                    self.cap.release()
                    self.cap = None
                    return False
                time.sleep(0.1)
            
            # Log actual camera properties
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"RTSP stream opened: {safe_url}")
            logger.info(f"Stream properties: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            return True
        
        except Exception as e:
            logger.error(f"Error opening RTSP stream: {e}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False
    
    def _get_safe_url(self, url):
        """
        Get a safe version of the URL with credentials removed
        
        Args:
            url (str): Original URL
            
        Returns:
            str: URL with credentials removed
        """
        try:
            # Parse URL
            parsed = urllib.parse.urlparse(url)
            
            # Check if URL has credentials
            if '@' in parsed.netloc:
                # Remove credentials from netloc
                netloc = parsed.netloc.split('@', 1)[1]
                
                # Reconstruct URL
                return urllib.parse.urlunparse((
                    parsed.scheme,
                    netloc,
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment
                ))
            
            return url
        except:
            # If parsing fails, return original URL
            return url
    
    def _close_camera(self):
        """Close the camera and release resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("RTSP stream closed")
    
    def _capture_loop(self):
        """
        Main capture loop for RTSP camera
        
        This method continuously reads frames from the RTSP stream and updates the frame buffer.
        It handles reconnection attempts if the stream disconnects.
        """
        retries = 0
        consecutive_failures = 0
        
        while self.is_running:
            # Check if camera is open, try to open if not
            if self.cap is None or not self.cap.isOpened():
                if self.max_retries >= 0 and retries >= self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) reached, giving up on RTSP stream")
                    break
                
                logger.info(f"Trying to open RTSP stream (attempt {retries + 1})")
                if self._open_camera():
                    retries = 0  # Reset retry counter on successful connection
                    consecutive_failures = 0
                else:
                    retries += 1
                    time.sleep(self.retry_interval)
                    continue
            
            # Read frame from stream
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 5:  # If we fail to get frames multiple times, reconnect
                        logger.warning(f"Failed to get frame from RTSP stream ({consecutive_failures} consecutive failures)")
                        self._close_camera()
                    time.sleep(0.1)
                    continue
                
                # Reset consecutive failures on successful frame capture
                consecutive_failures = 0
                
                # Preprocess frame
                processed_frame = self._preprocess_frame(frame)
                if processed_frame is not None:
                    self._set_frame(processed_frame)
                
                # Control frame rate (less aggressive for network streams)
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error capturing frame from RTSP stream: {e}")
                consecutive_failures += 1
                if consecutive_failures > 5:
                    self._close_camera()
                time.sleep(0.1)
        
        # Clean up when loop exits
        self._close_camera()
    
    def stop(self):
        """Stop the camera capture thread and release resources"""
        super().stop()
        self._close_camera() 