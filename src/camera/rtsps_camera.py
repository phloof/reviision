"""
RTSPS (Secure RTSP) Camera implementation for Retail Analytics System
"""

import os
import time
import logging
import cv2
import tempfile
import shutil
import urllib.parse
import subprocess
from pathlib import Path
from .rtsp_camera import RTSPCamera

logger = logging.getLogger(__name__)

class RTSPSCamera(RTSPCamera):
    """
    RTSPS Camera implementation for secure IP cameras
    
    This class extends the RTSPCamera class to handle secure RTSP connections
    with SSL/TLS support. It handles certificate validation and secure connections.
    """
    
    def __init__(self, config):
        """
        Initialize RTSPS camera with the provided configuration
        
        Args:
            config (dict): Camera configuration dictionary
        """
        # Call parent init but let it handle URL validation differently
        super(RTSPCamera, self).__init__(config)
        
        # RTSPS specific settings
        self.url = config.get('url')
        self.credential_ref = config.get('credential_ref')
        if not self.url and not self.credential_ref:
            raise ValueError("RTSPS URL or credential_ref must be provided in the configuration")
        
        # Certificate settings
        self.cert_file = config.get('cert_file')
        self.key_file = config.get('key_file')
        self.ca_file = config.get('ca_file')
        self.verify_ssl = config.get('verify_ssl', True)
        
        # Initialize other properties from RTSPCamera
        self.cap = None
        self.retry_interval = config.get('retry_interval', 5.0)
        self.max_retries = config.get('max_retries', -1)
        self.connection_timeout = config.get('connection_timeout', 15.0)  # Extended timeout for SSL handshake
        self.buffer_size = config.get('buffer_size', 1)
        
        # RTSPS uses TCP by default
        self.rtsp_transport = 'tcp'
        self.cap_options = [
            cv2.CAP_PROP_BUFFERSIZE, self.buffer_size,
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'),
            cv2.CAP_PROP_RTSP_TRANSPORT, 0  # Use TCP (0 = TCP, 1 = UDP)
        ]
        
        # Extra options
        extra_options = config.get('cap_options', [])
        if extra_options and isinstance(extra_options, list) and len(extra_options) % 2 == 0:
            self.cap_options.extend(extra_options)
        
        # Environment variables for OpenSSL settings
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        
        # Set up temporary directory for certificates if needed
        self.temp_dir = None
        if self.cert_file or self.key_file or self.ca_file:
            self.temp_dir = tempfile.mkdtemp(prefix='rtsps_')
            logger.debug(f"Created temporary directory for certificates: {self.temp_dir}")
        
        # Try to get certificates from credential manager if available
        self._get_certificates_from_credentials()
    
    def _get_certificates_from_credentials(self):
        """
        Get certificate files from credential manager if available
        """
        credential_manager = self.config.get('credential_manager')
        if not credential_manager:
            return
        
        # If credential reference is specified, try to get certificates
        if self.credential_ref:
            try:
                creds = credential_manager.get_service_credentials(self.credential_ref)
                
                # Check for certificate data in credentials
                cert_data = creds.get('cert')
                key_data = creds.get('key')
                ca_data = creds.get('ca')
                
                # Create temporary directory if needed
                if (cert_data or key_data or ca_data) and not self.temp_dir:
                    self.temp_dir = tempfile.mkdtemp(prefix='rtsps_')
                    logger.debug(f"Created temporary directory for credentials: {self.temp_dir}")
                
                # Save certificate data to files
                if cert_data and self.temp_dir:
                    cert_path = os.path.join(self.temp_dir, 'client.crt')
                    with open(cert_path, 'w') as f:
                        f.write(cert_data)
                    os.chmod(cert_path, 0o600)
                    self.cert_file = cert_path
                
                if key_data and self.temp_dir:
                    key_path = os.path.join(self.temp_dir, 'client.key')
                    with open(key_path, 'w') as f:
                        f.write(key_data)
                    os.chmod(key_path, 0o600)
                    self.key_file = key_path
                
                if ca_data and self.temp_dir:
                    ca_path = os.path.join(self.temp_dir, 'ca.crt')
                    with open(ca_path, 'w') as f:
                        f.write(ca_data)
                    os.chmod(ca_path, 0o600)
                    self.ca_file = ca_path
                
                # Get SSL verification setting
                if 'verify_ssl' in creds:
                    self.verify_ssl = creds.get('verify_ssl')
                
            except Exception as e:
                logger.error(f"Error getting certificates from credentials: {e}")
    
    def _setup_certificates(self):
        """
        Set up certificates for secure connection
        
        This method prepares certificate files for use with OpenCV's FFmpeg backend.
        
        Returns:
            bool: True if certificates were set up successfully, False otherwise
        """
        if not self.temp_dir and not (self.cert_file or self.key_file or self.ca_file):
            return True  # No certificates to set up
        
        try:
            # Set environment variables for SSL certificates
            if self.cert_file:
                if self.temp_dir and not self.cert_file.startswith(self.temp_dir):
                    cert_path = os.path.join(self.temp_dir, 'client.crt')
                    shutil.copy(self.cert_file, cert_path)
                    self.cert_file = cert_path
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] += ' client_certificate;' + self.cert_file
                logger.debug(f"Using client certificate: {self.cert_file}")
            
            if self.key_file:
                if self.temp_dir and not self.key_file.startswith(self.temp_dir):
                    key_path = os.path.join(self.temp_dir, 'client.key')
                    shutil.copy(self.key_file, key_path)
                    self.key_file = key_path
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] += ' client_key;' + self.key_file
                logger.debug(f"Using client key: {self.key_file}")
            
            if self.ca_file:
                if self.temp_dir and not self.ca_file.startswith(self.temp_dir):
                    ca_path = os.path.join(self.temp_dir, 'ca.crt')
                    shutil.copy(self.ca_file, ca_path)
                    self.ca_file = ca_path
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] += ' ca_file;' + self.ca_file
                logger.debug(f"Using CA certificate: {self.ca_file}")
            
            # Set SSL verification option
            if not self.verify_ssl:
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] += ' tls_verify;0'
                logger.warning("SSL certificate verification is disabled")
            
            return True
        
        except Exception as e:
            logger.error(f"Error setting up certificates: {e}")
            return False
    
    def _open_camera(self):
        """
        Open the RTSPS camera with secure connection
        
        Returns:
            bool: True if camera was opened successfully, False otherwise
        """
        try:
            # Get URL with authentication
            url = self._get_url_with_auth()
            
            # Log URL without credentials for security
            safe_url = self._get_safe_url(url)
            logger.info(f"Connecting to secure RTSPS stream: {safe_url}")
            
            # Set up certificates
            if not self._setup_certificates():
                logger.error("Failed to set up certificates, aborting connection")
                return False
            
            # Check if ffmpeg supports SSL
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            if 'enable-openssl' not in result.stdout.lower() and 'enable-gnutls' not in result.stdout.lower():
                logger.warning("FFmpeg may not support SSL. RTSPS connections might fail.")
            
            # Create capture with specific options for RTSPS
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            
            # Set capture options
            for i in range(0, len(self.cap_options), 2):
                self.cap.set(self.cap_options[i], self.cap_options[i+1])
            
            # Wait for the connection to establish
            start_time = time.time()
            while not self.cap.isOpened():
                if time.time() - start_time > self.connection_timeout:
                    logger.error(f"Connection timeout to RTSPS stream: {safe_url}")
                    self.cap.release()
                    self.cap = None
                    return False
                time.sleep(0.1)
            
            # Log actual camera properties
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"RTSPS stream opened: {safe_url}")
            logger.info(f"Stream properties: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            return True
        
        except Exception as e:
            logger.error(f"Error opening RTSPS stream: {e}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False
    
    def _cleanup_temp_dir(self):
        """Clean up temporary directory used for certificates"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Removed temporary directory: {self.temp_dir}")
                self.temp_dir = None
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {e}")
    
    def stop(self):
        """Stop the camera capture thread and release resources"""
        super().stop()
        self._cleanup_temp_dir()
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up"""
        self._cleanup_temp_dir() 