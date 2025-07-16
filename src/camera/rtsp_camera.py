"""
RTSP Camera implementation for Retail Analytics System
"""

import time
import logging
import cv2
import urllib.parse
import threading
import queue
from .base import BaseCamera

logger = logging.getLogger(__name__)

try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False

class RTSPCamera(BaseCamera):
    """
    RTSP Camera implementation for IP cameras
    
    This class handles video capture from IP cameras via RTSP protocol
    using OpenCV's VideoCapture interface with secure credential handling
    and H.264 error resilience.
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
        self.frame_queue = queue.Queue(maxsize=1)
        self.grabber_thread = None
        self.grabber_running = False
        
        # Additional RTSP camera specific settings
        self.retry_interval = config.get('retry_interval', 5.0)  # seconds between reconnection attempts
        self.max_retries = config.get('max_retries', -1)  # -1 means infinite retries
        self.connection_timeout = config.get('connection_timeout', 10.0)  # seconds to wait for connection
        self.buffer_size = config.get('buffer_size', 1)  # OpenCV buffer size (1 = no buffering)
        
        # Set RTSP transport method
        self.rtsp_transport = config.get('rtsp_transport', 'tcp')  # tcp or udp
        
        # H.264 error handling settings
        self.h264_error_threshold = config.get('h264_error_threshold', 20)  # Max consecutive H.264 errors
        self.frame_skip_on_error = config.get('frame_skip_on_error', True)  # Skip corrupted frames
        self.recovery_frame_count = config.get('recovery_frame_count', 5)  # Frames to read for recovery

        # Frame validation settings
        self.min_frame_size = config.get('min_frame_size', 100)  # Minimum frame size in pixels
        self.max_decode_errors = config.get('max_decode_errors', 50)  # Max decode errors before reconnect

        # Tracking variables for H.264 errors
        self.h264_error_count = 0
        self.last_good_frame = None
        self.frame_lock = threading.Lock()

        # Base cap options (always available) - optimized for H.264 stability
        self.cap_options = [
            cv2.CAP_PROP_BUFFERSIZE, self.buffer_size,
            cv2.CAP_PROP_FPS, config.get('fps', 15),  # Reduced FPS for stability
        ]

        # Add H.264 fourcc if available
        try:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            self.cap_options.extend([cv2.CAP_PROP_FOURCC, fourcc])
        except AttributeError:
            # Fallback if VideoWriter_fourcc is not available
            logger.debug("VideoWriter_fourcc not available, using default codec")

        # Add H.264 specific options for better error handling
        self.cap_options.extend([
            cv2.CAP_PROP_CONVERT_RGB, 1,  # Ensure RGB conversion
        ])

        # Add resolution settings if specified
        if 'resolution' in config:
            width, height = config['resolution']
            self.cap_options.extend([
                cv2.CAP_PROP_FRAME_WIDTH, width,
                cv2.CAP_PROP_FRAME_HEIGHT, height
            ])
        
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
    
    def _validate_frame(self, frame):
        """
        Validate frame quality and detect H.264 corruption

        Args:
            frame: OpenCV frame to validate

        Returns:
            bool: True if frame is valid, False otherwise
        """
        if frame is None:
            return False

        # Check frame size
        if frame.size == 0:
            return False

        # Check frame dimensions
        if len(frame.shape) < 2 or frame.shape[0] < self.min_frame_size or frame.shape[1] < self.min_frame_size:
            return False

        # Check for completely black or white frames (common H.264 corruption)
        if frame.mean() < 1 or frame.mean() > 254:
            return False

        # Check for frame with no variation (frozen frame indicator)
        if frame.std() < 1:
            return False

        return True

    def _handle_h264_error(self):
        """
        Handle H.264 decoding errors by implementing recovery strategies

        Returns:
            bool: True if recovery was attempted, False if reconnection needed
        """
        self.h264_error_count += 1

        if self.h264_error_count > self.h264_error_threshold:
            logger.warning(f"H.264 error threshold reached ({self.h264_error_count} errors), reconnecting")
            return False

        # Try to recover by skipping frames
        if self.frame_skip_on_error and self.cap is not None:
            logger.debug(f"H.264 error #{self.h264_error_count}, attempting frame skip recovery")

            # Skip several frames to get past corrupted section
            for _ in range(self.recovery_frame_count):
                if self.cap.isOpened():
                    try:
                        self.cap.grab()  # Fast frame skip without decoding
                    except:
                        break

            return True

        return False

    def _reset_error_counters(self):
        """Reset H.264 error counters after successful frame"""
        self.h264_error_count = 0

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
        Open the RTSP camera with H.264 error handling

        Returns:
            bool: True if camera was opened successfully, False otherwise
        """
        try:
            # Get URL with authentication
            url = self._get_url_with_auth()
            
            # Log URL without credentials for security
            safe_url = self._get_safe_url(url)
            logger.info(f"Connecting to RTSP stream: {safe_url}")
            
            # Create capture with specific options for RTSP and H.264 stability
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            
            # Set capture options for H.264 stability
            for i in range(0, len(self.cap_options), 2):
                prop = self.cap_options[i]
                value = self.cap_options[i+1]
                if not self.cap.set(prop, value):
                    logger.debug(f"Failed to set property {prop} to {value}")

            # Wait for the connection to establish
            start_time = time.time()
            while not self.cap.isOpened():
                if time.time() - start_time > self.connection_timeout:
                    logger.error(f"Connection timeout to RTSP stream: {safe_url}")
                    self.cap.release()
                    self.cap = None
                    return False
                time.sleep(0.1)
            
            # Reset error counters on successful connection
            self._reset_error_counters()

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
    
    def _start_grabber(self):
        if self.grabber_thread and self.grabber_thread.is_alive():
            return
        self.grabber_running = True
        backend = self.config.get('backend', 'opencv')
        if backend == 'pyav' and PYAV_AVAILABLE:
            self.grabber_thread = threading.Thread(target=self._grabber_loop_pyav, daemon=True)
        else:
            self.grabber_thread = threading.Thread(target=self._grabber_loop, daemon=True)
        self.grabber_thread.start()

    def _stop_grabber(self):
        self.grabber_running = False
        if self.grabber_thread:
            self.grabber_thread.join(timeout=2)
        self.grabber_thread = None

    def _grabber_loop(self):
        while self.grabber_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue
            # Always keep only the latest frame
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass
            except Exception as e:
                logger.error(f"Grabber thread error: {e}")
                break

    def _grabber_loop_pyav(self):
        url = self._get_url_with_auth()
        options = {
            'rtsp_transport': self.config.get('rtsp_transport', 'tcp'),
            'stimeout': str(int(self.config.get('connection_timeout', 10.0) * 1000000)),
        }
        try:
            container = av.open(url, options=options)
            stream = next(s for s in container.streams if s.type == 'video')
            for frame in container.decode(stream):
                if not self.grabber_running:
                    break
                img = frame.to_ndarray(format='bgr24')
                try:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(img)
                except queue.Full:
                    pass
                except Exception as e:
                    logger.error(f"PyAV grabber thread error: {e}")
                    break
        except Exception as e:
            logger.error(f"PyAV failed to open RTSP stream: {e}")

    def _capture_loop(self):
        retries = 0
        consecutive_failures = 0
        decode_errors = 0
        process_latest_only = self.config.get('process_latest_only', True)
        last_frame_time = time.time()
        watchdog_timeout = self.config.get('watchdog_timeout', 10.0)

        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                if self.max_retries >= 0 and retries >= self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) reached, giving up on RTSP stream")
                    break
                logger.info(f"Trying to open RTSP stream (attempt {retries + 1})")
                if self._open_camera():
                    retries = 0
                    consecutive_failures = 0
                    decode_errors = 0
                    self._start_grabber()
                else:
                    retries += 1
                    time.sleep(self.retry_interval)
                    continue

            try:
                # Watchdog: if no frame for too long, force reconnect
                if time.time() - last_frame_time > watchdog_timeout:
                    logger.warning("No frame received for watchdog timeout, reconnecting RTSP stream")
                    self._close_camera()
                    self._stop_grabber()
                    continue

                # Get the latest frame from the queue
                try:
                    frame = self.frame_queue.get(timeout=1)
                    last_frame_time = time.time()
                except queue.Empty:
                    consecutive_failures += 1
                    time.sleep(0.05)
                    continue

                if not self._validate_frame(frame):
                    consecutive_failures += 1
                    if consecutive_failures > 5:
                        if self._handle_h264_error():
                            time.sleep(0.01)
                            continue
                    if consecutive_failures > 20:
                        logger.warning(f"Received corrupted frames ({consecutive_failures} times), reconnecting")
                        self._close_camera()
                        self._stop_grabber()
                    if self.last_good_frame is not None and self.frame_skip_on_error:
                        with self.frame_lock:
                            processed_frame = self._preprocess_frame(self.last_good_frame)
                            if processed_frame is not None:
                                self._set_frame(processed_frame)
                    time.sleep(0.02)
                    continue

                consecutive_failures = 0
                self._reset_error_counters()
                with self.frame_lock:
                    self.last_good_frame = frame.copy()
                processed_frame = self._preprocess_frame(frame)
                if processed_frame is not None:
                    self._set_frame(processed_frame)
                time.sleep(0.01)

            except Exception as e:
                logger.error(f"Error capturing frame from RTSP stream: {e}")
                consecutive_failures += 1
                if "h264" in str(e).lower() or "decode" in str(e).lower():
                    if not self._handle_h264_error():
                        self._close_camera()
                        self._stop_grabber()
                elif consecutive_failures > 5:
                    self._close_camera()
                    self._stop_grabber()
                time.sleep(0.1)
        self._close_camera()
        self._stop_grabber()

    def _close_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("RTSP stream closed")
        self._stop_grabber()

    def stop(self):
        super().stop()
        self._close_camera()
        self._stop_grabber()
