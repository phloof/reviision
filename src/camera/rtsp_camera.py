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

# Check if enhanced components are available
try:
    from .rtsp_connection_manager import RTSPConnectionManager
    from .h264_validator import H264StreamValidator
    from .adaptive_buffer_manager import AdaptiveBufferManager
    from .stream_health_monitor import StreamHealthMonitor
    from .network_optimizer import NetworkOptimizer
    from .pyav_enhanced_handler import PyAVEnhancedHandler
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    ENHANCED_COMPONENTS_AVAILABLE = False

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
        
        # Force resolution settings
        self.force_resolution = config.get('force_resolution', True)
        self.target_resolution = tuple(config.get('resolution', [1280, 720]))
        self.fast_decode = config.get('fast_decode', True)
        self.drop_frames = config.get('drop_frames', True)
        
        # Stream URL alternatives for different resolutions
        self.stream_urls = self._generate_stream_urls()
        self.current_stream_index = 0
        
        # H.264 error handling settings
        self.h264_error_threshold = config.get('h264_error_threshold', 20)  # Max consecutive H.264 errors
        self.frame_skip_on_error = config.get('frame_skip_on_error', True)  # Skip corrupted frames
        self.recovery_frame_count = config.get('recovery_frame_count', 5)  # Frames to read for recovery

        # Frame validation settings
        self.min_frame_size = config.get('min_frame_size', 100)  # Minimum frame size in pixels
        self.max_decode_errors = config.get('max_decode_errors', 50)  # Max decode errors before reconnect

        # Enhanced features configuration (default: enabled to fix H.264 errors)
        self.enable_enhanced_features = config.get('enable_enhanced_features', True) and ENHANCED_COMPONENTS_AVAILABLE
        
        # Initialize enhanced components if available and enabled
        if self.enable_enhanced_features:
            self._init_enhanced_components()
            logger.info("Enhanced H.264 error recovery enabled")
        else:
            logger.info("Using legacy RTSP camera mode")

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
    
    def _init_enhanced_components(self):
        """Initialize enhanced H.264 error recovery components"""
        try:
            # Connection Manager
            from .rtsp_connection_manager import RTSPConnectionManager, ConnectionConfig
            from .h264_validator import H264StreamValidator
            from .adaptive_buffer_manager import AdaptiveBufferManager
            from .stream_health_monitor import StreamHealthMonitor
            from .network_optimizer import NetworkOptimizer
            from .pyav_enhanced_handler import PyAVEnhancedHandler
            
            conn_config = ConnectionConfig(
                connection_timeout=self.connection_timeout,
                keepalive_interval=10.0,
                max_retries=self.max_retries,
                retry_base_delay=2.0,
                health_check_interval=5.0,
                frame_timeout=self.connection_timeout
            )
            
            self.connection_manager = RTSPConnectionManager(conn_config)
            self.h264_validator = H264StreamValidator()
            self.buffer_manager = AdaptiveBufferManager()
            self.health_monitor = StreamHealthMonitor()
            self.network_optimizer = NetworkOptimizer()
            self.pyav_handler = PyAVEnhancedHandler(self.url)
            
            logger.info("Enhanced components initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize enhanced components: {e}")
            self.enable_enhanced_features = False
    
    def _generate_stream_urls(self):
        """
        Generate alternative RTSP stream URLs to try different resolution streams
        
        Returns:
            list: List of alternative stream URLs to try
        """
        base_url = self.url or ""
        width, height = self.target_resolution
        fps = self.config.get('fps', 15)
        
        # Common RTSP stream endpoint variations for different cameras
        stream_variations = [
            # Main stream with resolution parameters
            f"?width={width}&height={height}&fps={fps}",
            # Sub-stream endpoints (typically lower resolution)
            "/stream2",
            "/h264Preview_01_sub", 
            "/live1.sdp",
            "/live2.sdp",
            "/videoMain",
            "/videoSub",
            "/cam/realmonitor?channel=1&subtype=1",  # Dahua sub-stream
            "/axis-media/media.amp?resolution={width}x{height}",  # Axis
            # Original URL without parameters
            ""
        ]
        
        urls = []
        for variation in stream_variations:
            if base_url:
                if variation.startswith('?'):
                    # Add parameters to existing URL
                    if '?' in base_url:
                        url = base_url + '&' + variation[1:]
                    else:
                        url = base_url + variation
                elif variation.startswith('/'):
                    # Replace path
                    parsed = urllib.parse.urlparse(base_url)
                    url = urllib.parse.urlunparse((
                        parsed.scheme, parsed.netloc, variation,
                        parsed.params, parsed.query, parsed.fragment
                    ))
                else:
                    # Original URL
                    url = base_url
                urls.append(url)
        
        return urls if urls else [base_url]
    
    def _force_frame_resize(self, frame):
        """
        Force frame to target resolution regardless of source resolution
        
        Args:
            frame: Input frame of any resolution
            
        Returns:
            np.ndarray: Frame resized to target resolution
        """
        if frame is None:
            return None
            
        if not self.force_resolution:
            return frame
            
        current_height, current_width = frame.shape[:2]
        target_width, target_height = self.target_resolution
        
        # If frame is already at target resolution, return as-is
        if current_width == target_width and current_height == target_height:
            return frame
            
        # Log resolution mismatch only occasionally to avoid spam
        if hasattr(self, '_last_logged_resolution'):
            if self._last_logged_resolution != (current_width, current_height):
                logger.warning(f"Camera streaming at {current_width}x{current_height}, forcing resize to {target_width}x{target_height}")
                self._last_logged_resolution = (current_width, current_height)
        else:
            logger.warning(f"Camera streaming at {current_width}x{current_height}, forcing resize to {target_width}x{target_height}")
            self._last_logged_resolution = (current_width, current_height)
        
        # Force resize to target resolution
        try:
            if self.fast_decode:
                # Use faster interpolation for real-time processing
                resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            else:
                # Use higher quality interpolation
                resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            
            return resized_frame
        except Exception as e:
            logger.error(f"Error resizing frame: {e}")
            return frame
    
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
                self.cap.set(prop, value)

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
        """Enhanced PyAV grabber loop with robust error handling and optimization"""
        url = self._get_url_with_auth()
        
        # Try to force resolution by modifying URL
        target_width, target_height = self.target_resolution
        if '?' in url:
            url = f"{url}&width={target_width}&height={target_height}&fps=15"
        else:
            url = f"{url}?width={target_width}&height={target_height}&fps=15"
        
        logger.info(f"PyAV trying URL with resolution parameters: {self._get_safe_url(url)}")
        
        # Build comprehensive options for PyAV - optimized for stability
        options = {
            'rtsp_transport': self.config.get('rtsp_transport', 'tcp'),
            'stimeout': str(int(self.config.get('connection_timeout', 15.0) * 1000000)),
        }
        
        # Add PyAV-specific options from config
        pyav_options = self.config.get('pyav_options', {})
        options.update(pyav_options)
        
        # Enhanced stability options for better connection handling
        options.update({
            'rtsp_flags': 'prefer_tcp',
            'fflags': 'nobuffer+flush_packets',
            'flags': 'low_delay',
            'max_delay': '500000',  # Increased to 500ms max delay for stability
            'reorder_queue_size': '0',  # Disable reordering for live streams
            'buffer_size': '1048576',  # Increased to 1MB buffer for stability
            'analyzeduration': '2000000',  # 2 seconds analysis for better stream detection
            'probesize': '1048576',  # 1MB probe size for better format detection
            'reconnect': '1',  # Enable automatic reconnection
            'reconnect_at_eof': '1',  # Reconnect on end of file
            'reconnect_streamed': '1',  # Reconnect for streamed content
            'reconnect_delay_max': '2',  # Maximum 2 second delay between reconnects
        })
        
        container = None
        stream = None
        consecutive_errors = 0
        max_consecutive_errors = 10
        frame_count = 0  # ensure defined for finally block
        
        try:
            logger.info(f"PyAV opening RTSP stream: {self._get_safe_url(url)}")
            logger.debug(f"PyAV options: {options}")
            
            container = av.open(url, options=options)
            
            # Find video stream
            video_streams = [s for s in container.streams if s.type == 'video']
            if not video_streams:
                logger.error("No video streams found in RTSP source")
                return
                
            stream = video_streams[0]
            
            # Log stream properties
            logger.info(f"PyAV stream opened - Codec: {stream.codec_context.name}, "
                       f"Size: {stream.codec_context.width}x{stream.codec_context.height}, "
                       f"FPS: {stream.average_rate}")
            
            frame_count = 0
            last_log_time = time.time()
            
            # Main frame processing loop
            for packet in container.demux(stream):
                if not self.grabber_running:
                    logger.info("PyAV grabber stopping...")
                    break
                    
                try:
                    # Decode packet into frames
                    frames = packet.decode()
                    
                    for frame in frames:
                        if not self.grabber_running:
                            break
                            
                        # Convert frame to numpy array
                        img = frame.to_ndarray(format='bgr24')
                        
                        # Apply forced resizing if needed
                        resized_img = self._force_frame_resize(img)
                        final_img = resized_img if resized_img is not None else img
                        
                        # Queue the frame
                        try:
                            # Keep only the latest frame
                            if self.frame_queue.full():
                                try:
                                    self.frame_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            
                            self.frame_queue.put_nowait(final_img)
                            consecutive_errors = 0  # Reset error counter on success
                            frame_count += 1
                            
                            # Log progress periodically
                            current_time = time.time()
                            if current_time - last_log_time > 30:  # Every 30 seconds
                                logger.debug(f"PyAV processed {frame_count} frames, queue size: {self.frame_queue.qsize()}")
                                last_log_time = current_time
                                
                        except queue.Full:
                            # Queue full is normal for live streams
                            pass
                        except Exception as e:
                            logger.warning(f"Error queuing PyAV frame: {e}")
                            consecutive_errors += 1
                            
                except av.error.InvalidDataError as e:
                    logger.debug(f"PyAV invalid data (skipping): {e}")
                    consecutive_errors += 1
                except av.error.EOFError:
                    logger.warning("PyAV reached end of stream")
                    break
                except Exception as e:
                    logger.warning(f"PyAV decode error: {e}")
                    consecutive_errors += 1
                
                # Break if too many consecutive errors
                if consecutive_errors > max_consecutive_errors:
                    logger.error(f"Too many consecutive PyAV errors ({consecutive_errors}), restarting after back-off")
                    time.sleep(5.0)  # back-off before reconnect
                    break
                    
        except av.error.HTTPNotFoundError:
            logger.error(f"PyAV HTTP 404: Stream not found - {self._get_safe_url(url)}")
        except av.error.HTTPUnauthorizedError:
            logger.error(f"PyAV HTTP 401: Unauthorized access - check credentials")
        except av.error.ConnectionResetError as e:
            logger.error(f"PyAV connection reset: {e}")
        except av.error.TimeoutError:
            logger.error(f"PyAV timeout connecting to stream")
        except Exception as e:
            logger.error(f"PyAV unexpected error: {e}")
        finally:
            # Clean up resources
            if container:
                try:
                    container.close()
                except:
                    pass
            try:
                logger.info(f"PyAV grabber loop ended, processed {frame_count} frames")
            except NameError:
                logger.info("PyAV grabber loop ended before frame processing started")

    def _capture_loop(self):
        """Main capture loop - delegates to enhanced or legacy mode"""
        if self.enable_enhanced_features:
            self._enhanced_capture_loop()
        else:
            self._legacy_capture_loop()
    
    def _enhanced_capture_loop(self):
        """Enhanced capture loop with H.264 error recovery"""
        while self.is_running:
            try:
                # Check connection state
                if not self.connection_manager.is_connected() and not self.connection_manager.is_connecting():
                    if self.connection_manager.can_attempt_connection():
                        if not self.connection_manager.start_connection():
                            retry_delay = self.connection_manager.get_retry_delay()
                            time.sleep(retry_delay)
                            continue
                    else:
                        time.sleep(1.0)
                        continue

                # Process frames if connected
                if self.connection_manager.is_connected():
                    success = self._process_next_frame()
                    if not success:
                        self._handle_frame_processing_error()

                time.sleep(0.001)

            except Exception as e:
                logger.error(f"Enhanced capture loop error: {e}")
                self._handle_capture_error(e)
                time.sleep(0.1)
    
    def _legacy_capture_loop(self):
        """Legacy capture loop for backward compatibility"""
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
                    # Reset watchdog timer when connection is successfully (re)opened
                    last_frame_time = time.time()
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
                
                # Force frame resize if needed
                resized_frame = self._force_frame_resize(frame)
                
                with self.frame_lock:
                    self.last_good_frame = resized_frame.copy() if resized_frame is not None else frame.copy()
                
                processed_frame = self._preprocess_frame(resized_frame if resized_frame is not None else frame)
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

    def _process_next_frame(self):
        """Process next frame with enhanced validation"""
        try:
            # Get latest frame from grabber queue
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                return False
            
            if frame is None:
                return False
            
            # Validate frame with H.264 validator if available
            if hasattr(self, 'h264_validator') and self.h264_validator:
                if not self.h264_validator.validate_frame_data(frame):
                    return False
            
            # Update frame buffer
            with self.frame_lock:
                self.frame_buffer = frame
                self.last_good_frame = frame.copy()
            
            return True
            
        except Exception as e:
            return False
    
    def _handle_frame_processing_error(self):
        """Handle frame processing errors"""
        self.h264_error_count += 1
        if self.h264_error_count > self.h264_error_threshold:
            logger.warning("Too many frame processing errors, attempting recovery")
            if hasattr(self, 'connection_manager'):
                self.connection_manager.trigger_recovery("Frame processing errors")
            self.h264_error_count = 0
    
    def _handle_capture_error(self, error):
        """Handle capture loop errors"""
        logger.error(f"Capture error: {error}")
        if hasattr(self, 'connection_manager'):
            self.connection_manager.trigger_recovery(f"Capture error: {error}")

    def stop(self):
        # Stop enhanced components if enabled
        if self.enable_enhanced_features and hasattr(self, 'connection_manager'):
            try:
                if hasattr(self, 'health_monitor') and self.health_monitor:
                    self.health_monitor.stop_monitoring()
                if hasattr(self, 'buffer_manager') and self.buffer_manager:
                    self.buffer_manager.stop()
                if hasattr(self, 'connection_manager') and self.connection_manager:
                    self.connection_manager.disconnect("Camera stopping")
            except Exception as e:
                logger.warning(f"Error stopping enhanced components: {e}")
        
        super().stop()
        self._close_camera()
        self._stop_grabber()
