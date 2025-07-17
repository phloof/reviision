"""
Enhanced PyAV Handler with robust error handling and recovery
"""

import time
import logging
import threading
from typing import Optional, Dict, Any, Callable, List, Tuple
from dataclasses import dataclass
from enum import Enum
import av
from av.error import *

logger = logging.getLogger(__name__)


class PyAVErrorType(Enum):
    """PyAV error categorization"""
    NETWORK = "network"
    CODEC = "codec"
    FORMAT = "format"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    STREAM = "stream"
    UNKNOWN = "unknown"


@dataclass
class PyAVMetrics:
    """PyAV performance and error metrics"""
    total_packets_processed: int = 0
    successful_packets: int = 0
    decode_errors: int = 0
    network_errors: int = 0
    timeout_errors: int = 0
    resource_errors: int = 0
    container_resets: int = 0
    stream_switches: int = 0
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    average_packet_time_ms: float = 0.0
    peak_memory_usage_mb: float = 0.0
    container_uptime_s: float = 0.0


class PyAVEnhancedHandler:
    """
    Enhanced PyAV handler with comprehensive error handling and recovery
    
    Features:
    - Proper container lifecycle management
    - Progressive timeout strategies
    - Memory usage monitoring
    - Intelligent error categorization and recovery
    - Resource cleanup and leak prevention
    - Performance optimization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize enhanced PyAV handler
        
        Args:
            config: Handler configuration
        """
        self.config = config or {}
        
        # Configuration
        self.initial_timeout_s = self.config.get('initial_timeout_s', 5.0)
        self.max_timeout_s = self.config.get('max_timeout_s', 30.0)
        self.timeout_multiplier = self.config.get('timeout_multiplier', 1.5)
        self.max_memory_mb = self.config.get('max_memory_mb', 256)
        self.container_reset_threshold = self.config.get('container_reset_threshold', 10)
        
        # State tracking
        self.current_timeout = self.initial_timeout_s
        self.container: Optional[av.InputContainer] = None
        self.video_stream: Optional[av.VideoStream] = None
        self.handler_lock = threading.RLock()
        self.metrics = PyAVMetrics()
        
        # Error tracking
        self.consecutive_errors = 0
        self.error_history: List[Tuple[float, str, str]] = []  # (timestamp, error_type, description)
        self.last_successful_packet_time = time.time()
        
        # Performance tracking
        self.container_start_time: Optional[float] = None
        self.packet_timing_history: List[float] = []
        
        # Callbacks
        self.error_callback: Optional[Callable] = None
        self.recovery_callback: Optional[Callable] = None
        self.metrics_callback: Optional[Callable] = None
        
        logger.info("PyAVEnhancedHandler initialized")
    
    def set_callbacks(self, error_callback: Callable = None, 
                     recovery_callback: Callable = None,
                     metrics_callback: Callable = None):
        """
        Set handler callbacks
        
        Args:
            error_callback: Called on errors (error_type, description, severity)
            recovery_callback: Called on recovery attempts (recovery_type, success)
            metrics_callback: Called on metrics updates (metrics)
        """
        self.error_callback = error_callback
        self.recovery_callback = recovery_callback
        self.metrics_callback = metrics_callback
    
    def open_container(self, url: str, options: Dict[str, str] = None) -> bool:
        """
        Open PyAV container with enhanced error handling
        
        Args:
            url: Stream URL
            options: PyAV options
            
        Returns:
            bool: True if container opened successfully
        """
        with self.handler_lock:
            try:
                # Close existing container if open
                self._close_container_internal()
                
                # Prepare options with timeout
                final_options = self._prepare_options(options or {})
                
                logger.info(f"Opening PyAV container: {self._safe_url(url)}")
                logger.debug(f"PyAV options: {final_options}")
                
                # Open container with timeout
                start_time = time.time()
                self.container = av.open(url, options=final_options)
                open_time = time.time() - start_time
                
                # Find video stream
                video_streams = [s for s in self.container.streams if s.type == 'video']
                if not video_streams:
                    logger.error("No video streams found in container")
                    self._close_container_internal()
                    return False
                
                self.video_stream = video_streams[0]
                self.container_start_time = time.time()
                
                # Log container information
                self._log_container_info()
                
                # Reset error tracking on successful open
                self.consecutive_errors = 0
                self.current_timeout = self.initial_timeout_s
                
                logger.info(f"PyAV container opened successfully in {open_time:.2f}s")
                return True
                
            except TimeoutError as e:
                self._handle_error("timeout", f"Container open timeout: {e}", "error")
                self._escalate_timeout()
                return False
            except ConnectionResetError as e:
                self._handle_error("network", f"Connection reset: {e}", "error")
                return False
            except HTTPNotFoundError as e:
                self._handle_error("network", f"Stream not found (404): {e}", "error")
                return False
            except HTTPUnauthorizedError as e:
                self._handle_error("network", f"Unauthorized access (401): {e}", "error")
                return False
            except Exception as e:
                error_type = self._categorize_error(e)
                self._handle_error(error_type, f"Container open error: {e}", "error")
                return False
    
    def get_next_packet(self) -> Optional[av.Packet]:
        """
        Get next packet with enhanced error handling
        
        Returns:
            av.Packet or None if error/end of stream
        """
        if not self.container or not self.video_stream:
            return None
        
        try:
            start_time = time.time()
            
            # Get packet from demuxer
            for packet in self.container.demux(self.video_stream):
                process_time = time.time() - start_time
                
                # Update metrics
                with self.handler_lock:
                    self.metrics.total_packets_processed += 1
                    self.metrics.successful_packets += 1
                    self.last_successful_packet_time = time.time()
                    
                    # Update timing metrics
                    self.packet_timing_history.append(process_time * 1000)  # Convert to ms
                    if len(self.packet_timing_history) > 100:
                        self.packet_timing_history.pop(0)
                    
                    self.metrics.average_packet_time_ms = sum(self.packet_timing_history) / len(self.packet_timing_history)
                    
                    # Reset consecutive errors on success
                    self.consecutive_errors = 0
                
                return packet
            
            # End of stream reached
            logger.info("PyAV reached end of stream")
            return None
            
        except InvalidDataError as e:
            self._handle_decode_error(f"Invalid data: {e}")
            return None
        except EOFError as e:
            logger.info("PyAV EOF reached")
            return None
        except TimeoutError as e:
            self._handle_error("timeout", f"Packet timeout: {e}", "warning")
            return None
        except Exception as e:
            error_type = self._categorize_error(e)
            self._handle_error(error_type, f"Packet error: {e}", "warning")
            return None
    
    def decode_packet(self, packet: av.Packet) -> List[av.VideoFrame]:
        """
        Decode packet with error handling
        
        Args:
            packet: Packet to decode
            
        Returns:
            List of decoded frames
        """
        if not packet:
            return []
        
        try:
            frames = packet.decode()
            
            # Update metrics
            with self.handler_lock:
                self.metrics.successful_packets += len(frames)
            
            return frames
            
        except InvalidDataError as e:
            self._handle_decode_error(f"Decode error: {e}")
            return []
        except Exception as e:
            error_type = self._categorize_error(e)
            self._handle_error(error_type, f"Frame decode error: {e}", "warning")
            return []
    
    def close_container(self):
        """Close container with proper cleanup"""
        with self.handler_lock:
            self._close_container_internal()
            logger.info("PyAV container closed")
    
    def _close_container_internal(self):
        """Internal container cleanup"""
        if self.container:
            try:
                self.container.close()
                
                # Update uptime metrics
                if self.container_start_time:
                    uptime = time.time() - self.container_start_time
                    self.metrics.container_uptime_s += uptime
                    
            except Exception as e:
                logger.warning(f"Error closing container: {e}")
            finally:
                self.container = None
                self.video_stream = None
                self.container_start_time = None
    
    def _prepare_options(self, options: Dict[str, str]) -> Dict[str, str]:
        """
        Prepare PyAV options with enhanced settings
        
        Args:
            options: Base options
            
        Returns:
            Enhanced options dictionary
        """
        # Start with provided options
        final_options = options.copy()
        
        # Add timeout
        timeout_microseconds = int(self.current_timeout * 1000000)
        final_options['stimeout'] = str(timeout_microseconds)
        
        # Enhanced stability options
        enhanced_options = {
            'rtsp_flags': 'prefer_tcp',
            'fflags': 'nobuffer+flush_packets+discardcorrupt',
            'flags': 'low_delay',
            'max_delay': '500000',  # 500ms max delay
            'reorder_queue_size': '0',  # Disable reordering for live streams
            'buffer_size': '2097152',  # 2MB buffer for stability
            'analyzeduration': '1000000',  # 1 second analysis
            'probesize': '2097152',  # 2MB probe size
            'reconnect': '1',
            'reconnect_at_eof': '1',
            'reconnect_streamed': '1',
            'reconnect_delay_max': '2',
            'max_muxing_queue_size': '1024',
            'err_detect': 'ignore_err',  # Continue on errors
            'skip_frame': 'default',  # Skip frames if needed
        }
        
        # Add enhanced options (don't override existing)
        for key, value in enhanced_options.items():
            if key not in final_options:
                final_options[key] = value
        
        # Memory management
        final_options['max_alloc'] = str(self.max_memory_mb * 1024 * 1024)
        
        return final_options
    
    def _log_container_info(self):
        """Log container and stream information"""
        if not self.container or not self.video_stream:
            return
        
        try:
            codec_name = self.video_stream.codec_context.name
            width = self.video_stream.codec_context.width
            height = self.video_stream.codec_context.height
            fps = float(self.video_stream.average_rate) if self.video_stream.average_rate else 0
            duration = self.container.duration if self.container.duration else 0
            
            logger.info(f"Container info - Codec: {codec_name}, Resolution: {width}x{height}, FPS: {fps:.1f}")
            if duration > 0:
                duration_s = duration / av.time_base
                logger.info(f"Stream duration: {duration_s:.1f}s")
                
        except Exception as e:
            logger.debug(f"Error logging container info: {e}")
    
    def _handle_error(self, error_type: str, description: str, severity: str):
        """
        Handle errors with categorization and recovery
        
        Args:
            error_type: Type of error
            description: Error description
            severity: Error severity
        """
        current_time = time.time()
        
        with self.handler_lock:
            # Update metrics
            if error_type == "network":
                self.metrics.network_errors += 1
            elif error_type == "codec":
                self.metrics.decode_errors += 1
            elif error_type == "timeout":
                self.metrics.timeout_errors += 1
            elif error_type == "resource":
                self.metrics.resource_errors += 1
            
            # Track consecutive errors
            self.consecutive_errors += 1
            
            # Add to error history
            self.error_history.append((current_time, error_type, description))
            # Keep only recent errors
            if len(self.error_history) > 100:
                self.error_history.pop(0)
        
        # Log error
        if severity == "error":
            logger.error(f"PyAV {error_type} error: {description}")
        elif severity == "warning":
            logger.warning(f"PyAV {error_type} warning: {description}")
        else:
            logger.debug(f"PyAV {error_type} info: {description}")
        
        # Call error callback
        if self.error_callback:
            try:
                self.error_callback(error_type, description, severity)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")
        
        # Check if recovery is needed
        if self.consecutive_errors >= self.container_reset_threshold:
            self._attempt_recovery("Too many consecutive errors")
    
    def _handle_decode_error(self, description: str):
        """Handle decode-specific errors"""
        self._handle_error("codec", description, "warning")
        
        # For decode errors, we might want to seek or skip
        if self.container and self.video_stream:
            try:
                # Try to seek to next keyframe
                self.container.seek(0, stream=self.video_stream, any_frame=False)
                logger.debug("Seeked to keyframe after decode error")
            except Exception as e:
                logger.debug(f"Failed to seek after decode error: {e}")
    
    def _categorize_error(self, error: Exception) -> str:
        """
        Categorize PyAV errors
        
        Args:
            error: Exception to categorize
            
        Returns:
            Error category string
        """
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Network errors
        if any(term in error_type.lower() for term in ['connection', 'http', 'network', 'socket']):
            return "network"
        if any(term in error_message for term in ['connection', 'network', 'unreachable', 'timeout']):
            return "network"
        
        # Codec errors
        if any(term in error_type.lower() for term in ['decode', 'codec', 'format']):
            return "codec"
        if any(term in error_message for term in ['decode', 'codec', 'invalid', 'corrupt']):
            return "codec"
        
        # Timeout errors
        if 'timeout' in error_type.lower() or 'timeout' in error_message:
            return "timeout"
        
        # Resource errors
        if any(term in error_type.lower() for term in ['memory', 'resource', 'allocation']):
            return "resource"
        if any(term in error_message for term in ['memory', 'allocation', 'resource']):
            return "resource"
        
        # Stream errors
        if any(term in error_type.lower() for term in ['stream', 'eof', 'format']):
            return "stream"
        
        return "unknown"
    
    def _escalate_timeout(self):
        """Escalate timeout for next connection attempt"""
        with self.handler_lock:
            old_timeout = self.current_timeout
            self.current_timeout = min(
                self.current_timeout * self.timeout_multiplier,
                self.max_timeout_s
            )
            logger.info(f"Escalated timeout: {old_timeout:.1f}s -> {self.current_timeout:.1f}s")
    
    def _attempt_recovery(self, reason: str):
        """
        Attempt recovery from errors
        
        Args:
            reason: Reason for recovery
        """
        with self.handler_lock:
            self.metrics.recovery_attempts += 1
            
            logger.warning(f"Attempting PyAV recovery: {reason}")
            
            try:
                # Close and reset container
                self._close_container_internal()
                self.metrics.container_resets += 1
                
                # Reset error counters
                self.consecutive_errors = 0
                
                # Call recovery callback
                if self.recovery_callback:
                    success = self.recovery_callback("container_reset", reason)
                    if success:
                        self.metrics.successful_recoveries += 1
                        logger.info("PyAV recovery successful")
                    else:
                        logger.warning("PyAV recovery failed")
                
            except Exception as e:
                logger.error(f"PyAV recovery error: {e}")
    
    def _safe_url(self, url: str) -> str:
        """Get safe URL for logging (remove credentials)"""
        try:
            import urllib.parse
            parsed = urllib.parse.urlparse(url)
            if '@' in parsed.netloc:
                netloc = parsed.netloc.split('@', 1)[1]
                return urllib.parse.urlunparse((
                    parsed.scheme, netloc, parsed.path,
                    parsed.params, parsed.query, parsed.fragment
                ))
            return url
        except:
            return url
    
    def get_metrics(self) -> PyAVMetrics:
        """Get current PyAV metrics"""
        with self.handler_lock:
            return self.metrics
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get error summary and patterns
        
        Returns:
            Dict with error analysis
        """
        with self.handler_lock:
            current_time = time.time()
            recent_errors = [e for e in self.error_history if current_time - e[0] < 300]  # Last 5 minutes
            
            error_types = {}
            for _, error_type, _ in recent_errors:
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            return {
                'total_errors': len(self.error_history),
                'recent_errors': len(recent_errors),
                'error_types': error_types,
                'consecutive_errors': self.consecutive_errors,
                'last_successful_packet': self.last_successful_packet_time,
                'current_timeout': self.current_timeout,
                'container_uptime': time.time() - self.container_start_time if self.container_start_time else 0
            }
    
    def reset_timeout(self):
        """Reset timeout to initial value (call after successful operation)"""
        with self.handler_lock:
            if self.current_timeout > self.initial_timeout_s:
                logger.info(f"Resetting timeout from {self.current_timeout:.1f}s to {self.initial_timeout_s:.1f}s")
                self.current_timeout = self.initial_timeout_s
    
    def is_container_healthy(self) -> bool:
        """
        Check if container is in a healthy state
        
        Returns:
            bool: True if container appears healthy
        """
        with self.handler_lock:
            if not self.container or not self.video_stream:
                return False
            
            # Check recent error rate
            current_time = time.time()
            recent_errors = [e for e in self.error_history if current_time - e[0] < 60]  # Last minute
            
            # Too many recent errors = unhealthy
            if len(recent_errors) > 10:
                return False
            
            # No recent successful packets = unhealthy
            if current_time - self.last_successful_packet_time > 10:
                return False
            
            # Too many consecutive errors = unhealthy
            if self.consecutive_errors > 5:
                return False
            
            return True 