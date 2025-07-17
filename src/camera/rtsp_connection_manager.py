"""
RTSP Connection Manager with robust state management and health monitoring
"""

import time
import logging
import threading
import enum
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ConnectionState(enum.Enum):
    """Connection state enumeration"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECOVERING = "recovering"
    FAILED = "failed"


@dataclass
class ConnectionMetrics:
    """Connection health and performance metrics"""
    connection_attempts: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    last_connection_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    total_uptime: float = 0.0
    total_downtime: float = 0.0
    average_connection_time: float = 0.0
    frames_received: int = 0
    frames_dropped: int = 0
    bytes_received: int = 0
    network_errors: int = 0
    h264_errors: int = 0
    consecutive_failures: int = 0
    last_keepalive_time: Optional[float] = None
    keepalive_failures: int = 0


@dataclass
class ConnectionConfig:
    """Connection configuration parameters"""
    # Timeout settings
    connection_timeout: float = 15.0
    keepalive_interval: float = 10.0
    keepalive_timeout: float = 5.0
    
    # Retry settings
    max_retries: int = 10
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0
    retry_exponential_backoff: bool = True
    
    # Progressive timeout settings
    initial_timeout: float = 5.0
    max_timeout: float = 30.0
    timeout_multiplier: float = 1.5
    
    # Health monitoring
    health_check_interval: float = 5.0
    max_consecutive_failures: int = 5
    frame_timeout: float = 10.0
    
    # Recovery settings
    recovery_frame_count: int = 10
    recovery_timeout: float = 15.0


class RTSPConnectionManager:
    """
    Robust RTSP connection manager with state machine and health monitoring
    
    Implements explicit connection states, progressive timeouts, health monitoring,
    and intelligent recovery strategies for RTSP streaming.
    """
    
    def __init__(self, config: ConnectionConfig = None):
        """
        Initialize connection manager
        
        Args:
            config: Connection configuration parameters
        """
        self.config = config or ConnectionConfig()
        self.state = ConnectionState.DISCONNECTED
        self.metrics = ConnectionMetrics()
        
        # State management
        self.state_lock = threading.RLock()
        self.state_change_callbacks: Dict[ConnectionState, list] = {
            state: [] for state in ConnectionState
        }
        
        # Connection tracking
        self.connection_start_time: Optional[float] = None
        self.current_timeout = self.config.initial_timeout
        self.current_retry_delay = self.config.retry_base_delay
        
        # Health monitoring
        self.last_frame_time: Optional[float] = None
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.health_monitor_running = False
        
        # Connection callbacks
        self.connect_callback: Optional[Callable] = None
        self.disconnect_callback: Optional[Callable] = None
        self.keepalive_callback: Optional[Callable] = None
        
        logger.info("RTSPConnectionManager initialized")
    
    def set_state(self, new_state: ConnectionState, reason: str = ""):
        """
        Change connection state with proper transitions and callbacks
        
        Args:
            new_state: Target connection state
            reason: Reason for state change
        """
        with self.state_lock:
            old_state = self.state
            if old_state == new_state:
                return
            
            # Validate state transitions
            valid_transitions = {
                ConnectionState.DISCONNECTED: [ConnectionState.CONNECTING],
                ConnectionState.CONNECTING: [ConnectionState.CONNECTED, ConnectionState.FAILED, ConnectionState.DISCONNECTED],
                ConnectionState.CONNECTED: [ConnectionState.RECOVERING, ConnectionState.DISCONNECTED, ConnectionState.FAILED],
                ConnectionState.RECOVERING: [ConnectionState.CONNECTED, ConnectionState.FAILED, ConnectionState.DISCONNECTED],
                ConnectionState.FAILED: [ConnectionState.CONNECTING, ConnectionState.DISCONNECTED]
            }
            
            if new_state not in valid_transitions.get(old_state, []):
                logger.warning(f"Invalid state transition from {old_state.value} to {new_state.value}")
                return
            
            self.state = new_state
            
            # Update metrics based on state change
            current_time = time.time()
            
            if new_state == ConnectionState.CONNECTING:
                self.metrics.connection_attempts += 1
                self.connection_start_time = current_time
                
            elif new_state == ConnectionState.CONNECTED:
                self.metrics.successful_connections += 1
                self.metrics.consecutive_failures = 0
                self.metrics.last_connection_time = current_time
                
                if self.connection_start_time:
                    connection_time = current_time - self.connection_start_time
                    if self.metrics.average_connection_time == 0:
                        self.metrics.average_connection_time = connection_time
                    else:
                        self.metrics.average_connection_time = (
                            self.metrics.average_connection_time * 0.8 + connection_time * 0.2
                        )
                
                # Reset timeout on successful connection
                self.current_timeout = self.config.initial_timeout
                self.current_retry_delay = self.config.retry_base_delay
                
            elif new_state in [ConnectionState.FAILED, ConnectionState.DISCONNECTED]:
                self.metrics.failed_connections += 1
                self.metrics.consecutive_failures += 1
                self.metrics.last_failure_time = current_time
                
                # Increase timeouts for next attempt
                if self.config.retry_exponential_backoff:
                    self.current_timeout = min(
                        self.current_timeout * self.config.timeout_multiplier,
                        self.config.max_timeout
                    )
                    self.current_retry_delay = min(
                        self.current_retry_delay * 2,
                        self.config.retry_max_delay
                    )
            
            logger.info(f"Connection state changed: {old_state.value} -> {new_state.value} ({reason})")
            
            # Execute state change callbacks
            for callback in self.state_change_callbacks.get(new_state, []):
                try:
                    callback(old_state, new_state, reason)
                except Exception as e:
                    logger.error(f"Error in state change callback: {e}")
    
    def get_state(self) -> ConnectionState:
        """Get current connection state"""
        with self.state_lock:
            return self.state
    
    def is_connected(self) -> bool:
        """Check if connection is in connected state"""
        return self.get_state() == ConnectionState.CONNECTED
    
    def is_connecting(self) -> bool:
        """Check if connection is in connecting state"""
        return self.get_state() == ConnectionState.CONNECTING
    
    def can_attempt_connection(self) -> bool:
        """Check if connection attempt is allowed"""
        state = self.get_state()
        return state in [ConnectionState.DISCONNECTED, ConnectionState.FAILED]
    
    def should_retry(self) -> bool:
        """Check if retry is allowed based on configuration and metrics"""
        if self.config.max_retries <= 0:
            return True  # Infinite retries
        
        return self.metrics.consecutive_failures < self.config.max_consecutive_failures
    
    def get_retry_delay(self) -> float:
        """Get current retry delay"""
        return self.current_retry_delay
    
    def get_connection_timeout(self) -> float:
        """Get current connection timeout"""
        return self.current_timeout
    
    def register_state_callback(self, state: ConnectionState, callback: Callable):
        """
        Register callback for state changes
        
        Args:
            state: State to monitor for
            callback: Function to call on state change (old_state, new_state, reason)
        """
        self.state_change_callbacks[state].append(callback)
    
    def set_connection_callbacks(self, 
                               connect_callback: Callable = None,
                               disconnect_callback: Callable = None,
                               keepalive_callback: Callable = None):
        """
        Set connection management callbacks
        
        Args:
            connect_callback: Function to call for connecting
            disconnect_callback: Function to call for disconnecting
            keepalive_callback: Function to call for keepalive checks
        """
        self.connect_callback = connect_callback
        self.disconnect_callback = disconnect_callback
        self.keepalive_callback = keepalive_callback
    
    def start_connection(self) -> bool:
        """
        Start connection attempt
        
        Returns:
            bool: True if connection attempt started, False otherwise
        """
        if not self.can_attempt_connection():
            logger.warning(f"Cannot start connection in state {self.get_state().value}")
            return False
        
        if not self.should_retry():
            logger.error(f"Max retries exceeded ({self.metrics.consecutive_failures})")
            return False
        
        self.set_state(ConnectionState.CONNECTING, "Manual connection attempt")
        
        if self.connect_callback:
            try:
                success = self.connect_callback()
                if success:
                    self.set_state(ConnectionState.CONNECTED, "Connection successful")
                    self._start_health_monitoring()
                else:
                    self.set_state(ConnectionState.FAILED, "Connection callback failed")
                return success
            except Exception as e:
                logger.error(f"Connection callback error: {e}")
                self.set_state(ConnectionState.FAILED, f"Connection error: {e}")
                return False
        else:
            logger.warning("No connection callback set")
            self.set_state(ConnectionState.FAILED, "No connection callback")
            return False
    
    def disconnect(self, reason: str = "Manual disconnect"):
        """
        Disconnect and clean up resources
        
        Args:
            reason: Reason for disconnection
        """
        self._stop_health_monitoring()
        
        if self.disconnect_callback:
            try:
                self.disconnect_callback()
            except Exception as e:
                logger.error(f"Disconnect callback error: {e}")
        
        self.set_state(ConnectionState.DISCONNECTED, reason)
    
    def report_frame_received(self, frame_size: int = 0):
        """
        Report successful frame reception
        
        Args:
            frame_size: Size of received frame in bytes
        """
        current_time = time.time()
        self.last_frame_time = current_time
        self.metrics.frames_received += 1
        self.metrics.bytes_received += frame_size
        
        # Update connection state if recovering
        if self.get_state() == ConnectionState.RECOVERING:
            self.set_state(ConnectionState.CONNECTED, "Frame received during recovery")
    
    def report_frame_dropped(self):
        """Report dropped frame"""
        self.metrics.frames_dropped += 1
    
    def report_network_error(self):
        """Report network error"""
        self.metrics.network_errors += 1
        
        # Check if we should enter recovery mode
        if self.get_state() == ConnectionState.CONNECTED:
            self.set_state(ConnectionState.RECOVERING, "Network error detected")
    
    def report_h264_error(self):
        """Report H.264 decoding error"""
        self.metrics.h264_errors += 1
    
    def get_metrics(self) -> ConnectionMetrics:
        """Get current connection metrics"""
        return self.metrics
    
    def get_connection_quality_score(self) -> float:
        """
        Calculate connection quality score (0.0 to 1.0)
        
        Returns:
            float: Quality score based on various metrics
        """
        if self.metrics.connection_attempts == 0:
            return 0.0
        
        # Success rate factor (0-1)
        success_rate = self.metrics.successful_connections / self.metrics.connection_attempts
        
        # Frame rate factor (0-1)
        if self.metrics.frames_received == 0:
            frame_success_rate = 0.0
        else:
            frame_success_rate = self.metrics.frames_received / (
                self.metrics.frames_received + self.metrics.frames_dropped
            )
        
        # Error rate factor (0-1)
        total_errors = self.metrics.network_errors + self.metrics.h264_errors
        total_operations = self.metrics.frames_received + total_errors
        if total_operations == 0:
            error_rate = 0.0
        else:
            error_rate = 1.0 - (total_errors / total_operations)
        
        # Weighted average
        quality_score = (
            success_rate * 0.4 +
            frame_success_rate * 0.4 +
            error_rate * 0.2
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def _start_health_monitoring(self):
        """Start health monitoring thread"""
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            return
        
        self.health_monitor_running = True
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name="RTSPHealthMonitor"
        )
        self.health_monitor_thread.start()
        logger.debug("Health monitoring started")
    
    def _stop_health_monitoring(self):
        """Stop health monitoring thread"""
        self.health_monitor_running = False
        if self.health_monitor_thread:
            self.health_monitor_thread.join(timeout=2)
        logger.debug("Health monitoring stopped")
    
    def _health_monitor_loop(self):
        """Health monitoring loop"""
        while self.health_monitor_running:
            try:
                current_time = time.time()
                
                # Check frame timeout
                if (self.last_frame_time and 
                    current_time - self.last_frame_time > self.config.frame_timeout):
                    
                    if self.get_state() == ConnectionState.CONNECTED:
                        logger.warning(f"No frames received for {self.config.frame_timeout}s")
                        self.set_state(ConnectionState.RECOVERING, "Frame timeout")
                
                # Perform keepalive check
                if (not self.metrics.last_keepalive_time or
                    current_time - self.metrics.last_keepalive_time > self.config.keepalive_interval):
                    
                    self._perform_keepalive()
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    def _perform_keepalive(self):
        """Perform keepalive check"""
        if not self.keepalive_callback:
            return
        
        try:
            self.metrics.last_keepalive_time = time.time()
            success = self.keepalive_callback()
            
            if not success:
                self.metrics.keepalive_failures += 1
                logger.warning(f"Keepalive failed ({self.metrics.keepalive_failures} times)")
                
                if self.metrics.keepalive_failures >= 3:
                    self.set_state(ConnectionState.RECOVERING, "Multiple keepalive failures")
            else:
                self.metrics.keepalive_failures = 0
                
        except Exception as e:
            logger.error(f"Keepalive callback error: {e}")
            self.metrics.keepalive_failures += 1 