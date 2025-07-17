"""
Stream Health Monitor for comprehensive RTSP stream quality assessment
"""

import time
import logging
import threading
import math
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class StreamHealth(Enum):
    """Stream health status levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"
    OFFLINE = "offline"


@dataclass
class QualityMetrics:
    """Comprehensive stream quality metrics"""
    # Bitrate metrics
    current_bitrate_bps: float = 0.0
    average_bitrate_bps: float = 0.0
    peak_bitrate_bps: float = 0.0
    bitrate_stability: float = 0.0  # 0.0 = unstable, 1.0 = stable
    
    # Frame metrics
    total_frames: int = 0
    keyframes: int = 0
    dropped_frames: int = 0
    corrupted_frames: int = 0
    frame_rate_fps: float = 0.0
    target_fps: float = 0.0
    frame_rate_stability: float = 0.0
    
    # Quality metrics
    average_frame_quality: float = 0.0  # 0.0 = poor, 1.0 = excellent
    resolution_consistency: float = 0.0
    color_depth_consistency: float = 0.0
    
    # Network metrics
    network_latency_ms: float = 0.0
    jitter_ms: float = 0.0
    packet_loss_rate: float = 0.0
    connection_stability: float = 0.0
    
    # Error metrics
    h264_errors: int = 0
    network_errors: int = 0
    decoder_errors: int = 0
    timeout_errors: int = 0
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    
    # Timing metrics
    keyframe_interval_s: float = 0.0
    last_keyframe_time: Optional[float] = None
    stream_uptime_s: float = 0.0
    total_downtime_s: float = 0.0
    
    # Overall scores (0.0 to 1.0)
    overall_quality_score: float = 0.0
    reliability_score: float = 0.0
    performance_score: float = 0.0


@dataclass
class FrameQualityInfo:
    """Information about frame quality"""
    timestamp: float
    size: int
    is_keyframe: bool
    corruption_score: float
    processing_time_ms: float
    resolution: Tuple[int, int]
    quality_score: float = 0.0  # Calculated based on various factors


@dataclass
class NetworkEvent:
    """Network event information"""
    timestamp: float
    event_type: str  # 'connection', 'disconnection', 'error', 'recovery'
    description: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    duration_ms: Optional[float] = None


class StreamHealthMonitor:
    """
    Comprehensive stream health monitor with real-time quality assessment
    
    Features:
    - Real-time bitrate and frame rate monitoring
    - Frame quality analysis and scoring
    - Network performance tracking
    - Error pattern detection and analysis
    - Predictive health scoring
    - Alert generation for quality issues
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize stream health monitor
        
        Args:
            config: Monitor configuration parameters
        """
        self.config = config or {}
        
        # Configuration
        self.target_fps = self.config.get('target_fps', 15.0)
        self.target_bitrate = self.config.get('target_bitrate_bps', 1000000)  # 1 Mbps
        self.quality_window_size = self.config.get('quality_window_size', 100)
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'low_quality': 0.3,
            'high_error_rate': 0.1,
            'poor_stability': 0.5,
            'connection_issues': 0.8
        })
        
        # Monitoring state
        self.metrics = QualityMetrics()
        self.metrics.target_fps = self.target_fps
        self.monitoring_lock = threading.RLock()
        self.monitoring_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Data collection
        self.frame_history: deque = deque(maxlen=self.quality_window_size)
        self.bitrate_history: deque = deque(maxlen=50)
        self.latency_history: deque = deque(maxlen=30)
        self.error_history: deque = deque(maxlen=200)
        self.network_events: deque = deque(maxlen=100)
        
        # Timing tracking
        self.stream_start_time: Optional[float] = None
        self.last_frame_time: Optional[float] = None
        self.last_keyframe_time: Optional[float] = None
        self.last_quality_update: float = time.time()
        
        # Statistics calculation
        self.fps_calculator = deque(maxlen=30)  # Last 30 frame timestamps
        self.bytes_received_this_second = 0
        self.last_bitrate_calculation = time.time()
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        self.quality_change_callbacks: List[Callable] = []
        
        logger.info("StreamHealthMonitor initialized")
    
    def start_monitoring(self):
        """Start health monitoring"""
        if self.running:
            return
        
        self.running = True
        self.stream_start_time = time.time()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="StreamHealthMonitor"
        )
        self.monitoring_thread.start()
        logger.info("Stream health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        logger.info("Stream health monitoring stopped")
    
    def report_frame(self, frame_data: bytes, is_keyframe: bool = False, 
                    corruption_score: float = 0.0, processing_time_ms: float = 0.0,
                    resolution: Tuple[int, int] = None):
        """
        Report a received frame for quality analysis
        
        Args:
            frame_data: Frame binary data
            is_keyframe: Whether this is a keyframe
            corruption_score: Corruption probability (0.0-1.0)
            processing_time_ms: Time taken to process frame
            resolution: Frame resolution (width, height)
        """
        current_time = time.time()
        
        with self.monitoring_lock:
            # Create frame quality info
            frame_info = FrameQualityInfo(
                timestamp=current_time,
                size=len(frame_data) if frame_data else 0,
                is_keyframe=is_keyframe,
                corruption_score=corruption_score,
                processing_time_ms=processing_time_ms,
                resolution=resolution or (0, 0)
            )
            
            # Calculate frame quality score
            frame_info.quality_score = self._calculate_frame_quality(frame_info)
            
            # Add to history
            self.frame_history.append(frame_info)
            
            # Update frame metrics
            self.metrics.total_frames += 1
            if is_keyframe:
                self.metrics.keyframes += 1
                self._update_keyframe_metrics(current_time)
            
            if corruption_score > 0.5:
                self.metrics.corrupted_frames += 1
            
            # Update FPS calculation
            self.fps_calculator.append(current_time)
            self.last_frame_time = current_time
            
            # Update bitrate calculation
            if frame_data:
                self.bytes_received_this_second += len(frame_data)
            
            # Update quality metrics
            self._update_quality_metrics()
    
    def report_network_event(self, event_type: str, description: str, 
                           severity: str = "info", duration_ms: float = None):
        """
        Report a network event
        
        Args:
            event_type: Type of event ('connection', 'disconnection', 'error', 'recovery')
            description: Event description
            severity: Event severity ('info', 'warning', 'error', 'critical')
            duration_ms: Event duration in milliseconds
        """
        current_time = time.time()
        
        event = NetworkEvent(
            timestamp=current_time,
            event_type=event_type,
            description=description,
            severity=severity,
            duration_ms=duration_ms
        )
        
        with self.monitoring_lock:
            self.network_events.append(event)
            
            # Update error counters
            if event_type == 'error':
                if 'h264' in description.lower() or 'decode' in description.lower():
                    self.metrics.h264_errors += 1
                elif 'network' in description.lower() or 'connection' in description.lower():
                    self.metrics.network_errors += 1
                elif 'timeout' in description.lower():
                    self.metrics.timeout_errors += 1
                else:
                    self.metrics.decoder_errors += 1
            elif event_type == 'recovery':
                if severity in ['info', 'warning']:
                    self.metrics.successful_recoveries += 1
                self.metrics.recovery_attempts += 1
        
        # Trigger alerts for critical events
        if severity == 'critical':
            self._trigger_alert(f"Critical network event: {description}")
    
    def report_network_latency(self, latency_ms: float, jitter_ms: float = None):
        """
        Report network latency measurement
        
        Args:
            latency_ms: Network latency in milliseconds
            jitter_ms: Network jitter in milliseconds
        """
        with self.monitoring_lock:
            self.latency_history.append(latency_ms)
            self.metrics.network_latency_ms = latency_ms
            
            if jitter_ms is not None:
                self.metrics.jitter_ms = jitter_ms
    
    def get_current_health_status(self) -> StreamHealth:
        """
        Get current stream health status
        
        Returns:
            StreamHealth enum value
        """
        with self.monitoring_lock:
            score = self.metrics.overall_quality_score
            
            if score >= 0.9:
                return StreamHealth.EXCELLENT
            elif score >= 0.7:
                return StreamHealth.GOOD
            elif score >= 0.5:
                return StreamHealth.FAIR
            elif score >= 0.3:
                return StreamHealth.POOR
            elif score > 0.0:
                return StreamHealth.CRITICAL
            else:
                return StreamHealth.OFFLINE
    
    def get_quality_trends(self, window_minutes: int = 5) -> Dict[str, List[float]]:
        """
        Get quality trends over specified time window
        
        Args:
            window_minutes: Time window in minutes
            
        Returns:
            Dict with trend data
        """
        window_seconds = window_minutes * 60
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        with self.monitoring_lock:
            # Filter recent data
            recent_frames = [f for f in self.frame_history if f.timestamp >= cutoff_time]
            recent_bitrates = [b for b in self.bitrate_history if b[0] >= cutoff_time]
            recent_events = [e for e in self.network_events if e.timestamp >= cutoff_time]
            
            # Calculate trends
            quality_trend = [f.quality_score for f in recent_frames]
            bitrate_trend = [b[1] for b in recent_bitrates]  # (timestamp, bitrate)
            error_trend = [1 if e.severity in ['error', 'critical'] else 0 for e in recent_events]
            
            return {
                'quality_scores': quality_trend,
                'bitrates_bps': bitrate_trend,
                'error_events': error_trend,
                'timestamps': [f.timestamp for f in recent_frames]
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary
        
        Returns:
            Dict with performance data
        """
        with self.monitoring_lock:
            current_time = time.time()
            uptime = current_time - self.stream_start_time if self.stream_start_time else 0
            
            # Calculate rates
            error_rate = 0.0
            if self.metrics.total_frames > 0:
                error_rate = (self.metrics.corrupted_frames + self.metrics.dropped_frames) / self.metrics.total_frames
            
            recovery_rate = 0.0
            if self.metrics.recovery_attempts > 0:
                recovery_rate = self.metrics.successful_recoveries / self.metrics.recovery_attempts
            
            return {
                'health_status': self.get_current_health_status().value,
                'overall_scores': {
                    'quality': self.metrics.overall_quality_score,
                    'reliability': self.metrics.reliability_score,
                    'performance': self.metrics.performance_score
                },
                'frame_stats': {
                    'total_frames': self.metrics.total_frames,
                    'keyframes': self.metrics.keyframes,
                    'dropped_frames': self.metrics.dropped_frames,
                    'corrupted_frames': self.metrics.corrupted_frames,
                    'current_fps': self.metrics.frame_rate_fps,
                    'target_fps': self.metrics.target_fps
                },
                'bitrate_stats': {
                    'current_bps': self.metrics.current_bitrate_bps,
                    'average_bps': self.metrics.average_bitrate_bps,
                    'peak_bps': self.metrics.peak_bitrate_bps,
                    'stability': self.metrics.bitrate_stability
                },
                'network_stats': {
                    'latency_ms': self.metrics.network_latency_ms,
                    'jitter_ms': self.metrics.jitter_ms,
                    'packet_loss_rate': self.metrics.packet_loss_rate,
                    'connection_stability': self.metrics.connection_stability
                },
                'error_stats': {
                    'h264_errors': self.metrics.h264_errors,
                    'network_errors': self.metrics.network_errors,
                    'error_rate': error_rate,
                    'recovery_rate': recovery_rate
                },
                'timing_stats': {
                    'uptime_s': uptime,
                    'downtime_s': self.metrics.total_downtime_s,
                    'keyframe_interval_s': self.metrics.keyframe_interval_s
                }
            }
    
    def add_alert_callback(self, callback: Callable[[str, str], None]):
        """
        Add callback for quality alerts
        
        Args:
            callback: Function to call with (alert_type, message)
        """
        self.alert_callbacks.append(callback)
    
    def add_quality_change_callback(self, callback: Callable[[StreamHealth, float], None]):
        """
        Add callback for quality changes
        
        Args:
            callback: Function to call with (health_status, quality_score)
        """
        self.quality_change_callbacks.append(callback)
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        last_update_time = time.time()
        update_interval = 1.0  # Update every second
        
        while self.running:
            try:
                current_time = time.time()
                
                if current_time - last_update_time >= update_interval:
                    self._update_periodic_metrics()
                    self._check_alert_conditions()
                    last_update_time = current_time
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
    
    def _update_keyframe_metrics(self, current_time: float):
        """Update keyframe-related metrics"""
        if self.last_keyframe_time:
            interval = current_time - self.last_keyframe_time
            if self.metrics.keyframe_interval_s == 0:
                self.metrics.keyframe_interval_s = interval
            else:
                # Exponential moving average
                self.metrics.keyframe_interval_s = (
                    self.metrics.keyframe_interval_s * 0.8 + interval * 0.2
                )
        
        self.last_keyframe_time = current_time
        self.metrics.last_keyframe_time = current_time
    
    def _calculate_frame_quality(self, frame_info: FrameQualityInfo) -> float:
        """
        Calculate quality score for a frame
        
        Args:
            frame_info: Frame information
            
        Returns:
            float: Quality score (0.0-1.0)
        """
        quality_score = 1.0
        
        # Corruption penalty
        quality_score -= frame_info.corruption_score * 0.5
        
        # Size-based quality (very small frames are likely corrupted)
        if frame_info.size > 0:
            if frame_info.is_keyframe:
                # Keyframes should be reasonably large
                if frame_info.size < 1000:  # Less than 1KB is suspicious
                    quality_score -= 0.3
            else:
                # P/B frames can be smaller
                if frame_info.size < 100:  # Less than 100 bytes is suspicious
                    quality_score -= 0.3
        else:
            quality_score = 0.0  # No data = no quality
        
        # Processing time penalty (very long processing suggests problems)
        if frame_info.processing_time_ms > 100:  # More than 100ms is slow
            quality_score -= min(0.2, frame_info.processing_time_ms / 1000)
        
        # Resolution consistency (if we track resolution)
        if frame_info.resolution != (0, 0):
            # This would need baseline resolution to compare against
            pass
        
        return max(0.0, min(1.0, quality_score))
    
    def _update_quality_metrics(self):
        """Update quality metrics based on recent data"""
        if not self.frame_history:
            return
        
        recent_frames = list(self.frame_history)
        
        # Update average frame quality
        quality_scores = [f.quality_score for f in recent_frames]
        if quality_scores:
            self.metrics.average_frame_quality = np.mean(quality_scores)
        
        # Update frame rate
        if len(self.fps_calculator) >= 2:
            time_span = self.fps_calculator[-1] - self.fps_calculator[0]
            if time_span > 0:
                fps = (len(self.fps_calculator) - 1) / time_span
                self.metrics.frame_rate_fps = fps
                
                # Calculate frame rate stability
                target_fps = self.metrics.target_fps
                if target_fps > 0:
                    fps_ratio = min(fps / target_fps, 1.0)
                    self.metrics.frame_rate_stability = fps_ratio
    
    def _update_periodic_metrics(self):
        """Update metrics that require periodic calculation"""
        current_time = time.time()
        
        with self.monitoring_lock:
            # Update bitrate
            time_since_last = current_time - self.last_bitrate_calculation
            if time_since_last >= 1.0:  # At least 1 second
                current_bitrate = (self.bytes_received_this_second * 8) / time_since_last
                self.metrics.current_bitrate_bps = current_bitrate
                
                # Add to history
                self.bitrate_history.append((current_time, current_bitrate))
                
                # Update average bitrate
                if self.metrics.average_bitrate_bps == 0:
                    self.metrics.average_bitrate_bps = current_bitrate
                else:
                    self.metrics.average_bitrate_bps = (
                        self.metrics.average_bitrate_bps * 0.9 + current_bitrate * 0.1
                    )
                
                # Update peak bitrate
                if current_bitrate > self.metrics.peak_bitrate_bps:
                    self.metrics.peak_bitrate_bps = current_bitrate
                
                # Calculate bitrate stability
                if len(self.bitrate_history) >= 10:
                    recent_bitrates = [b[1] for b in list(self.bitrate_history)[-10:]]
                    if recent_bitrates:
                        mean_bitrate = np.mean(recent_bitrates)
                        std_bitrate = np.std(recent_bitrates)
                        if mean_bitrate > 0:
                            stability = 1.0 - min(1.0, std_bitrate / mean_bitrate)
                            self.metrics.bitrate_stability = stability
                
                # Reset counter
                self.bytes_received_this_second = 0
                self.last_bitrate_calculation = current_time
            
            # Update uptime
            if self.stream_start_time:
                self.metrics.stream_uptime_s = current_time - self.stream_start_time
            
            # Calculate overall scores
            self._calculate_overall_scores()
    
    def _calculate_overall_scores(self):
        """Calculate overall quality, reliability, and performance scores"""
        # Quality score based on frame quality and bitrate stability
        quality_factors = [
            self.metrics.average_frame_quality * 0.4,
            self.metrics.frame_rate_stability * 0.3,
            self.metrics.bitrate_stability * 0.3
        ]
        self.metrics.overall_quality_score = np.mean([f for f in quality_factors if f > 0])
        
        # Reliability score based on errors and recoveries
        if self.metrics.total_frames > 0:
            error_rate = (self.metrics.corrupted_frames + self.metrics.dropped_frames) / self.metrics.total_frames
            reliability = max(0.0, 1.0 - error_rate * 2.0)  # Errors heavily penalize reliability
        else:
            reliability = 0.0
        
        recovery_bonus = 0.0
        if self.metrics.recovery_attempts > 0:
            recovery_rate = self.metrics.successful_recoveries / self.metrics.recovery_attempts
            recovery_bonus = recovery_rate * 0.2  # Up to 20% bonus for good recovery
        
        self.metrics.reliability_score = min(1.0, reliability + recovery_bonus)
        
        # Performance score based on network metrics
        network_factors = []
        if self.metrics.network_latency_ms > 0:
            # Good latency < 50ms, poor > 200ms
            latency_score = max(0.0, 1.0 - (self.metrics.network_latency_ms - 50) / 150)
            network_factors.append(latency_score)
        
        if self.metrics.jitter_ms > 0:
            # Good jitter < 10ms, poor > 50ms
            jitter_score = max(0.0, 1.0 - (self.metrics.jitter_ms - 10) / 40)
            network_factors.append(jitter_score)
        
        packet_loss_score = max(0.0, 1.0 - self.metrics.packet_loss_rate * 10)
        network_factors.append(packet_loss_score)
        
        if network_factors:
            self.metrics.performance_score = np.mean(network_factors)
        else:
            self.metrics.performance_score = 0.5  # Neutral if no network data
    
    def _check_alert_conditions(self):
        """Check for conditions that should trigger alerts"""
        # Low quality alert
        if (self.metrics.overall_quality_score < self.alert_thresholds['low_quality'] and
            self.metrics.total_frames > 10):  # Only after some frames
            self._trigger_alert("low_quality", f"Stream quality is low ({self.metrics.overall_quality_score:.2f})")
        
        # High error rate alert
        if self.metrics.total_frames > 0:
            error_rate = (self.metrics.corrupted_frames + self.metrics.dropped_frames) / self.metrics.total_frames
            if error_rate > self.alert_thresholds['high_error_rate']:
                self._trigger_alert("high_error_rate", f"High error rate detected ({error_rate:.2%})")
        
        # Poor stability alert
        if (self.metrics.frame_rate_stability < self.alert_thresholds['poor_stability'] and
            len(self.fps_calculator) > 10):
            self._trigger_alert("poor_stability", f"Frame rate unstable ({self.metrics.frame_rate_stability:.2f})")
        
        # Connection issues alert
        recent_network_errors = sum(1 for e in self.network_events if 
                                  e.timestamp > time.time() - 60 and e.severity in ['error', 'critical'])
        if recent_network_errors > 5:
            self._trigger_alert("connection_issues", f"Multiple network errors ({recent_network_errors} in last minute)")
    
    def _trigger_alert(self, alert_type: str, message: str):
        """
        Trigger alert callbacks
        
        Args:
            alert_type: Type of alert
            message: Alert message
        """
        logger.warning(f"Stream alert [{alert_type}]: {message}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, message)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def get_metrics(self) -> QualityMetrics:
        """Get current quality metrics"""
        with self.monitoring_lock:
            return self.metrics 