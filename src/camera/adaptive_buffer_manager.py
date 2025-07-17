"""
Adaptive Buffer Manager for dynamic RTSP stream buffering and packet management
"""

import time
import logging
import threading
import queue
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BufferMetrics:
    """Buffer performance and health metrics"""
    total_packets_received: int = 0
    packets_buffered: int = 0
    packets_dropped: int = 0
    packets_reordered: int = 0
    buffer_overflows: int = 0
    buffer_underflows: int = 0
    average_buffer_size: float = 0.0
    peak_buffer_size: int = 0
    buffer_efficiency: float = 0.0
    reorder_window_hits: int = 0
    late_packet_drops: int = 0
    duplicate_packets: int = 0
    corruption_recoveries: int = 0
    adaptive_adjustments: int = 0
    network_jitter_ms: float = 0.0
    packet_loss_rate: float = 0.0


@dataclass
class PacketInfo:
    """Information about a buffered packet"""
    data: bytes
    timestamp: float
    sequence_number: Optional[int] = None
    size: int = 0
    is_keyframe: bool = False
    corruption_score: float = 0.0
    arrival_time: float = field(default_factory=time.time)
    expected_arrival_time: Optional[float] = None


class BufferStrategy:
    """Buffer strategy configuration"""
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        # Basic buffer settings
        self.initial_size = config.get('initial_buffer_size', 32)
        self.min_size = config.get('min_buffer_size', 8)
        self.max_size = config.get('max_buffer_size', 256)
        
        # Adaptive settings
        self.enable_adaptive_sizing = config.get('enable_adaptive_sizing', True)
        self.size_adjustment_threshold = config.get('size_adjustment_threshold', 0.1)
        self.adjustment_factor = config.get('adjustment_factor', 1.2)
        
        # Reordering settings
        self.reorder_window_ms = config.get('reorder_window_ms', 100.0)
        self.max_reorder_distance = config.get('max_reorder_distance', 16)
        
        # Timeout settings
        self.packet_timeout_ms = config.get('packet_timeout_ms', 500.0)
        self.keyframe_timeout_ms = config.get('keyframe_timeout_ms', 2000.0)
        
        # Quality settings
        self.drop_threshold_score = config.get('drop_threshold_score', 0.8)
        self.prefer_keyframes = config.get('prefer_keyframes', True)
        self.aggressive_dropping = config.get('aggressive_dropping', False)


class AdaptiveBufferManager:
    """
    Adaptive buffer manager for RTSP streaming with dynamic sizing and quality management
    
    Features:
    - Dynamic buffer sizing based on network conditions
    - Packet reordering tolerance with sliding window
    - Frame dropout detection and recovery
    - Quality-based packet dropping
    - Network jitter adaptation
    - Corruption recovery mechanisms
    """
    
    def __init__(self, strategy: BufferStrategy = None):
        """
        Initialize adaptive buffer manager
        
        Args:
            strategy: Buffer strategy configuration
        """
        self.strategy = strategy or BufferStrategy()
        self.metrics = BufferMetrics()
        
        # Threading
        self.buffer_lock = threading.RLock()
        self.management_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Buffer storage
        self.main_buffer: deque = deque(maxlen=self.strategy.max_size)
        self.reorder_buffer: Dict[int, PacketInfo] = {}
        self.keyframe_buffer: deque = deque(maxlen=10)  # Keep recent keyframes
        
        # Sequence tracking
        self.expected_sequence = 0
        self.sequence_wrap_count = 0
        self.last_sequence_time = time.time()
        
        # Adaptive parameters
        self.current_buffer_target = self.strategy.initial_size
        self.network_quality_score = 1.0
        self.jitter_estimator = 0.0
        self.bandwidth_estimator = 0.0
        
        # Statistics for adaptation
        self.packet_arrival_times: deque = deque(maxlen=100)
        self.buffer_size_history: deque = deque(maxlen=50)
        self.quality_history: deque = deque(maxlen=20)
        
        logger.info("AdaptiveBufferManager initialized")
    
    def start(self):
        """Start the buffer management thread"""
        if self.running:
            return
        
        self.running = True
        self.management_thread = threading.Thread(
            target=self._management_loop,
            daemon=True,
            name="AdaptiveBufferManager"
        )
        self.management_thread.start()
        logger.info("Buffer management started")
    
    def stop(self):
        """Stop the buffer management thread"""
        self.running = False
        if self.management_thread:
            self.management_thread.join(timeout=2)
        logger.info("Buffer management stopped")
    
    def add_packet(self, data: bytes, timestamp: float = None, 
                   sequence_number: int = None, is_keyframe: bool = False,
                   corruption_score: float = 0.0) -> bool:
        """
        Add packet to buffer with adaptive management
        
        Args:
            data: Packet data
            timestamp: Packet timestamp
            sequence_number: Sequence number for reordering
            is_keyframe: Whether this is a keyframe packet
            corruption_score: Corruption probability (0.0-1.0)
            
        Returns:
            bool: True if packet was accepted, False if dropped
        """
        if not data:
            return False
        
        current_time = time.time()
        
        # Create packet info
        packet = PacketInfo(
            data=data,
            timestamp=timestamp or current_time,
            sequence_number=sequence_number,
            size=len(data),
            is_keyframe=is_keyframe,
            corruption_score=corruption_score,
            arrival_time=current_time
        )
        
        with self.buffer_lock:
            self.metrics.total_packets_received += 1
            
            # Update arrival time statistics
            self.packet_arrival_times.append(current_time)
            
            # Check for corruption
            if corruption_score > self.strategy.drop_threshold_score:
                self.metrics.packets_dropped += 1
                logger.debug(f"Dropped corrupted packet (score: {corruption_score:.2f})")
                return False
            
            # Handle sequence numbering and reordering
            if sequence_number is not None:
                if not self._handle_sequenced_packet(packet):
                    return False
            else:
                # No sequence number, add directly
                if not self._add_to_main_buffer(packet):
                    return False
            
            # Special handling for keyframes
            if is_keyframe:
                self._handle_keyframe_packet(packet)
            
            # Update network quality estimation
            self._update_network_quality(packet)
            
            return True
    
    def get_packet(self, timeout: float = None) -> Optional[PacketInfo]:
        """
        Get next packet from buffer
        
        Args:
            timeout: Maximum time to wait for packet
            
        Returns:
            PacketInfo or None if no packet available
        """
        start_time = time.time()
        
        while True:
            with self.buffer_lock:
                # Try to get from reorder buffer first (in-order packets)
                packet = self._get_from_reorder_buffer()
                if packet:
                    self.metrics.packets_buffered += 1
                    return packet
                
                # Get from main buffer
                if self.main_buffer:
                    packet = self.main_buffer.popleft()
                    self.metrics.packets_buffered += 1
                    return packet
                
                # No packets available
                if timeout is None:
                    return None
            
            # Wait if timeout specified
            if timeout and (time.time() - start_time) >= timeout:
                self.metrics.buffer_underflows += 1
                return None
            
            time.sleep(0.001)  # Short wait before retry
    
    def get_latest_packet(self) -> Optional[PacketInfo]:
        """
        Get the most recent packet, dropping older ones
        
        Returns:
            PacketInfo or None if no packet available
        """
        with self.buffer_lock:
            if not self.main_buffer:
                return None
            
            # Drop all but the latest packet
            dropped_count = len(self.main_buffer) - 1
            if dropped_count > 0:
                self.metrics.packets_dropped += dropped_count
                # Keep only the last packet
                latest_packet = self.main_buffer[-1]
                self.main_buffer.clear()
                self.main_buffer.append(latest_packet)
                logger.debug(f"Dropped {dropped_count} old packets for latest-only mode")
            
            return self.main_buffer.popleft()
    
    def get_keyframe_packet(self) -> Optional[PacketInfo]:
        """
        Get most recent keyframe packet
        
        Returns:
            PacketInfo or None if no keyframe available
        """
        with self.buffer_lock:
            if self.keyframe_buffer:
                return self.keyframe_buffer[-1]  # Most recent keyframe
            return None
    
    def flush_buffer(self):
        """Flush all buffered packets"""
        with self.buffer_lock:
            dropped_count = len(self.main_buffer) + len(self.reorder_buffer)
            self.main_buffer.clear()
            self.reorder_buffer.clear()
            self.metrics.packets_dropped += dropped_count
            logger.debug(f"Flushed {dropped_count} packets from buffer")
    
    def _handle_sequenced_packet(self, packet: PacketInfo) -> bool:
        """
        Handle packet with sequence number for reordering
        
        Args:
            packet: Packet to handle
            
        Returns:
            bool: True if packet was handled successfully
        """
        seq_num = packet.sequence_number
        
        if seq_num == self.expected_sequence:
            # Perfect order - add to main buffer
            self.expected_sequence += 1
            return self._add_to_main_buffer(packet)
        
        elif seq_num > self.expected_sequence:
            # Future packet - add to reorder buffer
            if seq_num - self.expected_sequence <= self.strategy.max_reorder_distance:
                self.reorder_buffer[seq_num] = packet
                self.metrics.packets_reordered += 1
                return True
            else:
                # Too far in the future - assume packet loss and skip
                logger.debug(f"Skipping packet sequence {self.expected_sequence} to {seq_num - 1}")
                self.expected_sequence = seq_num + 1
                self.metrics.packet_loss_rate = self._calculate_packet_loss_rate()
                return self._add_to_main_buffer(packet)
        
        else:
            # Late packet
            if self.expected_sequence - seq_num <= self.strategy.max_reorder_distance:
                # Still within reorder window
                if seq_num not in self.reorder_buffer:  # Avoid duplicates
                    self.reorder_buffer[seq_num] = packet
                    self.metrics.packets_reordered += 1
                    return True
                else:
                    self.metrics.duplicate_packets += 1
                    return False
            else:
                # Too late - drop
                self.metrics.late_packet_drops += 1
                return False
    
    def _get_from_reorder_buffer(self) -> Optional[PacketInfo]:
        """
        Get next in-order packet from reorder buffer
        
        Returns:
            PacketInfo or None if no in-order packet available
        """
        if self.expected_sequence in self.reorder_buffer:
            packet = self.reorder_buffer.pop(self.expected_sequence)
            self.expected_sequence += 1
            self.metrics.reorder_window_hits += 1
            return packet
        return None
    
    def _add_to_main_buffer(self, packet: PacketInfo) -> bool:
        """
        Add packet to main buffer with overflow protection
        
        Args:
            packet: Packet to add
            
        Returns:
            bool: True if packet was added, False if dropped due to overflow
        """
        if len(self.main_buffer) >= self.current_buffer_target:
            if self.strategy.aggressive_dropping or not packet.is_keyframe:
                # Drop oldest packet to make room
                if self.main_buffer:
                    self.main_buffer.popleft()
                    self.metrics.buffer_overflows += 1
            else:
                # Keyframe and not aggressive - drop this packet instead
                self.metrics.packets_dropped += 1
                return False
        
        self.main_buffer.append(packet)
        
        # Update buffer size statistics
        current_size = len(self.main_buffer)
        self.buffer_size_history.append(current_size)
        
        if current_size > self.metrics.peak_buffer_size:
            self.metrics.peak_buffer_size = current_size
        
        # Update average buffer size
        if self.metrics.average_buffer_size == 0:
            self.metrics.average_buffer_size = current_size
        else:
            self.metrics.average_buffer_size = (
                self.metrics.average_buffer_size * 0.95 + current_size * 0.05
            )
        
        return True
    
    def _handle_keyframe_packet(self, packet: PacketInfo):
        """
        Special handling for keyframe packets
        
        Args:
            packet: Keyframe packet
        """
        # Always keep recent keyframes for recovery
        self.keyframe_buffer.append(packet)
        
        # Keyframes reset sequence expectations in some protocols
        if packet.sequence_number is not None:
            # Reset sequence tracking on keyframe for robustness
            self.expected_sequence = packet.sequence_number + 1
    
    def _update_network_quality(self, packet: PacketInfo):
        """
        Update network quality estimation based on packet characteristics
        
        Args:
            packet: Received packet
        """
        current_time = time.time()
        
        # Calculate jitter based on arrival time variance
        if len(self.packet_arrival_times) > 1:
            intervals = []
            times = list(self.packet_arrival_times)
            for i in range(1, len(times)):
                intervals.append(times[i] - times[i-1])
            
            if intervals:
                mean_interval = np.mean(intervals)
                jitter = np.std(intervals)
                
                # Update jitter estimator (exponential moving average)
                if self.jitter_estimator == 0:
                    self.jitter_estimator = jitter
                else:
                    self.jitter_estimator = self.jitter_estimator * 0.8 + jitter * 0.2
                
                self.metrics.network_jitter_ms = self.jitter_estimator * 1000
        
        # Estimate bandwidth
        if packet.size > 0:
            # Simple bandwidth estimation based on recent packets
            recent_packets = list(self.packet_arrival_times)[-10:]  # Last 10 packets
            if len(recent_packets) > 1:
                time_span = recent_packets[-1] - recent_packets[0]
                if time_span > 0:
                    # Rough bandwidth estimate
                    total_size = packet.size * len(recent_packets)
                    bandwidth_bps = total_size / time_span
                    
                    if self.bandwidth_estimator == 0:
                        self.bandwidth_estimator = bandwidth_bps
                    else:
                        self.bandwidth_estimator = (
                            self.bandwidth_estimator * 0.9 + bandwidth_bps * 0.1
                        )
        
        # Calculate overall network quality score
        jitter_factor = max(0, 1.0 - (self.jitter_estimator / 0.1))  # Good if jitter < 100ms
        loss_factor = max(0, 1.0 - self.metrics.packet_loss_rate)
        
        self.network_quality_score = (jitter_factor + loss_factor) / 2.0
        self.quality_history.append(self.network_quality_score)
    
    def _calculate_packet_loss_rate(self) -> float:
        """
        Calculate packet loss rate based on sequence gaps
        
        Returns:
            float: Packet loss rate (0.0 to 1.0)
        """
        if self.metrics.total_packets_received == 0:
            return 0.0
        
        expected_packets = self.expected_sequence + len(self.reorder_buffer)
        actual_packets = self.metrics.total_packets_received
        
        if expected_packets > actual_packets:
            loss_rate = (expected_packets - actual_packets) / expected_packets
            return min(1.0, max(0.0, loss_rate))
        
        return 0.0
    
    def _management_loop(self):
        """Buffer management and adaptation loop"""
        last_adaptation_time = time.time()
        adaptation_interval = 5.0  # Adapt every 5 seconds
        
        while self.running:
            try:
                current_time = time.time()
                
                # Periodic adaptation
                if current_time - last_adaptation_time >= adaptation_interval:
                    self._adapt_buffer_size()
                    self._cleanup_reorder_buffer()
                    last_adaptation_time = current_time
                
                # Clean up old packets from reorder buffer
                self._timeout_reorder_packets()
                
                time.sleep(1.0)  # Run every second
                
            except Exception as e:
                logger.error(f"Buffer management error: {e}")
    
    def _adapt_buffer_size(self):
        """Adapt buffer size based on network conditions"""
        if not self.strategy.enable_adaptive_sizing:
            return
        
        with self.buffer_lock:
            # Analyze recent buffer performance
            if len(self.buffer_size_history) < 10:
                return
            
            avg_size = np.mean(list(self.buffer_size_history))
            overflow_rate = self.metrics.buffer_overflows / max(1, self.metrics.total_packets_received)
            underflow_rate = self.metrics.buffer_underflows / max(1, self.metrics.total_packets_received)
            
            # Decide on adjustment
            new_target = self.current_buffer_target
            
            if overflow_rate > self.strategy.size_adjustment_threshold:
                # Too many overflows - increase buffer
                new_target = min(
                    self.strategy.max_size,
                    int(self.current_buffer_target * self.strategy.adjustment_factor)
                )
                logger.debug(f"Increasing buffer size due to overflows: {self.current_buffer_target} -> {new_target}")
            
            elif underflow_rate > self.strategy.size_adjustment_threshold:
                # Too many underflows - might decrease buffer (if network is good)
                if self.network_quality_score > 0.8:
                    new_target = max(
                        self.strategy.min_size,
                        int(self.current_buffer_target / self.strategy.adjustment_factor)
                    )
                    logger.debug(f"Decreasing buffer size due to good network: {self.current_buffer_target} -> {new_target}")
            
            elif avg_size < self.current_buffer_target * 0.5 and self.network_quality_score > 0.9:
                # Buffer is consistently underutilized and network is excellent
                new_target = max(
                    self.strategy.min_size,
                    int(self.current_buffer_target * 0.8)
                )
                logger.debug(f"Optimizing buffer size for excellent network: {self.current_buffer_target} -> {new_target}")
            
            if new_target != self.current_buffer_target:
                self.current_buffer_target = new_target
                self.main_buffer = deque(self.main_buffer, maxlen=new_target)
                self.metrics.adaptive_adjustments += 1
    
    def _cleanup_reorder_buffer(self):
        """Clean up old entries from reorder buffer"""
        current_time = time.time()
        timeout_threshold = self.strategy.packet_timeout_ms / 1000.0
        
        expired_sequences = []
        for seq_num, packet in self.reorder_buffer.items():
            if current_time - packet.arrival_time > timeout_threshold:
                expired_sequences.append(seq_num)
        
        for seq_num in expired_sequences:
            self.reorder_buffer.pop(seq_num)
            self.metrics.late_packet_drops += 1
    
    def _timeout_reorder_packets(self):
        """Handle timeout of packets in reorder buffer"""
        current_time = time.time()
        timeout_threshold = self.strategy.reorder_window_ms / 1000.0
        
        with self.buffer_lock:
            expired_sequences = []
            for seq_num, packet in self.reorder_buffer.items():
                if current_time - packet.arrival_time > timeout_threshold:
                    expired_sequences.append(seq_num)
            
            # Move expired packets to main buffer or drop them
            for seq_num in expired_sequences:
                packet = self.reorder_buffer.pop(seq_num)
                if not self._add_to_main_buffer(packet):
                    self.metrics.packets_dropped += 1
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """
        Get current buffer status
        
        Returns:
            Dict with buffer status information
        """
        with self.buffer_lock:
            return {
                'main_buffer_size': len(self.main_buffer),
                'reorder_buffer_size': len(self.reorder_buffer),
                'keyframe_buffer_size': len(self.keyframe_buffer),
                'target_buffer_size': self.current_buffer_target,
                'network_quality_score': self.network_quality_score,
                'network_jitter_ms': self.metrics.network_jitter_ms,
                'packet_loss_rate': self.metrics.packet_loss_rate,
                'buffer_efficiency': self._calculate_buffer_efficiency()
            }
    
    def _calculate_buffer_efficiency(self) -> float:
        """
        Calculate buffer efficiency score
        
        Returns:
            float: Efficiency score (0.0 to 1.0)
        """
        if self.metrics.total_packets_received == 0:
            return 0.0
        
        successful_rate = self.metrics.packets_buffered / self.metrics.total_packets_received
        overflow_penalty = self.metrics.buffer_overflows / max(1, self.metrics.total_packets_received)
        underflow_penalty = self.metrics.buffer_underflows / max(1, self.metrics.total_packets_received)
        
        efficiency = successful_rate - (overflow_penalty + underflow_penalty) * 0.5
        return max(0.0, min(1.0, efficiency))
    
    def get_metrics(self) -> BufferMetrics:
        """Get current buffer metrics"""
        with self.buffer_lock:
            self.metrics.buffer_efficiency = self._calculate_buffer_efficiency()
            return self.metrics 