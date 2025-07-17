"""
Network Optimizer for RTSP streaming with advanced network configuration
"""

import time
import logging
import socket
import struct
import threading
import subprocess
import platform
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class NetworkProtocol(Enum):
    """Network protocol types"""
    TCP = "tcp"
    UDP = "udp"
    HTTP = "http"
    RTSP = "rtsp"
    RTP = "rtp"


@dataclass
class NetworkMetrics:
    """Network performance metrics"""
    bandwidth_bps: float = 0.0
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    packet_loss_rate: float = 0.0
    mtu_size: int = 1500
    rtt_ms: float = 0.0
    connection_stability: float = 0.0
    tcp_window_size: int = 0
    congestion_events: int = 0
    retransmissions: int = 0


@dataclass
class OptimizationConfig:
    """Network optimization configuration"""
    # TCP settings
    tcp_keepalive_enabled: bool = True
    tcp_keepalive_idle: int = 10
    tcp_keepalive_interval: int = 5
    tcp_keepalive_probes: int = 3
    tcp_nodelay: bool = True
    tcp_window_scaling: bool = True
    
    # Buffer settings
    socket_recv_buffer: int = 262144  # 256KB
    socket_send_buffer: int = 262144  # 256KB
    
    # RTP settings
    rtp_packet_size: int = 1400
    rtp_jitter_buffer_ms: int = 100
    
    # Bandwidth settings
    adaptive_bitrate: bool = True
    min_bitrate_bps: int = 100000  # 100 Kbps
    max_bitrate_bps: int = 10000000  # 10 Mbps
    bandwidth_probe_interval: int = 30  # seconds
    
    # Quality settings
    quality_adaptation: bool = True
    resolution_fallback: bool = True
    fps_adaptation: bool = True
    
    # Timeout settings
    connection_timeout: float = 15.0
    read_timeout: float = 10.0
    keepalive_timeout: float = 30.0


class NetworkOptimizer:
    """
    Advanced network optimizer for RTSP streaming
    
    Features:
    - TCP keepalive and socket optimization
    - RTP packet size optimization
    - Bandwidth monitoring and adaptation
    - MTU discovery and optimization
    - Network quality assessment
    - Adaptive streaming configuration
    """
    
    def __init__(self, config: OptimizationConfig = None):
        """
        Initialize network optimizer
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.metrics = NetworkMetrics()
        
        # State tracking
        self.optimization_lock = threading.RLock()
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Network measurements
        self.bandwidth_history: List[Tuple[float, float]] = []  # (timestamp, bandwidth)
        self.latency_history: List[Tuple[float, float]] = []   # (timestamp, latency)
        self.quality_history: List[Tuple[float, float]] = []   # (timestamp, quality_score)
        
        # Adaptive settings
        self.current_bitrate = self.config.max_bitrate_bps
        self.current_quality = 1.0
        self.last_adaptation_time = time.time()
        
        # Platform-specific optimizations
        self.platform = platform.system().lower()
        
        logger.info("NetworkOptimizer initialized")
    
    def optimize_socket(self, sock: socket.socket, protocol: NetworkProtocol) -> bool:
        """
        Apply network optimizations to socket
        
        Args:
            sock: Socket to optimize
            protocol: Network protocol type
            
        Returns:
            bool: True if optimizations applied successfully
        """
        try:
            with self.optimization_lock:
                success = True
                
                # TCP-specific optimizations
                if protocol in [NetworkProtocol.TCP, NetworkProtocol.RTSP]:
                    success &= self._optimize_tcp_socket(sock)
                
                # UDP-specific optimizations
                elif protocol in [NetworkProtocol.UDP, NetworkProtocol.RTP]:
                    success &= self._optimize_udp_socket(sock)
                
                # Generic socket optimizations
                success &= self._optimize_generic_socket(sock)
                
                if success:
                    logger.info(f"Socket optimizations applied for {protocol.value}")
                else:
                    logger.warning(f"Some socket optimizations failed for {protocol.value}")
                
                return success
                
        except Exception as e:
            logger.error(f"Socket optimization error: {e}")
            return False
    
    def _optimize_tcp_socket(self, sock: socket.socket) -> bool:
        """Apply TCP-specific optimizations"""
        success = True
        
        try:
            # TCP_NODELAY - disable Nagle's algorithm for low latency
            if self.config.tcp_nodelay:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                logger.debug("TCP_NODELAY enabled")
        except Exception as e:
            logger.debug(f"Failed to set TCP_NODELAY: {e}")
            success = False
        
        try:
            # TCP Keepalive
            if self.config.tcp_keepalive_enabled:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
                # Platform-specific keepalive settings
                if self.platform == 'linux':
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, self.config.tcp_keepalive_idle)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, self.config.tcp_keepalive_interval)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, self.config.tcp_keepalive_probes)
                elif self.platform == 'darwin':  # macOS
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, self.config.tcp_keepalive_idle)
                elif self.platform == 'windows':
                    # Windows TCP keepalive configuration
                    sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, self.config.tcp_keepalive_idle * 1000, 
                                                         self.config.tcp_keepalive_interval * 1000))
                
                logger.debug("TCP keepalive configured")
        except Exception as e:
            logger.debug(f"Failed to configure TCP keepalive: {e}")
            success = False
        
        return success
    
    def _optimize_udp_socket(self, sock: socket.socket) -> bool:
        """Apply UDP-specific optimizations"""
        success = True
        
        try:
            # Increase UDP buffer sizes for better performance
            udp_buffer_size = max(self.config.socket_recv_buffer, 1024 * 1024)  # At least 1MB for UDP
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, udp_buffer_size)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, udp_buffer_size)
            logger.debug(f"UDP buffer size set to {udp_buffer_size}")
        except Exception as e:
            logger.debug(f"Failed to set UDP buffer size: {e}")
            success = False
        
        return success
    
    def _optimize_generic_socket(self, sock: socket.socket) -> bool:
        """Apply generic socket optimizations"""
        success = True
        
        try:
            # Socket buffer sizes
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.config.socket_recv_buffer)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.config.socket_send_buffer)
            logger.debug(f"Socket buffers set to {self.config.socket_recv_buffer} bytes")
        except Exception as e:
            logger.debug(f"Failed to set socket buffers: {e}")
            success = False
        
        try:
            # Socket timeouts
            sock.settimeout(self.config.read_timeout)
            logger.debug(f"Socket timeout set to {self.config.read_timeout}s")
        except Exception as e:
            logger.debug(f"Failed to set socket timeout: {e}")
            success = False
        
        try:
            # Reuse address
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            logger.debug("SO_REUSEADDR enabled")
        except Exception as e:
            logger.debug(f"Failed to set SO_REUSEADDR: {e}")
            success = False
        
        return success
    
    def discover_mtu(self, target_host: str, target_port: int = 554) -> int:
        """
        Discover Path MTU to target host
        
        Args:
            target_host: Target hostname or IP
            target_port: Target port
            
        Returns:
            int: Discovered MTU size
        """
        try:
            logger.info(f"Discovering MTU to {target_host}:{target_port}")
            
            # Start with common MTU sizes and work down
            test_sizes = [1500, 1472, 1400, 1300, 1200, 1000, 576]
            
            for mtu_size in test_sizes:
                if self._test_mtu_size(target_host, target_port, mtu_size):
                    logger.info(f"Discovered MTU size: {mtu_size}")
                    with self.optimization_lock:
                        self.metrics.mtu_size = mtu_size
                    return mtu_size
            
            # Fallback to safe size
            fallback_mtu = 576
            logger.warning(f"MTU discovery failed, using fallback: {fallback_mtu}")
            with self.optimization_lock:
                self.metrics.mtu_size = fallback_mtu
            return fallback_mtu
            
        except Exception as e:
            logger.error(f"MTU discovery error: {e}")
            return 1500  # Standard Ethernet MTU
    
    def _test_mtu_size(self, host: str, port: int, mtu_size: int) -> bool:
        """Test if specific MTU size works"""
        try:
            # Create test socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(2.0)
            
            # Create test packet (MTU size minus IP/UDP headers)
            test_data_size = mtu_size - 28  # 20 bytes IP header + 8 bytes UDP header
            test_data = b'x' * max(0, test_data_size)
            
            # Try to send packet
            sock.sendto(test_data, (host, port))
            sock.close()
            
            return True
            
        except Exception:
            return False
    
    def measure_bandwidth(self, target_host: str, target_port: int = 554, 
                         duration: float = 5.0) -> float:
        """
        Measure bandwidth to target host
        
        Args:
            target_host: Target hostname or IP
            target_port: Target port
            duration: Measurement duration in seconds
            
        Returns:
            float: Measured bandwidth in bps
        """
        try:
            logger.debug(f"Measuring bandwidth to {target_host}:{target_port}")
            
            start_time = time.time()
            total_bytes = 0
            
            # Create test connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            
            try:
                sock.connect((target_host, target_port))
                
                # Send test data for specified duration
                test_data = b'x' * 1024  # 1KB test packets
                
                while time.time() - start_time < duration:
                    try:
                        bytes_sent = sock.send(test_data)
                        total_bytes += bytes_sent
                    except socket.timeout:
                        break
                    except Exception:
                        break
                
                sock.close()
                
            except Exception as e:
                logger.debug(f"Bandwidth test connection failed: {e}")
                sock.close()
                return 0.0
            
            # Calculate bandwidth
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                bandwidth_bps = (total_bytes * 8) / elapsed_time
                
                with self.optimization_lock:
                    self.metrics.bandwidth_bps = bandwidth_bps
                    self.bandwidth_history.append((time.time(), bandwidth_bps))
                    
                    # Keep only recent history
                    cutoff_time = time.time() - 300  # 5 minutes
                    self.bandwidth_history = [(t, b) for t, b in self.bandwidth_history if t >= cutoff_time]
                
                logger.debug(f"Measured bandwidth: {bandwidth_bps:.0f} bps")
                return bandwidth_bps
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Bandwidth measurement error: {e}")
            return 0.0
    
    def measure_latency(self, target_host: str, samples: int = 5) -> Tuple[float, float]:
        """
        Measure network latency and jitter
        
        Args:
            target_host: Target hostname or IP
            samples: Number of ping samples
            
        Returns:
            Tuple of (latency_ms, jitter_ms)
        """
        try:
            logger.debug(f"Measuring latency to {target_host}")
            
            latencies = []
            
            for _ in range(samples):
                start_time = time.time()
                
                try:
                    # Simple TCP connect test for latency
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2.0)
                    
                    result = sock.connect_ex((target_host, 80))  # Try HTTP port
                    
                    if result == 0:  # Connection successful
                        latency_ms = (time.time() - start_time) * 1000
                        latencies.append(latency_ms)
                    
                    sock.close()
                    
                except Exception:
                    pass
                
                time.sleep(0.1)  # Small delay between samples
            
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                jitter = max(latencies) - min(latencies) if len(latencies) > 1 else 0.0
                
                with self.optimization_lock:
                    self.metrics.latency_ms = avg_latency
                    self.metrics.jitter_ms = jitter
                    self.latency_history.append((time.time(), avg_latency))
                    
                    # Keep only recent history
                    cutoff_time = time.time() - 300  # 5 minutes
                    self.latency_history = [(t, l) for t, l in self.latency_history if t >= cutoff_time]
                
                logger.debug(f"Measured latency: {avg_latency:.1f}ms, jitter: {jitter:.1f}ms")
                return avg_latency, jitter
            
            return 0.0, 0.0
            
        except Exception as e:
            logger.error(f"Latency measurement error: {e}")
            return 0.0, 0.0
    
    def get_optimal_rtp_packet_size(self) -> int:
        """
        Calculate optimal RTP packet size based on MTU
        
        Returns:
            int: Optimal RTP packet size
        """
        with self.optimization_lock:
            # RTP packet size = MTU - IP header (20) - UDP header (8) - RTP header (12)
            rtp_packet_size = self.metrics.mtu_size - 40
            
            # Ensure it's within reasonable bounds
            rtp_packet_size = max(200, min(rtp_packet_size, self.config.rtp_packet_size))
            
            return rtp_packet_size
    
    def adapt_streaming_quality(self, current_bitrate: float, error_rate: float, 
                              frame_rate: float) -> Dict[str, Any]:
        """
        Adapt streaming quality based on network conditions
        
        Args:
            current_bitrate: Current streaming bitrate
            error_rate: Current error rate
            frame_rate: Current frame rate
            
        Returns:
            Dict with recommended settings
        """
        if not self.config.quality_adaptation:
            return {}
        
        with self.optimization_lock:
            current_time = time.time()
            
            # Don't adapt too frequently
            if current_time - self.last_adaptation_time < 10:  # 10 seconds minimum
                return {}
            
            recommendations = {}
            
            # Calculate network quality score
            quality_score = self._calculate_network_quality_score()
            
            # Bitrate adaptation
            if self.config.adaptive_bitrate:
                if quality_score < 0.3:  # Poor quality
                    new_bitrate = max(self.config.min_bitrate_bps, current_bitrate * 0.7)
                    recommendations['bitrate'] = new_bitrate
                elif quality_score > 0.8:  # Good quality
                    new_bitrate = min(self.config.max_bitrate_bps, current_bitrate * 1.2)
                    recommendations['bitrate'] = new_bitrate
            
            # Frame rate adaptation
            if self.config.fps_adaptation:
                if error_rate > 0.1:  # High error rate
                    recommendations['fps'] = max(5, frame_rate * 0.8)
                elif quality_score > 0.9:  # Excellent quality
                    recommendations['fps'] = min(30, frame_rate * 1.1)
            
            # Resolution adaptation
            if self.config.resolution_fallback and quality_score < 0.2:
                recommendations['resolution_scale'] = 0.75  # Reduce resolution by 25%
            
            if recommendations:
                self.last_adaptation_time = current_time
                logger.info(f"Quality adaptation: {recommendations}")
            
            return recommendations
    
    def _calculate_network_quality_score(self) -> float:
        """
        Calculate overall network quality score (0.0 to 1.0)
        
        Returns:
            float: Quality score
        """
        score_factors = []
        
        # Bandwidth factor
        if self.metrics.bandwidth_bps > 0:
            bandwidth_score = min(1.0, self.metrics.bandwidth_bps / self.config.max_bitrate_bps)
            score_factors.append(bandwidth_score)
        
        # Latency factor (lower is better)
        if self.metrics.latency_ms > 0:
            latency_score = max(0.0, 1.0 - (self.metrics.latency_ms - 50) / 200)  # 50ms good, 250ms poor
            score_factors.append(latency_score)
        
        # Jitter factor (lower is better)
        if self.metrics.jitter_ms > 0:
            jitter_score = max(0.0, 1.0 - self.metrics.jitter_ms / 50)  # 50ms jitter = score 0
            score_factors.append(jitter_score)
        
        # Packet loss factor
        if hasattr(self.metrics, 'packet_loss_rate'):
            loss_score = max(0.0, 1.0 - self.metrics.packet_loss_rate * 10)
            score_factors.append(loss_score)
        
        if score_factors:
            return sum(score_factors) / len(score_factors)
        else:
            return 0.5  # Neutral score if no data
    
    def start_monitoring(self, target_host: str, target_port: int = 554):
        """
        Start network monitoring
        
        Args:
            target_host: Target hostname to monitor
            target_port: Target port
        """
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(target_host, target_port),
            daemon=True,
            name="NetworkMonitor"
        )
        self.monitor_thread.start()
        logger.info(f"Network monitoring started for {target_host}:{target_port}")
    
    def stop_monitoring(self):
        """Stop network monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Network monitoring stopped")
    
    def _monitoring_loop(self, target_host: str, target_port: int):
        """Network monitoring loop"""
        last_bandwidth_check = 0
        last_latency_check = 0
        
        while self.is_monitoring:
            try:
                current_time = time.time()
                
                # Periodic bandwidth measurement
                if current_time - last_bandwidth_check >= self.config.bandwidth_probe_interval:
                    self.measure_bandwidth(target_host, target_port, duration=2.0)
                    last_bandwidth_check = current_time
                
                # Periodic latency measurement
                if current_time - last_latency_check >= 30:  # Every 30 seconds
                    self.measure_latency(target_host)
                    last_latency_check = current_time
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Network monitoring error: {e}")
                time.sleep(10)  # Wait longer on error
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Get network optimization recommendations
        
        Returns:
            Dict with optimization recommendations
        """
        with self.optimization_lock:
            recommendations = {
                'rtp_packet_size': self.get_optimal_rtp_packet_size(),
                'tcp_settings': {
                    'keepalive_enabled': self.config.tcp_keepalive_enabled,
                    'nodelay': self.config.tcp_nodelay,
                    'buffer_size': self.config.socket_recv_buffer
                },
                'quality_settings': {
                    'adaptive_bitrate': self.config.adaptive_bitrate,
                    'current_quality_score': self._calculate_network_quality_score()
                }
            }
            
            # Add MTU-based recommendations
            if self.metrics.mtu_size < 1500:
                recommendations['mtu_warning'] = f"Low MTU detected ({self.metrics.mtu_size}), consider fragmentation"
            
            # Add latency-based recommendations
            if self.metrics.latency_ms > 100:
                recommendations['latency_warning'] = f"High latency ({self.metrics.latency_ms:.1f}ms), consider TCP optimizations"
            
            return recommendations
    
    def get_metrics(self) -> NetworkMetrics:
        """Get current network metrics"""
        with self.optimization_lock:
            return self.metrics 