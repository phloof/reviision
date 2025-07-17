"""
H.264 Stream Validator for robust NAL unit handling and corruption detection
"""

import time
import logging
import struct
import threading
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class NALUnitType(Enum):
    """H.264 NAL Unit types"""
    UNSPECIFIED = 0
    CODED_SLICE_NON_IDR = 1
    CODED_SLICE_A = 2
    CODED_SLICE_B = 3
    CODED_SLICE_C = 4
    CODED_SLICE_IDR = 5
    SEI = 6
    SPS = 7  # Sequence Parameter Set
    PPS = 8  # Picture Parameter Set
    ACCESS_UNIT_DELIMITER = 9
    END_OF_SEQUENCE = 10
    END_OF_STREAM = 11
    FILLER_DATA = 12


@dataclass
class NALUnit:
    """NAL Unit information"""
    type: NALUnitType
    size: int
    data: bytes
    is_keyframe: bool = False
    is_valid: bool = True
    corruption_score: float = 0.0


@dataclass
class StreamValidationMetrics:
    """Stream validation metrics"""
    total_nals_processed: int = 0
    valid_nals: int = 0
    corrupted_nals: int = 0
    i_frames_found: int = 0
    p_frames_found: int = 0
    b_frames_found: int = 0
    sps_found: int = 0
    pps_found: int = 0
    size_errors: int = 0
    format_errors: int = 0
    sequence_errors: int = 0
    last_keyframe_time: Optional[float] = None
    keyframe_interval: float = 0.0
    average_nal_size: float = 0.0
    corruption_patterns: Dict[str, int] = None
    
    def __post_init__(self):
        if self.corruption_patterns is None:
            self.corruption_patterns = {}


class H264StreamValidator:
    """
    H.264 stream validator with NAL unit analysis and corruption detection
    
    Provides comprehensive H.264 stream validation including:
    - NAL unit size and format validation
    - Frame boundary detection
    - I-frame seeking on corruption
    - Corruption pattern analysis
    - Stream health assessment
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize H.264 stream validator
        
        Args:
            config: Validation configuration parameters
        """
        self.config = config or {}
        
        # Validation thresholds
        self.min_nal_size = self.config.get('min_nal_size', 4)
        self.max_nal_size = self.config.get('max_nal_size', 1024 * 1024)  # 1MB
        self.min_keyframe_size = self.config.get('min_keyframe_size', 64)
        self.max_corruption_score = self.config.get('max_corruption_score', 0.7)
        
        # Frame validation settings
        self.keyframe_interval_threshold = self.config.get('keyframe_interval_threshold', 5.0)
        self.sequence_error_threshold = self.config.get('sequence_error_threshold', 10)
        
        # Buffer settings for frame reconstruction
        self.buffer_size = self.config.get('buffer_size', 1024 * 1024)  # 1MB
        self.reorder_buffer_size = self.config.get('reorder_buffer_size', 16)
        
        # Metrics and state
        self.metrics = StreamValidationMetrics()
        self.validation_lock = threading.RLock()
        
        # NAL unit sequence tracking
        self.nal_buffer: List[NALUnit] = []
        self.last_sps: Optional[bytes] = None
        self.last_pps: Optional[bytes] = None
        self.frame_sequence_number = 0
        
        # Corruption pattern detection
        self.corruption_patterns = {
            'size_47_bytes': 0,  # Specific issue mentioned in plan
            'incomplete_start_codes': 0,
            'invalid_nal_types': 0,
            'sequence_breaks': 0,
            'missing_parameter_sets': 0,
            'keyframe_corruption': 0,
            'packet_reordering': 0
        }
        
        logger.info("H264StreamValidator initialized")
    
    def validate_nal_unit(self, data: bytes, offset: int = 0) -> Optional[NALUnit]:
        """
        Validate a single NAL unit
        
        Args:
            data: Raw NAL unit data
            offset: Offset into the data
            
        Returns:
            NALUnit object if valid, None if invalid
        """
        if not data or len(data) < self.min_nal_size:
            self.metrics.size_errors += 1
            return None
        
        try:
            # Check for H.264 start code (0x00000001 or 0x000001)
            start_code_4 = data[offset:offset+4] == b'\x00\x00\x00\x01'
            start_code_3 = data[offset:offset+3] == b'\x00\x00\x01'
            
            if start_code_4:
                nal_header_offset = offset + 4
            elif start_code_3:
                nal_header_offset = offset + 3
            else:
                # No valid start code found
                self.corruption_patterns['incomplete_start_codes'] += 1
                self.metrics.format_errors += 1
                return None
            
            if nal_header_offset >= len(data):
                return None
            
            # Parse NAL header
            nal_header = data[nal_header_offset]
            forbidden_zero_bit = (nal_header >> 7) & 0x01
            nal_ref_idc = (nal_header >> 5) & 0x03
            nal_unit_type_val = nal_header & 0x1F
            
            # Validate forbidden zero bit
            if forbidden_zero_bit != 0:
                self.corruption_patterns['invalid_nal_types'] += 1
                self.metrics.format_errors += 1
                return None
            
            # Validate NAL unit type
            try:
                nal_type = NALUnitType(nal_unit_type_val)
            except ValueError:
                self.corruption_patterns['invalid_nal_types'] += 1
                self.metrics.format_errors += 1
                return None
            
            # Find next start code to determine NAL unit size
            next_start = self._find_next_start_code(data, nal_header_offset + 1)
            if next_start == -1:
                nal_size = len(data) - nal_header_offset
            else:
                nal_size = next_start - nal_header_offset
            
            # Validate NAL unit size
            if nal_size < self.min_nal_size or nal_size > self.max_nal_size:
                # Special handling for the 47-byte corruption mentioned in the plan
                if nal_size == 47:
                    self.corruption_patterns['size_47_bytes'] += 1
                    logger.debug("Detected 47-byte NAL unit corruption")
                self.metrics.size_errors += 1
                return None
            
            # Extract NAL unit data
            nal_data = data[nal_header_offset:nal_header_offset + nal_size]
            
            # Determine if this is a keyframe
            is_keyframe = nal_type == NALUnitType.CODED_SLICE_IDR
            
            # Additional validation for keyframes
            if is_keyframe and nal_size < self.min_keyframe_size:
                self.corruption_patterns['keyframe_corruption'] += 1
                return None
            
            # Calculate corruption score
            corruption_score = self._calculate_corruption_score(nal_data, nal_type)
            
            # Create NAL unit object
            nal_unit = NALUnit(
                type=nal_type,
                size=nal_size,
                data=nal_data,
                is_keyframe=is_keyframe,
                is_valid=corruption_score < self.max_corruption_score,
                corruption_score=corruption_score
            )
            
            # Update metrics
            with self.validation_lock:
                self.metrics.total_nals_processed += 1
                if nal_unit.is_valid:
                    self.metrics.valid_nals += 1
                else:
                    self.metrics.corrupted_nals += 1
                
                # Update frame type counters
                if nal_type == NALUnitType.CODED_SLICE_IDR:
                    self.metrics.i_frames_found += 1
                    current_time = time.time()
                    if self.metrics.last_keyframe_time:
                        interval = current_time - self.metrics.last_keyframe_time
                        if self.metrics.keyframe_interval == 0:
                            self.metrics.keyframe_interval = interval
                        else:
                            self.metrics.keyframe_interval = (
                                self.metrics.keyframe_interval * 0.8 + interval * 0.2
                            )
                    self.metrics.last_keyframe_time = current_time
                    
                elif nal_type == NALUnitType.CODED_SLICE_NON_IDR:
                    self.metrics.p_frames_found += 1
                elif nal_type in [NALUnitType.CODED_SLICE_A, NALUnitType.CODED_SLICE_B]:
                    self.metrics.b_frames_found += 1
                elif nal_type == NALUnitType.SPS:
                    self.metrics.sps_found += 1
                    self.last_sps = nal_data
                elif nal_type == NALUnitType.PPS:
                    self.metrics.pps_found += 1
                    self.last_pps = nal_data
                
                # Update average NAL size
                if self.metrics.average_nal_size == 0:
                    self.metrics.average_nal_size = nal_size
                else:
                    self.metrics.average_nal_size = (
                        self.metrics.average_nal_size * 0.9 + nal_size * 0.1
                    )
            
            return nal_unit
            
        except Exception as e:
            logger.debug(f"NAL unit validation error: {e}")
            self.metrics.format_errors += 1
            return None
    
    def _find_next_start_code(self, data: bytes, start_offset: int) -> int:
        """
        Find the next H.264 start code in the data
        
        Args:
            data: Binary data to search
            start_offset: Offset to start searching from
            
        Returns:
            int: Offset of next start code, or -1 if not found
        """
        # Look for 0x000001 or 0x00000001
        for i in range(start_offset, len(data) - 3):
            if data[i:i+3] == b'\x00\x00\x01':
                return i
            elif i < len(data) - 4 and data[i:i+4] == b'\x00\x00\x00\x01':
                return i
        return -1
    
    def _calculate_corruption_score(self, nal_data: bytes, nal_type: NALUnitType) -> float:
        """
        Calculate corruption probability score for NAL unit
        
        Args:
            nal_data: NAL unit binary data
            nal_type: NAL unit type
            
        Returns:
            float: Corruption score (0.0 = clean, 1.0 = definitely corrupted)
        """
        if not nal_data or len(nal_data) < 4:
            return 1.0
        
        score = 0.0
        
        # Check for excessive zero bytes (common corruption pattern)
        zero_ratio = nal_data.count(0) / len(nal_data)
        if zero_ratio > 0.8:
            score += 0.4
        
        # Check for excessive 0xFF bytes (another corruption pattern)
        ff_ratio = nal_data.count(0xFF) / len(nal_data)
        if ff_ratio > 0.5:
            score += 0.3
        
        # Check entropy (low entropy suggests corruption)
        entropy = self._calculate_entropy(nal_data)
        if entropy < 2.0:  # Very low entropy
            score += 0.2
        
        # Check for valid RBSP structure for parameter sets
        if nal_type in [NALUnitType.SPS, NALUnitType.PPS]:
            if not self._validate_parameter_set_structure(nal_data):
                score += 0.5
        
        # Check for impossible patterns
        if len(nal_data) > 4:
            # Look for impossible bit patterns
            if nal_data[1:4] == b'\x00\x00\x00':  # Shouldn't have start code inside
                score += 0.3
        
        return min(1.0, score)
    
    def _calculate_entropy(self, data: bytes) -> float:
        """
        Calculate Shannon entropy of byte data
        
        Args:
            data: Binary data
            
        Returns:
            float: Entropy value
        """
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        for count in byte_counts:
            if count > 0:
                probability = count / data_len
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _validate_parameter_set_structure(self, nal_data: bytes) -> bool:
        """
        Basic validation of SPS/PPS structure
        
        Args:
            nal_data: NAL unit data
            
        Returns:
            bool: True if structure appears valid
        """
        if len(nal_data) < 4:
            return False
        
        # Basic checks for parameter set structure
        # This is a simplified validation - full validation would require
        # complete H.264 bitstream parsing
        
        # Check that we have some reasonable data after the header
        if len(nal_data) < 8:
            return False
        
        # Parameter sets should not be too small or too large
        if len(nal_data) > 1000:  # Very large parameter set is suspicious
            return False
        
        return True
    
    def validate_frame_data(self, frame_data: bytes) -> Tuple[bool, List[NALUnit]]:
        """
        Validate complete frame data containing multiple NAL units
        
        Args:
            frame_data: Complete frame binary data
            
        Returns:
            Tuple of (is_valid, list of NAL units)
        """
        nal_units = []
        offset = 0
        is_valid = True
        
        while offset < len(frame_data):
            nal_unit = self.validate_nal_unit(frame_data, offset)
            
            if nal_unit is None:
                # Skip to next potential start code
                next_start = self._find_next_start_code(frame_data, offset + 1)
                if next_start == -1:
                    break
                offset = next_start
                is_valid = False
                continue
            
            nal_units.append(nal_unit)
            offset += nal_unit.size + (4 if frame_data[offset:offset+4] == b'\x00\x00\x00\x01' else 3)
            
            if not nal_unit.is_valid:
                is_valid = False
        
        return is_valid, nal_units
    
    def find_next_keyframe(self, stream_data: bytes, start_offset: int = 0) -> int:
        """
        Find the next I-frame (keyframe) in the stream data
        
        Args:
            stream_data: Stream binary data
            start_offset: Offset to start searching from
            
        Returns:
            int: Offset of next keyframe, or -1 if not found
        """
        offset = start_offset
        
        while offset < len(stream_data):
            nal_unit = self.validate_nal_unit(stream_data, offset)
            
            if nal_unit and nal_unit.is_keyframe and nal_unit.is_valid:
                return offset
            
            # Move to next potential NAL unit
            next_start = self._find_next_start_code(stream_data, offset + 1)
            if next_start == -1:
                break
            offset = next_start
        
        return -1
    
    def should_seek_to_keyframe(self) -> bool:
        """
        Determine if seeking to next keyframe is recommended
        
        Returns:
            bool: True if seeking is recommended
        """
        with self.validation_lock:
            # Check corruption rate
            if self.metrics.total_nals_processed == 0:
                return False
            
            corruption_rate = self.metrics.corrupted_nals / self.metrics.total_nals_processed
            
            # Seek if corruption rate is high
            if corruption_rate > 0.3:
                return True
            
            # Seek if too long since last keyframe
            if (self.metrics.last_keyframe_time and 
                time.time() - self.metrics.last_keyframe_time > self.keyframe_interval_threshold):
                return True
            
            # Seek if parameter sets are missing
            if self.metrics.i_frames_found > 0 and (not self.last_sps or not self.last_pps):
                self.corruption_patterns['missing_parameter_sets'] += 1
                return True
            
            return False
    
    def get_stream_health_score(self) -> float:
        """
        Calculate overall stream health score (0.0 to 1.0)
        
        Returns:
            float: Health score
        """
        with self.validation_lock:
            if self.metrics.total_nals_processed == 0:
                return 0.0
            
            # Valid NAL ratio
            valid_ratio = self.metrics.valid_nals / self.metrics.total_nals_processed
            
            # Frame type distribution score
            total_frames = (self.metrics.i_frames_found + 
                          self.metrics.p_frames_found + 
                          self.metrics.b_frames_found)
            
            if total_frames == 0:
                frame_score = 0.0
            else:
                # Expect some reasonable distribution of frame types
                i_ratio = self.metrics.i_frames_found / total_frames
                frame_score = 1.0 if 0.05 <= i_ratio <= 0.5 else 0.5
            
            # Parameter set availability
            param_score = 1.0 if (self.last_sps and self.last_pps) else 0.0
            
            # Weighted health score
            health_score = (
                valid_ratio * 0.6 +
                frame_score * 0.2 +
                param_score * 0.2
            )
            
            return max(0.0, min(1.0, health_score))
    
    def get_metrics(self) -> StreamValidationMetrics:
        """Get current validation metrics"""
        with self.validation_lock:
            # Update corruption patterns in metrics
            self.metrics.corruption_patterns.update(self.corruption_patterns)
            return self.metrics
    
    def reset_metrics(self):
        """Reset all validation metrics"""
        with self.validation_lock:
            self.metrics = StreamValidationMetrics()
            self.corruption_patterns = {key: 0 for key in self.corruption_patterns}
            logger.info("H.264 validation metrics reset") 