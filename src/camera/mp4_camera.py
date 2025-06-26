"""
Video File Camera implementation for Retail Analytics System
"""

import time
import logging
import os
from pathlib import Path
import cv2
from .base import BaseCamera

logger = logging.getLogger(__name__)

class VideoFileCamera(BaseCamera):
    """
    Video File Camera implementation for video file sources
    
    This class handles video capture from video files (MP4, MPG) via OpenCV's VideoCapture interface.
    It can be used to process pre-recorded footage for analytics.
    """
    
    def __init__(self, config):
        """
        Initialize video file camera with the provided configuration
        
        Args:
            config (dict): Camera configuration dictionary with the following keys:
                - file_path: Path to the video file (MP4, MPG) (required)
                - loop: Whether to loop the video when it ends (default: False)
                - playback_speed: Speed multiplier for playback (default: 1.0)
        """
        super().__init__(config)
        self.file_path = config.get('file_path')
        self.loop = config.get('loop', False)
        self.playback_speed = config.get('playback_speed', 1.0)
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0
        
        if not self.file_path:
            logger.error("No file path provided for VideoFileCamera")
            raise ValueError("file_path is required for VideoFileCamera")
        
        # Resolve path relative to project root
        if not os.path.isabs(self.file_path):
            # Get project root (two levels up from src/camera)
            project_root = Path(__file__).parent.parent.parent
            self.file_path = str(project_root / self.file_path)
            
        if not os.path.exists(self.file_path):
            logger.error(f"Video file not found: {self.file_path}")
            raise FileNotFoundError(f"Video file not found: {self.file_path}")
            
        # Check if file extension is supported
        _, ext = os.path.splitext(self.file_path)
        if ext.lower() not in ['.mp4', '.mpg', '.mpeg']:
            logger.warning(f"File extension {ext} might not be supported. Supported formats are: .mp4, .mpg, .mpeg")
    
    def _open_video(self):
        """
        Open the video file
        
        Returns:
            bool: True if video was opened successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.file_path)
            
            # Check if video is open
            if not self.cap.isOpened():
                logger.error(f"Failed to open video file: {self.file_path}")
                return False
            
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.resolution = [width, height]
            
            logger.info(f"Video file opened: {self.file_path}")
            logger.info(f"Video properties: {width}x{height}, {self.fps} FPS, {self.total_frames} frames")
            logger.info(f"Playback settings: speed={self.playback_speed}, loop={self.loop}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error opening video file {self.file_path}: {e}")
            return False
    
    def _close_video(self):
        """Close the video file and release resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info(f"Video file closed: {self.file_path}")
    
    def _capture_loop(self):
        """
        Main capture loop for video file camera
        
        This method continuously reads frames from the video file and updates the frame buffer.
        It handles looping the video if configured to do so.
        """
        if not self._open_video():
            logger.error("Failed to open video file, exiting capture loop")
            return
        
        consecutive_failures = 0
        max_failures = 10  # Maximum consecutive failures before giving up
        
        while self.is_running:
            try:
                # Read frame from video
                ret, frame = self.cap.read()
                
                # Handle successful frame read
                if ret and frame is not None:
                    consecutive_failures = 0  # Reset failure counter
                    self.current_frame += 1
                    
                    # Preprocess frame
                    processed_frame = self._preprocess_frame(frame)
                    if processed_frame is not None:
                        self._set_frame(processed_frame)
                    
                    # Control playback speed
                    if self.fps > 0:
                        delay = 1.0 / (self.fps * self.playback_speed)
                        time.sleep(max(0.016, delay))  # Minimum 16ms delay (~60fps max)
                    else:
                        time.sleep(0.033)  # Default 30fps if fps is invalid
                else:
                    # Handle end of video or read failure
                    if self.loop and self.current_frame > 0:  # Only loop if we've actually read some frames
                        logger.info(f"End of video reached: {self.file_path}")
                        logger.info("Looping video from beginning")
                        
                        # Reset to beginning with error checking
                        success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        if success:
                            self.current_frame = 0
                            consecutive_failures = 0  # Reset failure counter on successful loop
                            # Add a small delay to prevent tight looping
                            time.sleep(0.1)
                            continue
                        else:
                            logger.warning("Failed to seek to beginning, attempting to restart video file")
                            if self._restart_video():
                                consecutive_failures = 0  # Reset failure counter on successful restart
                                time.sleep(0.1)
                                continue
                            else:
                                logger.error("Failed to restart video file for looping")
                                consecutive_failures += 1
                    else:
                        consecutive_failures += 1
                        
                        if not self.loop:
                            logger.info("Video playback complete")
                            break
                        elif self.current_frame == 0:
                            logger.error("Failed to read any frames from video file")
                            break
                    
                    if consecutive_failures > max_failures:
                        logger.error(f"Too many consecutive failures ({max_failures}), stopping camera")
                        break
                    
                    # Add delay for failed reads to prevent tight loop
                    time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error reading frame from video file: {e}")
                consecutive_failures += 1
                if consecutive_failures > max_failures:
                    logger.error(f"Too many consecutive errors ({max_failures}), stopping camera")
                    break
                time.sleep(0.1)  # Small delay before retrying
        
        # Clean up when loop exits
        self._close_video()
    
    def get_progress(self):
        """
        Get current playback progress
        
        Returns:
            float: Progress as a percentage (0-100)
        """
        if self.total_frames > 0:
            return (self.current_frame / self.total_frames) * 100
        return 0
    
    def seek(self, position_percent):
        """
        Seek to a specific position in the video
        
        Args:
            position_percent (float): Position as a percentage (0-100)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.cap is None or not self.cap.isOpened():
            return False
            
        if position_percent < 0 or position_percent > 100:
            logger.error(f"Invalid seek position: {position_percent}%")
            return False
            
        frame_number = int((position_percent / 100) * self.total_frames)
        success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        if success:
            self.current_frame = frame_number
            logger.info(f"Seeked to position {position_percent:.1f}% (frame {frame_number})")
        else:
            logger.error(f"Failed to seek to position {position_percent:.1f}%")
        
        return success
    
    def stop(self):
        """Stop the camera capture thread and release resources"""
        super().stop()
        self._close_video()

    def _restart_video(self):
        """
        Restart the video from the beginning by re-opening the file
        
        Returns:
            bool: True if restart was successful, False otherwise
        """
        try:
            logger.info("Restarting video file from beginning")
            self._close_video()
            if self._open_video():
                self.current_frame = 0
                return True
            else:
                logger.error("Failed to restart video file")
                return False
        except Exception as e:
            logger.error(f"Error restarting video file: {e}")
            return False

# For backward compatibility
MP4Camera = VideoFileCamera 