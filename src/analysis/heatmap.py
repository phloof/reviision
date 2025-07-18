"""
Heatmap Generator class for Retail Analytics System
"""

import logging
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class HeatmapGenerator:
    """
    Heatmap generator that visualizes customer density and activity
    
    This class generates various types of heatmaps to visualize customer
    movement patterns, dwell times, and zone popularity in the retail space.
    """
    
    def __init__(self, config):
        """
        Initialize the heatmap generator with the provided configuration
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        
        # Heatmap parameters
        self.resolution = config.get('resolution', (640, 480))
        self.alpha = config.get('alpha', 0.6)  # Transparency of heatmap overlay
        self.blur_radius = config.get('blur_radius', 15)  # Gaussian blur radius
        self.point_decay = config.get('point_decay', 0.99)  # Decay factor for accumulated points
        self.max_accumulate_frames = config.get('max_accumulate_frames', 300)  # Maximum frames to accumulate
        
        # Enhanced opacity controls
        self.dynamic_opacity = config.get('dynamic_opacity', True)
        self.opacity_presets = config.get('opacity_presets', {
            'subtle': 0.3,
            'moderate': 0.6,
            'strong': 0.9
        })
        self.current_opacity_preset = config.get('default_opacity_preset', 'moderate')
        
        # Load colormap configuration
        self.colormap_config = self._load_colormap_config()
        self.colormap = self.colormap_config.get('default_colormap', 'jet')  # Default from config or fallback to jet
        
        # Store type-specific colormaps
        self.position_colormap = self._get_type_colormap('position')
        self.dwell_colormap = self._get_type_colormap('dwell')
        self.zone_colormap = self._get_type_colormap('zone')
        self.comparison_colormap = self._get_type_colormap('comparison')
        
        # Initialize accumulation maps
        self.position_map = None
        self.dwell_map = None
        self.reset_maps()
        
        # Counter for frame accumulation
        self.frame_count = 0
        
        logger.info(f"Heatmap generator initialized with resolution {self.resolution}")
    
    def _load_colormap_config(self):
        """Load colormap configuration from JSON file"""
        try:
            # Try to find the config file in expected locations
            config_paths = [
                Path(__file__).parent.parent / 'web' / 'static' / 'colormap_config.json',  # From source dir
                Path('/app/src/web/static/colormap_config.json'),  # For containerized app
                Path('src/web/static/colormap_config.json')  # Relative to working dir
            ]
            
            for path in config_paths:
                if path.exists():
                    with open(path, 'r') as f:
                        return json.load(f)
            
            logger.warning("Colormap configuration file not found, using defaults")
            return {
                "default_colormap": "jet",
                "colormaps": {
                    "position": {"name": "hot"},
                    "dwell": {"name": "plasma"},
                    "zone": {"name": "viridis"},
                    "comparison": {"name": "coolwarm"}
                }
            }
        except Exception as e:
            logger.error(f"Error loading colormap configuration: {e}")
            return {"default_colormap": "jet", "colormaps": {}}
    
    def _get_type_colormap(self, heatmap_type):
        """Get colormap name for a specific heatmap type"""
        if not self.colormap_config or 'colormaps' not in self.colormap_config:
            return self.colormap
            
        type_config = self.colormap_config['colormaps'].get(heatmap_type, {})
        return type_config.get('name', self.colormap)
    
    def reset_maps(self):
        """Reset accumulated heatmap data"""
        self.position_map = np.zeros(self.resolution, dtype=np.float32)
        self.dwell_map = np.zeros(self.resolution, dtype=np.float32)
        self.zone_activity = defaultdict(int)
        self.frame_count = 0
        logger.debug("Heatmap data reset")
    
    def update_position_map(self, tracks, frame_shape=None):
        """
        Update position heatmap with current tracking data
        
        Args:
            tracks (list): List of person tracks with IDs and bounding boxes
            frame_shape (tuple, optional): Frame dimensions (height, width)
            
        Returns:
            numpy.ndarray: Updated position heatmap
        """
        if not tracks:
            return self.position_map
        
        # Update resolution if needed and reset maps
        if frame_shape and frame_shape[:2] != self.resolution:
            self.resolution = frame_shape[:2]
            self.reset_maps()
        
        # Apply decay to existing map
        self.position_map *= self.point_decay
        
        # Increment frame counter and reset if needed
        self.frame_count += 1
        if self.frame_count > self.max_accumulate_frames:
            self.reset_maps()
        
        # Create temporary map for current positions
        height, width = self.resolution
        temp_map = np.zeros((height, width), dtype=np.float32)
        
        # Add positions to temporary map
        for track in tracks:
            bbox = track['bbox']
            
            # Get foot position (bottom center of bounding box)
            x = (bbox[0] + bbox[2]) // 2
            y = bbox[3]
            
            # Ensure coordinates are within bounds
            if 0 <= x < width and 0 <= y < height:
                # Add point with intensity based on confidence
                confidence = track.get('confidence', 1.0)
                intensity = max(0.5, min(1.0, confidence))
                
                # Add to temporary map
                cv2.circle(temp_map, (x, y), 10, intensity, -1)
        
        # Apply Gaussian blur to smooth the temporary map
        temp_map = cv2.GaussianBlur(temp_map, (self.blur_radius, self.blur_radius), 0)
        
        # Add temporary map to accumulated map
        self.position_map += temp_map
        
        # Normalize the map to prevent overflow
        max_val = np.max(self.position_map)
        if max_val > 0:
            self.position_map = np.clip(self.position_map / max_val, 0, 1)
        
        return self.position_map
    
    def update_dwell_map(self, dwell_data):
        """
        Update dwell time heatmap with current dwell data
        
        Args:
            dwell_data (list): List of dwell points with position and intensity
            
        Returns:
            numpy.ndarray: Updated dwell time heatmap
        """
        if not dwell_data:
            return self.dwell_map
        
        # Create temporary map for current dwell points
        height, width = self.resolution
        temp_map = np.zeros((height, width), dtype=np.float32)
        
        # Add dwell points to temporary map
        for point in dwell_data:
            position = point['position']
            intensity = point['intensity']
            x, y = int(position[0]), int(position[1])
            
            # Ensure coordinates are within bounds
            if 0 <= x < width and 0 <= y < height:
                # Make the radius proportional to dwell time up to a maximum
                radius = min(30, 5 + int(intensity / 3))
                weight = min(1.0, intensity / 60.0)  # Normalize to 1.0 at 60 seconds
                
                # Add to temporary map
                cv2.circle(temp_map, (x, y), radius, weight, -1)
        
        # Apply Gaussian blur to smooth the temporary map
        temp_map = cv2.GaussianBlur(temp_map, (self.blur_radius, self.blur_radius), 0)
        
        # Accumulate dwell map with less decay
        self.dwell_map = self.dwell_map * 0.95 + temp_map
        
        # Normalize the map to prevent overflow
        max_val = np.max(self.dwell_map)
        if max_val > 0:
            self.dwell_map = np.clip(self.dwell_map / max_val, 0, 1)
        
        return self.dwell_map
    
    def update_zone_activity(self, zones, zone_stats):
        """
        Update zone activity data for zone-based heatmap
        
        Args:
            zones (dict): Dictionary of zone definitions
            zone_stats (dict): Dictionary of zone statistics
        """
        # Update zone activity counts
        for zone_id, stats in zone_stats.items():
            if zone_id not in zones:
                continue
                
            # Use active visitors or total dwells as activity metric
            activity = stats.get('active_visitors', 0)
            if activity == 0:
                activity = stats.get('total_dwells', 0)
            
            self.zone_activity[zone_id] = activity
    
    def generate_position_heatmap(self, frame):
        """
        Generate position heatmap overlay on frame
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Frame with position heatmap overlay
        """
        if frame is None or np.max(self.position_map) == 0:
            return frame
        
        return self._apply_heatmap_to_frame(frame, self.position_map, self.position_colormap)
    
    def generate_dwell_heatmap(self, frame):
        """
        Generate dwell time heatmap overlay on frame
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Frame with dwell time heatmap overlay
        """
        if frame is None or np.max(self.dwell_map) == 0:
            return frame
        
        return self._apply_heatmap_to_frame(frame, self.dwell_map, self.dwell_colormap)
    
    def generate_zone_heatmap(self, frame, zones):
        """
        Generate zone activity heatmap overlay on frame
        
        Args:
            frame (numpy.ndarray): Input frame
            zones (dict): Dictionary of zone definitions
            
        Returns:
            numpy.ndarray: Frame with zone activity heatmap overlay
        """
        if frame is None or not zones or not self.zone_activity:
            return frame
        
        # Create a copy of the frame
        output_frame = frame.copy()
        
        # Get max activity for normalization
        max_activity = max(self.zone_activity.values()) if self.zone_activity else 1
        
        # Draw each zone with color based on activity
        for zone_id, zone_info in zones.items():
            if 'bounds' not in zone_info:
                continue
                
            x1, y1, x2, y2 = zone_info['bounds']
            
            # Get normalized activity (0-1)
            activity = self.zone_activity.get(zone_id, 0)
            normalized_activity = activity / max_activity if max_activity > 0 else 0
            
            # Convert to color using zone colormap
            cmap = plt.get_cmap(self.zone_colormap)
            color_rgba = cmap(normalized_activity)
            color_bgr = (
                int(color_rgba[2] * 255),
                int(color_rgba[1] * 255),
                int(color_rgba[0] * 255)
            )
            
            # Create overlay for this zone
            overlay = output_frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
            
            # Apply overlay with transparency
            cv2.addWeighted(overlay, self.alpha, output_frame, 1 - self.alpha, 0, output_frame)
            
            # Draw zone boundary
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color_bgr, 2)
            
            # Draw zone name and activity
            zone_name = zone_info.get('name', zone_id)
            label = f"{zone_name}: {activity}"
            cv2.putText(output_frame, label, (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output_frame
    
    def _apply_heatmap_to_frame(self, frame, heatmap, colormap_name=None):
        """
        Apply heatmap overlay to frame
        
        Args:
            frame (numpy.ndarray): Input frame
            heatmap (numpy.ndarray): Heatmap data
            colormap_name (str, optional): Name of colormap to use, default is self.colormap
            
        Returns:
            numpy.ndarray: Frame with heatmap overlay
        """
        # Resize heatmap to match frame dimensions if needed
        frame_h, frame_w = frame.shape[:2]
        if heatmap.shape[:2] != (frame_h, frame_w):
            heatmap = cv2.resize(heatmap, (frame_w, frame_h))
        
        # Create a copy of the frame
        output_frame = frame.copy()
        
        # Apply colormap to convert heatmap to BGR
        cmap = plt.get_cmap(colormap_name or self.colormap)
        heatmap_rgb = cmap(heatmap)
        heatmap_rgb = np.delete(heatmap_rgb, 3, 2)  # Remove alpha channel
        heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)
        heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
        
        # Create mask for non-zero regions
        mask = (heatmap > 0.05).astype(np.uint8)
        mask_3channel = cv2.merge([mask, mask, mask])
        
        # Apply heatmap only to masked regions with transparency
        try:
            # Create blended image
            blended = cv2.addWeighted(heatmap_bgr, self.alpha,
                                       output_frame, 1 - self.alpha, 0)
            # Apply mask manually
            output_frame[mask == 1] = blended[mask == 1]
        except Exception as e:
            # Fallback without mask if OpenCV version mismatches
            output_frame = cv2.addWeighted(heatmap_bgr, self.alpha,
                                           output_frame, 1 - self.alpha, 0)
 
        return output_frame
    
    def generate_comparison_heatmap(self, frame):
        """
        Generate a composite heatmap showing both position and dwell time
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            numpy.ndarray: Frame with composite heatmap overlay
        """
        if frame is None:
            return frame
        
        # Use position map for blue channel, dwell map for red channel
        height, width = frame.shape[:2]
        
        # Resize heatmaps if needed
        position_map = cv2.resize(self.position_map, (width, height)) if self.position_map.shape[:2] != (height, width) else self.position_map
        dwell_map = cv2.resize(self.dwell_map, (width, height)) if self.dwell_map.shape[:2] != (height, width) else self.dwell_map
        
        # Create RGB channels
        blue_channel = position_map
        red_channel = dwell_map
        green_channel = np.zeros_like(blue_channel)
        
        # Create RGB heatmap
        heatmap = np.stack([red_channel, green_channel, blue_channel], axis=2)
        
        # Create a mask for non-zero regions
        mask = ((red_channel > 0.05) | (blue_channel > 0.05)).astype(np.float32)
        mask = np.stack([mask, mask, mask], axis=2)
        
        # Apply to frame
        output_frame = frame.copy()
        heatmap_bgr = (heatmap * 255).astype(np.uint8)
        
        # Get intensity scale from colormap config
        intensity_scale = 0.8
        if self.colormap_config and 'colormaps' in self.colormap_config:
            comp_config = self.colormap_config['colormaps'].get('comparison', {})
            intensity_scale = comp_config.get('intensity_scale', 0.8)
        
        # Blend with frame using potentially customized alpha
        cv2.addWeighted(
            heatmap_bgr, self.alpha * intensity_scale,
            output_frame, 1 - (self.alpha * intensity_scale),
            0, output_frame, mask
        )
        
        return output_frame
    
    def get_position_heatmap_data(self):
        """
        Get raw position heatmap data
        
        Returns:
            numpy.ndarray: Position heatmap data
        """
        return self.position_map.copy()
    
    def get_dwell_heatmap_data(self):
        """
        Get raw dwell heatmap data
        
        Returns:
            numpy.ndarray: Dwell time heatmap data
        """
        return self.dwell_map.copy()
    
    def get_zone_activity_data(self):
        """
        Get zone activity data
        
        Returns:
            dict: Dictionary of zone ID to activity level
        """
        return dict(self.zone_activity)
    
    def get_available_colormaps(self):
        """
        Get list of available colormaps
        
        Returns:
            list: List of colormap information dictionaries
        """
        if not self.colormap_config or 'available_colormaps' not in self.colormap_config:
            # Return basic list of standard matplotlib colormaps
            return [
                {"name": "jet", "category": "sequential"},
                {"name": "viridis", "category": "perceptual"},
                {"name": "plasma", "category": "perceptual"},
                {"name": "hot", "category": "sequential"}
            ]
        
        return self.colormap_config['available_colormaps']
    
    def set_opacity(self, opacity_value):
        """
        Set heatmap opacity dynamically
        
        Args:
            opacity_value (float): Opacity value between 0.0 and 1.0
        """
        if 0.0 <= opacity_value <= 1.0:
            self.alpha = opacity_value
            logger.debug(f"Heatmap opacity set to {opacity_value}")
        else:
            logger.warning(f"Invalid opacity value: {opacity_value}. Must be between 0.0 and 1.0")
    
    def set_opacity_preset(self, preset_name):
        """
        Set opacity using a preset name
        
        Args:
            preset_name (str): Preset name ('subtle', 'moderate', 'strong')
        """
        if preset_name in self.opacity_presets:
            self.alpha = self.opacity_presets[preset_name]
            self.current_opacity_preset = preset_name
            logger.debug(f"Heatmap opacity preset set to '{preset_name}' ({self.alpha})")
        else:
            logger.warning(f"Unknown opacity preset: {preset_name}")
    
    def get_opacity_settings(self):
        """
        Get current opacity settings and available presets
        
        Returns:
            dict: Opacity configuration
        """
        return {
            'current_opacity': self.alpha,
            'current_preset': self.current_opacity_preset,
            'available_presets': self.opacity_presets,
            'dynamic_opacity_enabled': self.dynamic_opacity
        }
    
    def generate_path_heatmap(self, frame, path_data):
        """
        Generate path overlay heatmap
        
        Args:
            frame (numpy.ndarray): Input frame
            path_data (list): List of path points with coordinates and metadata
            
        Returns:
            numpy.ndarray: Frame with path heatmap overlay
        """
        if frame is None or not path_data:
            return frame
        
        # Create path visualization
        height, width = frame.shape[:2]
        path_map = np.zeros((height, width), dtype=np.float32)
        
        # Draw paths
        for path_point in path_data:
            x = int(path_point.get('x', 0))
            y = int(path_point.get('y', 0))
            intensity = path_point.get('confidence', 0.5)
            
            # Ensure coordinates are within bounds
            if 0 <= x < width and 0 <= y < height:
                # Add path point with trail effect
                cv2.circle(path_map, (x, y), 3, intensity, -1)
        
        # Apply Gaussian blur for smooth trails
        if np.max(path_map) > 0:
            path_map = cv2.GaussianBlur(path_map, (9, 9), 0)
        
        # Get path-specific opacity
        path_opacity = self.alpha
        if self.colormap_config and 'colormaps' in self.colormap_config:
            path_config = self.colormap_config['colormaps'].get('paths', {})
            path_opacity = path_config.get('default_opacity', self.alpha)
        
        # Apply path heatmap to frame
        return self._apply_heatmap_to_frame(frame, path_map, 'viridis') 