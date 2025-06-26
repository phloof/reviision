"""
Person Detector class using YOLOv8 for Retail Analytics System
"""

import os
import time
import logging
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path

logger = logging.getLogger(__name__)

class PersonDetector:
    """
    Person detector class using YOLOv8 model
    
    This class handles the detection of persons in frames using the YOLOv8 model.
    It provides methods for detecting persons and extracting bounding boxes.
    """
    
    def __init__(self, config):
        """
        Initialize the person detector with the provided configuration
        
        Args:
            config (dict): Detector configuration dictionary
        """
        self.config = config
        self.model_path = config.get('model_path', 'models/yolov8n.pt')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.device = config.get('device', 'cpu')  # 'cpu', 'cuda', 'cuda:0', etc.
        self.nms_threshold = config.get('nms_threshold', 0.45)
        self.image_size = config.get('image_size', 640)
        
        # Person class ID in COCO dataset is 0
        self.person_class_id = 0
        
        # Performance settings
        self.detection_interval = config.get('detection_interval', 1)  # Run detection every N frames
        self.frame_count = 0
        
        # Initialize the model
        self._load_model()
        
        logger.info(f"Person detector initialized with confidence threshold {self.confidence_threshold}")
    
    def _load_model(self):
        """
        Load the YOLOv8 model
        
        Raises:
            FileNotFoundError: If the model file does not exist
            RuntimeError: If the model fails to load
        """
        model_path = Path(self.model_path)
        if not model_path.exists():
            error_msg = f"YOLOv8 model not found at {model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLOv8 model loaded from {model_path}")
        except Exception as e:
            error_msg = f"Failed to load YOLOv8 model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def detect(self, frame):
        """
        Detect persons in the given frame
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            list: List of detected persons with bounding boxes, confidence scores
                 Each item is a dict with: 'bbox', 'confidence'
        """
        if frame is None:
            return []
        
        # Skip frames if detection interval is set
        self.frame_count += 1
        if self.detection_interval > 1 and (self.frame_count % self.detection_interval) != 0:
            return []
        
        try:
            start_time = time.time()
            
            # Run inference
            results = self.model(frame, 
                                 conf=self.confidence_threshold, 
                                 iou=self.nms_threshold,
                                 classes=[self.person_class_id],  # Only detect persons
                                 device=self.device,
                                 imgsz=self.image_size)
            
            # Extract and format detections
            detections = []
            if results and len(results) > 0:
                # Extract boxes, process first result only (single image inference)
                result = results[0]
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence score
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Format detection
                    detection = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(confidence)
                    }
                    detections.append(detection)
            
            elapsed_time = time.time() - start_time
            if detections:
                logger.debug(f"Detected {len(detections)} persons in {elapsed_time:.4f} seconds")
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during person detection: {e}")
            return []
    
    def visualize_detections(self, frame, detections):
        """
        Draw bounding boxes around detected persons
        
        Args:
            frame (numpy.ndarray): Input frame
            detections (list): List of detection dictionaries
            
        Returns:
            numpy.ndarray: Frame with bounding boxes drawn
        """
        if frame is None or not detections:
            return frame
        
        # Create a copy of the frame
        output_frame = frame.copy()
        
        # Draw bounding boxes
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Draw rectangle around person
            cv2.rectangle(output_frame, 
                          (bbox[0], bbox[1]), 
                          (bbox[2], bbox[3]), 
                          (0, 255, 0), 2)
            
            # Draw label with confidence
            label = f"Person: {confidence:.2f}"
            cv2.putText(output_frame, 
                        label, 
                        (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 255, 0), 
                        2)
        
        return output_frame 