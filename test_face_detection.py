#!/usr/bin/env python3
"""
Test script for face detection pipeline
"""

import os
import sys
import cv2
import yaml
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the new face detection modules
from detection import get_detector, get_tracker, count_people_from_detections
from analysis import DemographicAnalyzer

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config():
    """Load configuration"""
    config_path = Path('src/config.yaml')
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_face_detection_pipeline():
    """Test the face detection pipeline"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config()
        logger.info(f"Loaded config with mode: {config.get('mode', 'not set')}")
        
        # Initialize detector and tracker
        detection_config = config['detection'].copy()
        tracking_config = config['tracking'].copy()
        
        # Set mode
        mode = config.get('mode', 'face')
        detection_config['mode'] = mode
        tracking_config['mode'] = mode
        
        # Set model directory for face detection
        if mode == 'face':
            detection_config['model_dir'] = str(Path(__file__).parent / 'models')
        
        logger.info(f"Initializing {mode} detection pipeline...")
        
        # Get detector and tracker
        detector = get_detector(detection_config)
        tracker = get_tracker(tracking_config)
        
        logger.info("✓ Detector and tracker initialized successfully")
        
        # Initialize demographic analyzer
        demo_analyzer = DemographicAnalyzer(config['analysis']['demographics'])
        logger.info("✓ Demographic analyzer initialized successfully")
        
        # Test with a sample image (if available)
        test_images = [
            'testData/asianstoremulti.mp4',  # Can extract frame from video
            'test_image.jpg',
            'sample.png'
        ]
        
        test_image_path = None
        for img_path in test_images:
            if os.path.exists(img_path):
                test_image_path = img_path
                break
        
        if test_image_path and test_image_path.endswith('.mp4'):
            # Extract frame from video
            cap = cv2.VideoCapture(test_image_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                logger.info(f"Testing with frame from video: {test_image_path}")
                
                # Run detection
                detections = detector.detect(frame)
                logger.info(f"✓ Detected {len(detections)} {mode}s")
                
                # Count people
                people_count = count_people_from_detections(detections, mode)
                logger.info(f"✓ People count: {people_count}")
                
                # Run tracking
                tracks = tracker.update(detections, frame)
                logger.info(f"✓ Active tracks: {len(tracks)}")
                
                # Test demographic analysis
                demographics = demo_analyzer.process(frame, tracks)
                logger.info(f"✓ Demographics analyzed for {len(demographics)} people")
                
                # Visualize results
                vis_frame = detector.visualize_detections(frame, detections)
                vis_frame = demo_analyzer.visualize_demographics(vis_frame, tracks)
                
                # Save result
                output_path = f'test_output_{mode}_detection.jpg'
                cv2.imwrite(output_path, vis_frame)
                logger.info(f"✓ Visualization saved to: {output_path}")
                
            else:
                logger.warning("Could not read frame from video")
        
        logger.info(f"✓ Face detection pipeline test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Face detection pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_face_detection_pipeline()
    sys.exit(0 if success else 1) 