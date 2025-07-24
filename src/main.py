 #!/usr/bin/env python3
"""
ReViision - Retail Vision + Reiterative Improvement
Main entry point for the application
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO

# Centralised logging utility
from utils.logger import setup_logging as core_setup_logging

# Add src directory to path for relative imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.config import ConfigManager
from camera import get_camera, stop_camera
from detection import get_detector, get_tracker, PersonDetector, PersonTracker
from analysis import DemographicAnalyzer, DwellTimeAnalyzer, HeatmapGenerator
from database import get_database
from web import create_app

def setup_logging(debug: bool = False):
    """Wrapper that delegates to :pyfunc:`utils.logger.setup_logging`."""
    log_level = "DEBUG" if debug else "INFO"
    project_root = Path(__file__).parent.parent
    log_file = project_root / "reviision.log"
    core_setup_logging(level=log_level, log_file=log_file)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ReViision - Retail Vision + Reiterative Improvement')
    # Default config path in src directory
    default_config = Path(__file__).parent / 'config.yaml'
    parser.add_argument('--config', type=str, default=str(default_config),
                        help='Path to configuration file')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--setup-credentials', action='store_true',
                        help='Run interactive credential setup')
    parser.add_argument('--import-env-credentials', action='store_true',
                        help='Import credentials from environment variables')
    return parser.parse_args()

def setup_credentials(config_manager):
    """
    Run interactive credential setup
    
    Args:
        config_manager (ConfigManager): Configuration manager
        
    Returns:
        bool: True if credentials were set up successfully
    """
    logger = logging.getLogger(__name__)
    try:
        import getpass
        logger.info("=== RTSP Camera Credential Setup ===")
        
        url = input("RTSP URL: ")
        if not url:
            logger.warning("URL is required. Aborting.")
            return False
        
        username = input("Username (leave empty if not required): ")
        
        if username:
            password = getpass.getpass("Password: ")
        else:
            password = ""
        
        logger.info("Setting up credentials...")
        if config_manager.set_rtsp_credentials(url, username, password):
            logger.info("Credentials set up successfully.")
            
            # Optionally set up a config entry using these credentials
            setup_config = input("Add this camera to config.yaml? (y/n): ").lower() == 'y'
            if setup_config:
                camera_id = input("Camera ID (leave empty to use hash): ")
                if not camera_id:
                    from hashlib import md5
                    camera_id = md5(url.encode()).hexdigest()[:8]
                
                # Generate credential reference
                credential_ref = f"rtsp_{camera_id}"
                
                # Update config
                config = config_manager.get_config()
                if 'cameras' not in config:
                    config['cameras'] = {}
                
                config['cameras'][camera_id] = {
                    'type': 'rtsp',
                    'credential_ref': credential_ref,
                    'fps': 30,
                    'resolution': [1280, 720]
                }
                
                config_manager.save_config(config)
                logger.info(f"Camera '{camera_id}' added to configuration.")
            
            return True
        else:
            logger.error("Failed to set up credentials.")
            return False
            
    except KeyboardInterrupt:
        logger.info("\nSetup aborted.")
        return False
    except Exception as e:
        logger.error(f"Error in credential setup: {e}")
        return False

def download_model_if_needed(model_path):
    """Download YOLOv8 model if it doesn't exist"""
    logger = logging.getLogger(__name__)
    if not os.path.exists(model_path):
        logger.info("YOLOv8 model not found, downloading...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model = YOLO('yolov8n.pt')
        model.save(model_path)
        logger.info(f"Model saved to: {model_path}")

def load_config(config_path=None):
    """Load configuration from YAML file"""
    if config_path is None:
        # Default to src/config.yaml (local config)
        config_path = Path(__file__).parent / 'config.yaml'
    
    # Convert to Path object if string
    if isinstance(config_path, str):
        config_path = Path(config_path)
    
    # Ensure the path exists
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main function to initialize and run the ReViision system"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    if args.debug:
        logger.debug("Debug mode enabled")
    
    # Initialize config manager
    config_path = Path(args.config)
    config_dir = config_path.parent
    config_manager = ConfigManager(config_dir=str(config_dir))
    
    # Handle credential setup if requested
    if args.setup_credentials:
        success = setup_credentials(config_manager)
        return 0 if success else 1
    
    # Import credentials from environment if requested
    if args.import_env_credentials:
        count = config_manager.import_credentials_from_env()
        logger.info(f"Imported {count} credentials from environment variables")
    
    # Load configuration
    config = load_config(args.config)
    
    try:
        # Initialize database connection
        db = get_database(config['database'])
        
        # Initialize camera
        camera_config = config['camera'].copy()
        # Add credential manager to camera config for secure access
        camera_config['credential_manager'] = config_manager.get_credential_manager()
        camera = get_camera(camera_config)
        
        # Download model if needed (only for person detection mode)
        project_root = Path(__file__).parent.parent
        model_path = project_root / 'models' / 'yolov8n.pt'
        
        # Initialize detection and tracking based on mode
        detection_mode = config.get('mode', 'face')  # Default to face detection
        detection_config = config['detection'].copy()
        tracking_config = config['tracking'].copy()
        
        # Set detection mode in both configs
        detection_config['mode'] = detection_mode
        tracking_config['mode'] = detection_mode
        
        if detection_mode.lower() == 'person':
            # Only download YOLO model for person detection
            download_model_if_needed(str(model_path))
            detection_config['model_path'] = str(model_path)
        elif detection_mode.lower() == 'face':
            # Set model directory for InsightFace models
            detection_config['model_dir'] = str(project_root / 'models')
        
        # Get detector and tracker based on mode
        detector = get_detector(detection_config)
        tracker = get_tracker(tracking_config)
        
        # Initialize analysis modules
        from analysis.zone_manager import ZoneManager
        zone_manager = ZoneManager(db)
        demographic_analyzer = DemographicAnalyzer(config['analysis']['demographics'])
        dwell_time_analyzer = DwellTimeAnalyzer(config['analysis']['dwell_time'], zone_manager, db)
        heatmap_generator = HeatmapGenerator(config['analysis']['heatmap'])
        
        # Initialize new analysis modules if enabled
        path_analyzer = None
        correlation_analyzer = None
        
        if config['analysis'].get('path', {}).get('enabled', True):
            from analysis import PathAnalyzer
            path_analyzer = PathAnalyzer(config['analysis']['path'], db)
            logger.info("PathAnalyzer initialized")
        
        if config['analysis'].get('correlation', {}).get('enabled', True):
            from analysis import CorrelationAnalyzer
            correlation_analyzer = CorrelationAnalyzer(config['analysis']['correlation'], db)
            logger.info("CorrelationAnalyzer initialized")
        
        # Initialize Flask web application
        app = create_app(config, db)
        app.zone_manager = zone_manager
        
        # Start processing pipeline in background thread
        logger.info("Starting ReViision system...")
        
        # Run web server (SocketIO if available)
        if hasattr(app, 'run_with_socketio'):
            app.run_with_socketio(host=config['web']['host'],port=config['web']['port'],debug=args.debug)
        else:
            app.run(host=config['web']['host'],port=config['web']['port'],debug=args.debug)
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        stop_camera()  # Stop camera on shutdown
    except Exception as e:
        logger.error(f"Error in main application: {e}", exc_info=True)
        stop_camera()  # Stop camera on error
        return 1
    finally:
        # Ensure camera is stopped
        try:
            stop_camera()
        except Exception as e:
            logger.error(f"Error stopping camera during cleanup: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 