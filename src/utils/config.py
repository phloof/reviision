"""
Configuration Manager for ReViision
Loads and validates configuration with support for secure credentials
"""

import os
import yaml
import json
import logging
import jsonschema
from pathlib import Path
from .credentials import CredentialManager

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Configuration manager that loads, validates and provides access to configuration
    
    This class integrates with CredentialManager to securely handle sensitive information
    like RTSP URLs, usernames, passwords, and API keys.
    """
    
    def __init__(self, config_dir='./config'):
        """
        Initialize the configuration manager
        
        Args:
            config_dir (str): Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / 'config.yaml'
        self.schema_file = self.config_dir / 'config_schema.json'
        
        # Initialize credential manager
        self.credential_manager = CredentialManager(config_dir=config_dir)
        
        # Initialize configuration
        self.config = {}
        self.default_config = {
            'camera': {
                'type': 'usb',
                'device': '/dev/video0',
                'resolution': [640, 480],
                'fps': 30
            },
            'detection': {
                'model_path': 'models/yolov8n.pt',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },
            'tracking': {
                'max_age': 30,
                'min_hits': 3,
                'iou_threshold': 0.3
            },
            'analysis': {
                'demographics': {
                    'min_face_size': 50,
                    'confidence_threshold': 0.8,
                    'detection_interval': 30
                },
                'path': {
                    'max_path_history': 100,
                    'max_path_time': 3600,
                    'min_path_points': 5
                },
                'dwell_time': {
                    'min_dwell_time': 3.0,
            
                },
                'heatmap': {
                    'resolution': [640, 480],
                    'alpha': 0.6,
                    'blur_radius': 15
                },
                'correlation': {
                    'min_data_points': 10,
                    'significance_threshold': 0.2,
                    'cache_expiry': 3600
                }
            },
            'database': {
                'type': 'sqlite',
                'path': 'data/reviision.db'
            },
            'web': {
                'host': '127.0.0.1',
                'port': 5000,
                'debug': False
            }
        }
    
    def load_config(self, config_path=None):
        """
        Load configuration from file
        
        Args:
            config_path (str, optional): Path to configuration file
            
        Returns:
            dict: Loaded configuration
        """
        if config_path:
            config_file = Path(config_path)
        else:
            config_file = self.config_file
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    self.config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {config_file}")
            else:
                logger.warning(f"Configuration file {config_file} not found, using defaults")
                self.config = self.default_config.copy()
            
            # Validate configuration
            self._validate_config()
            
            # Process credential placeholders
            self._process_credentials()
            
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
            self.config = self.default_config.copy()
            return self.config
    
    def _validate_config(self):
        """
        Validate configuration against schema
        
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if self.schema_file.exists():
                with open(self.schema_file, 'r') as f:
                    schema = json.load(f)
                
                jsonschema.validate(instance=self.config, schema=schema)
                logger.debug("Configuration validated successfully")
                return True
            else:
                logger.warning(f"Schema file {self.schema_file} not found, skipping validation")
                return False
                
        except jsonschema.exceptions.ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error validating configuration: {e}")
            return False
    
    def _process_credentials(self):
        """
        Process credential placeholders in the configuration
        
        Replaces credential placeholders in the form ${service:key} with actual values
        from the credential manager.
        """
        def process_dict(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    process_dict(v)
                elif isinstance(v, list):
                    process_list(d, k)
                elif isinstance(v, str) and v.startswith('${') and v.endswith('}'):
                    # Extract service and key
                    cred_ref = v[2:-1]
                    if ':' in cred_ref:
                        service, key = cred_ref.split(':', 1)
                        # Get credential
                        cred_value = self.credential_manager.get_credential(service, key)
                        if cred_value is not None:
                            d[k] = cred_value
                        else:
                            logger.warning(f"Credential {cred_ref} not found")
        
        def process_list(parent, key):
            for i, item in enumerate(parent[key]):
                if isinstance(item, dict):
                    process_dict(item)
                elif isinstance(item, list):
                    process_list(parent[key], i)
                elif isinstance(item, str) and item.startswith('${') and item.endswith('}'):
                    # Extract service and key
                    cred_ref = item[2:-1]
                    if ':' in cred_ref:
                        service, key = cred_ref.split(':', 1)
                        # Get credential
                        cred_value = self.credential_manager.get_credential(service, key)
                        if cred_value is not None:
                            parent[key][i] = cred_value
                        else:
                            logger.warning(f"Credential {cred_ref} not found")
        
        # Process the config dictionary
        process_dict(self.config)
    
    def get_config(self, section=None):
        """
        Get the configuration
        
        Args:
            section (str, optional): Configuration section to get
            
        Returns:
            dict: Configuration or section
        """
        if not self.config:
            self.load_config()
        
        if section:
            return self.config.get(section, {})
        return self.config
    
    def save_config(self, config=None, config_path=None):
        """
        Save configuration to file
        
        Args:
            config (dict, optional): Configuration to save
            config_path (str, optional): Path to save configuration
            
        Returns:
            bool: True if successful, False otherwise
        """
        if config:
            self.config = config
        
        if config_path:
            save_path = Path(config_path)
        else:
            save_path = self.config_file
        
        try:
            # Create parent directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get_credential_manager(self):
        """
        Get the credential manager
        
        Returns:
            CredentialManager: Credential manager instance
        """
        return self.credential_manager
    
    def set_rtsp_credentials(self, url, username, password, options=None):
        """
        Set RTSP camera credentials
        
        Args:
            url (str): RTSP URL
            username (str): RTSP username
            password (str): RTSP password
            options (dict, optional): Additional RTSP options
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Generate a unique service name based on the URL
        from hashlib import md5
        service = f"rtsp_{md5(url.encode()).hexdigest()[:8]}"
        
        # Set credentials
        success = True
        if not self.credential_manager.set_credential(service, 'url', url):
            success = False
        if not self.credential_manager.set_credential(service, 'username', username):
            success = False
        if not self.credential_manager.set_credential(service, 'password', password):
            success = False
        
        if options and isinstance(options, dict):
            if not self.credential_manager.set_credential(service, 'options', json.dumps(options)):
                success = False
        
        return success
    
    def get_rtsp_credentials(self, camera_id):
        """
        Get RTSP camera credentials
        
        Args:
            camera_id (str): Camera identifier or URL
            
        Returns:
            dict: RTSP credentials
        """
        # Try to find service by camera ID first
        service = f"rtsp_{camera_id}"
        creds = self.credential_manager.get_service_credentials(service)
        
        if not creds:
            # Try to find by URL hash
            from hashlib import md5
            service = f"rtsp_{md5(camera_id.encode()).hexdigest()[:8]}"
            creds = self.credential_manager.get_service_credentials(service)
        
        # Convert options from JSON if present
        if 'options' in creds and isinstance(creds['options'], str):
            try:
                creds['options'] = json.loads(creds['options'])
            except:
                pass
        
        return creds
    
    def import_credentials_from_env(self):
        """
        Import credentials from environment variables
        
        Returns:
            int: Number of credentials imported
        """
        return self.credential_manager.import_from_env()
    
    def get_safe_config(self):
        """
        Get a safe version of the configuration with sensitive information redacted
        
        Returns:
            dict: Safe configuration
        """
        import copy
        safe_config = copy.deepcopy(self.config)
        
        # Function to recursively redact sensitive fields
        def redact_sensitive(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    redact_sensitive(v)
                elif isinstance(v, str) and any(s in k.lower() for s in ['password', 'key', 'secret', 'token']):
                    d[k] = '********'
        
        redact_sensitive(safe_config)
        return safe_config 