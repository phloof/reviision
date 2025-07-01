"""
Configuration manager for ReViision Pi Test Bench
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import os

class ConfigManager:
    """Manages configuration loading and updates"""
    
    def __init__(self, config_path: str = "config/pi_config.yaml"):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger('reviision_pi.config_manager')
        self._config = None
        self._config_mtime = 0
    
    def load_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            force_reload: Force reload even if file hasn't changed
            
        Returns:
            Configuration dictionary
        """
        try:
            # Check if file has been modified
            if self.config_path.exists():
                current_mtime = self.config_path.stat().st_mtime
                
                if not force_reload and self._config and current_mtime == self._config_mtime:
                    return self._config
                
                with open(self.config_path, 'r') as f:
                    self._config = yaml.safe_load(f)
                
                self._config_mtime = current_mtime
                self.logger.info(f"Configuration loaded from {self.config_path}")
                
            else:
                self.logger.warning(f"Configuration file not found: {self.config_path}")
                self._config = self._get_default_config()
            
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            return self._config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            if self._config:
                return self._config
            return self._get_default_config()
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        Save configuration to YAML file
        
        Args:
            config: Configuration dictionary to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create config directory if it doesn't exist
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup of existing config
            if self.config_path.exists():
                backup_path = self.config_path.with_suffix('.yaml.backup')
                backup_path.write_text(self.config_path.read_text())
            
            # Write new config
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            self._config = config
            self._config_mtime = self.config_path.stat().st_mtime
            
            self.logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary of updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config = self.get_config()
            self._deep_update(config, updates)
            return self.save_config(config)
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return False
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Recursively update nested dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        if not self._config:
            return
        
        # Server host override
        if 'REVIISION_SERVER_HOST' in os.environ:
            if 'network' not in self._config:
                self._config['network'] = {}
            if 'server' not in self._config['network']:
                self._config['network']['server'] = {}
            self._config['network']['server']['host'] = os.environ['REVIISION_SERVER_HOST']
        
        # Server port override
        if 'REVIISION_SERVER_PORT' in os.environ:
            if 'network' not in self._config:
                self._config['network'] = {}
            if 'server' not in self._config['network']:
                self._config['network']['server'] = {}
            self._config['network']['server']['port'] = int(os.environ['REVIISION_SERVER_PORT'])
        
        # WiFi credentials override
        if 'WIFI_SSID' in os.environ:
            if 'network' not in self._config:
                self._config['network'] = {}
            if 'hotspot' not in self._config['network']:
                self._config['network']['hotspot'] = {}
            self._config['network']['hotspot']['ssid'] = os.environ['WIFI_SSID']
        
        if 'WIFI_PASSWORD' in os.environ:
            if 'network' not in self._config:
                self._config['network'] = {}
            if 'hotspot' not in self._config['network']:
                self._config['network']['hotspot'] = {}
            self._config['network']['hotspot']['password'] = os.environ['WIFI_PASSWORD']
        
        # Logging level override
        if 'LOG_LEVEL' in os.environ:
            if 'logging' not in self._config:
                self._config['logging'] = {}
            self._config['logging']['level'] = os.environ['LOG_LEVEL']
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'network': {
                'hotspot': {
                    'ssid': 'ReViision-TestBench',
                    'password': 'testbench2024',
                    'interface': 'wlan0',
                    'ip_range': '192.168.4.0/24',
                    'gateway': '192.168.4.1',
                    'dhcp_range': '192.168.4.2,192.168.4.20'
                },
                'server': {
                    'host': '192.168.4.10',
                    'port': 5000,
                    'protocol': 'http',
                    'username': 'admin',
                    'password': 'admin',
                    'timeout': 10,
                    'retry_attempts': 3,
                    'retry_delay': 5
                }
            },
            'display': {
                'type': 'epd4in26',
                'width': 800,
                'height': 480,
                'refresh_interval': 30,
                'full_refresh_interval': 300,
                'gpio': {
                    'din': 10,
                    'clk': 11,
                    'cs': 8,
                    'dc': 25,
                    'rst': 17,
                    'busy': 24
                }
            },
            'data': {
                'polling': {
                    'analytics': 15,
                    'system_status': 60,
                    'network_status': 120
                },
                'cache': {
                    'max_age': 3600,
                    'max_entries': 1000
                }
            },
            'logging': {
                'level': 'INFO',
                'log_file': 'logs/pi_testbench.log',
                'max_size': '10MB',
                'backup_count': 5
            },
            'system': {
                'auto_start': True,
                'startup_delay': 30,
                'health_check': {
                    'enabled': True,
                    'interval': 300
                },
                'max_cpu_usage': 80,
                'max_memory_usage': 80,
                'temperature': {
                    'warning_threshold': 70,
                    'critical_threshold': 80
                }
            }
        }
    
    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> tuple[bool, list]:
        """
        Validate configuration
        
        Args:
            config: Configuration to validate (uses current if None)
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if config is None:
            config = self.get_config()
        
        errors = []
        
        # Validate network configuration
        if 'network' not in config:
            errors.append("Missing 'network' section")
        else:
            network = config['network']
            
            if 'hotspot' not in network:
                errors.append("Missing 'network.hotspot' section")
            else:
                hotspot = network['hotspot']
                required_hotspot = ['ssid', 'password', 'interface']
                for field in required_hotspot:
                    if field not in hotspot:
                        errors.append(f"Missing 'network.hotspot.{field}'")
            
            if 'server' not in network:
                errors.append("Missing 'network.server' section")
            else:
                server = network['server']
                required_server = ['host', 'port']
                for field in required_server:
                    if field not in server:
                        errors.append(f"Missing 'network.server.{field}'")
        
        # Validate display configuration
        if 'display' not in config:
            errors.append("Missing 'display' section")
        else:
            display = config['display']
            required_display = ['width', 'height', 'gpio']
            for field in required_display:
                if field not in display:
                    errors.append(f"Missing 'display.{field}'")
        
        return len(errors) == 0, errors 