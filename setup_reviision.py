#!/usr/bin/env python3
"""
ReViision Centralized Setup Script
==================================

This script configures ReViision for different deployment environments.
It reads from setup_config.yaml and sets up all necessary components.

Usage:
    python setup_reviision.py [environment_name]
    
Examples:
    python setup_reviision.py                    # Interactive setup
    python setup_reviision.py pi_testbench      # Use pi_testbench config
    python setup_reviision.py production        # Use production config
"""

import os
import sys
import yaml
import subprocess
import platform
import base64
import secrets
from pathlib import Path
from typing import Dict, Any, Optional

class ReViisionSetup:
    def __init__(self):
        self.config_file = "setup_config.yaml"
        self.app_config_file = "src/config.yaml"
        self.setup_config = None
        self.environment = None
        
    def load_setup_config(self) -> bool:
        """Load the setup configuration file."""
        try:
            if not os.path.exists(self.config_file):
                print(f"❌ Setup configuration file '{self.config_file}' not found!")
                print("Please create setup_config.yaml with your environment settings.")
                return False
                
            with open(self.config_file, 'r') as f:
                self.setup_config = yaml.safe_load(f)
            
            print(f"✅ Loaded setup configuration from {self.config_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading setup configuration: {e}")
            return False
    
    def list_environments(self) -> list:
        """List available environments."""
        environments = [self.setup_config.get('environment', 'default')]
        
        if 'environments' in self.setup_config:
            environments.extend(self.setup_config['environments'].keys())
        
        return environments
    
    def select_environment(self, env_name: Optional[str] = None) -> bool:
        """Select and load environment configuration."""
        if env_name:
            if env_name == self.setup_config.get('environment'):
                self.environment = self.setup_config
                print(f"✅ Using default environment: {env_name}")
                return True
            elif 'environments' in self.setup_config and env_name in self.setup_config['environments']:
                # Merge environment-specific config with base config
                self.environment = self.merge_configs(
                    self.setup_config, 
                    self.setup_config['environments'][env_name]
                )
                self.environment['environment'] = env_name
                print(f"✅ Using environment: {env_name}")
                return True
            else:
                print(f"❌ Environment '{env_name}' not found!")
                return False
        
        # Interactive selection
        environments = self.list_environments()
        print("\n📋 Available environments:")
        for i, env in enumerate(environments, 1):
            print(f"  {i}. {env}")
        
        try:
            choice = input(f"\nSelect environment (1-{len(environments)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(environments):
                return self.select_environment(environments[idx])
            else:
                print("❌ Invalid selection!")
                return False
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Setup cancelled.")
            return False
    
    def merge_configs(self, base: dict, override: dict) -> dict:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def generate_encryption_key(self) -> str:
        """Generate a new encryption key."""
        key_bytes = secrets.token_bytes(32)
        return base64.urlsafe_b64encode(key_bytes).decode('utf-8')
    
    def setup_encryption(self) -> bool:
        """Set up encryption key."""
        print("\n🔐 Setting up encryption...")
        
        encryption_key = self.environment['app'].get('encryption_key', '')
        
        if not encryption_key:
            print("Generating new encryption key...")
            encryption_key = self.generate_encryption_key()
            print(f"Generated key: {encryption_key}")
        else:
            print("Using provided encryption key.")
        
        # Set environment variable
        try:
            if platform.system() == "Windows":
                # Windows PowerShell
                cmd = f'$env:REVIISION_KEY="{encryption_key}"'
                print(f"Setting Windows environment variable...")
                print(f"Run this command in your PowerShell: {cmd}")
                
                # Also try to set it programmatically
                os.environ['REVIISION_KEY'] = encryption_key
                
            else:
                # Linux/macOS
                os.environ['REVIISION_KEY'] = encryption_key
                print(f"Set REVIISION_KEY environment variable")
                
                # Add to bashrc/profile
                bashrc_line = f'export REVIISION_KEY="{encryption_key}"'
                print(f"Add this to your ~/.bashrc or ~/.profile: {bashrc_line}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error setting encryption key: {e}")
            return False
    
    def setup_port_forwarding(self) -> bool:
        """Set up port forwarding if enabled."""
        if not self.environment['network'].get('enable_port_forwarding', False):
            print("Port forwarding disabled, skipping...")
            return True
            
        print("\n🌐 Setting up port forwarding...")
        
        try:
            pi_ip = self.environment['network']['pi_ip']
            camera_ip = self.environment['network']['camera_ip']
            camera_port = self.environment['network']['camera_port']
            external_port = self.environment['network']['external_port']
            
            if platform.system() != "Linux":
                print("⚠️  Port forwarding setup is only supported on Linux.")
                print(f"Manual setup required:")
                print(f"  Forward port {external_port} on {pi_ip} to {camera_ip}:{camera_port}")
                return True
            
            # Create iptables rules
            commands = [
                f"sudo iptables -t nat -A PREROUTING -p tcp --dport {external_port} -j DNAT --to-destination {camera_ip}:{camera_port}",
                f"sudo iptables -A FORWARD -p tcp -d {camera_ip} --dport {camera_port} -j ACCEPT",
                "sudo iptables-save > /etc/iptables/rules.v4"
            ]
            
            for cmd in commands:
                print(f"Running: {cmd}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"⚠️  Command failed: {result.stderr}")
            
            print(f"✅ Port forwarding configured: {pi_ip}:{external_port} -> {camera_ip}:{camera_port}")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up port forwarding: {e}")
            return False
    
    def update_app_config(self) -> bool:
        """Update the application config.yaml file."""
        print("\n⚙️  Updating application configuration...")
        
        try:
            # Load existing app config
            app_config = {}
            if os.path.exists(self.app_config_file):
                with open(self.app_config_file, 'r') as f:
                    app_config = yaml.safe_load(f)
            
            # Update camera configuration
            network = self.environment['network']
            app_settings = self.environment['app']
            
            # Determine camera URL based on port forwarding
            if network.get('enable_port_forwarding', False):
                camera_host = network['pi_ip']
                camera_port = network['external_port']
            else:
                camera_host = network['camera_ip']
                camera_port = network['camera_port']
            
            camera_url = app_settings['camera_url_template'].format(
                username=network['camera_username'],
                password=network['camera_password'],
                host=camera_host,
                port=camera_port
            )
            
            # Update configuration
            app_config.update({
                'camera': {
                    'type': app_settings['camera_type'],
                    'url': camera_url,
                    'username': network['camera_username'],
                    'password': network['camera_password']
                },
                'database': {
                    'type': 'sqlite',
                    'path': app_settings['database_path']
                },
                'logging': {
                    'level': self.environment['deployment']['log_level']
                }
            })
            
            # Write updated config
            os.makedirs(os.path.dirname(self.app_config_file), exist_ok=True)
            with open(self.app_config_file, 'w') as f:
                yaml.dump(app_config, f, default_flow_style=False, indent=2)
            
            print(f"✅ Updated {self.app_config_file}")
            print(f"   Camera URL: {camera_url}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating app configuration: {e}")
            return False
    
    def setup_service(self) -> bool:
        """Set up systemd service if enabled."""
        if not self.environment['deployment'].get('create_systemd_service', False):
            print("Systemd service creation disabled, skipping...")
            return True
            
        if platform.system() != "Linux":
            print("⚠️  Systemd service setup is only supported on Linux.")
            return True
            
        print("\n🔧 Setting up systemd service...")
        
        try:
            service_name = self.environment['deployment']['service_name']
            working_dir = os.getcwd()
            
            service_content = f"""[Unit]
Description=ReViision Detection System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory={working_dir}
Environment=REVIISION_KEY={os.environ.get('REVIISION_KEY', '')}
ExecStart=/usr/bin/python3 {working_dir}/src/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
            
            service_file = f"/etc/systemd/system/{service_name}.service"
            
            # Write service file
            with open(f"{service_name}.service", 'w') as f:
                f.write(service_content)
            
            print(f"Created {service_name}.service")
            print("To install the service, run:")
            print(f"  sudo mv {service_name}.service {service_file}")
            print(f"  sudo systemctl daemon-reload")
            print(f"  sudo systemctl enable {service_name}")
            print(f"  sudo systemctl start {service_name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error setting up service: {e}")
            return False
    
    def validate_setup(self) -> bool:
        """Validate the setup configuration."""
        print("\n✅ Validating setup...")
        
        # Check required files
        required_files = [self.app_config_file, "src/main.py"]
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"❌ Required file missing: {file_path}")
                return False
        
        # Check network connectivity (basic)
        try:
            import socket
            network = self.environment['network']
            
            # Test if we can resolve the camera IP
            socket.gethostbyname(network['camera_ip'])
            print(f"✅ Camera IP {network['camera_ip']} is reachable")
            
        except Exception as e:
            print(f"⚠️  Network validation warning: {e}")
        
        print("✅ Setup validation completed")
        return True
    
    def run_setup(self, environment_name: Optional[str] = None) -> bool:
        """Run the complete setup process."""
        print("🚀 ReViision Setup Starting...")
        print("=" * 50)
        
        # Load configuration
        if not self.load_setup_config():
            return False
        
        # Select environment
        if not self.select_environment(environment_name):
            return False
        
        print(f"\n📝 Setup Summary for '{self.environment['environment']}':")
        print(f"   Pi IP: {self.environment['network']['pi_ip']}")
        print(f"   Camera IP: {self.environment['network']['camera_ip']}")
        print(f"   Port Forwarding: {self.environment['network'].get('enable_port_forwarding', False)}")
        
        # Confirm setup
        try:
            confirm = input("\nProceed with setup? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Setup cancelled.")
                return False
        except KeyboardInterrupt:
            print("\nSetup cancelled.")
            return False
        
        # Run setup steps
        steps = [
            ("Encryption", self.setup_encryption),
            ("Port Forwarding", self.setup_port_forwarding),
            ("Application Config", self.update_app_config),
            ("System Service", self.setup_service),
            ("Validation", self.validate_setup)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*20} {step_name} {'='*20}")
            if not step_func():
                print(f"❌ Setup failed at step: {step_name}")
                return False
        
        print("\n" + "="*50)
        print("🎉 ReViision setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Start the application: python src/main.py")
        print("2. Access the web interface at: http://localhost:5000")
        print("3. Check logs for any issues")
        
        return True

def main():
    setup = ReViisionSetup()
    
    # Get environment from command line args
    environment_name = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        success = setup.run_setup(environment_name)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 