#!/usr/bin/env python3
"""
ReViision Comprehensive Setup Script
====================================

This script configures ReViision for different deployment environments including:
- Main system (Desktop/Server) setup
- Pi testbench deployment
- Credential management
- Service configuration
- Environment-specific optimisation

Usage:
    python setup_reviision.py [options]

Examples:
    python setup_reviision.py                           # Interactive setup
    python setup_reviision.py --env desktop_production  # Production desktop setup
    python setup_reviision.py --env pi_testbench        # Pi testbench setup
    python setup_reviision.py --setup-credentials       # Setup credentials only
    python setup_reviision.py --validate                # Validate existing setup
"""

import os
import sys
import yaml
import subprocess
import platform
import base64
import secrets
import argparse
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from cryptography.fernet import Fernet

class ReViisionSetup:
    def __init__(self):
        self.config_file = "setup_config.yaml"
        self.main_config_file = "src/config.yaml"
        self.pi_config_file = "pi_testbench/config/pi_config.yaml"
        self.setup_config = None
        self.environment = None
        self.deployment_type = None  # 'main_system' or 'pi_testbench'

        # Paths
        self.project_root = Path.cwd()
        self.src_dir = self.project_root / "src"
        self.pi_dir = self.project_root / "pi_testbench"
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"

    def load_setup_config(self) -> bool:
        """Load the setup configuration file."""
        try:
            if not self.config_file.exists() if isinstance(self.config_file, Path) else not os.path.exists(self.config_file):
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

    def list_environments(self) -> List[str]:
        """List available environments."""
        environments = []

        # Add default environment
        default_env = self.setup_config.get('environment', 'desktop_development')
        environments.append(default_env)

        # Add environments from environments section
        if 'environments' in self.setup_config:
            environments.extend(self.setup_config['environments'].keys())

        return list(set(environments))  # Remove duplicates

    def select_environment(self, env_name: Optional[str] = None) -> bool:
        """Select and load environment configuration."""
        if env_name:
            # Check if it's the default environment
            if env_name == self.setup_config.get('environment', 'desktop_development'):
                self.environment = self.setup_config
                self.environment['environment'] = env_name
                print(f"✅ Using default environment: {env_name}")
                return self._determine_deployment_type()

            # Check if it's in environments section
            elif 'environments' in self.setup_config and env_name in self.setup_config['environments']:
                # Start with base config and merge environment-specific config
                base_config = {
                    'main_system': self.setup_config.get('main_system', {}),
                    'pi_testbench': self.setup_config.get('pi_testbench', {}),
                    'credential_templates': self.setup_config.get('credential_templates', {})
                }

                env_config = self.setup_config['environments'][env_name]
                self.environment = self.merge_configs(base_config, env_config)
                self.environment['environment'] = env_name
                print(f"✅ Using environment: {env_name}")
                return self._determine_deployment_type()
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

    def _determine_deployment_type(self) -> bool:
        """Determine if this is main system or pi testbench deployment."""
        env_name = self.environment.get('environment', '')

        # Auto-detect based on environment name
        if 'pi' in env_name.lower():
            self.deployment_type = 'pi_testbench'
        elif 'desktop' in env_name.lower() or 'production' in env_name.lower() or 'development' in env_name.lower():
            self.deployment_type = 'main_system'
        else:
            # Interactive selection
            print("\n🎯 Select deployment type:")
            print("  1. Main System (Desktop/Server)")
            print("  2. Pi Testbench")

            try:
                choice = input("Select deployment type (1-2): ").strip()
                if choice == '1':
                    self.deployment_type = 'main_system'
                elif choice == '2':
                    self.deployment_type = 'pi_testbench'
                else:
                    print("❌ Invalid selection!")
                    return False
            except (ValueError, KeyboardInterrupt):
                print("\n❌ Setup cancelled.")
                return False

        print(f"✅ Deployment type: {self.deployment_type}")
        return True

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
        return Fernet.generate_key().decode()

    def setup_directories(self) -> bool:
        """Create necessary directories."""
        print("\n📁 Setting up directories...")

        try:
            directories = [
                self.data_dir,
                self.data_dir / "faces",
                self.models_dir,
                self.src_dir / "config",
            ]

            if self.deployment_type == 'pi_testbench':
                directories.extend([
                    self.pi_dir / "logs",
                    self.pi_dir / "config",
                ])

            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created directory: {directory}")

            return True

        except Exception as e:
            print(f"❌ Error creating directories: {e}")
            return False

    def setup_encryption(self) -> bool:
        """Set up encryption key and credential management."""
        print("\n🔐 Setting up encryption and credentials...")

        try:
            config_section = self.environment.get(self.deployment_type, {}).get('app', {})
            encryption_key = config_section.get('encryption_key', '')
            encryption_passphrase = config_section.get('encryption_passphrase', '')

            if not encryption_key and not encryption_passphrase:
                print("Generating new encryption key...")
                encryption_key = self.generate_encryption_key()
                print(f"Generated key: {encryption_key[:20]}...")
            elif encryption_passphrase:
                print("Using provided encryption passphrase.")
            else:
                print("Using provided encryption key.")

            # Set environment variables
            env_vars = {}
            if encryption_key:
                env_vars['RETAIL_ANALYTICS_KEY'] = encryption_key
                os.environ['RETAIL_ANALYTICS_KEY'] = encryption_key

            if encryption_passphrase:
                env_vars['RETAIL_ANALYTICS_PASSPHRASE'] = encryption_passphrase
                os.environ['RETAIL_ANALYTICS_PASSPHRASE'] = encryption_passphrase

            # Create environment file for persistence
            env_file = self.project_root / '.env'
            env_lines = []

            if env_file.exists():
                with open(env_file, 'r') as f:
                    env_lines = f.readlines()

            # Update or add environment variables
            for key, value in env_vars.items():
                found = False
                for i, line in enumerate(env_lines):
                    if line.startswith(f"{key}="):
                        env_lines[i] = f"{key}={value}\n"
                        found = True
                        break
                if not found:
                    env_lines.append(f"{key}={value}\n")

            with open(env_file, 'w') as f:
                f.writelines(env_lines)

            print(f"✅ Environment variables saved to {env_file}")

            # Platform-specific instructions
            if platform.system() == "Windows":
                print("\n📝 Windows Setup Instructions:")
                print("Add these to your PowerShell profile or set as system environment variables:")
                for key, value in env_vars.items():
                    print(f"  $env:{key}=\"{value}\"")
            else:
                print("\n📝 Linux/macOS Setup Instructions:")
                print("Add these to your ~/.bashrc or ~/.profile:")
                for key, value in env_vars.items():
                    print(f"  export {key}=\"{value}\"")

            return True

        except Exception as e:
            print(f"❌ Error setting up encryption: {e}")
            return False

    def setup_credentials(self) -> bool:
        """Set up credential management system."""
        print("\n🔑 Setting up credential management...")

        try:
            # Import credential manager
            sys.path.insert(0, str(self.src_dir))
            from utils.credentials import CredentialManager

            # Initialize credential manager
            config_dir = self.src_dir / "config"
            credential_manager = CredentialManager(config_dir=str(config_dir))

            # Get credential templates for this environment
            templates = self.environment.get('credential_templates', {})

            if not templates:
                print("No credential templates found, skipping credential setup...")
                return True

            print(f"Found {len(templates)} credential template(s)")

            # Interactive credential setup
            for template_name, template_config in templates.items():
                print(f"\n📝 Setting up credentials for: {template_name}")

                service = template_config['service']
                keys = template_config['keys']
                example = template_config.get('example', {})

                print(f"Service: {service}")
                print(f"Required keys: {', '.join(keys)}")

                if example:
                    print("Example values:")
                    for key, value in example.items():
                        print(f"  {key}: {value}")

                # Ask user if they want to set up this credential
                try:
                    setup = input(f"\nSet up credentials for {template_name}? (y/N): ").strip().lower()
                    if setup != 'y':
                        continue
                except KeyboardInterrupt:
                    print("\nSkipping credential setup...")
                    continue

                # Collect credential values
                for key in keys:
                    try:
                        if 'password' in key.lower():
                            import getpass
                            value = getpass.getpass(f"Enter {key}: ")
                        else:
                            value = input(f"Enter {key}: ").strip()

                        if value:
                            credential_manager.set_credential(service, key, value)
                            print(f"✅ Set {key} for {service}")
                    except KeyboardInterrupt:
                        print(f"\nSkipping {key}...")
                        continue

            print("✅ Credential setup completed")
            return True

        except Exception as e:
            print(f"❌ Error setting up credentials: {e}")
            return False

    def update_main_config(self) -> bool:
        """Update the main system configuration."""
        print("\n⚙️  Updating main system configuration...")

        try:
            config_section = self.environment.get('main_system', {})
            app_config = config_section.get('app', {})

            # Load existing config or create new one
            if self.main_config_file.exists():
                with open(self.main_config_file, 'r') as f:
                    main_config = yaml.safe_load(f) or {}
            else:
                main_config = {}

            # Update camera configuration
            camera_config = app_config.get('camera', {})
            if camera_config:
                main_config['camera'] = self.merge_configs(
                    main_config.get('camera', {}),
                    camera_config
                )

            # Update database configuration
            database_config = app_config.get('database', {})
            if database_config:
                main_config['database'] = self.merge_configs(
                    main_config.get('database', {}),
                    database_config
                )

            # Update web configuration
            web_config = app_config.get('web', {})
            if web_config:
                # Generate secret key if not provided
                if not web_config.get('secret_key'):
                    web_config['secret_key'] = secrets.token_urlsafe(32)

                main_config['web'] = self.merge_configs(
                    main_config.get('web', {}),
                    web_config
                )

            # Update analysis configuration
            analysis_config = app_config.get('analysis', {})
            if analysis_config:
                main_config['analysis'] = self.merge_configs(
                    main_config.get('analysis', {}),
                    analysis_config
                )

            # Update performance settings
            deployment_config = config_section.get('deployment', {})
            performance_config = deployment_config.get('performance', {})
            if performance_config:
                main_config['performance'] = self.merge_configs(
                    main_config.get('performance', {}),
                    performance_config
                )

            # Update logging
            log_level = deployment_config.get('log_level', 'INFO')
            main_config['logging'] = main_config.get('logging', {})
            main_config['logging']['level'] = log_level

            # Write updated config
            self.main_config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.main_config_file, 'w') as f:
                yaml.dump(main_config, f, default_flow_style=False, indent=2)

            print(f"✅ Updated {self.main_config_file}")
            return True

        except Exception as e:
            print(f"❌ Error updating main system configuration: {e}")
            return False

    def update_pi_config(self) -> bool:
        """Update the Pi testbench configuration."""
        print("\n⚙️  Updating Pi testbench configuration...")

        try:
            config_section = self.environment.get('pi_testbench', {})

            # Load existing Pi config or create new one
            if self.pi_config_file.exists():
                with open(self.pi_config_file, 'r') as f:
                    pi_config = yaml.safe_load(f) or {}
            else:
                pi_config = {}

            # Update network configuration
            network_config = config_section.get('network', {})
            if network_config:
                pi_config['network'] = self.merge_configs(
                    pi_config.get('network', {}),
                    network_config
                )

            # Update data configuration
            data_config = config_section.get('data', {})
            if data_config:
                pi_config['data'] = self.merge_configs(
                    pi_config.get('data', {}),
                    data_config
                )

            # Update system configuration
            system_config = config_section.get('system', {})
            if system_config:
                pi_config['system'] = self.merge_configs(
                    pi_config.get('system', {}),
                    system_config
                )

            # Update logging
            deployment_config = config_section.get('deployment', {})
            log_level = deployment_config.get('log_level', 'INFO')
            pi_config['logging'] = pi_config.get('logging', {})
            pi_config['logging']['level'] = log_level

            # Write updated config
            self.pi_config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.pi_config_file, 'w') as f:
                yaml.dump(pi_config, f, default_flow_style=False, indent=2)

            print(f"✅ Updated {self.pi_config_file}")
            return True

        except Exception as e:
            print(f"❌ Error updating Pi testbench configuration: {e}")
            return False

    def setup_service(self) -> bool:
        """Set up systemd service if enabled."""
        config_section = self.environment.get(self.deployment_type, {})
        deployment_config = config_section.get('deployment', {})

        if not deployment_config.get('create_systemd_service', False):
            print("Systemd service creation disabled, skipping...")
            return True

        if platform.system() != "Linux":
            print("⚠️  Systemd service setup is only supported on Linux.")
            return True

        print("\n🔧 Setting up systemd service...")

        try:
            service_name = deployment_config['service_name']
            service_user = deployment_config.get('service_user', 'reviision')
            working_dir = str(self.project_root)

            # Determine the correct executable path and environment
            if self.deployment_type == 'main_system':
                exec_start = f"{working_dir}/src/main.py"
                description = "ReViision Retail Analytics System"
                env_vars = [
                    f"RETAIL_ANALYTICS_KEY={os.environ.get('RETAIL_ANALYTICS_KEY', '')}",
                    f"RETAIL_ANALYTICS_PASSPHRASE={os.environ.get('RETAIL_ANALYTICS_PASSPHRASE', '')}",
                ]
            else:  # pi_testbench
                exec_start = f"{working_dir}/pi_testbench/main.py"
                description = "ReViision Pi Testbench System"
                env_vars = [
                    f"PYTHONPATH={working_dir}/pi_testbench/src",
                    "PYTHONUNBUFFERED=1",
                ]

            # Create service content
            service_content = f"""[Unit]
Description={description}
Documentation=https://github.com/your-org/reviision
After=multi-user.target
After=network-online.target
Wants=network-online.target
Conflicts=shutdown.target
StartLimitBurst=3
StartLimitIntervalSec=60

[Service]
Type=simple
User={service_user}
Group={service_user}
WorkingDirectory={working_dir}
"""

            # Add environment variables
            for env_var in env_vars:
                if env_var.split('=')[1]:  # Only add if value is not empty
                    service_content += f"Environment={env_var}\n"

            service_content += f"""
ExecStart=/usr/bin/python3 {exec_start}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier={service_name}

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths={working_dir}

[Install]
WantedBy=multi-user.target
"""

            service_file_name = f"{service_name}.service"
            service_file_path = f"/etc/systemd/system/{service_file_name}"

            # Write service file to current directory
            with open(service_file_name, 'w') as f:
                f.write(service_content)

            print(f"✅ Created {service_file_name}")
            print("\n📋 To install the service, run these commands:")
            print(f"  sudo cp {service_file_name} {service_file_path}")
            print(f"  sudo systemctl daemon-reload")
            print(f"  sudo systemctl enable {service_name}")
            print(f"  sudo systemctl start {service_name}")
            print(f"\n📊 To check service status:")
            print(f"  sudo systemctl status {service_name}")
            print(f"  sudo journalctl -u {service_name} -f")

            return True

        except Exception as e:
            print(f"❌ Error setting up service: {e}")
            return False

    def download_models(self) -> bool:
        """Download required ML models."""
        print("\n📥 Downloading required models...")

        try:
            # Only download models for main system
            if self.deployment_type != 'main_system':
                print("Pi testbench doesn't require model downloads, skipping...")
                return True

            # Create models directory
            self.models_dir.mkdir(parents=True, exist_ok=True)

            # Download YOLOv8 model
            yolo_model_path = self.models_dir / "yolov8n.pt"
            if not yolo_model_path.exists():
                print("Downloading YOLOv8 model...")
                try:
                    from ultralytics import YOLO
                    # This will automatically download the model
                    model = YOLO('yolov8n.pt')
                    # Move to our models directory
                    import shutil
                    shutil.move('yolov8n.pt', str(yolo_model_path))
                    print(f"✅ Downloaded YOLOv8 model to {yolo_model_path}")
                except Exception as e:
                    print(f"⚠️  Could not download YOLOv8 model: {e}")
                    print("Model will be downloaded automatically on first run")
            else:
                print(f"✅ YOLOv8 model already exists at {yolo_model_path}")

            return True

        except Exception as e:
            print(f"❌ Error downloading models: {e}")
            return False

    def validate_setup(self) -> bool:
        """Validate the setup configuration."""
        print("\n✅ Validating setup...")

        try:
            # Check required files based on deployment type
            if self.deployment_type == 'main_system':
                required_files = [
                    self.src_dir / "main.py",
                    self.main_config_file,
                ]
                required_dirs = [
                    self.src_dir,
                    self.data_dir,
                ]
            else:  # pi_testbench
                required_files = [
                    self.pi_dir / "main.py",
                    self.pi_config_file,
                ]
                required_dirs = [
                    self.pi_dir / "src",
                    self.pi_dir / "config",
                ]

            # Check required files
            for file_path in required_files:
                if not file_path.exists():
                    print(f"❌ Required file missing: {file_path}")
                    return False
                else:
                    print(f"✅ Found required file: {file_path}")

            # Check required directories
            for dir_path in required_dirs:
                if not dir_path.exists():
                    print(f"❌ Required directory missing: {dir_path}")
                    return False
                else:
                    print(f"✅ Found required directory: {dir_path}")

            # Check Python dependencies
            print("Checking Python dependencies...")
            try:
                import yaml
                import cv2
                import numpy
                print("✅ Core dependencies available")

                if self.deployment_type == 'main_system':
                    import torch
                    from ultralytics import YOLO
                    print("✅ ML dependencies available")

            except ImportError as e:
                print(f"⚠️  Missing Python dependency: {e}")
                print("Run: pip install -r requirements.txt")

            # Validate configuration files
            print("Validating configuration files...")
            if self.deployment_type == 'main_system' and self.main_config_file.exists():
                with open(self.main_config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    if config:
                        print("✅ Main configuration file is valid YAML")
                    else:
                        print("⚠️  Main configuration file is empty")

            if self.deployment_type == 'pi_testbench' and self.pi_config_file.exists():
                with open(self.pi_config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    if config:
                        print("✅ Pi configuration file is valid YAML")
                    else:
                        print("⚠️  Pi configuration file is empty")

            print("✅ Setup validation completed")
            return True

        except Exception as e:
            print(f"❌ Error during validation: {e}")
            return False

    def run_setup(self, args) -> bool:
        """Run the complete setup process."""
        print("🚀 ReViision Comprehensive Setup Starting...")
        print("=" * 60)

        # Load configuration
        if not self.load_setup_config():
            return False

        # Handle special modes
        if args.setup_credentials:
            return self.setup_credentials()

        if args.validate:
            return self.validate_setup()

        # Select environment
        if not self.select_environment(args.env):
            return False

        # Display setup summary
        print(f"\n📝 Setup Summary:")
        print(f"   Environment: {self.environment['environment']}")
        print(f"   Deployment Type: {self.deployment_type}")

        config_section = self.environment.get(self.deployment_type, {})
        if self.deployment_type == 'pi_testbench':
            network_config = config_section.get('network', {})
            hotspot_config = network_config.get('hotspot', {})
            server_config = network_config.get('server', {})
            print(f"   Hotspot SSID: {hotspot_config.get('ssid', 'N/A')}")
            print(f"   Server Host: {server_config.get('host', 'N/A')}")
        else:
            app_config = config_section.get('app', {})
            camera_config = app_config.get('camera', {})
            web_config = app_config.get('web', {})
            print(f"   Camera Type: {camera_config.get('type', 'N/A')}")
            print(f"   Web Port: {web_config.get('port', 'N/A')}")

        # Confirm setup
        if not args.yes:
            try:
                confirm = input("\nProceed with setup? (y/N): ").strip().lower()
                if confirm != 'y':
                    print("Setup cancelled.")
                    return False
            except KeyboardInterrupt:
                print("\nSetup cancelled.")
                return False

        # Define setup steps based on deployment type
        if self.deployment_type == 'main_system':
            steps = [
                ("Directory Setup", self.setup_directories),
                ("Encryption & Credentials", self.setup_encryption),
                ("Credential Management", self.setup_credentials),
                ("Main Configuration", self.update_main_config),
                ("Model Download", self.download_models),
                ("System Service", self.setup_service),
                ("Validation", self.validate_setup)
            ]
        else:  # pi_testbench
            steps = [
                ("Directory Setup", self.setup_directories),
                ("Encryption & Credentials", self.setup_encryption),
                ("Pi Configuration", self.update_pi_config),
                ("System Service", self.setup_service),
                ("Validation", self.validate_setup)
            ]

        # Run setup steps
        for step_name, step_func in steps:
            print(f"\n{'='*20} {step_name} {'='*20}")
            if not step_func():
                print(f"❌ Setup failed at step: {step_name}")
                return False

        # Success message
        print("\n" + "="*60)
        print("🎉 ReViision setup completed successfully!")

        # Provide next steps based on deployment type
        print("\n📋 Next steps:")
        if self.deployment_type == 'main_system':
            print("1. Start the application:")
            print("   cd src && python main.py")
            print("2. Access the web interface:")
            web_config = config_section.get('app', {}).get('web', {})
            host = web_config.get('host', 'localhost')
            port = web_config.get('port', 5000)
            if host == '0.0.0.0':
                host = 'localhost'
            print(f"   http://{host}:{port}")
            print("3. Default login: admin/admin")
        else:  # pi_testbench
            print("1. Start the Pi testbench:")
            print("   cd pi_testbench && python main.py")
            print("2. Or install as service (Linux only):")
            service_name = config_section.get('deployment', {}).get('service_name', 'reviision-pi')
            print(f"   sudo systemctl start {service_name}")
            print("3. Connect devices to the WiFi hotspot:")
            hotspot_config = config_section.get('network', {}).get('hotspot', {})
            print(f"   SSID: {hotspot_config.get('ssid', 'ReViision-TestBench')}")
            print(f"   Password: {hotspot_config.get('password', 'testbench2024')}")

        print("\n📚 For more information, see README.md")
        return True

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ReViision Comprehensive Setup Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_reviision.py                           # Interactive setup
  python setup_reviision.py --env desktop_production  # Production desktop setup
  python setup_reviision.py --env pi_testbench        # Pi testbench setup
  python setup_reviision.py --setup-credentials       # Setup credentials only
  python setup_reviision.py --validate                # Validate existing setup
  python setup_reviision.py --list-environments       # List available environments
        """
    )

    parser.add_argument(
        '--env', '--environment',
        help='Environment name to use for setup'
    )

    parser.add_argument(
        '--setup-credentials',
        action='store_true',
        help='Setup credentials only'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate existing setup'
    )

    parser.add_argument(
        '--list-environments',
        action='store_true',
        help='List available environments'
    )

    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompts'
    )

    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    setup = ReViisionSetup()

    try:
        # Handle list environments
        if args.list_environments:
            if setup.load_setup_config():
                environments = setup.list_environments()
                print("📋 Available environments:")
                for env in environments:
                    print(f"  • {env}")
            return

        # Run setup
        success = setup.run_setup(args)
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()