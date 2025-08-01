# ReViision
## Retail Vision + Reiterative Improvement

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

**ReViision** is a comprehensive retail analytics platform that leverages advanced computer vision and machine learning to provide actionable insights into customer behaviour. The system enables retailers to optimise store layouts, improve customer experience, and make data-driven decisions through real-time analysis of foot traffic patterns, demographic insights, and behavioural analytics.

## Key Features

### Computer Vision & Analytics
- **Person Detection & Tracking**: YOLOv8-powered real-time person detection with advanced multi-object tracking
- **Demographic Analysis**: Age, gender, and emotion recognition using InsightFace and DeepFace models
- **Path Analysis**: Customer movement pattern detection and common route identification
- **Dwell Time Analysis**: Zone-based time spent analysis with configurable areas of interest
- **Heat Map Generation**: Visual traffic density mapping for store layout optimisation
- **Correlation Analysis**: Statistical analysis of relationships between demographics and behaviour

### Camera & Input Support
- **Multiple Camera Types**: USB cameras, RTSP/RTSPS IP cameras, and video file analysis
- **Secure Streaming**: Encrypted RTSP support with credential management
- **Real-time Processing**: Low-latency frame analysis with configurable performance settings
- **Cross-platform Compatibility**: Windows, Linux, and macOS support

### Web Interface & Visualisation
- **Interactive Dashboard**: Real-time analytics with customisable charts and graphs
- **User Management**: Role-based access control (Admin, Manager, Viewer)
- **Export Capabilities**: Data export in multiple formats (CSV, JSON, PDF reports)
- **Mobile-responsive Design**: Access analytics from any device

### Security & Privacy
- **Encrypted Credential Storage**: Secure management of camera and database credentials
- **Privacy-first Design**: Demographic analysis without storing personally identifiable information
- **Local Processing**: On-premises data processing with optional cloud integration
- **Audit Logging**: Comprehensive activity logging for compliance and monitoring

### Deployment Options
- **Desktop/Server Installation**: Full-featured deployment for high-performance analytics
- **Raspberry Pi Testbench**: Portable field deployment with WiFi hotspot management
- **Docker Support**: Containerised deployment for easy scaling and management

## 🏗️ System Architecture

ReViision follows a modular, service-oriented architecture designed for scalability and maintainability:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Camera Layer  │    │  Processing     │    │   Analysis      │
│                 │    │     Layer       │    │     Layer       │
│ • USB Cameras   │───▶│ • Person        │───▶│ • Demographics  │
│ • RTSP Streams  │    │   Detection     │    │ • Path Analysis │
│ • Video Files   │    │ • Tracking      │    │ • Dwell Time    │
│ • Live Feeds    │    │ • Re-ID         │    │ • Heatmaps      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │    Database     │    │   Security      │
│                 │    │     Layer       │    │     Layer       │
│ • Dashboard     │◀───│ • SQLite        │    │ • Encryption    │
│ • API Endpoints │    │ • Analytics     │    │ • Authentication│
│ • Real-time     │    │ • Metadata      │    │ • Audit Logs   │
│   Streaming     │    │ • Reporting     │    │ • Access Control│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

- **Camera Module**: Unified interface supporting multiple camera types with factory pattern
- **Detection Module**: YOLOv8-based person detection with Kalman filter tracking
- **Analysis Module**: Comprehensive behavioural analysis including demographics and patterns
- **Web Module**: Flask-based interface with real-time streaming and interactive dashboards
- **Database Module**: SQLite storage with optimised schema for analytics data
- **Security Module**: Encrypted credential management and secure communication protocols

## 📋 Requirements

### Hardware Requirements

#### Minimum System Requirements
- **CPU**: Intel i5 or AMD Ryzen 5 (4+ cores recommended)
- **RAM**: 8GB 
- **Storage**: 10GB free space (additional space for analytics data)
- **GPU**: Optional NVIDIA GPU with CUDA support for enhanced performance

#### Camera Requirements
- **USB Cameras**: Any UVC-compatible USB camera
- **IP Cameras**: RTSP/RTSPS-enabled cameras (H.264/H.265 support recommended)
- **Supported Formats**: MP4, AVI, MOV video files for analysis

#### Raspberry Pi Requirements (Testbench Deployment)
- **Model**: Raspberry Pi 4B (4GB RAM minimum, 8GB recommended)
- **Storage**: 32GB+ Class 10 microSD card
- **Network**: WiFi adapter for dual-interface setup or Ethernet connection
- **Power**: 5V 3A power supply

### Software Requirements

#### Operating System Support
- **Windows**: 10/11 (64-bit)
- **Linux**: Ubuntu 20.04+, Debian 11+, CentOS 8+
- **macOS**: 10.15+ (Intel/Apple Silicon)
- **Raspberry Pi OS**: 64-bit (recommended for Pi deployment)

#### Python Environment
- **Python**: 3.8 to 3.11 (3.10-3.11 recommended)
- **Package Manager**: pip 21.0+ or conda
- **Virtual Environment**: venv or conda environment recommended

#### Dependencies
Core dependencies are automatically installed via `requirements.txt`:
- **Computer Vision**: OpenCV 4.8+, ultralytics 8.0+
- **Machine Learning**: PyTorch 2.1+, TensorFlow 2.15, InsightFace 0.7+
- **Web Framework**: Flask 3.0+, Flask-SocketIO 5.3+
- **Database**: SQLAlchemy 2.0+
- **Security**: cryptography 41.0+, PyJWT 2.8+

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/reviision.git
cd reviision
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Required Models
```bash
# YOLOv8 model will be downloaded automatically on first run
# For manual download:
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt
```

### 4. Configure the System
```bash
# Copy and edit configuration
cp src/config.yaml src/config_local.yaml
# Edit src/config_local.yaml with your camera settings
```

### 5. Run the Application
```bash
cd src
python main.py --config config_local.yaml
```

### 6. Access the Web Interface
Open your browser and navigate to:
- **Local access**: http://localhost:5000
- **Network access**: http://YOUR_IP:5000

**Default login credentials:**
- Username: `admin`
- Password: `admin`

> ⚠️ **Security Note**: Change default credentials immediately in production environments.

## 📦 Installation Guide

### Desktop/Server Installation

#### 1. System Preparation
```bash
# Update system packages (Linux)
sudo apt update && sudo apt upgrade -y

# Install system dependencies (Ubuntu/Debian)
sudo apt install -y python3-pip python3-venv git build-essential \
    libopencv-dev python3-opencv ffmpeg

# For CentOS/RHEL
sudo yum groupinstall -y "Development Tools"
sudo yum install -y python3-pip python3-devel opencv-devel ffmpeg
```

#### 2. Python Environment Setup
```bash
# Clone repository
git clone https://github.com/your-org/reviision.git
cd reviision

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. GPU Support (Optional)
For NVIDIA GPU acceleration:
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install CUDA-enabled OpenCV (optional)
pip uninstall opencv-python
pip install opencv-python-headless
```

#### 4. Database Initialisation
```bash
# Database will be created automatically on first run
# For manual initialisation:
cd src
python -c "from database import get_database; db = get_database({'type': 'sqlite', 'path': '../data/reviision.db'})"
```

### Raspberry Pi Testbench Installation

The Pi Testbench provides a portable, field-deployable version of ReViision with WiFi hotspot capabilities.

#### 1. Prepare Raspberry Pi
```bash
# Flash Raspberry Pi OS Lite (64-bit) to SD card
# Enable SSH and configure WiFi during imaging

# SSH into Pi and update
sudo apt update && sudo apt upgrade -y
```

#### 2. Automated Setup
```bash
# Clone repository
git clone https://github.com/your-org/reviision.git
cd reviision/pi_testbench

# Run automated setup (SSH-safe version)
bash scripts/complete_network_setup_ssh_safe.sh
```

#### 3. Manual Setup (Alternative)
```bash
# Install dependencies
sudo apt install -y hostapd dnsmasq python3-pip python3-venv \
    wireless-tools net-tools iptables-persistent

# Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure network services
sudo bash scripts/complete_network_setup.sh

# Set up systemd service
sudo cp services/reviision-pi.service /etc/systemd/system/
sudo systemctl enable reviision-pi
sudo systemctl start reviision-pi
```

#### 4. Pi Testbench Features
- **WiFi Hotspot**: Creates `ReViision-TestBench` network
- **Dual WiFi**: Connects to internet while serving hotspot
- **Remote Access**: Web interface accessible at `192.168.4.1:5000`
- **Data Synchronisation**: Forwards analytics to main server
- **System Monitoring**: Health checks and performance monitoring

## ⚙️ Configuration

### Basic Configuration

ReViision uses YAML configuration files for system settings. The main configuration file is `src/config.yaml`.

#### Camera Configuration
```yaml
camera:
  type: usb                    # Camera type: usb, rtsp, rtsps, mp4
  device: 0                    # USB device ID or video file path
  fps: 30                      # Frames per second
  resolution: [1920, 1080]     # Camera resolution

  # For RTSP cameras
  url: "rtsp://camera-ip/stream"
  username: "camera_user"      # Use credential_ref for security
  password: "camera_pass"      # Use credential_ref for security

  # Performance settings
  buffer_size: 1               # Frame buffer size
  connection_timeout: 15.0     # Connection timeout in seconds
```

#### Detection & Analysis Configuration
```yaml
detection:
  # Person detection settings
  person:
    model: yolov8n.pt          # YOLO model file
    confidence_threshold: 0.5   # Detection confidence
    iou_threshold: 0.45        # Non-maximum suppression threshold
    device: cpu                # cpu or cuda

  # Face detection settings
  face:
    model_name: buffalo_l      # InsightFace model
    min_face_size: 15          # Minimum face size in pixels
    confidence_threshold: 0.3   # Face detection confidence

tracking:
  max_age: 30                  # Maximum frames to keep lost tracks
  min_hits: 3                  # Minimum detections before confirming track
  iou_threshold: 0.3           # Intersection over Union threshold
  reid_enabled: true           # Enable re-identification

analysis:
  demographics:
    enabled: true
    detection_interval: 2      # Analyse every N frames
    min_face_size: 15         # Minimum face size for analysis

  path:
    enabled: true
    max_path_history: 100     # Maximum path points to store
    min_path_points: 5        # Minimum points for valid path

  dwell_time:
    enabled: true
    min_dwell_time: 3.0       # Minimum seconds for dwell detection

  heatmap:
    enabled: true
    resolution: [640, 480]    # Heatmap resolution
    decay_factor: 0.1         # Heat decay rate
```

#### Web Interface Configuration
```yaml
web:
  host: 0.0.0.0               # Bind address (0.0.0.0 for all interfaces)
  port: 5000                  # Web server port
  debug: false                # Debug mode
  secret_key: "your-secret-key"  # Session encryption key

  # Authentication settings
  session_timeout_hours: 2    # Session timeout
  max_login_attempts: 5       # Maximum failed login attempts
  lockout_duration_minutes: 15  # Account lockout duration
```

### Secure Credential Management

ReViision includes a comprehensive credential management system for secure handling of sensitive information.

#### Setting Up Credentials
```bash
# Interactive credential setup
python src/main.py --setup-credentials

# Import from environment variables
export RA_CRED_RTSP_URL="rtsp://camera.local/stream"
export RA_CRED_RTSP_USERNAME="admin"
export RA_CRED_RTSP_PASSWORD="password123"
python src/main.py --import-env-credentials
```

#### Encryption Key Setup
```bash
# Generate encryption key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Set encryption key
export RETAIL_ANALYTICS_KEY="your-generated-key"
# OR use passphrase
export RETAIL_ANALYTICS_PASSPHRASE="your-secure-passphrase"
```

#### Using Credentials in Configuration
```yaml
camera:
  type: rtsp
  credential_ref: rtsp_cam1    # Reference to stored credentials

# OR use placeholder syntax
database:
  host: localhost
  username: ${db:username}     # Replaced with actual credential
  password: ${db:password}     # Replaced with actual credential
```

### Environment-Specific Configuration

Create environment-specific configuration files:
- `config_development.yaml` - Development settings
- `config_production.yaml` - Production settings
- `config_testing.yaml` - Testing settings

```bash
# Use specific configuration
python src/main.py --config config_production.yaml
```

## 🖥️ Usage

### Starting the System

#### Basic Usage
```bash
cd src
python main.py
```

#### Command-Line Options
```bash
python main.py --help

# Common options:
python main.py --config config_production.yaml  # Use specific config
python main.py --debug                          # Enable debug logging
python main.py --setup-credentials              # Interactive credential setup
python main.py --import-env-credentials         # Import from environment
```

### Web Interface

#### Accessing the Dashboard
1. Start the ReViision system
2. Open web browser and navigate to `http://localhost:5000`
3. Log in with your credentials

#### Default User Accounts
| Username | Password | Role    | Permissions |
|----------|----------|---------|-------------|
| admin    | admin    | Admin   | Full system access, user management |
| manager  | manager  | Manager | Analytics access, configuration |
| viewer   | viewer   | Viewer  | Read-only analytics access |

> ⚠️ **Important**: Change default passwords immediately in production environments.

#### Dashboard Features

**Real-time Analytics**
- Live camera feed with detection overlays
- Real-time person count and demographics
- Current activity heatmap
- Zone-based dwell time monitoring

**Historical Analysis**
- Traffic patterns over time
- Demographic trends and insights
- Path analysis and common routes
- Correlation analysis between variables

**Configuration Management**
- Camera settings and stream configuration
- Detection and tracking parameter tuning
- Zone definition and management
- User account administration

### API Endpoints

ReViision provides RESTful API endpoints for integration with external systems:

#### Analytics Endpoints
```bash
# Get current analytics summary
GET /api/analytics/summary

# Get demographic data
GET /api/analytics/demographics?start_date=2024-01-01&end_date=2024-01-31

# Get traffic patterns
GET /api/analytics/traffic_patterns?interval=hourly

# Get heatmap data
GET /api/analytics/heatmap?zone_id=1

# Get path analysis
GET /api/analytics/paths?min_length=5
```

#### System Endpoints
```bash
# System status
GET /api/system/status

# Camera stream
GET /api/camera/stream

# Configuration
GET /api/config
POST /api/config
```

#### Authentication
```bash
# Login
POST /api/auth/login
{
  "username": "admin",
  "password": "admin"
}

# Logout
POST /api/auth/logout
```

### Data Export

#### Export Options
- **CSV**: Tabular data for spreadsheet analysis
- **JSON**: Structured data for API integration
- **PDF**: Formatted reports for presentations

#### Export Methods
```bash
# Via Web Interface
Dashboard → Export → Select Format → Download

# Via API
GET /api/export/demographics?format=csv&start_date=2024-01-01
GET /api/export/traffic_patterns?format=json&interval=daily
GET /api/export/report?format=pdf&template=summary
```

### Performance Optimisation

#### System Tuning
```yaml
# config.yaml performance settings
performance:
  frame_analysis:
    min_interval_ms: 100        # Minimum time between analyses
    frame_skip_factor: 3        # Process every Nth frame
    enable_motion_detection: true  # Skip processing when no motion

  yolo:
    batch_size: 1              # Batch size for detection
    half_precision: true       # Use FP16 for faster inference

  database:
    batch_insert_size: 100     # Batch database operations
    connection_pool_size: 5    # Database connection pool
```

#### Hardware Optimisation
```bash
# GPU acceleration (NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Update config for GPU
detection:
  person:
    device: cuda              # Use GPU for detection
  face:
    ctx_id: 0                 # GPU context for face analysis
```

## 🔒 Security & Privacy

### Security Features

#### Credential Management
- **Encrypted Storage**: All credentials encrypted using Fernet symmetric encryption
- **Environment Variables**: Secure key management via environment variables
- **Access Control**: Role-based permissions (Admin, Manager, Viewer)
- **Session Management**: Configurable session timeouts and login attempt limits

#### Network Security
- **HTTPS Support**: TLS encryption for web interface
- **Secure RTSP**: RTSPS support for encrypted camera streams
- **Network Isolation**: Recommended VLAN segmentation for camera networks
- **Firewall Configuration**: Minimal port exposure with configurable access

#### Data Protection
- **Privacy-First Design**: No personally identifiable information stored
- **Local Processing**: On-premises analytics with optional cloud integration
- **Audit Logging**: Comprehensive activity logging for compliance
- **Data Retention**: Configurable data retention policies

### Security Best Practices

#### Production Deployment
```bash
# 1. Change default credentials immediately
# 2. Set strong encryption keys
export RETAIL_ANALYTICS_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"

# 3. Use HTTPS with proper certificates
# 4. Configure firewall rules
sudo ufw allow 5000/tcp  # Web interface
sudo ufw enable

# 5. Set restrictive file permissions
chmod 600 config/credentials.enc
chmod 700 config/
```

#### Network Security
- **Camera Isolation**: Place cameras on separate VLAN
- **Access Control**: Limit web interface access to authorised networks
- **Regular Updates**: Keep system and dependencies updated
- **Monitoring**: Enable audit logging and monitor for suspicious activity

#### Privacy Compliance
- **Data Minimisation**: Only collect necessary analytics data
- **Anonymisation**: Demographic analysis without identity storage
- **Consent Management**: Implement appropriate consent mechanisms
- **Data Subject Rights**: Provide data access and deletion capabilities

## 🛠️ Development

### Development Environment Setup

#### Prerequisites
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black isort flake8 mypy pre-commit

# Set up pre-commit hooks
pre-commit install
```

#### Code Quality Tools
```bash
# Code formatting
black src/
isort src/

# Linting
flake8 src/
mypy src/

# Testing
pytest tests/ --cov=src/
```

### Project Structure
```
reviision/
├── src/                     # Main application source
│   ├── main.py             # Application entry point
│   ├── config.yaml         # Default configuration
│   ├── camera/             # Camera interface modules
│   ├── detection/          # Object detection and tracking
│   ├── analysis/           # Behavioural analysis modules
│   ├── web/                # Web interface and API
│   ├── database/           # Database abstraction layer
│   └── utils/              # Utility functions and helpers
├── pi_testbench/           # Raspberry Pi deployment
│   ├── src/                # Pi-specific source code
│   ├── scripts/            # Setup and management scripts
│   ├── services/           # Systemd service files
│   └── config/             # Pi configuration files
├── tests/                  # Test suite
├── docs/                   # Documentation
├── models/                 # ML model storage
├── data/                   # Database and data files
└── requirements.txt        # Python dependencies
```

### Contributing Guidelines

#### Code Standards
- **Python Style**: Follow PEP 8 with Black formatting
- **Documentation**: Comprehensive docstrings for all public functions
- **Testing**: Minimum 80% test coverage for new features
- **Type Hints**: Use type annotations for better code clarity

#### Development Workflow
1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** Pull Request with detailed description

#### Testing
```bash
# Run full test suite
pytest tests/

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/performance/   # Performance tests

# Generate coverage report
pytest --cov=src/ --cov-report=html
```

### API Development

#### Adding New Endpoints
```python
# src/web/routes.py
@web_bp.route('/api/new-feature')
@require_auth('manager')
def new_feature():
    """New API endpoint with proper authentication"""
    try:
        # Implementation
        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        logger.error(f"Error in new_feature: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
```

#### Database Schema Changes
```python
# src/database/sqlite_db.py
def create_new_table(self):
    """Add new table with proper constraints"""
    self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS new_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            -- Add columns with appropriate constraints
        )
    ''')
```
## 🔧 Troubleshooting

### Common Issues

#### Installation Problems

**PyTorch Installation Fails**
```bash
# Solution: Use platform-specific installation
# For CUDA support:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For macOS with Apple Silicon:
pip install torch torchvision
```

**OpenCV Import Error**
```bash
# Solution: Reinstall OpenCV
pip uninstall opencv-python opencv-python-headless
pip install opencv-python

# For headless servers:
pip install opencv-python-headless
```

**Permission Denied Errors**
```bash
# Solution: Fix file permissions
chmod +x scripts/*.sh
chmod 600 config/credentials.enc
sudo chown -R $USER:$USER data/
```

#### Runtime Issues

**Camera Connection Failed**
```yaml
# Check camera configuration in config.yaml
camera:
  type: usb
  device: 0  # Try different device numbers: 0, 1, 2...

# For RTSP cameras, verify URL and credentials
camera:
  type: rtsp
  url: "rtsp://camera-ip:554/stream"
  # Test with VLC or ffplay first
```

**High CPU/Memory Usage**
```yaml
# Optimise performance settings
performance:
  frame_analysis:
    frame_skip_factor: 5      # Process every 5th frame
    min_interval_ms: 200      # Increase processing interval

detection:
  person:
    image_size: 416           # Reduce from 640 for faster processing
```

**Web Interface Not Accessible**
```bash
# Check if service is running
python src/main.py --debug

# Verify network configuration
netstat -tlnp | grep 5000

# Check firewall settings
sudo ufw status
sudo ufw allow 5000/tcp
```

#### Database Issues

**Database Locked Error**
```bash
# Solution: Check for multiple instances
ps aux | grep python | grep main.py
kill <process_id>

# Reset database if corrupted
mv data/reviision.db data/reviision.db.backup
# Restart application to recreate database
```

**Migration Errors**
```bash
# Solution: Manual database reset
cd src
python -c "
from database import get_database
import os
if os.path.exists('../data/reviision.db'):
    os.remove('../data/reviision.db')
db = get_database({'type': 'sqlite', 'path': '../data/reviision.db'})
print('Database recreated successfully')
"
```

### Performance Tuning

#### System Optimisation
```yaml
# config.yaml optimisation
performance:
  frame_analysis:
    enable_motion_detection: true    # Skip processing when no motion
    motion_threshold: 1000          # Adjust sensitivity
    cache_duration_ms: 1000         # Cache results

  yolo:
    batch_size: 1                   # Reduce for lower memory usage
    half_precision: true            # Use FP16 for speed

camera:
  fps: 15                          # Reduce FPS for better performance
  buffer_size: 1                   # Minimal buffering
```

#### Hardware Recommendations
- **CPU**: Intel i7 or AMD Ryzen 7 for multiple cameras
- **RAM**: 16GB+ for optimal performance
- **GPU**: NVIDIA GTX 1660+ for GPU acceleration
- **Storage**: SSD for database and model storage

### Raspberry Pi Specific Issues

#### WiFi Hotspot Not Working
```bash
# Check network services
sudo systemctl status hostapd
sudo systemctl status dnsmasq

# Restart network services
sudo systemctl restart reviision-network-prep
sudo systemctl restart hostapd
sudo systemctl restart dnsmasq
```

#### Pi Performance Issues
```bash
# Check system resources
htop
free -h
df -h

# Optimise Pi configuration
echo "arm_freq=1800" | sudo tee -a /boot/config.txt
echo "gpu_mem=128" | sudo tee -a /boot/config.txt
sudo reboot
```

### Logging and Debugging

#### Enable Debug Logging
```bash
# Run with debug mode
python src/main.py --debug

# Check log files
tail -f reviision.log
tail -f pi_testbench/logs/pi_testbench.log
```

#### Log Locations
- **Main System**: `reviision.log`
- **Pi Testbench**: `pi_testbench/logs/pi_testbench.log`
- **Web Server**: `web.log`
- **Database**: `database.log`

## 📚 Additional Resources

### Third-Party Licenses
- **YOLOv8**: AGPL-3.0 License
- **InsightFace**: MIT License
- **OpenCV**: Apache 2.0 License
- **Flask**: BSD-3-Clause License

## 🙏 Acknowledgements

- **Ultralytics** for the YOLOv8 object detection framework
- **InsightFace** for facial analysis capabilities
- **DeepFace** for advanced facial recognition and analysis
- **OpenCV** community for computer vision tools
- **Flask** team for the web framework

---

**ReViision** - Transforming retail analytics through intelligent computer vision.
