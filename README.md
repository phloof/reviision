# ReViision
## Retail vision + reiterative improvement

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

ReViision is a retail analytics platform that uses computer vision and machine learning to deliver real‑time insights into customer behaviour. It helps retailers optimise layouts, improve customer experience, and support data‑driven decisions with live and historical analytics.

## Key features

- **Computer vision and analytics**: Person detection and multi‑object tracking (YOLOv8), age/gender/emotion analysis (InsightFace/DeepFace), path and dwell analysis, heatmaps, correlation analysis
- **Camera and input support**: USB, RTSP/RTSPS IP cameras, and video files; low‑latency processing; Windows/macOS/Linux support
- **Web interface**: Interactive dashboard with charts, role‑based access (Admin/Manager/Viewer), export to CSV/JSON/PDF
- **Security and privacy**: Encrypted credential storage, on‑prem processing with optional cloud, audit logging
- **Deployment options**: Desktop/server, Raspberry Pi testbench, Docker

## System architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Camera Layer │──▶│ Processing   │──▶│ Analysis     │
│ USB/RTSP/Vid │   │ Detection/   │   │ Demographics │
│ Live Feeds   │   │ Tracking/ReID│   │ Paths/Dwell  │
└──────────────┘   └──────────────┘   └──────────────┘
          │                    │
          └────────────────────┴──────────────────────▶
                               ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Web UI/API   │   │ Database     │   │ Security     │
│ Dashboard    │◀──│ SQLite/Meta  │   │ Auth/Encrypt │
│ Streaming    │   │ Reporting    │   │ Audit/Access │
└──────────────┘   └──────────────┘   └──────────────┘
```

Core modules: Camera (factory), Detection (YOLOv8 + Kalman), Analysis (behavioural), Web (Flask + Socket.IO), Database (SQLite/SQLAlchemy), Security (encryption and RBAC).

## Requirements

- **CPU/RAM**: Intel i5/Ryzen 5+, 8 GB RAM minimum
- **Storage**: 10 GB free (more for analytics data)
- **GPU (optional)**: NVIDIA CUDA for acceleration
- **OS**: Windows 10/11, Ubuntu 20.04+, Debian 11+, CentOS 8+, macOS 10.15+, Raspberry Pi OS 64‑bit
- **Python**: 3.8–3.11 (3.10–3.11 recommended)

Core dependencies are installed via `requirements.txt` (OpenCV, Ultralytics, PyTorch/TensorFlow, Flask/Socket.IO, SQLAlchemy, cryptography, PyJWT).

## Quick start

1) Clone and enter the repo
```bash
git clone https://github.com/your-org/reviision.git
cd reviision
```

2) Create a virtual environment and install deps
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
# source venv/bin/activate
pip install -r requirements.txt
```

3) Configure and run
```bash
cp src/config.yaml src/config_local.yaml
# Edit src/config_local.yaml for your camera(s)
cd src
python main.py --config config_local.yaml
```

4) Open the web UI
- Local: http://localhost:5000
- Network: http://YOUR_IP:5000

Default credentials: `admin` / `admin`  (change immediately in production).

## Installation

### Desktop/server

System packages (Ubuntu/Debian):
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git build-essential libopencv-dev python3-opencv ffmpeg
```

Python environment:
```bash
git clone https://github.com/your-org/reviision.git
cd reviision
python3 -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
# venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional GPU (NVIDIA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# For headless servers
pip uninstall -y opencv-python
pip install opencv-python-headless
```

Database initialisation (auto on first run; optional manual):
```bash
cd src
python -c "from database import get_database; db = get_database({'type': 'sqlite', 'path': '../data/reviision.db'})"
```

### Raspberry Pi testbench

A portable, field‑deployable build with Wi‑Fi hotspot and health monitoring.

Automated setup:
```bash
git clone https://github.com/your-org/reviision.git
cd reviision/pi_testbench
bash scripts/complete_network_setup_ssh_safe.sh
```

Manual alternative:
```bash
sudo apt install -y hostapd dnsmasq python3-pip python3-venv wireless-tools net-tools iptables-persistent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
sudo bash scripts/complete_network_setup.sh
sudo cp services/reviision-pi.service /etc/systemd/system/
sudo systemctl enable --now reviision-pi
```

Pi features: hotspot (`ReViision-TestBench`), dual Wi‑Fi, dashboard at `192.168.4.1:5000`, data sync, system monitoring.

## Configuration

Main file: `src/config.yaml` (use a local copy per environment).

Camera example:
```yaml
camera:
  type: usb                 # usb | rtsp | rtsps | mp4
  device: 0                 # device ID or video path
  fps: 30
  resolution: [1920, 1080]
  # RTSP
  url: "rtsp://camera-ip/stream"
  username: "camera_user"   # prefer credential_ref
  password: "camera_pass"   # prefer credential_ref
  buffer_size: 1
  connection_timeout: 15.0
```

Detection/analysis:
```yaml
detection:
  person:
    model: yolov8n.pt
    confidence_threshold: 0.5
    iou_threshold: 0.45
    device: cpu              # cpu | cuda
  face:
    model_name: buffalo_l
    min_face_size: 15
    confidence_threshold: 0.3
tracking:
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3
  reid_enabled: true
analysis:
  demographics:
    enabled: true
    detection_interval: 2
  path:
    enabled: true
    max_path_history: 100
  dwell_time:
    enabled: true
    min_dwell_time: 3.0
  heatmap:
    enabled: true
    resolution: [640, 480]
    decay_factor: 0.1
```

Web/auth:
```yaml
web:
  host: 0.0.0.0
  port: 5000
  debug: false
  secret_key: "your-secret-key"
  session_timeout_hours: 2
  max_login_attempts: 5
  lockout_duration_minutes: 15
```

Secure credentials:
```bash
# Interactive
python src/main.py --setup-credentials
# Import from environment
export RA_CRED_RTSP_URL="rtsp://camera.local/stream"
export RA_CRED_RTSP_USERNAME="admin"
export RA_CRED_RTSP_PASSWORD="password123"
python src/main.py --import-env-credentials
```

Encryption key:
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
export RETAIL_ANALYTICS_KEY="<generated>"
# or
export RETAIL_ANALYTICS_PASSPHRASE="<secure-passphrase>"
```

Environment files: `config_development.yaml`, `config_production.yaml`, `config_testing.yaml`.
```bash
python src/main.py --config config_production.yaml
```

## Usage

Start:
```bash
cd src
python main.py
```

CLI options:
```bash
python main.py --help
python main.py --config config_production.yaml
python main.py --debug
python main.py --setup-credentials
python main.py --import-env-credentials
```

Web interface:
1. Start ReViision
2. Open http://localhost:5000
3. Login with your credentials (change defaults in production)

## API overview

Analytics:
```http
GET /api/analytics/summary
GET /api/analytics/demographics?start_date=2024-01-01&end_date=2024-01-31
GET /api/analytics/traffic_patterns?interval=hourly
GET /api/analytics/heatmap?zone_id=1
GET /api/analytics/paths?min_length=5
```

System:
```http
GET /api/system/status
GET /api/camera/stream
GET /api/config
POST /api/config
```

Auth:
```http
POST /api/auth/login
POST /api/auth/logout
```

Data export:
```http
GET /api/export/demographics?format=csv&start_date=2024-01-01
GET /api/export/traffic_patterns?format=json&interval=daily
GET /api/export/report?format=pdf&template=summary
```

## Performance optimisation

Config:
```yaml
performance:
  frame_analysis:
    min_interval_ms: 100
    frame_skip_factor: 3
    enable_motion_detection: true
  yolo:
    batch_size: 1
    half_precision: true
  database:
    batch_insert_size: 100
    connection_pool_size: 5
```

GPU usage:
```yaml
detection:
  person:
    device: cuda
  face:
    ctx_id: 0
```

## Security and privacy

- **Credentials**: Encrypted at rest; environment‑based key management
- **RBAC**: Admin/Manager/Viewer roles with session controls
- **Network**: HTTPS/TLS support, RTSPS for cameras, firewall/VLAN isolation
- **Privacy**: No personally identifiable information is stored; on‑prem processing; audit logging and retention policies

Production checklist:
```bash
# Change default credentials and set strong keys
export RETAIL_ANALYTICS_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
# Enable HTTPS, restrict firewall
sudo ufw allow 5000/tcp
sudo ufw enable
# Restrict file permissions
chmod 600 config/credentials.enc
chmod 700 config/
```

## Development

Setup:
```bash
pip install -r requirements.txt
pip install pytest pytest-cov black isort flake8 mypy pre-commit
pre-commit install
```

Quality and tests:
```bash
black src/ && isort src/
flake8 src/ && mypy src/
pytest tests/ --cov=src/
```

Project structure:
```
reviision/
├── src/                    # Application source
│   ├── main.py            # Entry point
│   ├── config.yaml        # Default config
│   ├── camera/            # Camera interfaces
│   ├── detection/         # Detection and tracking
│   ├── analysis/          # Behavioural analysis
│   ├── web/               # Web UI and API
│   ├── database/          # Persistence layer
│   └── utils/             # Helpers
├── pi_testbench/          # Raspberry Pi deployment
├── tests/                 # Test suite
├── models/                # Model files
├── data/                  # Data and DB
└── requirements.txt       # Python deps
```

Contributing:
- Follow PEP 8; format with Black; type‑hint public APIs
- Aim for ≥80% coverage on new features
- Fork → branch → commit → push → open a PR with context

## Troubleshooting

- **PyTorch install fails**: use platform‑specific wheels
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu    # CPU
```

- **OpenCV import error**: reinstall
```bash
pip uninstall -y opencv-python opencv-python-headless
pip install opencv-python  # or opencv-python-headless for servers
```

- **Permission errors**:
```bash
chmod +x scripts/*.sh
chmod 600 config/credentials.enc
sudo chown -R $USER:$USER data/
```

- **Camera connection failed**: verify `config.yaml`; test RTSP with VLC/ffplay

- **High CPU/memory**: increase `frame_skip_factor`, reduce image size/FPS

- **Web UI unreachable**: run with `--debug`; check port/firewall

- **SQLite locked**: stop extra processes; back up and recreate the DB

## Licensing and acknowledgements

Third‑party licences include YOLOv8 (AGPL‑3.0), InsightFace (MIT), OpenCV (Apache‑2.0), Flask (BSD‑3‑Clause).

Acknowledgements: Ultralytics (YOLOv8), InsightFace, DeepFace, OpenCV community, Flask team.
