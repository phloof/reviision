# ReViision Raspberry Pi Test Bench

This directory contains the Raspberry Pi 4B setup for the ReViision test bench system. The Pi serves as a WiFi access point for the Tapo C220 camera and displays real-time analytics on a 4.26" e-paper hat.

## Recommended Operating System

**Raspberry Pi OS Lite (64-bit)** is the recommended OS for this project:

- **Download**: [Raspberry Pi OS Lite](https://www.raspberrypi.org/software/operating-systems/)
- **Benefits**: 
  - Lightweight (no desktop environment)
  - Official Pi foundation support
  - Excellent hardware compatibility
  - Regular security updates
  - Optimized for Pi hardware

**Installation using Raspberry Pi Imager**:
1. Download and install [Raspberry Pi Imager](https://www.raspberrypi.org/software/)
2. Select "Raspberry Pi OS Lite (64-bit)"
3. Configure WiFi and SSH in advanced options
4. Flash to SD card

## Components

- **Raspberry Pi 4B**: Main controller and WiFi hotspot
- **Tapo C220 Camera**: RTSP streaming camera  
- **4.26" E-Paper HAT**: Display for real-time analytics
- **Windows Computer**: Main ReViision analysis server

## Features

### WiFi Hotspot with Internet Forwarding
- Creates a dedicated network for the test bench
- Forwards internet traffic from Ethernet or secondary WiFi
- Allows Tapo camera to access cloud services
- Configurable SSID and password
- Support for WPA2 and WPA2 Enterprise

### E-Paper Display
- Real-time analytics dashboard
- Low power consumption
- Clear visibility in various lighting conditions
- Displays:
  - Current visitor count
  - Demographics breakdown
  - Dwell time statistics
  - System status
  - Network information

### Data Integration
- Connects to main ReViision server via WiFi
- Real-time data fetching from analytics APIs
- Automatic failover and reconnection
- Local data caching for offline display

### Auto-Start on Boot
- Systemd service for automatic startup
- Graceful shutdown handling
- Service monitoring and restart
- Log management and rotation

## Hardware Setup

### E-Paper HAT Connection
The 4.26" e-paper HAT connects via GPIO pins:
- VCC → 3.3V (Pin 1)
- GND → Ground (Pin 6)
- DIN → GPIO 10 (Pin 19)
- CLK → GPIO 11 (Pin 23)
- CS → GPIO 8 (Pin 24)
- DC → GPIO 25 (Pin 22)
- RST → GPIO 17 (Pin 11)
- BUSY → GPIO 24 (Pin 18)

### Network Topology
```
Internet → Raspberry Pi (WiFi AP + Ethernet/WiFi) → Tapo C220 Camera
                    ↓
              Windows Computer
              (ReViision Server)
```

## Quick Installation

### Automated Installation
```bash
# Clone or copy the project to the Pi
cd pi_testbench

# Run the installation script
bash scripts/install.sh
```

The installation script will:
1. Update system packages
2. Install required dependencies
3. Configure Python environment
4. Set up systemd service for auto-start
5. Configure network (optional)
6. Enable hardware interfaces (SPI/I2C)

### Manual Installation

#### 1. System Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y hostapd dnsmasq python3-pip python3-venv git \
  wireless-tools net-tools iptables-persistent

# Enable SPI for e-paper display
sudo raspi-config nonint do_spi 0

# Install Python dependencies
cd pi_testbench
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. Auto-Start Configuration
```bash
# Copy systemd service file
sudo cp services/reviision-pi.service /etc/systemd/system/

# Enable auto-start
sudo systemctl daemon-reload
sudo systemctl enable reviision-pi

# Start the service
sudo systemctl start reviision-pi
```

#### 3. Network Setup
```bash
# Run network configuration script
sudo bash scripts/setup_network.sh
```

## Configuration

### Main Configuration
Edit `config/pi_config.yaml` to match your setup:
- ReViision server IP and port
- WiFi network name and password
- Display refresh intervals
- API endpoints

### Environment Variables
Create `.env` file for runtime overrides:
```bash
# Network configuration
REVIISION_SERVER_HOST=192.168.4.10
REVIISION_SERVER_PORT=5000
WIFI_SSID=ReViision-TestBench
WIFI_PASSWORD=testbench2024

# Logging
LOG_LEVEL=INFO

# Hardware (for testing without hardware)
DISABLE_EPAPER=false
DISABLE_GPIO=false
```

### WPA2 Enterprise Configuration

For enterprise WiFi networks, uncomment and configure the enterprise section in `config/pi_config.yaml`:

```yaml
network:
  internet:
    wifi:
      # Standard configuration
      interface: "wlan1"
      ssid: "Enterprise-WiFi"
      dhcp: true
      
      # WPA2 Enterprise configuration
      enterprise:
        enabled: true
        # Authentication method: 'peap', 'ttls', 'tls'
        eap_method: "peap"
        # User credentials
        identity: "username@company.com"
        anonymous_identity: "anonymous@company.com"
        password: "your-password"
        # Phase 2 auth: 'mschapv2', 'gtc', 'md5'
        phase2_auth: "mschapv2"
        
        # Certificate configuration (for TLS or validation)
        certificates:
          ca_cert: "/etc/ssl/certs/company-ca.crt"
          # For TLS authentication (uncomment if needed):
          # client_cert: "/etc/ssl/certs/client.crt"
          # private_key: "/etc/ssl/private/client.key"
          # private_key_passwd: "key-password"
        
        # Security settings
        security:
          domain_suffix_match: "radius.company.com"
          ca_cert_verification: true
```

**Certificate Installation**:
```bash
# Copy certificates to the Pi
sudo cp company-ca.crt /etc/ssl/certs/
sudo cp client.crt /etc/ssl/certs/        # If using TLS
sudo cp client.key /etc/ssl/private/      # If using TLS

# Set proper permissions
sudo chmod 644 /etc/ssl/certs/*.crt
sudo chmod 600 /etc/ssl/private/*.key
```

## Service Management

### Starting/Stopping the Service
```bash
# Start service
sudo systemctl start reviision-pi

# Stop service
sudo systemctl stop reviision-pi

# Restart service
sudo systemctl restart reviision-pi

# Check status
sudo systemctl status reviision-pi

# Enable auto-start (run on boot)
sudo systemctl enable reviision-pi

# Disable auto-start
sudo systemctl disable reviision-pi
```

### Viewing Logs
```bash
# View recent logs
sudo journalctl -u reviision-pi -n 50

# Follow live logs
sudo journalctl -u reviision-pi -f

# Using convenience scripts
bash scripts/logs.sh           # Recent logs
bash scripts/logs.sh follow    # Live logs
```

### Using Convenience Scripts
```bash
# Check complete system status
bash scripts/status.sh

# Restart the service
bash scripts/restart.sh

# View logs
bash scripts/logs.sh
```

## Usage

### Automatic Operation
After installation and reboot:
1. The Pi creates a WiFi hotspot: "ReViision-TestBench"
2. The service starts automatically
3. Connect your Tapo camera to the hotspot
4. Connect your Windows computer to the hotspot
5. The e-paper display shows real-time analytics

### Manual Operation
```bash
# Start manually (if auto-start is disabled)
cd /home/pi/ReViision/pi_testbench
source venv/bin/activate
python3 main.py
```

### Network Configuration Options

#### 1. Ethernet + WiFi Hotspot (Recommended)
- Pi connects to internet via Ethernet
- Broadcasts WiFi hotspot for devices
- Best performance and reliability

#### 2. Dual WiFi Setup
- Requires USB WiFi adapter
- One WiFi for internet, one for hotspot
- More flexibility in placement

#### 3. Hotspot Only
- No internet forwarding
- Isolated test environment
- Camera operates offline only

## Troubleshooting

### Service Issues
```bash
# Check service status
sudo systemctl status reviision-pi

# View detailed logs
sudo journalctl -u reviision-pi -n 100

# Restart service
sudo systemctl restart reviision-pi

# Check Python dependencies
cd /home/pi/ReViision/pi_testbench
source venv/bin/activate
python3 -c "import yaml, requests, PIL, numpy, psutil, netifaces"
```

### Network Issues
```bash
# Check hotspot status
sudo systemctl status hostapd
sudo systemctl status dnsmasq

# Check network interfaces
ip addr show
iwconfig

# Test internet connectivity
ping 8.8.8.8

# View iptables rules
sudo iptables -L -n -v
```

### Hardware Issues
```bash
# Check SPI is enabled
lsmod | grep spi

# Check GPIO access
ls -la /dev/spidev*

# Test e-paper display
python3 scripts/test_display.py
```

### Common Solutions
1. **Service won't start**: Check Python dependencies and configuration file
2. **No WiFi hotspot**: Verify hostapd configuration and interface availability
3. **No internet forwarding**: Check iptables rules and upstream connectivity
4. **Display not working**: Verify SPI is enabled and GPIO connections
5. **High CPU usage**: Check for infinite loops in logs

## File Structure

```
pi_testbench/
├── main.py                    # Main application entry
├── requirements.txt           # Python dependencies
├── .env                      # Environment variables
├── config/                   # Configuration files
│   ├── pi_config.yaml       # Main configuration
│   ├── hostapd.conf         # WiFi hotspot settings
│   ├── dnsmasq.conf         # DHCP and DNS settings
│   └── dhcpcd.conf          # Network interface configuration
├── src/                      # Source code
│   ├── display/             # E-paper display drivers
│   ├── network/             # Network management
│   ├── data/                # Data collection from ReViision
│   └── utils/               # Utilities
├── services/                 # Systemd service files
│   └── reviision-pi.service # Auto-start service
├── scripts/                  # Helper scripts
│   ├── install.sh           # Installation script
│   ├── setup_network.sh     # Network configuration
│   ├── status.sh            # System status checker
│   ├── logs.sh              # Log viewer
│   └── restart.sh           # Service restart
└── logs/                     # Log files
```

## API Integration

The Pi connects to these ReViision API endpoints:
- `/api/analyze_frame`: Real-time frame analysis
- `/api/traffic_patterns`: Traffic pattern data
- `/api/facial_analysis`: Demographics data
- `/api/config`: System configuration

## Security

- WiFi network uses WPA2 encryption
- API connections use authentication tokens
- Firewall configured for minimal attack surface
- Service runs with limited privileges
- Regular security updates via systemd timers

## Performance Optimization

### System Settings
```bash
# Increase GPU memory split for display performance
echo "gpu_mem=128" | sudo tee -a /boot/config.txt

# Optimize for performance
echo "arm_freq=1800" | sudo tee -a /boot/config.txt
echo "over_voltage=6" | sudo tee -a /boot/config.txt

# Enable hardware acceleration
echo "dtoverlay=vc4-fkms-v3d" | sudo tee -a /boot/config.txt
```

### Application Settings
- Adjust polling intervals in config
- Enable/disable features as needed
- Monitor resource usage with `htop`

## Development

### Testing Without Hardware
Set environment variables to disable hardware:
```bash
export DISABLE_EPAPER=true
export DISABLE_GPIO=true
python3 main.py
```

### Local Development
```bash
# Install development dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/

# Code formatting
pip install black
black src/
``` 