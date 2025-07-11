# ReViision Raspberry Pi Test Bench

This directory contains the Raspberry Pi 4B setup for the ReViision test bench system. The Pi serves as a WiFi access point for the Tapo C220 camera and provides networking infrastructure for the test bench.

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

- **Raspberry Pi 4B**: Main controller and dual WiFi interface manager
- **Tapo C220 Camera**: RTSP streaming camera  
- **Windows Computer**: Main ReViision analysis server

## Features

### Dual WiFi Network Management
- **wlan0**: Internet connection (connects to existing WiFi network)
- **wlan1**: WiFi hotspot for test bench devices
- Creates a dedicated network for the test bench
- Forwards internet traffic from wlan0 to wlan1
- Allows Tapo camera to access cloud services while maintaining isolated test network
- Configurable SSID and password
- Support for WPA2 and WPA2 Enterprise on internet connection

### Data Integration
- Connects to main ReViision server via WiFi
- Real-time data fetching from analytics APIs
- Automatic failover and reconnection
- Local data caching for offline operation

### Auto-Start on Boot
- Systemd service for automatic startup
- Graceful shutdown handling
- Service monitoring and restart
- Log management and rotation
- Custom network preparation services

## Hardware Setup

### Network Topology
```
Internet (ORBI58) → wlan0 (Pi) → NAT Forwarding → wlan1 (Pi Hotspot) → Tapo C220 Camera
                              ↓                                      ↓
                        Windows Computer                    Test Bench Devices
                        (ReViision Server)                  (192.168.4.x network)
```

**Network Configuration:**
- **wlan0**: Internet connection to existing WiFi (DHCP or static)
- **wlan1**: Hotspot at 192.168.4.1/24 serving DHCP range 192.168.4.10-192.168.4.50
- **NAT Forwarding**: iptables rules forward traffic from wlan1 → wlan0 → Internet

## Quick Installation

### Comprehensive Automated Setup
```bash
# Clone or copy the project to the Pi
cd pi_testbench

# For SSH connections - SSH-safe version (recommended if connected via SSH)
bash scripts/complete_network_setup_ssh_safe.sh

# OR for direct console access - standard version
sudo bash scripts/complete_network_setup.sh
```

**SSH-Safe Setup (Recommended for SSH connections):**
The SSH-safe version is designed specifically for remote installation and includes:
- Automatic SSH session detection
- Warning prompts before network service restart
- 30-second countdown with cancel option
- Clear instructions for reconnection after network restart
- All configuration completed before any network services are restarted

**Standard Setup:**
Use only when you have direct console access (keyboard/monitor) or are prepared for immediate SSH disconnection.

The complete setup script will:
1. Update system packages
2. Install all required dependencies
3. Configure dual WiFi interfaces (wlan0 for internet, wlan1 for hotspot)
4. Set up NAT forwarding and firewall rules
5. Configure all network services (hostapd, dnsmasq, dhcpcd, wpa_supplicant)
6. Create custom systemd services for network management
7. Set up auto-start services with proper dependencies
8. Handle service conflicts and ensure proper startup order

### Legacy Installation Options

#### Manual Python Environment Setup
```bash
# If you need to set up only the Python environment manually
cd pi_testbench
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Manual Installation

#### 1. System Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y hostapd dnsmasq python3-pip python3-venv git \
  wireless-tools net-tools iptables-persistent

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

#### 3. Manual Network Setup
```bash
# Run complete network configuration script
sudo bash scripts/complete_network_setup.sh
```

## Configuration

### Main Configuration
Edit `config/pi_config.yaml` to match your setup:
- ReViision server IP and port
- WiFi network name and password
- API endpoints

### Network Configuration Files
- `config/dhcpcd.conf`: Interface configuration for wlan0 (internet) and wlan1 (hotspot)
- `config/hostapd.conf`: WiFi hotspot configuration for wlan1
- `config/dnsmasq.conf`: DHCP and DNS server configuration for wlan1
- `config/wpa_supplicant-wlan0.conf`: Internet WiFi connection configuration for wlan0

### Environment Variables
Create `.env` file for runtime overrides:
```bash
# Network configuration
REVIISION_SERVER_HOST=192.168.4.10
REVIISION_SERVER_PORT=5000
WIFI_SSID=ReViision-TestBench
WIFI_PASSWORD=testbench2024

# Internet WiFi (for complete_network_setup.sh)
INTERNET_WIFI_SSID=ORBI58
INTERNET_WIFI_PASSWORD=your_wifi_password

# Logging
LOG_LEVEL=INFO
```

### WPA2 Enterprise Configuration

For enterprise WiFi networks, configure the internet connection in `config/wpa_supplicant-wlan0.conf`:

```bash
# WPA2 Enterprise configuration for internet connection
network={
    ssid="Enterprise-WiFi"
    key_mgmt=WPA-EAP
    eap=PEAP
    identity="username@company.com"
    anonymous_identity="anonymous@company.com"
    password="your-password"
    phase2="auth=MSCHAPV2"
    ca_cert="/etc/ssl/certs/company-ca.crt"
    domain_suffix_match="radius.company.com"
}
```

**Certificate Installation**:
```bash
# Copy certificates to the Pi
sudo cp company-ca.crt /etc/ssl/certs/
sudo chmod 644 /etc/ssl/certs/company-ca.crt
```

## Service Management

### Network Services
The system uses several custom systemd services for network management:
- `reviision-network-prep.service`: Prepares network interfaces and dependencies
- `iptables-restore.service`: Configures NAT forwarding rules
- `hostapd.service`: WiFi hotspot service
- `dnsmasq.service`: DHCP and DNS server
- `dhcpcd.service`: Network interface management
- `wpa_supplicant@wlan0.service`: Internet WiFi connection

### Starting/Stopping the Main Service
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

### Network Service Management
```bash
# Check all network services
sudo systemctl status hostapd dnsmasq dhcpcd wpa_supplicant@wlan0

# Restart network services
sudo systemctl restart reviision-network-prep
sudo systemctl restart hostapd
sudo systemctl restart dnsmasq

# Check network service logs
sudo journalctl -u hostapd -n 20
sudo journalctl -u dnsmasq -n 20
sudo journalctl -u dhcpcd -n 20
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
After running the complete setup script and rebooting:
1. The Pi connects to your internet WiFi via wlan0
2. The Pi creates a WiFi hotspot "ReViision-TestBench" on wlan1
3. Internet traffic is forwarded from wlan1 → wlan0 via NAT
4. The service starts automatically
5. Connect your Tapo camera to the ReViision-TestBench hotspot
6. Connect your Windows computer to the ReViision-TestBench hotspot
7. The system provides network infrastructure for data collection

### Network Setup Verification
After running the SSH-safe setup script and reconnecting:
```bash
# Verify the dual WiFi setup is working correctly
bash scripts/verify_network_setup.sh
```

This verification script will check:
- All network services are running
- Both WiFi interfaces have correct IP addresses
- Hotspot is broadcasting and accessible
- Internet connectivity is working
- NAT forwarding rules are active
- Any connected devices via DHCP

### Updating WiFi Credentials
To change WiFi passwords or connect to different networks:
```bash
# Interactive script to update WiFi credentials
bash scripts/update_wifi_credentials.sh
```

This script allows you to:
- Update internet WiFi connection (wlan0) SSID and password
- Update hotspot (wlan1) SSID and password
- Configure WPA2 Enterprise authentication for internet connection
- Update both or just one network at a time
- Automatically backup existing configurations
- Handle SSH disconnection safely when changing internet WiFi
- Restart network services with proper timing

**Options available:**
1. Internet WiFi credentials only
2. Hotspot credentials only  
3. Both internet and hotspot credentials
4. Show current network status

### Manual Operation
```bash
# Start manually (if auto-start is disabled)
cd /home/admin/pi_testbench  # Note: updated path for admin user
source venv/bin/activate
python3 main.py
```

### Network Configuration Options

#### 1. Dual WiFi Setup (Implemented & Recommended)
- **wlan0**: Connects to existing WiFi network for internet access
- **wlan1**: Broadcasts WiFi hotspot for test bench devices
- **NAT Forwarding**: Routes traffic from hotspot to internet
- **Benefits**: Best performance, reliability, and isolation
- **Setup**: Use `complete_network_setup.sh` script

#### 2. Ethernet + WiFi Hotspot (Alternative)
- Pi connects to internet via Ethernet
- Broadcasts WiFi hotspot for devices
- Good performance and reliability
- Requires Ethernet connection

#### 3. Hotspot Only (Development/Testing)
- No internet forwarding
- Isolated test environment
- Camera operates offline only

## Troubleshooting

### Service Issues
```bash
# Check main service status
sudo systemctl status reviision-pi

# Check all network services
sudo systemctl status hostapd dnsmasq dhcpcd wpa_supplicant@wlan0 reviision-network-prep

# View detailed logs
sudo journalctl -u reviision-pi -n 100
sudo journalctl -u hostapd -n 50
sudo journalctl -u dnsmasq -n 50

# Restart service
sudo systemctl restart reviision-pi

# Check Python dependencies
cd /home/admin/pi_testbench
source venv/bin/activate
python3 -c "import yaml, requests, psutil, netifaces"
```

### Network Issues
```bash
# Check interface status
ip addr show wlan0 wlan1
iwconfig

# Check hotspot status
sudo systemctl status hostapd
sudo systemctl status dnsmasq

# Check internet connectivity
ping -I wlan0 8.8.8.8
ping -I wlan1 192.168.4.1

# View NAT forwarding rules
sudo iptables -L -n -v
sudo iptables -t nat -L -n -v

# Test DHCP on hotspot
sudo nmap -sn 192.168.4.0/24

# Check WiFi interface assignment
iw dev
```

### Common Solutions
1. **Service won't start**: Check Python dependencies and configuration file
2. **No WiFi hotspot**: Verify hostapd configuration and wlan1 interface availability
3. **No internet forwarding**: Check iptables rules and wlan0 connectivity
4. **High CPU usage**: Check for infinite loops in logs
5. **Network service conflicts**: Use `complete_network_setup.sh` to resolve dependencies
6. **Interface assignment issues**: Check that wlan0/wlan1 are properly configured

### Network-Specific Troubleshooting
```bash
# Restart all network services in correct order
sudo systemctl restart reviision-network-prep
sudo systemctl restart dhcpcd
sudo systemctl restart wpa_supplicant@wlan0
sudo systemctl restart hostapd
sudo systemctl restart dnsmasq

# Check interface IP assignments
ip addr show | grep -A 3 wlan

# Test hotspot connectivity
# From another device, connect to ReViision-TestBench and test:
ping 192.168.4.1        # Pi hotspot interface
ping 8.8.8.8            # Internet via NAT forwarding
```

## File Structure

```
pi_testbench/
├── main.py                           # Main application entry
├── requirements.txt                  # Python dependencies
├── .env                             # Environment variables
├── config/                          # Configuration files
│   ├── pi_config.yaml              # Main configuration
│   ├── hostapd.conf                # WiFi hotspot settings (wlan1)
│   ├── dnsmasq.conf                # DHCP and DNS settings (wlan1)
│   ├── dhcpcd.conf                 # Network interface configuration
│   └── wpa_supplicant-wlan0.conf   # Internet WiFi connection (wlan0)
├── src/                             # Source code
│   ├── network/                    # Network management
│   ├── data/                       # Data collection from ReViision
│   └── utils/                      # Utilities
├── services/                        # Systemd service files
│   └── reviision-pi.service        # Auto-start service
├── scripts/                         # Helper scripts
│   ├── complete_network_setup.sh   # Comprehensive dual WiFi setup
│   ├── complete_network_setup_ssh_safe.sh # SSH-safe network setup
│   ├── verify_network_setup.sh     # Network setup verification
│   ├── update_wifi_credentials.sh  # Update WiFi SSID/passwords
│   ├── status.sh                   # System status checker
│   ├── logs.sh                     # Log viewer
│   └── restart.sh                  # Service restart
└── logs/                            # Log files
```

## API Integration

The Pi connects to these ReViision API endpoints:
- `/api/analyze_frame`: Real-time frame analysis
- `/api/traffic_patterns`: Traffic pattern data
- `/api/facial_analysis`: Demographics data
- `/api/config`: System configuration

## Security

- WiFi hotspot uses WPA2 encryption
- Internet WiFi connection supports WPA2 and WPA2 Enterprise
- API connections use authentication tokens
- Firewall configured with NAT forwarding rules
- Service runs with limited privileges
- Regular security updates via systemd timers

## Performance Optimization

### System Settings
```bash
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