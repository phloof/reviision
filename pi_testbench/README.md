# ReViision Pi Testbench
## Portable Field Deployment System

[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4B-red.svg)](https://www.raspberrypi.org/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![WiFi Hotspot](https://img.shields.io/badge/WiFi-Hotspot%20Ready-green.svg)](https://github.com/your-org/reviision)

The **ReViision Pi Testbench** is a portable, field-deployable version of the ReViision retail analytics system designed for Raspberry Pi 4B. It provides a complete networking infrastructure with WiFi hotspot capabilities, enabling remote deployment and testing of retail analytics in any location.

## 🎯 Key Features

### 📡 **Dual WiFi Network Management**
- **Internet Connection (wlan0)**: Connects to existing WiFi networks for internet access
- **WiFi Hotspot (wlan1)**: Creates dedicated test bench network for cameras and devices
- **NAT Forwarding**: Seamless internet access for hotspot-connected devices
- **Enterprise WiFi Support**: WPA2 and WPA2 Enterprise authentication
- **Automatic Failover**: Robust connection management with auto-reconnection

### 🔄 **Data Integration & Synchronisation**
- **Real-time Analytics**: Live data collection from ReViision server APIs
- **Local Caching**: Offline operation capability with data synchronisation
- **API Integration**: Seamless connection to main ReViision analytics endpoints
- **Health Monitoring**: System status monitoring and performance metrics

### ⚙️ **System Management**
- **Auto-start Services**: Systemd integration for reliable boot-up
- **Service Monitoring**: Automatic restart and health checks
- **Log Management**: Comprehensive logging with rotation
- **Remote Access**: SSH and web interface access
- **Configuration Management**: YAML-based configuration with environment overrides

### 🛡️ **Security & Reliability**
- **WPA2 Encryption**: Secure WiFi hotspot with configurable credentials
- **Firewall Integration**: iptables-based NAT forwarding and security rules
- **Limited Privileges**: Service runs with minimal system permissions
- **Audit Logging**: Comprehensive activity logging for troubleshooting

## 🏗️ System Architecture

### Network Topology
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Internet      │    │  Raspberry Pi   │    │   Test Bench    │
│   (Existing     │───▶│     4B          │───▶│    Network      │
│    WiFi)        │    │                 │    │  (192.168.4.x)  │
└─────────────────┘    │  ┌───────────┐  │    └─────────────────┘
                       │  │   wlan0   │  │             │
                       │  │(Internet) │  │             │
                       │  └───────────┘  │             │
                       │        │        │             │
                       │   NAT Routing   │             │
                       │        │        │             │
                       │  ┌───────────┐  │             │
                       │  │   wlan1   │  │             │
                       │  │ (Hotspot) │  │             │
                       │  └───────────┘  │             │
                       └─────────────────┘             │
                                │                      │
                                └──────────────────────┘
                                         │
                       ┌─────────────────┼─────────────────┐
                       │                 │                 │
                ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
                │   Camera    │   │   Laptop    │   │   Mobile    │
                │ (RTSP/USB)  │   │  (Client)   │   │  (Client)   │
                └─────────────┘   └─────────────┘   └─────────────┘
```

### Component Overview
- **Raspberry Pi 4B**: Main controller with dual WiFi interface management
- **Network Services**: hostapd, dnsmasq, dhcpcd, wpa_supplicant
- **ReViision Client**: Data collection and API integration service
- **System Monitor**: Health checks and performance monitoring
- **Configuration Manager**: YAML-based settings with secure credential handling

## 📋 Requirements

### Hardware Requirements
- **Raspberry Pi 4B** (4GB RAM minimum, 8GB recommended)
- **MicroSD Card** (32GB+ Class 10 or better)
- **Power Supply** (5V 3A official Raspberry Pi power adapter)
- **WiFi Capability** (Built-in WiFi for dual-interface setup)
- **Optional**: USB WiFi adapter for enhanced dual-WiFi performance

### Recommended Operating System
**Raspberry Pi OS Lite (64-bit)** - Optimised for headless operation:

**Benefits:**
- Lightweight (no desktop environment)
- Official Raspberry Pi Foundation support
- Excellent hardware compatibility
- Regular security updates
- Optimised for Pi hardware

**Installation:**
1. Download [Raspberry Pi Imager](https://www.raspberrypi.org/software/)
2. Select "Raspberry Pi OS Lite (64-bit)"
3. Configure WiFi and SSH in advanced options
4. Flash to microSD card

### Network Configuration
- **Internet Connection**: Existing WiFi network with internet access
- **Hotspot Network**: 192.168.4.0/24 subnet (configurable)
- **DHCP Range**: 192.168.4.10-192.168.4.50 (configurable)
- **DNS Services**: Local DNS with internet forwarding

## 🚀 Quick Start

### 1. Prepare Raspberry Pi
```bash
# Flash Raspberry Pi OS Lite (64-bit) to microSD card
# Enable SSH and configure initial WiFi during imaging
# Boot Pi and SSH in

# Update system
sudo apt update && sudo apt upgrade -y
```

### 2. Clone Repository
```bash
# Clone the ReViision repository
git clone https://github.com/your-org/reviision.git
cd reviision/pi_testbench
```

### 3. Automated Setup (Recommended)
```bash
# For SSH connections (recommended)
bash scripts/complete_network_setup_ssh_safe.sh

# OR for direct console access
sudo bash scripts/complete_network_setup.sh
```

### 4. Verify Installation
```bash
# Check system status
bash scripts/status.sh

# Verify network setup
bash scripts/verify_network_setup.sh
```

### 5. Connect Devices
1. **Reboot the Pi**: `sudo reboot`
2. **Connect to hotspot**: Look for "ReViision-TestBench" WiFi network
3. **Default password**: `testbench2024`
4. **Access web interface**: http://192.168.4.1:5000

## 📦 Installation Guide

### Automated Installation

#### SSH-Safe Setup (Recommended)
The SSH-safe installation is designed for remote deployment:

```bash
cd pi_testbench
bash scripts/complete_network_setup_ssh_safe.sh
```

**Features:**
- ✅ Automatic SSH session detection
- ✅ Warning prompts before network service restart
- ✅ 30-second countdown with cancel option
- ✅ Clear reconnection instructions
- ✅ All configuration completed before network restart

#### Standard Setup
For direct console access or when SSH disconnection is acceptable:

```bash
sudo bash scripts/complete_network_setup.sh
```

**What the setup script does:**
1. **System Updates**: Updates all packages and installs dependencies
2. **Network Configuration**: Configures dual WiFi interfaces (wlan0/wlan1)
3. **Service Setup**: Installs and configures hostapd, dnsmasq, dhcpcd
4. **Firewall Rules**: Sets up NAT forwarding and iptables rules
5. **Python Environment**: Creates virtual environment and installs dependencies
6. **Systemd Services**: Configures auto-start services with proper dependencies
7. **Security**: Sets appropriate file permissions and service isolation

### Manual Installation

#### 1. System Dependencies
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y hostapd dnsmasq python3-pip python3-venv git \
    wireless-tools net-tools iptables-persistent

# Install optional monitoring tools
sudo apt install -y htop iotop nethogs
```

#### 2. Python Environment
```bash
# Navigate to pi_testbench directory
cd pi_testbench

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Network Configuration
```bash
# Run network configuration script
sudo bash scripts/complete_network_setup.sh

# OR configure manually using individual config files
sudo cp config/dhcpcd.conf /etc/dhcpcd.conf
sudo cp config/hostapd.conf /etc/hostapd/hostapd.conf
sudo cp config/dnsmasq.conf /etc/dnsmasq.conf
```

#### 4. Service Installation
```bash
# Copy systemd service file
sudo cp services/reviision-pi.service /etc/systemd/system/

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable reviision-pi

# Start the service
sudo systemctl start reviision-pi
```

## ⚙️ Configuration

### Main Configuration File

Edit `config/pi_config.yaml` to customise your deployment:

```yaml
# ReViision Pi Testbench Configuration
network:
  hotspot:
    ssid: "ReViision-TestBench"        # Hotspot network name
    password: "testbench2024"          # Hotspot password (WPA2)
    interface: "wlan1"                 # Hotspot interface
    ip_range: "192.168.4.0/24"         # Network subnet
    gateway: "192.168.4.1"             # Pi gateway IP
    dhcp_range: "192.168.4.10,192.168.4.50"  # DHCP IP range

  server:
    host: "192.168.4.10"               # ReViision server IP
    port: 5000                         # ReViision server port
    protocol: "http"                   # http or https
    timeout: 10                        # Connection timeout (seconds)
    retry_attempts: 3                  # Connection retry attempts
    retry_delay: 5                     # Delay between retries

data:
  polling:
    analytics: 15                      # Analytics polling interval (seconds)
    system_status: 60                  # System status check interval
    network_status: 120                # Network status check interval

  cache:
    max_age: 3600                      # Cache expiry (seconds)
    max_entries: 1000                  # Maximum cache entries

logging:
  level: "INFO"                        # DEBUG, INFO, WARNING, ERROR
  log_file: "logs/pi_testbench.log"    # Log file location
  max_size: "10MB"                     # Maximum log file size
  backup_count: 5                      # Number of backup log files

system:
  startup_delay: 30                    # Startup delay for network stabilisation
  health_check_interval: 300           # Health check interval (seconds)
  auto_restart: true                   # Auto-restart on failure
```

### Network Configuration Files

The system uses several network configuration files:

#### WiFi Hotspot Configuration (`config/hostapd.conf`)
```bash
# WiFi hotspot settings for wlan1
interface=wlan1
driver=nl80211
ssid=ReViision-TestBench
hw_mode=g
channel=7
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=testbench2024
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
```

#### DHCP Configuration (`config/dnsmasq.conf`)
```bash
# DHCP and DNS settings for hotspot
interface=wlan1
dhcp-range=192.168.4.10,192.168.4.50,255.255.255.0,24h
domain=local
address=/gw.local/192.168.4.1
```

#### Interface Configuration (`config/dhcpcd.conf`)
```bash
# Network interface configuration
interface wlan1
    static ip_address=192.168.4.1/24
    nohook wpa_supplicant
```

### Environment Variables

Create `.env` file for runtime configuration overrides:

```bash
# Network Configuration
REVIISION_SERVER_HOST=192.168.4.10
REVIISION_SERVER_PORT=5000
WIFI_SSID=ReViision-TestBench
WIFI_PASSWORD=testbench2024

# Internet WiFi (used by setup scripts)
INTERNET_WIFI_SSID=YourWiFiNetwork
INTERNET_WIFI_PASSWORD=your_wifi_password

# System Configuration
LOG_LEVEL=INFO
STARTUP_DELAY=30
HEALTH_CHECK_INTERVAL=300

# Security
API_TOKEN=your_api_token_here
ENCRYPTION_KEY=your_encryption_key
```

### Enterprise WiFi Configuration

For WPA2 Enterprise networks, configure `config/wpa_supplicant-wlan0.conf`:

```bash
# WPA2 Enterprise configuration for internet connection
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=AU

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
    priority=1
}

# Fallback to regular WPA2 if available
network={
    ssid="Backup-WiFi"
    psk="backup_password"
    priority=0
}
```

**Certificate Installation:**
```bash
# Copy enterprise certificates to the Pi
sudo cp company-ca.crt /etc/ssl/certs/
sudo chmod 644 /etc/ssl/certs/company-ca.crt
sudo update-ca-certificates
```

## 🖥️ Usage & Management

### Service Management

#### Main Application Service
```bash
# Start the ReViision Pi service
sudo systemctl start reviision-pi

# Stop the service
sudo systemctl stop reviision-pi

# Restart the service
sudo systemctl restart reviision-pi

# Check service status
sudo systemctl status reviision-pi

# Enable auto-start on boot
sudo systemctl enable reviision-pi

# Disable auto-start
sudo systemctl disable reviision-pi

# View service logs
sudo journalctl -u reviision-pi -f
```

#### Network Services
The system manages several network services:

| Service | Purpose | Interface |
|---------|---------|-----------|
| `hostapd` | WiFi hotspot | wlan1 |
| `dnsmasq` | DHCP & DNS server | wlan1 |
| `dhcpcd` | Network interface management | All |
| `wpa_supplicant@wlan0` | Internet WiFi connection | wlan0 |
| `reviision-network-prep` | Network preparation | All |

```bash
# Check all network services
sudo systemctl status hostapd dnsmasq dhcpcd wpa_supplicant@wlan0

# Restart network services in correct order
sudo systemctl restart reviision-network-prep
sudo systemctl restart dhcpcd
sudo systemctl restart wpa_supplicant@wlan0
sudo systemctl restart hostapd
sudo systemctl restart dnsmasq

# View network service logs
sudo journalctl -u hostapd -n 20
sudo journalctl -u dnsmasq -n 20
sudo journalctl -u dhcpcd -n 20
```

### Convenience Scripts

The Pi testbench includes several helper scripts for common tasks:

```bash
# Check complete system status
bash scripts/status.sh

# View system logs
bash scripts/logs.sh              # Recent logs
bash scripts/logs.sh follow       # Live log following

# Restart the main service
bash scripts/restart.sh

# Verify network setup
bash scripts/verify_network_setup.sh

# Update WiFi credentials
bash scripts/update_wifi_credentials.sh
```

### Network Management

#### Automatic Operation
After successful setup and reboot:

1. **Internet Connection**: Pi connects to configured WiFi via wlan0
2. **Hotspot Creation**: Pi broadcasts "ReViision-TestBench" on wlan1
3. **NAT Forwarding**: Internet traffic routed from wlan1 → wlan0
4. **Service Startup**: ReViision Pi service starts automatically
5. **Device Connection**: Cameras and clients connect to hotspot
6. **Data Collection**: System begins analytics data collection

#### Manual Operation
```bash
# Start manually (if auto-start disabled)
cd /home/admin/pi_testbench
source venv/bin/activate
python3 main.py

# Run with debug logging
python3 main.py --debug

# Run with custom configuration
python3 main.py --config config/custom_config.yaml
```

### WiFi Credential Management

#### Update WiFi Settings
```bash
# Interactive credential update
bash scripts/update_wifi_credentials.sh
```

**Available options:**
1. **Internet WiFi only**: Update wlan0 connection credentials
2. **Hotspot only**: Update wlan1 hotspot SSID/password
3. **Both networks**: Update internet and hotspot credentials
4. **Show status**: Display current network configuration

#### Manual WiFi Configuration
```bash
# Edit internet WiFi settings
sudo nano config/wpa_supplicant-wlan0.conf

# Edit hotspot settings
sudo nano config/hostapd.conf

# Restart network services after changes
sudo systemctl restart wpa_supplicant@wlan0
sudo systemctl restart hostapd
```

### System Monitoring

#### Real-time Monitoring
```bash
# System resources
htop

# Network traffic
sudo nethogs

# Disk I/O
sudo iotop

# Network interfaces
watch -n 2 'ip addr show wlan0 wlan1'

# Connected devices
watch -n 5 'sudo nmap -sn 192.168.4.0/24'
```

#### Log Monitoring
```bash
# Follow all logs
sudo journalctl -f

# Follow specific service
sudo journalctl -u reviision-pi -f

# View recent errors
sudo journalctl --since "1 hour ago" -p err

# View network service logs
sudo journalctl -u hostapd -u dnsmasq -u dhcpcd --since today
```

### Network Verification

#### Verify Setup
```bash
# Comprehensive network verification
bash scripts/verify_network_setup.sh
```

**Verification checks:**
- ✅ All network services running correctly
- ✅ WiFi interfaces have correct IP addresses
- ✅ Hotspot broadcasting and accessible
- ✅ Internet connectivity working
- ✅ NAT forwarding rules active
- ✅ DHCP clients and IP assignments
- ✅ DNS resolution functioning

#### Manual Network Checks
```bash
# Check interface status
ip addr show wlan0 wlan1

# Check WiFi interface details
iwconfig

# Test internet connectivity
ping -I wlan0 8.8.8.8

# Test hotspot connectivity
ping -I wlan1 192.168.4.1

# Check NAT forwarding rules
sudo iptables -L -n -v
sudo iptables -t nat -L -n -v

# Scan for connected devices
sudo nmap -sn 192.168.4.0/24
```

### Deployment Scenarios

#### 1. Dual WiFi Setup (Recommended)
**Configuration:**
- **wlan0**: Internet connection via existing WiFi
- **wlan1**: WiFi hotspot for test bench devices
- **NAT Forwarding**: Routes traffic from hotspot to internet

**Benefits:**
- ✅ Best performance and reliability
- ✅ Complete network isolation for test devices
- ✅ Internet access for cameras and clients
- ✅ Remote monitoring and management

**Setup:**
```bash
bash scripts/complete_network_setup_ssh_safe.sh
```

#### 2. Ethernet + WiFi Hotspot
**Configuration:**
- **eth0**: Internet connection via Ethernet cable
- **wlan0**: WiFi hotspot for test bench devices

**Benefits:**
- ✅ Stable wired internet connection
- ✅ Dedicated WiFi for hotspot
- ✅ Good performance for high-bandwidth applications

**Setup:**
```bash
# Modify network configuration for Ethernet
sudo nano config/dhcpcd.conf
# Add: interface eth0
# Remove wlan0 configuration
```

#### 3. Hotspot Only (Development/Testing)
**Configuration:**
- **wlan0**: WiFi hotspot only (no internet)
- Isolated test environment

**Benefits:**
- ✅ Simple setup
- ✅ Complete isolation
- ✅ No internet dependencies

**Use cases:**
- Development and testing
- Offline demonstrations
- Security-sensitive environments

## 🔧 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check service status and logs
sudo systemctl status reviision-pi
sudo journalctl -u reviision-pi -n 50

# Check Python environment
cd /home/admin/pi_testbench
source venv/bin/activate
python3 -c "import yaml, requests, psutil, netifaces; print('Dependencies OK')"

# Verify configuration file
python3 -c "import yaml; yaml.safe_load(open('config/pi_config.yaml'))"

# Check file permissions
ls -la config/
ls -la logs/
```

**Solutions:**
- Ensure virtual environment is properly created
- Verify all Python dependencies are installed
- Check configuration file syntax
- Ensure proper file permissions (logs directory writable)

#### WiFi Hotspot Not Working
```bash
# Check hostapd service
sudo systemctl status hostapd
sudo journalctl -u hostapd -n 20

# Check interface availability
iw dev
ip addr show wlan1

# Test hostapd configuration
sudo hostapd -d /etc/hostapd/hostapd.conf
```

**Solutions:**
- Verify wlan1 interface exists and is not in use
- Check hostapd configuration syntax
- Ensure WiFi adapter supports AP mode
- Restart network services in correct order

#### No Internet Forwarding
```bash
# Check internet connectivity on wlan0
ping -I wlan0 8.8.8.8

# Check NAT forwarding rules
sudo iptables -L -n -v
sudo iptables -t nat -L -n -v

# Check IP forwarding
cat /proc/sys/net/ipv4/ip_forward

# Test from hotspot client
# Connect device to hotspot and test:
ping 192.168.4.1    # Pi gateway
ping 8.8.8.8        # Internet via NAT
```

**Solutions:**
- Verify wlan0 has internet connectivity
- Check iptables NAT rules are configured
- Ensure IP forwarding is enabled
- Restart network services

#### High CPU/Memory Usage
```bash
# Monitor system resources
htop
free -h
df -h

# Check for problematic processes
ps aux | grep python
ps aux | grep hostapd

# Monitor network traffic
sudo nethogs
sudo iotop
```

**Solutions:**
- Reduce polling intervals in configuration
- Check for infinite loops in logs
- Monitor for memory leaks
- Consider reducing analytics frequency

### Network Troubleshooting

#### Interface Issues
```bash
# Check all network interfaces
ip addr show
iwconfig

# Check interface assignment
iw dev

# Reset network interfaces
sudo ip link set wlan0 down
sudo ip link set wlan1 down
sudo ip link set wlan0 up
sudo ip link set wlan1 up
```

#### Service Dependencies
```bash
# Check all network services
sudo systemctl status hostapd dnsmasq dhcpcd wpa_supplicant@wlan0 reviision-network-prep

# Restart services in correct order
sudo systemctl restart reviision-network-prep
sudo systemctl restart dhcpcd
sudo systemctl restart wpa_supplicant@wlan0
sudo systemctl restart hostapd
sudo systemctl restart dnsmasq
```

#### DHCP Issues
```bash
# Check DHCP leases
cat /var/lib/dhcp/dhcpcd.leases
sudo cat /var/lib/dhcpcd5/dhcpcd.leases

# Check dnsmasq DHCP
sudo journalctl -u dnsmasq -n 20

# Scan for connected devices
sudo nmap -sn 192.168.4.0/24

# Check DHCP configuration
sudo dnsmasq --test
```

### Performance Optimisation

#### System Optimisation
```bash
# Optimise Pi configuration
echo "arm_freq=1800" | sudo tee -a /boot/config.txt
echo "gpu_mem=128" | sudo tee -a /boot/config.txt
echo "dtoverlay=vc4-fkms-v3d" | sudo tee -a /boot/config.txt

# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable hciuart

# Optimise network settings
echo "net.core.rmem_max = 16777216" | sudo tee -a /etc/sysctl.conf
echo "net.core.wmem_max = 16777216" | sudo tee -a /etc/sysctl.conf
```

#### Application Optimisation
```yaml
# Optimise config/pi_config.yaml
data:
  polling:
    analytics: 30              # Increase polling interval
    system_status: 120         # Reduce status checks
    network_status: 300        # Reduce network checks

  cache:
    max_age: 7200             # Increase cache duration
    max_entries: 500          # Reduce cache size

logging:
  level: "WARNING"            # Reduce log verbosity
```

### Recovery Procedures

#### Complete Network Reset
```bash
# Stop all network services
sudo systemctl stop hostapd dnsmasq dhcpcd wpa_supplicant@wlan0

# Reset network configuration
sudo bash scripts/complete_network_setup.sh

# Reboot system
sudo reboot
```

#### Factory Reset
```bash
# Backup current configuration
cp -r config/ config_backup_$(date +%Y%m%d_%H%M%S)/

# Reset to default configuration
git checkout -- config/

# Reinstall system
bash scripts/complete_network_setup_ssh_safe.sh
```

#### Emergency SSH Access
If you lose SSH access after network changes:

1. **Physical Access**: Connect keyboard/monitor to Pi
2. **Serial Console**: Use UART pins for console access
3. **SD Card Recovery**: Mount SD card on another computer to fix configuration
4. **Network Recovery**: Reset network configuration via physical access

## 📁 Project Structure

```
pi_testbench/
├── 📄 main.py                           # Main application entry point
├── 📄 requirements.txt                  # Python dependencies
├── 📄 start_reviision.sh               # Service startup script
├── 📄 .env                             # Environment variables (create from env.sample)
├── 📄 env.sample                       # Environment variables template
│
├── 📁 config/                          # Configuration files
│   ├── 📄 pi_config.yaml              # Main application configuration
│   ├── 📄 hostapd.conf                # WiFi hotspot settings (wlan1)
│   ├── 📄 dnsmasq.conf                # DHCP and DNS settings (wlan1)
│   ├── 📄 dhcpcd.conf                 # Network interface configuration
│   └── 📄 wpa_supplicant-wlan0.conf   # Internet WiFi connection (wlan0)
│
├── 📁 src/                             # Source code modules
│   ├── 📁 network/                    # Network management
│   │   ├── 📄 __init__.py
│   │   └── 📄 hotspot_manager.py      # WiFi hotspot management
│   ├── 📁 data/                       # Data collection and integration
│   │   ├── 📄 __init__.py
│   │   └── 📄 reviision_client.py     # ReViision server client
│   └── 📁 utils/                      # Utility modules
│       ├── 📄 __init__.py
│       ├── 📄 config_manager.py       # Configuration management
│       ├── 📄 logger.py               # Logging utilities
│       └── 📄 system_monitor.py       # System monitoring
│
├── 📁 services/                        # Systemd service files
│   └── 📄 reviision-pi.service        # Auto-start service configuration
│
├── 📁 scripts/                         # Management and setup scripts
│   ├── 📄 complete_network_setup.sh           # Full dual WiFi setup
│   ├── 📄 complete_network_setup_ssh_safe.sh  # SSH-safe network setup
│   ├── 📄 verify_network_setup.sh             # Network verification
│   ├── 📄 update_wifi_credentials.sh          # WiFi credential management
│   ├── 📄 status.sh                           # System status checker
│   ├── 📄 logs.sh                             # Log viewer utility
│   └── 📄 restart.sh                          # Service restart utility
│
├── 📁 logs/                            # Log files (created automatically)
│   ├── 📄 pi_testbench.log            # Application logs
│   ├── 📄 startup.log                 # Startup logs
│   └── 📄 network.log                 # Network service logs
│
└── 📁 tests/                           # Test suite (optional)
    ├── 📄 test_network.py             # Network functionality tests
    ├── 📄 test_config.py              # Configuration tests
    └── 📄 test_integration.py         # Integration tests
```

## 🔌 API Integration

The Pi testbench integrates with the main ReViision system through these API endpoints:

### Analytics Endpoints
| Endpoint | Method | Purpose | Frequency |
|----------|--------|---------|-----------|
| `/api/analytics/summary` | GET | Current analytics overview | Every 15s |
| `/api/analytics/demographics` | GET | Demographic data | Every 30s |
| `/api/analytics/traffic_patterns` | GET | Traffic pattern data | Every 60s |
| `/api/analytics/heatmap` | GET | Heatmap data | Every 120s |

### System Endpoints
| Endpoint | Method | Purpose | Usage |
|----------|--------|---------|-------|
| `/api/system/status` | GET | System health check | Every 60s |
| `/api/system/config` | GET | Configuration data | On startup |
| `/api/camera/stream` | GET | Live camera feed | On demand |

### Authentication
```python
# API authentication example
headers = {
    'Authorization': f'Bearer {api_token}',
    'Content-Type': 'application/json'
}

response = requests.get(
    f'{server_url}/api/analytics/summary',
    headers=headers,
    timeout=10
)
```

## 🔒 Security Features

### Network Security
- **WPA2 Encryption**: WiFi hotspot uses WPA2-PSK encryption
- **Enterprise WiFi**: Support for WPA2 Enterprise authentication
- **Network Isolation**: Test bench devices isolated from main network
- **Firewall Rules**: iptables-based NAT forwarding with security rules
- **MAC Filtering**: Optional MAC address filtering for hotspot access

### Application Security
- **API Authentication**: Token-based authentication for ReViision server
- **Service Isolation**: Application runs with limited user privileges
- **Secure Configuration**: Sensitive credentials stored securely
- **Audit Logging**: Comprehensive logging for security monitoring
- **Auto-updates**: Systemd timers for security updates

### Best Practices
```bash
# Change default passwords
bash scripts/update_wifi_credentials.sh

# Enable firewall logging
sudo iptables -A INPUT -j LOG --log-prefix "INPUT: "
sudo iptables -A FORWARD -j LOG --log-prefix "FORWARD: "

# Monitor security logs
sudo journalctl -f | grep -E "(authentication|security|firewall)"

# Regular security updates
sudo apt update && sudo apt upgrade -y
```

## 🚀 Development & Testing

### Development Environment
```bash
# Set up development environment
cd pi_testbench
python3 -m venv venv_dev
source venv_dev/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio black isort flake8

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Code Quality
```bash
# Code formatting
black src/
isort src/

# Linting
flake8 src/

# Type checking
mypy src/
```

### Testing
```bash
# Run test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/ --cov-report=html

# Run specific test categories
pytest tests/test_network.py -v
pytest tests/test_integration.py -v
```

### Local Development
```bash
# Run application locally
python3 main.py --debug

# Run with custom configuration
python3 main.py --config config/dev_config.yaml

# Test network functionality
python3 -m pytest tests/test_network.py -v
```

## 📚 Additional Resources


### Useful Commands
```bash
# Quick system status
bash scripts/status.sh

# Network diagnostics
bash scripts/verify_network_setup.sh

# View live logs
bash scripts/logs.sh follow

# Update WiFi settings
bash scripts/update_wifi_credentials.sh
```

### Community Resources
- **GitHub Issues**: Report bugs and feature requests
- **GitHub Discussions**: Community support and questions
- **Wiki**: Additional documentation and tutorials


**ReViision Pi Testbench** - Portable retail analytics for field deployment.
nd support, please refer to the main [ReViision repository](https://github.com/your-org/reviision).