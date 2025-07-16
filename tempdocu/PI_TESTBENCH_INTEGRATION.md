# 🔧 Pi Testbench Integration Guide

## Overview
The **Pi Testbench** is a Raspberry Pi-based deployment system that allows ReViision to run in field conditions with WiFi hotspot management and remote monitoring capabilities.

## 🏗️ **System Architecture**

### Main ReViision (Desktop/Server)
- Full analytics and web interface
- High-performance processing
- Database management
- Development and testing

### Pi Testbench (Field Deployment)  
- Lightweight Raspberry Pi system
- WiFi hotspot management
- Data collection and forwarding
- System monitoring
- Remote access capabilities

## 📋 **Prerequisites**

### Hardware Requirements
- **Raspberry Pi 4B (4GB RAM minimum)**
- **MicroSD Card (32GB+ Class 10)**
- **USB Camera or RTSP-capable IP camera**
- **WiFi Adapter (for dual WiFi setup)** or **Ethernet connection**
- **Power Supply (5V 3A)**

### Software Requirements
- **Raspberry Pi OS (64-bit recommended)**
- **Python 3.9+**
- **Git**
- **Node.js (for some utilities)**

## 🚀 **Installation Steps**

### 1. **Raspberry Pi Setup**

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3-pip python3-venv git hostapd dnsmasq iptables

# Clone the repository
git clone https://your-repo-url.git /home/pi/ReViision
cd /home/pi/ReViision/pi_testbench
```

### 2. **Python Environment Setup**

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. **Network Configuration**

#### Option A: Dual WiFi Setup (Recommended)
```bash
# Copy network configuration scripts
sudo cp scripts/setup_dual_wifi.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/setup_dual_wifi.sh

# Run dual WiFi setup
sudo /usr/local/bin/setup_dual_wifi.sh
```

#### Option B: Ethernet + WiFi Hotspot
```bash
# Copy network configuration
sudo cp config/dhcpcd.conf /etc/dhcpcd.conf
sudo cp config/dnsmasq.conf /etc/dnsmasq.conf
sudo cp config/hostapd.conf /etc/hostapd/hostapd.conf

# Enable services
sudo systemctl enable hostapd
sudo systemctl enable dnsmasq
```

### 4. **ReViision Service Setup**

```bash
# Copy service file
sudo cp services/reviision-pi.service /etc/systemd/system/

# Enable and start service
sudo systemctl enable reviision-pi.service
sudo systemctl start reviision-pi.service
```

## ⚙️ **Configuration**

### Pi Testbench Configuration (`config/pi_config.yaml`)

```yaml
# Network Configuration
network:
  # WiFi Hotspot Settings
  hotspot:
    ssid: "ReViision-TestBench"
    password: "testbench2024"
    interface: "wlan0"
    ip_range: "192.168.4.0/24"
    gateway: "192.168.4.1"
    
  # Internet Connection
  internet:
    method: "ethernet"  # ethernet, wifi, or disabled
    
# ReViision Server Connection
reviision:
  # Main server connection
  server:
    host: "your-main-server.com"
    port: 5000
    api_key: "your-api-key"
    
  # Data collection settings
  data:
    collection_interval: 60  # seconds
    batch_size: 100
    retry_attempts: 3
    
# Camera Configuration (local processing)
camera:
  type: "usb"
  device: 0
  resolution: [1280, 720]
  fps: 15  # Lower FPS for Pi
  
# System Monitoring
monitoring:
  cpu_threshold: 80
  memory_threshold: 85
  temperature_threshold: 75
  disk_threshold: 90
```

### Main ReViision Camera Configuration for Pi
Update your main `src/config.yaml`:

```yaml
camera:
  type: usb                    # Use USB camera on Pi
  device: 0                    # First USB camera
  fps: 15                      # Reduced FPS for Pi performance
  resolution: [1280, 720]      # HD resolution
  retry_interval: 5.0
  max_retries: -1
  
  # Pi-specific optimizations
  buffer_size: 1               # Minimal buffering
  threading: true              # Enable threaded capture
  
# Analysis optimizations for Pi
analysis:
  demographics:
    detection_interval: 60     # Analyze every 60 frames (4 seconds at 15fps)
    min_face_size: 60          # Slightly larger minimum face
    confidence_threshold: 0.8  # Higher confidence threshold
    
detection:
  confidence_threshold: 0.6    # Higher confidence for fewer false positives
```

## 🔧 **Camera Type Switching on Pi**

### USB Camera Setup
```bash
# Check available cameras
ls /dev/video*

# Test camera
ffmpeg -f v4l2 -list_formats all -i /dev/video0
```

### IP Camera Setup
```yaml
camera:
  type: rtsp
  url: "rtsp://192.168.4.100:554/stream"
  timeout: 10.0
  retry_interval: 5.0
```

## 📊 **Monitoring and Management**

### System Status Commands
```bash
# Check Pi testbench status
sudo systemctl status reviision-pi

# View logs
sudo journalctl -u reviision-pi -f

# Monitor system resources
python3 src/utils/system_monitor.py

# Check network status
python3 src/network/hotspot_manager.py --status
```

### Web Interface Access
1. **Connect to Pi WiFi**: `ReViision-TestBench`
2. **Open browser**: `http://192.168.4.1:5000`
3. **Login**: Use your ReViision credentials

### Remote Management
```bash
# SSH access via hotspot
ssh pi@192.168.4.1

# Update configuration remotely
curl -X POST http://192.168.4.1:5000/api/config \
  -H "Content-Type: application/json" \
  -d '{"camera": {"type": "usb", "device": 0}}'
```

## 🔄 **Data Synchronization**

### Automatic Data Forwarding
The Pi testbench automatically forwards analytics data to your main server:

```python
# Data forwarding configuration
reviision_client = ReViisionClient({
    'server_url': 'https://your-main-server.com',
    'api_key': 'your-api-key',
    'sync_interval': 300,  # 5 minutes
    'offline_storage': True
})
```

### Manual Data Export
```bash
# Export analytics data
python3 scripts/export_data.py --format json --output /tmp/analytics.json

# Transfer to main server
scp /tmp/analytics.json user@main-server:/path/to/data/
```

## 🔧 **Troubleshooting**

### Camera Not Detected
```bash
# Check USB cameras
lsusb | grep -i camera

# Check video devices
ls -la /dev/video*

# Test camera access
sudo apt install v4l-utils
v4l2-ctl --list-devices
```

### WiFi Issues
```bash
# Restart network services
sudo systemctl restart hostapd
sudo systemctl restart dnsmasq

# Check WiFi interface status
iwconfig

# Reset network configuration
sudo /usr/local/bin/setup_dual_wifi.sh
```

### Performance Issues
```bash
# Monitor CPU temperature
vcgencmd measure_temp

# Check memory usage
free -h

# Optimize for Pi
sudo raspi-config
# > Advanced Options > Memory Split > Set to 128
```

### Service Issues
```bash
# Restart ReViision service
sudo systemctl restart reviision-pi

# Check service logs
sudo journalctl -u reviision-pi --since "1 hour ago"

# Manual service start (for debugging)
cd /home/pi/ReViision/pi_testbench
source venv/bin/activate
python3 main.py --debug
```

## 📱 **Mobile Access**

### Setup Mobile Hotspot Connection
1. Connect phone/tablet to `ReViision-TestBench` WiFi
2. Open browser to `http://192.168.4.1:5000`
3. Access full analytics dashboard
4. Configure camera settings remotely

### API Access
```bash
# Get system status
curl http://192.168.4.1:5000/api/status

# Update camera configuration
curl -X POST http://192.168.4.1:5000/api/config \
  -H "Content-Type: application/json" \
  -d '{"camera": {"type": "rtsp", "url": "rtsp://camera-ip/stream"}}'
```

## 🔐 **Security Considerations**

### WiFi Security
- Change default hotspot password
- Use WPA2 encryption
- Enable MAC address filtering if needed

### System Security
- Change default Pi password
- Enable SSH key authentication
- Configure firewall rules
- Regular system updates

## 🚀 **Quick Start Script**

Create a one-click deployment script:

```bash
#!/bin/bash
# quick_deploy.sh - One-click Pi testbench deployment

echo "🚀 Starting ReViision Pi Testbench Setup..."

# Update system
sudo apt update -y

# Install dependencies
sudo apt install -y python3-pip python3-venv git hostapd dnsmasq

# Setup network
sudo cp config/dhcpcd.conf /etc/dhcpcd.conf
sudo cp config/dnsmasq.conf /etc/dnsmasq.conf  
sudo cp config/hostapd.conf /etc/hostapd/hostapd.conf

# Setup service
sudo cp services/reviision-pi.service /etc/systemd/system/
sudo systemctl enable reviision-pi

echo "✅ Setup complete! Reboot to activate."
echo "📱 Connect to 'ReViision-TestBench' WiFi after reboot"
echo "🌐 Access: http://192.168.4.1:5000"
```

---

## Summary

The Pi Testbench integration provides:
- **🔧 Easy camera switching** through web interface
- **📡 WiFi hotspot** for field access  
- **📊 Real-time analytics** processing
- **🔄 Data synchronization** with main server
- **📱 Mobile-friendly** remote management
- **⚡ Optimized performance** for Raspberry Pi hardware

**Ready for field deployment with professional analytics capabilities!** 