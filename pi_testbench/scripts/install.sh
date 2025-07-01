#!/bin/bash
# ReViision Pi Test Bench Installation Script
# This script installs and configures the complete Pi test bench system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/home/pi/ReViision"
TESTBENCH_DIR="$PROJECT_DIR/pi_testbench"
SERVICE_NAME="reviision-pi"
USER="pi"

echo -e "${BLUE}ReViision Pi Test Bench Installation${NC}"
echo -e "${BLUE}====================================${NC}"
echo ""

# Check if running as pi user
if [[ "$(whoami)" != "pi" ]]; then
    echo -e "${RED}Error: This script must be run as the 'pi' user${NC}"
    echo "Usage: bash install.sh"
    exit 1
fi

# Check if we're on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo -e "${YELLOW}Warning: This doesn't appear to be a Raspberry Pi${NC}"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo -e "${GREEN}Step 1: Updating system packages...${NC}"
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo -e "${GREEN}Step 2: Installing system dependencies...${NC}"
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    hostapd \
    dnsmasq \
    iptables-persistent \
    wireless-tools \
    net-tools \
    curl \
    wget \
    htop \
    tmux \
    vim

# Install SPI and GPIO support
echo -e "${GREEN}Step 3: Configuring hardware interfaces...${NC}"
sudo raspi-config nonint do_spi 0  # Enable SPI
sudo raspi-config nonint do_i2c 0  # Enable I2C (might be useful)

# Create project directory
echo -e "${GREEN}Step 4: Setting up project directory...${NC}"
if [[ ! -d "$PROJECT_DIR" ]]; then
    mkdir -p "$PROJECT_DIR"
fi

# Check if we're already in the project directory
if [[ "$(pwd)" == *"pi_testbench"* ]]; then
    echo "Installing from current directory..."
    CURRENT_DIR="$(pwd)"
    if [[ ! -d "$TESTBENCH_DIR" ]]; then
        mkdir -p "$PROJECT_DIR"
        cp -r "$(dirname "$CURRENT_DIR")" "$PROJECT_DIR/"
    fi
else
    echo "Please run this script from the pi_testbench directory"
    exit 1
fi

cd "$TESTBENCH_DIR"

# Create Python virtual environment
echo -e "${GREEN}Step 5: Creating Python virtual environment...${NC}"
if [[ ! -d "venv" ]]; then
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
echo -e "${GREEN}Step 6: Installing Python dependencies...${NC}"
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo -e "${GREEN}Step 7: Creating directories...${NC}"
mkdir -p logs cache

# Set up configuration
echo -e "${GREEN}Step 8: Setting up configuration...${NC}"
if [[ ! -f "config/pi_config.yaml" ]]; then
    echo -e "${RED}Error: Configuration file not found!${NC}"
    exit 1
fi

# Set permissions
echo -e "${GREEN}Step 9: Setting permissions...${NC}"
chown -R pi:pi "$TESTBENCH_DIR"
chmod +x main.py
chmod +x scripts/*.sh

# Install systemd service
echo -e "${GREEN}Step 10: Installing systemd service...${NC}"
sudo cp services/reviision-pi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME

# Create environment file for service
cat > .env << EOF
# ReViision Pi Test Bench Environment Variables
# Uncomment and modify as needed

# Network configuration overrides
# REVIISION_SERVER_HOST=192.168.4.10
# REVIISION_SERVER_PORT=5000
# WIFI_SSID=ReViision-TestBench
# WIFI_PASSWORD=testbench2024

# Logging configuration
# LOG_LEVEL=INFO

# Hardware configuration
# DISABLE_EPAPER=false
# DISABLE_GPIO=false
EOF

# Network configuration prompt
echo ""
echo -e "${YELLOW}Network Configuration${NC}"
echo "Would you like to configure the network now?"
echo "This will set up WiFi hotspot and internet forwarding."
read -p "Configure network? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [[ -f "scripts/setup_network.sh" ]]; then
        sudo bash scripts/setup_network.sh
    else
        echo -e "${RED}Network setup script not found!${NC}"
    fi
else
    echo -e "${YELLOW}Network configuration skipped. You can run it later with:${NC}"
    echo "  sudo bash $TESTBENCH_DIR/scripts/setup_network.sh"
fi

# Test installation
echo ""
echo -e "${GREEN}Step 11: Testing installation...${NC}"
if python3 -c "import yaml, requests, PIL, numpy, psutil, netifaces; print('✓ All Python modules imported successfully')"; then
    echo -e "${GREEN}✓ Python dependencies OK${NC}"
else
    echo -e "${RED}✗ Python dependencies failed${NC}"
    exit 1
fi

# Create desktop shortcut (if desktop environment is available)
if command -v lxsession >/dev/null 2>&1; then
    echo -e "${GREEN}Step 12: Creating desktop shortcuts...${NC}"
    
    # Create desktop directory if it doesn't exist
    mkdir -p /home/pi/Desktop
    
    # Create start service shortcut
    cat > /home/pi/Desktop/start_reviision.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Start ReViision
Comment=Start ReViision Pi Test Bench
Exec=lxterminal -e "sudo systemctl start $SERVICE_NAME && sudo journalctl -u $SERVICE_NAME -f"
Icon=utilities-terminal
Terminal=false
Categories=Development;
EOF

    # Create stop service shortcut  
    cat > /home/pi/Desktop/stop_reviision.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Stop ReViision
Comment=Stop ReViision Pi Test Bench
Exec=lxterminal -e "sudo systemctl stop $SERVICE_NAME"
Icon=utilities-terminal
Terminal=false
Categories=Development;
EOF

    chmod +x /home/pi/Desktop/*.desktop
fi

# Create useful scripts
echo -e "${GREEN}Step 13: Creating utility scripts...${NC}"

# Status script
cat > scripts/status.sh << 'EOF'
#!/bin/bash
# Check ReViision Pi Test Bench status

echo "ReViision Pi Test Bench Status"
echo "=============================="
echo ""

# Service status
echo "Service Status:"
sudo systemctl status reviision-pi --no-pager -l

echo ""
echo "Network Status:"
echo "Hotspot (hostapd):"
sudo systemctl status hostapd --no-pager -l | head -n 3

echo "DHCP (dnsmasq):"  
sudo systemctl status dnsmasq --no-pager -l | head -n 3

echo ""
echo "Connected Clients:"
sudo grep "DHCPACK" /var/log/daemon.log | tail -n 5 | awk '{print $8, $9}' | sort | uniq

echo ""
echo "Internet Connectivity:"
if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
    echo "✓ Internet connection available"
else
    echo "✗ No internet connection"
fi

echo ""
echo "System Resources:"
echo "CPU Temperature: $(vcgencmd measure_temp 2>/dev/null | cut -d= -f2 || echo 'N/A')"
echo "Memory Usage: $(free -h | awk 'NR==2{printf "%.1f/%.1f GB (%.1f%%)\n", $3/1024/1024, $2/1024/1024, $3*100/$2 }')"
echo "Disk Usage: $(df -h / | awk 'NR==2{printf "%s/%s (%s)\n", $3, $2, $5}')"
EOF

# Log viewer script
cat > scripts/logs.sh << 'EOF'
#!/bin/bash
# View ReViision Pi Test Bench logs

echo "ReViision Pi Test Bench Logs"
echo "============================"
echo ""

if [[ "$1" == "follow" ]] || [[ "$1" == "-f" ]]; then
    echo "Following live logs (Ctrl+C to exit):"
    sudo journalctl -u reviision-pi -f
else
    echo "Recent logs (use 'bash logs.sh follow' for live view):"
    sudo journalctl -u reviision-pi --no-pager -l -n 50
fi
EOF

# Restart script
cat > scripts/restart.sh << 'EOF'
#!/bin/bash
# Restart ReViision Pi Test Bench service

echo "Restarting ReViision Pi Test Bench..."
sudo systemctl restart reviision-pi
echo "Service restarted. Checking status..."
sleep 2
sudo systemctl status reviision-pi --no-pager -l
EOF

chmod +x scripts/*.sh

# Final status
echo ""
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}===================${NC}"
echo ""
echo "Installation Summary:"
echo "• Project directory: $TESTBENCH_DIR"
echo "• Service name: $SERVICE_NAME"
echo "• Service status: $(sudo systemctl is-enabled $SERVICE_NAME)"
echo "• Auto-start: Enabled"
echo ""
echo "Useful Commands:"
echo "• Start service:    sudo systemctl start $SERVICE_NAME"
echo "• Stop service:     sudo systemctl stop $SERVICE_NAME" 
echo "• Restart service:  sudo systemctl restart $SERVICE_NAME"
echo "• View status:      bash scripts/status.sh"
echo "• View logs:        bash scripts/logs.sh"
echo "• Follow logs:      bash scripts/logs.sh follow"
echo ""
echo "Configuration files:"
echo "• Main config:      config/pi_config.yaml"
echo "• Environment:      .env"
echo ""

# Check if reboot is needed
if [[ ! -f /tmp/pi_network_configured ]]; then
    echo -e "${YELLOW}Network configuration may require a reboot to take effect.${NC}"
    read -p "Reboot now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Rebooting in 5 seconds..."
        sleep 5
        sudo reboot
    else
        echo "Remember to reboot before first use: sudo reboot"
    fi
else
    echo -e "${GREEN}Installation ready! The service will start automatically on next boot.${NC}"
    echo ""
    read -p "Start the service now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo systemctl start $SERVICE_NAME
        echo ""
        echo "Service started! View status with: bash scripts/status.sh"
    fi
fi 