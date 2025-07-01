#!/bin/bash
# Quick Setup for ReViision Pi Test Bench Auto-Start
# This script configures auto-startup without full installation

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}ReViision Pi Test Bench - Quick Auto-Start Setup${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Check if running as pi user
if [[ "$(whoami)" != "pi" ]]; then
    echo -e "${RED}Error: This script must be run as the 'pi' user${NC}"
    exit 1
fi

# Check if we're in the right directory
if [[ ! -f "main.py" ]] || [[ ! -f "requirements.txt" ]]; then
    echo -e "${RED}Error: Please run this script from the pi_testbench directory${NC}"
    exit 1
fi

PROJECT_DIR="/home/pi/ReViision"
TESTBENCH_DIR="$PROJECT_DIR/pi_testbench"
CURRENT_DIR="$(pwd)"

echo -e "${GREEN}Step 1: Setting up project directory...${NC}"
# Copy current directory to standard location if not already there
if [[ "$CURRENT_DIR" != "$TESTBENCH_DIR" ]]; then
    mkdir -p "$PROJECT_DIR"
    cp -r "$(dirname "$CURRENT_DIR")" "$PROJECT_DIR/"
    echo "Project copied to: $TESTBENCH_DIR"
else
    echo "Already in project directory: $TESTBENCH_DIR"
fi

cd "$TESTBENCH_DIR"

echo -e "${GREEN}Step 2: Installing Python dependencies...${NC}"
# Create virtual environment if it doesn't exist
if [[ ! -d "venv" ]]; then
    python3 -m venv venv
fi

# Install dependencies
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo -e "${GREEN}Step 3: Installing systemd service...${NC}"
# Copy service file
sudo cp services/reviision-pi.service /etc/systemd/system/

# Enable auto-start
sudo systemctl daemon-reload
sudo systemctl enable reviision-pi

echo -e "${GREEN}Step 4: Creating environment file...${NC}"
# Create .env file if it doesn't exist
if [[ ! -f ".env" ]]; then
    cat > .env << EOF
# ReViision Pi Test Bench Environment Variables
# Network configuration
REVIISION_SERVER_HOST=192.168.4.10
REVIISION_SERVER_PORT=5000
WIFI_SSID=ReViision-TestBench
WIFI_PASSWORD=testbench2024

# Logging
LOG_LEVEL=INFO

# Hardware (set to true for testing without hardware)
DISABLE_EPAPER=false
DISABLE_GPIO=false
EOF
    echo "Created .env configuration file"
else
    echo ".env file already exists"
fi

echo -e "${GREEN}Step 5: Setting permissions...${NC}"
chown -R pi:pi "$TESTBENCH_DIR"
chmod +x main.py
chmod +x scripts/*.sh

# Create logs directory
mkdir -p logs cache

echo ""
echo -e "${GREEN}Auto-Start Setup Complete!${NC}"
echo -e "${GREEN}=========================${NC}"
echo ""
echo "Service Status: $(sudo systemctl is-enabled reviision-pi 2>/dev/null || echo 'configured')"
echo "Project Location: $TESTBENCH_DIR"
echo ""
echo "Quick Commands:"
echo -e "• Start now:      ${YELLOW}sudo systemctl start reviision-pi${NC}"
echo -e "• Stop service:   ${YELLOW}sudo systemctl stop reviision-pi${NC}"
echo -e "• Check status:   ${YELLOW}sudo systemctl status reviision-pi${NC}"
echo -e "• View logs:      ${YELLOW}sudo journalctl -u reviision-pi -f${NC}"
echo ""

# Ask if user wants to start the service now
read -p "Start the service now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting ReViision Pi Test Bench service..."
    sudo systemctl start reviision-pi
    sleep 2
    echo ""
    echo "Service Status:"
    sudo systemctl status reviision-pi --no-pager -l
    echo ""
    echo -e "${GREEN}Service started! It will now start automatically on boot.${NC}"
    echo "View live logs with: sudo journalctl -u reviision-pi -f"
else
    echo ""
    echo -e "${YELLOW}Service configured but not started.${NC}"
    echo "Start it manually with: sudo systemctl start reviision-pi"
    echo "It will start automatically on next boot."
fi

echo ""
echo -e "${YELLOW}Note: For network configuration (WiFi hotspot), run:${NC}"
echo "  sudo bash scripts/setup_network.sh" 