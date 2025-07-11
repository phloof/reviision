#!/bin/bash

# ReViision Pi Test Bench - SSH-Safe Complete Network Setup
# ==========================================================
# Configuration: wlan0=Internet, wlan1=Hotspot
# This version is designed to be safe for SSH connections

set -e

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "This script should not be run as root"
    echo "Run as: bash scripts/complete_network_setup_ssh_safe.sh"
    exit 1
fi

# Check if we're in an SSH session
SSH_WARNING=false
if [ -n "$SSH_CLIENT" ] || [ -n "$SSH_TTY" ] || [ -n "$SSH_CONNECTION" ]; then
    SSH_WARNING=true
fi

# Configuration
INTERNET_SSID="${INTERNET_WIFI_SSID:-ORBI58}"
INTERNET_PASS="${INTERNET_WIFI_PASSWORD}"
HOTSPOT_SSID="${WIFI_SSID:-ReViision-TestBench}"
HOTSPOT_PASS="${WIFI_PASSWORD:-testbench2024}"

echo "ReViision Pi Test Bench - SSH-Safe Complete Network Setup"
echo "=========================================================="
echo "Configuration: wlan0=Internet, wlan1=Hotspot"
echo ""

if [ "$SSH_WARNING" = true ]; then
    echo "⚠️  WARNING: SSH SESSION DETECTED ⚠️"
    echo "You are connected via SSH. This script will restart network services"
    echo "at the end, which will temporarily disconnect your SSH session."
    echo ""
    echo "The script will:"
    echo "1. Install and configure all components first"
    echo "2. Configure dual WiFi interfaces (wlan0 for internet, wlan1 for hotspot)"
    echo "3. Set up NAT forwarding and firewall rules"
    echo "4. Configure all network services (hostapd, dnsmasq, dhcpcd, wpa_supplicant)"
    echo "5. Create custom systemd services for network management"
    echo "6. Set up auto-start services with proper dependencies"
    echo "7. Handle service conflicts and ensure proper startup order"
    echo ""
    echo "After disconnection, wait 2 minutes then reconnect to verify setup."
    echo ""
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
    echo ""
fi

# Get internet WiFi password if not set
if [ -z "$INTERNET_PASS" ]; then
    echo "Internet WiFi Configuration:"
    read -p "Enter password for '$INTERNET_SSID': " -s INTERNET_PASS
    echo ""
fi

echo "Configuration:"
echo "  Internet WiFi: $INTERNET_SSID (wlan0)"
echo "  Hotspot: $HOTSPOT_SSID (wlan1)"
echo "  Hotspot IP: 192.168.4.1/24"
echo ""

# 1. Update system
echo "Step 1/8: Updating system packages..."
sudo apt update -qq
sudo apt upgrade -y -qq

# 2. Install dependencies
echo "Step 2/8: Installing dependencies..."
sudo apt install -y -qq hostapd dnsmasq iptables-persistent wpasupplicant \
    python3-pip python3-venv git wireless-tools net-tools dhcpcd5

# 3. Disable NetworkManager (without stopping - to prevent SSH disconnection)
echo "Step 4/8: Disabling NetworkManager..."
sudo systemctl disable NetworkManager 2>/dev/null || true
sudo systemctl mask NetworkManager 2>/dev/null || true

# 4. Create configuration files
echo "Step 5/8: Creating network configuration files..."

# dhcpcd.conf
sudo tee /etc/dhcpcd.conf > /dev/null << 'EOF'
# Custom dhcpcd configuration for ReViision dual WiFi setup
# wlan0: Internet connection (DHCP)
# wlan1: Hotspot interface (Static IP)

# Standard options
option rapid_commit
option domain_name_servers, domain_name, domain_search, host_name
option classless_static_routes
option interface_mtu
require dhcp_server_identifier
slaac private

# wlan0: Internet connection via DHCP
interface wlan0
# Use DHCP for internet connection

# wlan1: Static IP for hotspot
interface wlan1
static ip_address=192.168.4.1/24
nohook wpa_supplicant
EOF

# hostapd.conf
sudo tee /etc/hostapd/hostapd.conf > /dev/null << EOF
# ReViision Test Bench WiFi Hotspot Configuration
# Interface and driver
interface=wlan1
driver=nl80211

# Network settings
ssid=$HOTSPOT_SSID
hw_mode=g
channel=7
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0

# Security settings
wpa=2
wpa_passphrase=$HOTSPOT_PASS
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP

# Country code (adjust as needed)
country_code=US
ieee80211n=1
ieee80211d=1
EOF

# dnsmasq.conf
sudo tee /etc/dnsmasq.conf > /dev/null << 'EOF'
# ReViision Test Bench DHCP and DNS Configuration

# Interface to bind to
interface=wlan1

# DHCP range for hotspot clients
dhcp-range=192.168.4.10,192.168.4.50,255.255.255.0,24h

# DNS settings
domain-needed
bogus-priv
no-resolv

# Use Google DNS for upstream
server=8.8.8.8
server=8.8.4.4

# Local domain
local=/testbench/
domain=testbench
expand-hosts

# DHCP options
dhcp-option=option:router,192.168.4.1
dhcp-option=option:dns-server,192.168.4.1

# Logging (comment out for production)
log-queries
log-dhcp

# Cache size
cache-size=1000
EOF

# wpa_supplicant for wlan0
sudo tee /etc/wpa_supplicant/wpa_supplicant-wlan0.conf > /dev/null << EOF
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=US

# Internet WiFi connection
network={
    ssid="$INTERNET_SSID"
    psk="$INTERNET_PASS"
    priority=1
}
EOF

# 6. Create systemd services
echo "Step 6/8: Creating custom systemd services..."

# iptables restore service
sudo tee /etc/systemd/system/iptables-restore.service > /dev/null << 'EOF'
[Unit]
Description=Restore iptables NAT rules for ReViision
After=network.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c 'iptables -t nat -A POSTROUTING -o wlan0 -j MASQUERADE && iptables -A FORWARD -i wlan0 -o wlan1 -m state --state RELATED,ESTABLISHED -j ACCEPT && iptables -A FORWARD -i wlan1 -o wlan0 -j ACCEPT'
ExecStop=/bin/bash -c 'iptables -t nat -F POSTROUTING && iptables -F FORWARD'

[Install]
WantedBy=multi-user.target
EOF

# Network preparation service
sudo tee /etc/systemd/system/reviision-network-prep.service > /dev/null << 'EOF'
[Unit]
Description=ReViision Network Preparation
After=network.target dhcpcd.service
Before=hostapd.service dnsmasq.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c 'echo 1 > /proc/sys/net/ipv4/ip_forward'
ExecStart=/bin/bash -c 'ip link set wlan1 down || true'
ExecStart=/bin/bash -c 'sleep 2'
ExecStart=/bin/bash -c 'ip link set wlan1 up || true'

[Install]
WantedBy=multi-user.target
EOF

# Service override configurations
sudo mkdir -p /etc/systemd/system/hostapd.service.d
sudo tee /etc/systemd/system/hostapd.service.d/override.conf > /dev/null << 'EOF'
[Unit]
After=dhcpcd.service reviision-network-prep.service
Wants=dhcpcd.service reviision-network-prep.service

[Service]
Restart=on-failure
RestartSec=5
EOF

sudo mkdir -p /etc/systemd/system/dnsmasq.service.d
sudo tee /etc/systemd/system/dnsmasq.service.d/override.conf > /dev/null << 'EOF'
[Unit]
After=hostapd.service reviision-network-prep.service
Wants=hostapd.service reviision-network-prep.service

[Service]
Restart=on-failure
RestartSec=5
EOF

# 7. Enable IP forwarding permanently
echo "Step 7/8: Enabling IP forwarding..."
echo 'net.ipv4.ip_forward=1' | sudo tee -a /etc/sysctl.conf > /dev/null

# 8. Enable services (but don't start yet - SSH safe)
echo "Step 8/8: Enabling services..."
sudo systemctl daemon-reload

# Check and enable dhcpcd service (handle different possible names)
if systemctl list-unit-files | grep -q "^dhcpcd\.service"; then
    echo "  Enabling dhcpcd.service"
    sudo systemctl enable dhcpcd
elif systemctl list-unit-files | grep -q "^dhcpcd5\.service"; then
    echo "  Enabling dhcpcd5.service"
    sudo systemctl enable dhcpcd5
elif systemctl list-unit-files | grep -q "^dhcpcd\.service"; then
    echo "  Installing and enabling dhcpcd"
    sudo apt install -y dhcpcd5
    sudo systemctl enable dhcpcd
else
    echo "  ⚠️  Warning: dhcpcd service not found, installing dhcpcd5..."
    sudo apt install -y dhcpcd5
    sudo systemctl enable dhcpcd
fi

sudo systemctl enable wpa_supplicant@wlan0
sudo systemctl enable iptables-restore
sudo systemctl enable reviision-network-prep
sudo systemctl enable hostapd
sudo systemctl enable dnsmasq
sudo systemctl disable wpa_supplicant 2>/dev/null || true
sudo systemctl disable wpa_supplicant@wlan1 2>/dev/null || true
sudo systemctl unmask hostapd

echo ""
echo "✅ Configuration complete!"
echo ""

# Final SSH warning and network restart
if [ "$SSH_WARNING" = true ]; then
    echo "🚨 FINAL WARNING: NETWORK RESTART REQUIRED 🚨"
    echo ""
    echo "All configuration is complete, but network services need to be restarted"
    echo "to activate the new dual WiFi setup. This WILL disconnect your SSH session."
    echo ""
    echo "After network restart:"
    echo "1. Wait 2-3 minutes for services to fully start"
    echo "2. Look for the '$HOTSPOT_SSID' WiFi network"
    echo "3. Reconnect via SSH to 192.168.4.1 (if connecting to hotspot)"
    echo "   OR to the Pi's wlan0 IP on your main network"
    echo ""
    echo "You can also restart manually later with:"
    if systemctl list-unit-files | grep -q "^dhcpcd\.service"; then
        echo "  sudo systemctl restart dhcpcd"
    elif systemctl list-unit-files | grep -q "^dhcpcd5\.service"; then
        echo "  sudo systemctl restart dhcpcd5"
    else
        echo "  sudo systemctl restart dhcpcd"
    fi
    echo "  sudo systemctl restart wpa_supplicant@wlan0"
    echo "  sudo systemctl restart iptables-restore"
    echo "  sudo systemctl restart reviision-network-prep"
    echo "  sudo systemctl restart hostapd"
    echo "  sudo systemctl restart dnsmasq"
    echo ""
    
    echo "Proceeding with network restart in 30 seconds..."
    echo "Press Ctrl+C to cancel and restart manually later."
    
    for i in {30..1}; do
        printf "\rRestarting network in %2d seconds... " $i
        sleep 1
    done
    echo ""
    echo ""
fi

# Stop and restart network services
echo "Stopping current network services..."
sudo systemctl stop hostapd 2>/dev/null || true
sudo systemctl stop dnsmasq 2>/dev/null || true
sudo systemctl stop NetworkManager 2>/dev/null || true
sudo systemctl stop wpa_supplicant 2>/dev/null || true
sudo systemctl stop wpa_supplicant@wlan0 2>/dev/null || true
sudo systemctl stop wpa_supplicant@wlan1 2>/dev/null || true

# Stop dhcpcd with correct service name
if systemctl list-unit-files | grep -q "^dhcpcd\.service"; then
    sudo systemctl stop dhcpcd 2>/dev/null || true
elif systemctl list-unit-files | grep -q "^dhcpcd5\.service"; then
    sudo systemctl stop dhcpcd5 2>/dev/null || true
else
    sudo systemctl stop dhcpcd 2>/dev/null || true
fi

echo "Starting network services with new configuration..."

# Start dhcpcd with correct service name
if systemctl list-unit-files | grep -q "^dhcpcd\.service"; then
    echo "  Starting dhcpcd.service"
    sudo systemctl start dhcpcd
elif systemctl list-unit-files | grep -q "^dhcpcd5\.service"; then
    echo "  Starting dhcpcd5.service" 
    sudo systemctl start dhcpcd5
else
    echo "  Starting dhcpcd (fallback)"
    sudo systemctl start dhcpcd
fi

sleep 2
sudo systemctl start wpa_supplicant@wlan0
sleep 2
sudo systemctl start iptables-restore
sudo systemctl start reviision-network-prep
sleep 3
sudo systemctl start hostapd
sleep 2
sudo systemctl start dnsmasq

echo ""
echo "🎉 ReViision Pi Network Setup Complete!"
echo ""
echo "Network Status:"

# Check dhcpcd status with correct service name
if systemctl list-unit-files | grep -q "^dhcpcd\.service"; then
    dhcpcd_status=$(sudo systemctl is-active dhcpcd)
    echo "  dhcpcd: $dhcpcd_status"
elif systemctl list-unit-files | grep -q "^dhcpcd5\.service"; then
    dhcpcd_status=$(sudo systemctl is-active dhcpcd5)
    echo "  dhcpcd5: $dhcpcd_status"
else
    dhcpcd_status=$(sudo systemctl is-active dhcpcd)
    echo "  dhcpcd: $dhcpcd_status"
fi

echo "  wpa_supplicant@wlan0: $(sudo systemctl is-active wpa_supplicant@wlan0)"
echo "  hostapd: $(sudo systemctl is-active hostapd)"
echo "  dnsmasq: $(sudo systemctl is-active dnsmasq)"
echo "  iptables-restore: $(sudo systemctl is-active iptables-restore)"
echo ""
echo "WiFi Networks:"
echo "  Internet: $INTERNET_SSID (wlan0)"
echo "  Hotspot: $HOTSPOT_SSID (wlan1) at 192.168.4.1"
echo ""
echo "Troubleshooting commands:"
echo "  ip addr show"
echo "  iwconfig"
echo "  sudo systemctl status hostapd"
echo "  sudo systemctl status dnsmasq"
echo ""
echo "The Pi will now provide internet access to devices connecting to the hotspot."

if [ "$SSH_WARNING" = true ]; then
    echo ""
    echo "Your SSH session will likely disconnect now due to network changes."
    echo "Wait 2-3 minutes, then reconnect to verify the setup."
fi 