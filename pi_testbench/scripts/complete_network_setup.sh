#!/bin/bash
# ReViision Pi Test Bench - COMPLETE Network Setup Script
# LOGICAL DUAL WIFI: wlan0=Internet, wlan1=Hotspot
# This script ensures everything works on boot automatically

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}ReViision Pi Test Bench - COMPLETE Network Setup${NC}"
echo "================================================="
echo -e "${BLUE}Configuration: wlan0=Internet, wlan1=Hotspot${NC}"
echo

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}This script should not be run as root${NC}" 
   exit 1
fi

# Get WiFi credentials for internet connection
echo -e "${YELLOW}Enter WiFi credentials for internet connection (wlan0):${NC}"
read -p "WiFi SSID: " WIFI_SSID
read -s -p "WiFi Password: " WIFI_PASSWORD
echo
echo

# Update system first
echo -e "${GREEN}Updating system packages...${NC}"
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y hostapd dnsmasq iptables-persistent wpasupplicant \
    python3-pip python3-venv git wireless-tools net-tools dhcpcd5

# STOP ALL network services
echo -e "${GREEN}Stopping all network services...${NC}"
sudo systemctl stop hostapd 2>/dev/null || true
sudo systemctl stop dnsmasq 2>/dev/null || true
sudo systemctl stop NetworkManager 2>/dev/null || true
sudo systemctl stop wpa_supplicant 2>/dev/null || true
sudo systemctl stop wpa_supplicant@wlan0 2>/dev/null || true
sudo systemctl stop wpa_supplicant@wlan1 2>/dev/null || true
sudo systemctl stop dhcpcd 2>/dev/null || true
sudo pkill wpa_supplicant 2>/dev/null || true
sudo pkill hostapd 2>/dev/null || true
sudo pkill dnsmasq 2>/dev/null || true

# Create backup directory and backup configs
echo -e "${GREEN}Backing up existing configurations...${NC}"
sudo mkdir -p /etc/reviision-backup
sudo cp /etc/dhcpcd.conf /etc/reviision-backup/ 2>/dev/null || true
sudo cp /etc/hostapd/hostapd.conf /etc/reviision-backup/ 2>/dev/null || true
sudo cp /etc/dnsmasq.conf /etc/reviision-backup/ 2>/dev/null || true
sudo cp /etc/wpa_supplicant/wpa_supplicant.conf /etc/reviision-backup/ 2>/dev/null || true

# Create directories
sudo mkdir -p /etc/hostapd
sudo mkdir -p /etc/wpa_supplicant
sudo mkdir -p /etc/systemd/system

# DISABLE NetworkManager if present (causes conflicts)
echo -e "${GREEN}Disabling NetworkManager to prevent conflicts...${NC}"
sudo systemctl disable NetworkManager 2>/dev/null || true
sudo systemctl mask NetworkManager 2>/dev/null || true

#============================================================================
# STEP 1: Configure dhcpcd (primary network manager)
#============================================================================
echo -e "${GREEN}Configuring dhcpcd for dual WiFi...${NC}"
sudo tee /etc/dhcpcd.conf > /dev/null << 'EOF'
# dhcpcd configuration for ReViision Pi Test Bench
# LOGICAL SETUP: wlan0=Internet, wlan1=Hotspot

# Standard options
hostname
clientid
persistent
option rapid_commit
option domain_name_servers, domain_name, domain_search, host_name
option classless_static_routes
option interface_mtu
require dhcp_server_identifier
slaac hwaddr
noipv6

# wlan0 - Internet connection (DHCP)
interface wlan0
# DHCP will be managed by wpa_supplicant

# wlan1 - Hotspot interface (STATIC IP)
interface wlan1
static ip_address=192.168.4.1/24
nohook wpa_supplicant

# eth0 - Ethernet fallback
interface eth0
EOF

#============================================================================
# STEP 2: Configure wpa_supplicant for internet (wlan0)
#============================================================================
echo -e "${GREEN}Configuring wpa_supplicant for internet connection...${NC}"
sudo tee /etc/wpa_supplicant/wpa_supplicant.conf > /dev/null << EOF
country=US
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
    ssid="${WIFI_SSID}"
    psk="${WIFI_PASSWORD}"
    priority=10
}
EOF

# Create wlan0-specific config
sudo tee /etc/wpa_supplicant/wpa_supplicant-wlan0.conf > /dev/null << EOF
country=US
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
    ssid="${WIFI_SSID}"
    psk="${WIFI_PASSWORD}"
    priority=10
}
EOF

#============================================================================
# STEP 3: Configure hostapd for hotspot (wlan1)
#============================================================================
echo -e "${GREEN}Configuring hostapd for WiFi hotspot...${NC}"
sudo tee /etc/hostapd/hostapd.conf > /dev/null << 'EOF'
# hostapd configuration for ReViision Test Bench Hotspot
# Interface: wlan1 (dedicated hotspot)

interface=wlan1
driver=nl80211

# Network settings
ssid=ReViision-TestBench
hw_mode=g
channel=7
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0

# Security settings
wpa=2
wpa_passphrase=testbench2024
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP

# Country and regulatory
country_code=US

# Performance settings
ieee80211n=1
ht_capab=[HT40][SHORT-GI-20][DSSS_CCK-40]

# Logging (minimal for performance)
logger_syslog=-1
logger_syslog_level=0
logger_stdout=-1
logger_stdout_level=0

# Stability settings
beacon_int=100
dtim_period=2
max_num_sta=10
disassoc_low_ack=1
EOF

# Configure hostapd daemon
echo 'DAEMON_CONF="/etc/hostapd/hostapd.conf"' | sudo tee /etc/default/hostapd > /dev/null
echo 'DAEMON_OPTS=""' | sudo tee -a /etc/default/hostapd > /dev/null

#============================================================================
# STEP 4: Configure dnsmasq for DHCP and DNS
#============================================================================
echo -e "${GREEN}Configuring dnsmasq for DHCP and DNS...${NC}"
sudo tee /etc/dnsmasq.conf > /dev/null << 'EOF'
# dnsmasq configuration for ReViision Test Bench
# Provides DHCP and DNS for hotspot clients on wlan1

# Bind to hotspot interface only
interface=wlan1
bind-interfaces

# DNS settings
domain-needed
bogus-priv
no-resolv
no-hosts
expand-hosts

# Local domain
local=/testbench.local/
domain=testbench.local

# DHCP settings
dhcp-range=192.168.4.2,192.168.4.20,255.255.255.0,24h
dhcp-option=option:router,192.168.4.1
dhcp-option=option:dns-server,192.168.4.1,8.8.8.8,8.8.4.4
dhcp-option=option:netmask,255.255.255.0
dhcp-option=option:broadcast,192.168.4.255

# Authoritative DHCP
dhcp-authoritative

# DNS forwarding (via wlan0 internet)
server=8.8.8.8
server=8.8.4.4
server=1.1.1.1

# Local addresses
address=/testbench.local/192.168.4.1
address=/pi.testbench.local/192.168.4.1
address=/reviision.testbench.local/192.168.4.1

# Performance and security
cache-size=1000
log-dhcp
stop-dns-rebind
rebind-localhost-ok
no-poll
EOF

#============================================================================
# STEP 5: Enable IP forwarding and configure iptables
#============================================================================
echo -e "${GREEN}Configuring IP forwarding and iptables...${NC}"

# Enable IP forwarding permanently
echo 'net.ipv4.ip_forward=1' | sudo tee /etc/sysctl.d/99-ip-forward.conf > /dev/null

# Clear existing iptables rules
sudo iptables -F
sudo iptables -t nat -F
sudo iptables -t mangle -F
sudo iptables -X 2>/dev/null || true
sudo iptables -t nat -X 2>/dev/null || true
sudo iptables -t mangle -X 2>/dev/null || true

# Configure NAT forwarding (wlan1 hotspot -> wlan0 internet)
sudo iptables -t nat -A POSTROUTING -o wlan0 -j MASQUERADE
sudo iptables -A FORWARD -i wlan1 -o wlan0 -j ACCEPT
sudo iptables -A FORWARD -i wlan0 -o wlan1 -m state --state RELATED,ESTABLISHED -j ACCEPT

# Allow local traffic on hotspot
sudo iptables -A INPUT -i wlan1 -j ACCEPT
sudo iptables -A OUTPUT -o wlan1 -j ACCEPT

# Save iptables rules
sudo mkdir -p /etc/iptables
sudo iptables-save | sudo tee /etc/iptables/rules.v4 > /dev/null

#============================================================================
# STEP 6: Create systemd services for proper startup order
#============================================================================
echo -e "${GREEN}Creating systemd services with proper dependencies...${NC}"

# Create iptables restore service
sudo tee /etc/systemd/system/iptables-restore.service > /dev/null << 'EOF'
[Unit]
Description=Restore iptables rules
After=network.target
Before=hostapd.service dnsmasq.service

[Service]
Type=oneshot
ExecStart=/sbin/iptables-restore /etc/iptables/rules.v4
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

# Create interface preparation service
sudo tee /etc/systemd/system/reviision-network-prep.service > /dev/null << 'EOF'
[Unit]
Description=ReViision Network Interface Preparation
After=network.target dhcpcd.service
Before=hostapd.service dnsmasq.service

[Service]
Type=oneshot
ExecStart=/bin/bash -c '
  # Ensure wlan1 is up and has static IP
  ip link set wlan1 up
  sleep 2
  # Force static IP on wlan1 (in case dhcpcd missed it)
  ip addr flush dev wlan1
  ip addr add 192.168.4.1/24 dev wlan1
  # Enable IP forwarding
  echo 1 > /proc/sys/net/ipv4/ip_forward
  exit 0
'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

# Override hostapd service to add dependencies
sudo mkdir -p /etc/systemd/system/hostapd.service.d
sudo tee /etc/systemd/system/hostapd.service.d/override.conf > /dev/null << 'EOF'
[Unit]
After=dhcpcd.service reviision-network-prep.service iptables-restore.service
Wants=reviision-network-prep.service iptables-restore.service

[Service]
Restart=always
RestartSec=5
EOF

# Override dnsmasq service to add dependencies
sudo mkdir -p /etc/systemd/system/dnsmasq.service.d
sudo tee /etc/systemd/system/dnsmasq.service.d/override.conf > /dev/null << 'EOF'
[Unit]
After=hostapd.service reviision-network-prep.service
Wants=hostapd.service reviision-network-prep.service

[Service]
Restart=always
RestartSec=5
EOF

#============================================================================
# STEP 7: Enable and configure all services
#============================================================================
echo -e "${GREEN}Enabling all services for automatic startup...${NC}"

# Enable services
echo "Enabling and starting services..."
sudo systemctl daemon-reload

# Check and enable dhcpcd service (handle different possible names)
if systemctl list-unit-files | grep -q "^dhcpcd\.service"; then
    echo "  Enabling dhcpcd.service"
    sudo systemctl enable dhcpcd
elif systemctl list-unit-files | grep -q "^dhcpcd5\.service"; then
    echo "  Enabling dhcpcd5.service"
    sudo systemctl enable dhcpcd5
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

# Disable conflicting services
sudo systemctl disable NetworkManager 2>/dev/null || true
sudo systemctl disable wpa_supplicant 2>/dev/null || true
sudo systemctl disable wpa_supplicant@wlan1 2>/dev/null || true

# Unmask hostapd
sudo systemctl unmask hostapd

# Install Python dependencies
echo -e "${GREEN}Installing Python dependencies...${NC}"
cd /home/admin/pi_testbench
sudo -u admin python3 -m venv venv
sudo -u admin ./venv/bin/pip install -r requirements.txt

# Install and configure systemd service
echo -e "${GREEN}Setting up systemd service...${NC}"
sudo cp services/reviision-pi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable reviision-pi

# Create log directory
sudo mkdir -p /home/admin/pi_testbench/logs
sudo chown admin:admin /home/admin/pi_testbench/logs

echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${YELLOW}Network services will be restarted now...${NC}" 