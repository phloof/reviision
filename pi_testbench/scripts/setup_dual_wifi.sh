#!/bin/bash
# ReViision Pi Test Bench - Optimized Dual WiFi Setup Script
# Sets up wlan0 as hotspot, wlan1 for internet connection

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}ReViision Pi Test Bench - Dual WiFi Setup${NC}"
echo "=============================================="

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}This script should not be run as root${NC}" 
   exit 1
fi

# Get WiFi credentials for internet connection
echo -e "${YELLOW}Enter WiFi credentials for internet connection (wlan1):${NC}"
read -p "WiFi SSID: " WIFI_SSID
read -s -p "WiFi Password: " WIFI_PASSWORD
echo

# Update system
echo -e "${GREEN}Updating system packages...${NC}"
sudo apt update && sudo apt upgrade -y

# Install required packages
echo -e "${GREEN}Installing required packages...${NC}"
sudo apt install -y hostapd dnsmasq iptables-persistent

# Stop services during configuration
echo -e "${GREEN}Stopping services for configuration...${NC}"
sudo systemctl stop hostapd
sudo systemctl stop dnsmasq
sudo systemctl stop dhcpcd

# Prepare directories and backup existing configurations
echo -e "${GREEN}Preparing directories and backing up existing configurations...${NC}"
sudo mkdir -p /etc/hostapd
sudo cp /etc/dhcpcd.conf /etc/dhcpcd.conf.backup 2>/dev/null || true
sudo cp /etc/hostapd/hostapd.conf /etc/hostapd/hostapd.conf.backup 2>/dev/null || true
sudo cp /etc/dnsmasq.conf /etc/dnsmasq.conf.backup 2>/dev/null || true

# Configure dhcpcd
echo -e "${GREEN}Configuring dhcpcd...${NC}"
sudo tee /etc/dhcpcd.conf > /dev/null << 'EOF'
# dhcpcd configuration for ReViision Pi Test Bench
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

# wlan0 - Hotspot interface (static IP)
interface wlan0
static ip_address=192.168.4.1/24
nohook wpa_supplicant

# wlan1 - Internet connection interface (DHCP)
interface wlan1

# eth0 - Ethernet interface (DHCP, fallback)
interface eth0
EOF

# Configure hostapd
echo -e "${GREEN}Configuring hostapd...${NC}"
sudo tee /etc/hostapd/hostapd.conf > /dev/null << 'EOF'
interface=wlan0
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
country_code=US
ieee80211n=1
ht_capab=[HT40][SHORT-GI-20][DSSS_CCK-40]
logger_syslog=-1
logger_syslog_level=1
beacon_int=100
dtim_period=2
max_num_sta=10
EOF

# Configure dnsmasq
echo -e "${GREEN}Configuring dnsmasq...${NC}"
sudo tee /etc/dnsmasq.conf > /dev/null << 'EOF'
interface=wlan0
domain-needed
bogus-priv
no-resolv
no-hosts
local=/testbench.local/
domain=testbench.local
expand-hosts
dhcp-range=192.168.4.2,192.168.4.20,255.255.255.0,24h
dhcp-option=option:router,192.168.4.1
dhcp-option=option:dns-server,192.168.4.1,8.8.8.8,8.8.4.4
log-dhcp
cache-size=1000
address=/testbench.local/192.168.4.1
server=8.8.8.8
server=8.8.4.4
server=1.1.1.1
dhcp-authoritative
bind-interfaces
EOF

# Configure hostapd daemon
echo 'DAEMON_CONF="/etc/hostapd/hostapd.conf"' | sudo tee /etc/default/hostapd > /dev/null

# Configure WiFi for internet connection (wlan1)
echo -e "${GREEN}Configuring WiFi for internet connection...${NC}"
sudo tee /etc/wpa_supplicant/wpa_supplicant-wlan1.conf > /dev/null << EOF
country=US
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
    ssid="${WIFI_SSID}"
    psk="${WIFI_PASSWORD}"
}
EOF

# Enable IP forwarding
echo -e "${GREEN}Enabling IP forwarding...${NC}"
echo 'net.ipv4.ip_forward=1' | sudo tee -a /etc/sysctl.conf

# Configure iptables for NAT
echo -e "${GREEN}Configuring iptables for internet sharing...${NC}"
sudo iptables -t nat -A POSTROUTING -o wlan1 -j MASQUERADE
sudo iptables -A FORWARD -i wlan0 -o wlan1 -j ACCEPT
sudo iptables -A FORWARD -i wlan1 -o wlan0 -m state --state RELATED,ESTABLISHED -j ACCEPT

# Save iptables rules
sudo mkdir -p /etc/iptables
sudo iptables-save | sudo tee /etc/iptables/rules.v4 > /dev/null

# Create iptables restore service
sudo tee /etc/systemd/system/iptables-restore.service > /dev/null << 'EOF'
[Unit]
Description=Restore iptables rules
After=network.target

[Service]
Type=oneshot
ExecStart=/sbin/iptables-restore /etc/iptables/rules.v4
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

# Enable services
echo -e "${GREEN}Enabling services...${NC}"
sudo systemctl unmask hostapd
sudo systemctl enable hostapd
sudo systemctl enable dnsmasq
sudo systemctl enable wpa_supplicant@wlan1
sudo systemctl enable iptables-restore

# Start services
echo -e "${GREEN}Starting services...${NC}"
sudo systemctl start dhcpcd
sleep 2
sudo systemctl start wpa_supplicant@wlan1
sleep 5
sudo systemctl start hostapd
sudo systemctl start dnsmasq
sudo systemctl start iptables-restore

# Check service status
echo -e "${GREEN}Checking service status...${NC}"
echo "dhcpcd: $(sudo systemctl is-active dhcpcd)"
echo "hostapd: $(sudo systemctl is-active hostapd)"
echo "dnsmasq: $(sudo systemctl is-active dnsmasq)"
echo "wpa_supplicant@wlan1: $(sudo systemctl is-active wpa_supplicant@wlan1)"

echo -e "${GREEN}Setup complete!${NC}"
echo
echo "Network Configuration:"
echo "- Hotspot: ReViision-TestBench (password: testbench2024)"
echo "- Hotspot IP: 192.168.4.1"
echo "- Internet via: ${WIFI_SSID}"
echo
echo "You can check network status with:"
echo "  iwconfig"
echo "  sudo systemctl status hostapd"
echo "  sudo systemctl status dnsmasq"
echo
echo -e "${YELLOW}Please reboot the system to ensure all changes take effect:${NC}"
echo "  sudo reboot" 