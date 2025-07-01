#!/bin/bash
# ReViision Pi Test Bench Network Setup Script
# This script configures the Raspberry Pi for dual network operation:
# - WiFi hotspot for local test bench network
# - Internet connection via Ethernet or secondary WiFi

set -e

echo "ReViision Pi Test Bench Network Setup"
echo "====================================="

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)" 
   exit 1
fi

# Detect Pi model and available interfaces
echo "Detecting network interfaces..."
INTERFACES=$(ip link show | grep -E '^[0-9]+:' | awk -F': ' '{print $2}' | grep -v lo)
echo "Available interfaces: $INTERFACES"

# Check for WiFi interfaces
WIFI_INTERFACES=$(iwconfig 2>/dev/null | grep -E '^(wlan|wlp)' | awk '{print $1}' || echo "")
WIFI_COUNT=$(echo "$WIFI_INTERFACES" | wc -w)

echo "WiFi interfaces detected: $WIFI_COUNT"
if [[ $WIFI_COUNT -gt 0 ]]; then
    echo "WiFi interfaces: $WIFI_INTERFACES"
fi

# Configuration options
echo ""
echo "Network Configuration Options:"
echo "1. Ethernet + WiFi Hotspot (Recommended)"
echo "2. Dual WiFi (Requires USB WiFi adapter)"
echo "3. WiFi Hotspot only (No internet forwarding)"

read -p "Select configuration [1-3]: " CONFIG_CHOICE

case $CONFIG_CHOICE in
    1)
        INTERNET_METHOD="ethernet"
        INTERNET_INTERFACE="eth0"
        HOTSPOT_INTERFACE="wlan0"
        echo "Selected: Ethernet for internet, WiFi for hotspot"
        ;;
    2)
        if [[ $WIFI_COUNT -lt 2 ]]; then
            echo "Error: Dual WiFi setup requires at least 2 WiFi interfaces"
            echo "Please connect a USB WiFi adapter and try again"
            exit 1
        fi
        INTERNET_METHOD="wifi"
        INTERNET_INTERFACE="wlan1"
        HOTSPOT_INTERFACE="wlan0"
        echo "Selected: Dual WiFi setup"
        
        # Get WiFi credentials for internet connection
        read -p "Enter WiFi SSID for internet connection: " WIFI_SSID
        read -s -p "Enter WiFi password: " WIFI_PASSWORD
        echo ""
        ;;
    3)
        INTERNET_METHOD="disabled"
        INTERNET_INTERFACE=""
        HOTSPOT_INTERFACE="wlan0"
        echo "Selected: Hotspot only (no internet forwarding)"
        ;;
    *)
        echo "Invalid selection"
        exit 1
        ;;
esac

# Update package list
echo "Updating package list..."
apt update

# Install required packages
echo "Installing required packages..."
apt install -y hostapd dnsmasq iptables-persistent

# Backup existing configurations
echo "Backing up existing configurations..."
cp /etc/dhcpcd.conf /etc/dhcpcd.conf.backup.$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
cp /etc/dnsmasq.conf /etc/dnsmasq.conf.backup.$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
cp /etc/hostapd/hostapd.conf /etc/hostapd/hostapd.conf.backup.$(date +%Y%m%d_%H%M%S) 2>/dev/null || true

# Configure dhcpcd
echo "Configuring dhcpcd..."
cat > /etc/dhcpcd.conf << EOF
# dhcpcd configuration for ReViision Pi Test Bench

hostname
clientid
persistent
option rapid_commit
option domain_name_servers, domain_name, domain_search, host_name
option classless_static_routes
option interface_mtu
require dhcp_server_identifier
slaac private

# Static IP for WiFi hotspot interface
interface $HOTSPOT_INTERFACE
static ip_address=192.168.4.1/24
nohook wpa_supplicant

EOF

# Add ethernet configuration if using ethernet for internet
if [[ "$INTERNET_METHOD" == "ethernet" ]]; then
    cat >> /etc/dhcpcd.conf << EOF
# Ethernet interface for internet connection
interface $INTERNET_INTERFACE
# DHCP configuration will be handled automatically

EOF
fi

echo "dhcpcd configured"

# Configure dnsmasq
echo "Configuring dnsmasq..."
cat > /etc/dnsmasq.conf << EOF
# dnsmasq configuration for ReViision Test Bench

interface=$HOTSPOT_INTERFACE
domain-needed
bogus-priv
no-resolv
no-hosts
local=/testbench.local/
domain=testbench.local
expand-hosts

# DHCP configuration
dhcp-range=192.168.4.2,192.168.4.20,255.255.255.0,24h
dhcp-option=option:router,192.168.4.1
dhcp-option=option:dns-server,192.168.4.1,8.8.8.8,8.8.4.4

log-dhcp
cache-size=1000

# Local domain mappings
address=/testbench.local/192.168.4.1
address=/pi.testbench.local/192.168.4.1

# Upstream DNS servers
server=8.8.8.8
server=8.8.4.4
EOF

echo "dnsmasq configured"

# Configure hostapd
echo "Configuring hostapd..."
mkdir -p /etc/hostapd

cat > /etc/hostapd/hostapd.conf << EOF
# hostapd configuration for ReViision Test Bench

interface=$HOTSPOT_INTERFACE
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

# Country code (adjust for your region)
country_code=US

# 802.11n support
ieee80211n=1
ht_capab=[HT40][SHORT-GI-20][DSSS_CCK-40]

# Logging
logger_syslog=-1
logger_syslog_level=2
logger_stdout=-1
logger_stdout_level=2

beacon_int=100
dtim_period=2
EOF

echo "hostapd configured"

# Configure hostapd daemon
echo 'DAEMON_CONF="/etc/hostapd/hostapd.conf"' > /etc/default/hostapd

# Configure WiFi for internet if selected
if [[ "$INTERNET_METHOD" == "wifi" ]]; then
    echo "Configuring WiFi for internet connection..."
    
    # Create wpa_supplicant configuration for internet WiFi
    cat > /etc/wpa_supplicant/wpa_supplicant-${INTERNET_INTERFACE}.conf << EOF
country=US
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
    ssid="$WIFI_SSID"
    psk="$WIFI_PASSWORD"
    key_mgmt=WPA-PSK
}
EOF

    chmod 600 /etc/wpa_supplicant/wpa_supplicant-${INTERNET_INTERFACE}.conf
    
    # Enable wpa_supplicant for the internet interface
    systemctl enable wpa_supplicant@${INTERNET_INTERFACE}.service
fi

# Enable IP forwarding
echo "Enabling IP forwarding..."
echo 'net.ipv4.ip_forward=1' >> /etc/sysctl.conf

# Configure iptables for internet sharing (if enabled)
if [[ "$INTERNET_METHOD" != "disabled" ]]; then
    echo "Configuring iptables for internet sharing..."
    
    # Create iptables rules script
    cat > /etc/iptables-rules.sh << EOF
#!/bin/bash
# ReViision Pi Test Bench iptables configuration

# Clear existing rules
iptables -t nat -F
iptables -F
iptables -X

# Create custom chains
iptables -N REVIISION_FORWARD 2>/dev/null || true
iptables -F REVIISION_FORWARD

# Enable masquerading for internet sharing
iptables -t nat -A POSTROUTING -o $INTERNET_INTERFACE -j MASQUERADE

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A FORWARD -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow hotspot network to access internet (specific ports)
iptables -A REVIISION_FORWARD -i $HOTSPOT_INTERFACE -o $INTERNET_INTERFACE -p tcp --dport 80 -j ACCEPT
iptables -A REVIISION_FORWARD -i $HOTSPOT_INTERFACE -o $INTERNET_INTERFACE -p tcp --dport 443 -j ACCEPT
iptables -A REVIISION_FORWARD -i $HOTSPOT_INTERFACE -o $INTERNET_INTERFACE -p tcp --dport 554 -j ACCEPT
iptables -A REVIISION_FORWARD -i $HOTSPOT_INTERFACE -o $INTERNET_INTERFACE -p udp --dport 53 -j ACCEPT
iptables -A REVIISION_FORWARD -i $HOTSPOT_INTERFACE -o $INTERNET_INTERFACE -p udp --dport 123 -j ACCEPT
iptables -A REVIISION_FORWARD -i $HOTSPOT_INTERFACE -o $INTERNET_INTERFACE -p icmp -j ACCEPT

# Apply custom forward chain
iptables -A FORWARD -j REVIISION_FORWARD

# Allow return traffic
iptables -A FORWARD -i $INTERNET_INTERFACE -o $HOTSPOT_INTERFACE -m state --state RELATED,ESTABLISHED -j ACCEPT

# Allow local services
iptables -A INPUT -i $HOTSPOT_INTERFACE -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -i $HOTSPOT_INTERFACE -p udp --dport 67:68 -j ACCEPT
iptables -A INPUT -i $HOTSPOT_INTERFACE -p udp --dport 53 -j ACCEPT
iptables -A INPUT -i $HOTSPOT_INTERFACE -p tcp --dport 53 -j ACCEPT

# Drop other forwarding
iptables -A FORWARD -j DROP
EOF

    chmod +x /etc/iptables-rules.sh
    
    # Create systemd service for iptables
    cat > /etc/systemd/system/reviision-iptables.service << EOF
[Unit]
Description=ReViision Pi Test Bench iptables rules
After=network.target

[Service]
Type=oneshot
ExecStart=/etc/iptables-rules.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

    systemctl enable reviision-iptables.service
fi

# Enable services
echo "Enabling services..."
systemctl enable hostapd
systemctl enable dnsmasq

# Disable wpa_supplicant on hotspot interface
systemctl disable wpa_supplicant.service

# Create startup script
cat > /usr/local/bin/reviision-network-start << EOF
#!/bin/bash
# ReViision network startup script

# Wait for interfaces to be ready
sleep 5

# Configure hotspot interface
ip link set $HOTSPOT_INTERFACE down
ip addr flush dev $HOTSPOT_INTERFACE
ip addr add 192.168.4.1/24 dev $HOTSPOT_INTERFACE
ip link set $HOTSPOT_INTERFACE up

# Apply iptables rules if internet forwarding is enabled
if [[ -f /etc/iptables-rules.sh ]]; then
    /etc/iptables-rules.sh
fi

# Start services
systemctl start hostapd
systemctl start dnsmasq
EOF

chmod +x /usr/local/bin/reviision-network-start

# Create systemd service for network startup
cat > /etc/systemd/system/reviision-network.service << EOF
[Unit]
Description=ReViision Network Setup
After=network.target
Before=hostapd.service dnsmasq.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/reviision-network-start
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl enable reviision-network.service

echo ""
echo "Network configuration complete!"
echo "================================"
echo ""
echo "Configuration Summary:"
echo "- Hotspot Interface: $HOTSPOT_INTERFACE"
echo "- Hotspot SSID: ReViision-TestBench"
echo "- Hotspot Password: testbench2024"
echo "- Hotspot Network: 192.168.4.0/24"
echo "- Internet Method: $INTERNET_METHOD"
if [[ "$INTERNET_METHOD" != "disabled" ]]; then
    echo "- Internet Interface: $INTERNET_INTERFACE"
    echo "- Internet Forwarding: Enabled"
fi
echo ""
echo "Next Steps:"
echo "1. Reboot the Raspberry Pi: sudo reboot"
echo "2. After reboot, the hotspot should be active"
echo "3. Connect devices to 'ReViision-TestBench' network"
echo "4. Run the Pi test bench application"
echo ""
echo "To check status after reboot:"
echo "- Hotspot: sudo systemctl status hostapd"
echo "- DHCP: sudo systemctl status dnsmasq"
echo "- Internet: ping 8.8.8.8"
echo ""

read -p "Reboot now? (y/n): " REBOOT_NOW
if [[ "$REBOOT_NOW" =~ ^[Yy]$ ]]; then
    echo "Rebooting..."
    reboot
fi 