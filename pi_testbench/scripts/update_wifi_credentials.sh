#!/bin/bash

# ReViision Pi WiFi Credentials Update Script
# ==========================================
# Update SSID and passwords for both internet connection and hotspot

set -e

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "This script should not be run as root"
    echo "Run as: bash scripts/update_wifi_credentials.sh"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}ReViision Pi WiFi Credentials Update${NC}"
echo "===================================="
echo ""

# Check if we're in an SSH session
SSH_WARNING=false
if [ -n "$SSH_CLIENT" ] || [ -n "$SSH_TTY" ] || [ -n "$SSH_CONNECTION" ]; then
    SSH_WARNING=true
    echo -e "${YELLOW}⚠️  SSH SESSION DETECTED${NC}"
    echo "Changes to internet WiFi may disconnect your session."
    echo ""
fi

# Show current configuration
echo -e "${BLUE}Current Configuration:${NC}"
echo ""

# Show current internet WiFi
echo "Internet WiFi (wlan0):"
if [ -f "/etc/wpa_supplicant/wpa_supplicant-wlan0.conf" ]; then
    current_internet_ssid=$(grep -o 'ssid="[^"]*"' /etc/wpa_supplicant/wpa_supplicant-wlan0.conf | sed 's/ssid="\(.*\)"/\1/' | head -1)
    echo "  Current SSID: ${current_internet_ssid:-'Not configured'}"
else
    echo "  Configuration file not found"
fi

echo ""

# Show current hotspot
echo "Hotspot (wlan1):"
if [ -f "/etc/hostapd/hostapd.conf" ]; then
    current_hotspot_ssid=$(grep "^ssid=" /etc/hostapd/hostapd.conf | cut -d'=' -f2)
    echo "  Current SSID: ${current_hotspot_ssid:-'Not configured'}"
else
    echo "  Configuration file not found"
fi

echo ""
echo "What would you like to update?"
echo "1) Internet WiFi credentials only"
echo "2) Hotspot credentials only"
echo "3) Both internet and hotspot credentials"
echo "4) Show current status and exit"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        update_internet=true
        update_hotspot=false
        ;;
    2)
        update_internet=false
        update_hotspot=true
        ;;
    3)
        update_internet=true
        update_hotspot=true
        ;;
    4)
        echo ""
        echo "Current network status:"
        ip addr show wlan0 wlan1 2>/dev/null | grep -E "(wlan[01]:|inet )" || echo "Network interfaces not available"
        echo ""
        echo "Service status:"
        
        # Determine dhcpcd service name
        if systemctl list-unit-files | grep -q "^dhcpcd\.service"; then
            dhcpcd_service="dhcpcd"
        elif systemctl list-unit-files | grep -q "^dhcpcd5\.service"; then
            dhcpcd_service="dhcpcd5"
        else
            dhcpcd_service="dhcpcd"
        fi
        
        echo "  $dhcpcd_service: $(sudo systemctl is-active $dhcpcd_service 2>/dev/null || echo 'inactive')"
        echo "  wpa_supplicant@wlan0: $(sudo systemctl is-active wpa_supplicant@wlan0 2>/dev/null || echo 'inactive')"
        echo "  hostapd: $(sudo systemctl is-active hostapd 2>/dev/null || echo 'inactive')"
        echo "  dnsmasq: $(sudo systemctl is-active dnsmasq 2>/dev/null || echo 'inactive')"
        echo ""
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""

# Backup current configurations
echo -e "${YELLOW}Creating backup of current configurations...${NC}"
sudo cp /etc/wpa_supplicant/wpa_supplicant-wlan0.conf /etc/wpa_supplicant/wpa_supplicant-wlan0.conf.backup.$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
sudo cp /etc/hostapd/hostapd.conf /etc/hostapd/hostapd.conf.backup.$(date +%Y%m%d_%H%M%S) 2>/dev/null || true
echo "Backups created in the same directories with timestamp suffixes."
echo ""

# Update Internet WiFi credentials
if [ "$update_internet" = true ]; then
    echo -e "${GREEN}Internet WiFi Configuration (wlan0):${NC}"
    echo "This WiFi connection provides internet access to the Pi."
    echo ""
    
    read -p "Enter new internet WiFi SSID: " new_internet_ssid
    if [ -z "$new_internet_ssid" ]; then
        echo "SSID cannot be empty. Exiting."
        exit 1
    fi
    
    read -p "Enter WiFi password: " -s new_internet_password
    echo ""
    
    if [ -z "$new_internet_password" ]; then
        echo "Password cannot be empty. Exiting."
        exit 1
    fi
    
    echo ""
    echo "Advanced options (press Enter for defaults):"
    read -p "Country code [US]: " country_code
    country_code=${country_code:-US}
    
    echo ""
    echo "Do you need WPA2 Enterprise authentication? (y/N): "
    read -n 1 -r enterprise_reply
    echo ""
    
    if [[ $enterprise_reply =~ ^[Yy]$ ]]; then
        echo ""
        echo "WPA2 Enterprise Configuration:"
        read -p "Identity (username): " enterprise_identity
        read -p "Anonymous identity [anonymous]: " enterprise_anonymous
        enterprise_anonymous=${enterprise_anonymous:-anonymous}
        read -p "EAP method [PEAP]: " eap_method
        eap_method=${eap_method:-PEAP}
        read -p "Phase 2 auth [MSCHAPV2]: " phase2_auth
        phase2_auth=${phase2_auth:-MSCHAPV2}
        
        # Create WPA2 Enterprise configuration
        sudo tee /etc/wpa_supplicant/wpa_supplicant-wlan0.conf > /dev/null << EOF
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=$country_code

# Internet WiFi connection (WPA2 Enterprise)
network={
    ssid="$new_internet_ssid"
    key_mgmt=WPA-EAP
    eap=$eap_method
    identity="$enterprise_identity"
    anonymous_identity="$enterprise_anonymous"
    password="$new_internet_password"
    phase2="auth=$phase2_auth"
    priority=1
}
EOF
    else
        # Create standard WPA2 configuration
        sudo tee /etc/wpa_supplicant/wpa_supplicant-wlan0.conf > /dev/null << EOF
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=$country_code

# Internet WiFi connection
network={
    ssid="$new_internet_ssid"
    psk="$new_internet_password"
    priority=1
}
EOF
    fi
    
    echo -e "${GREEN}✅ Internet WiFi configuration updated${NC}"
fi

# Update Hotspot credentials
if [ "$update_hotspot" = true ]; then
    echo ""
    echo -e "${GREEN}Hotspot Configuration (wlan1):${NC}"
    echo "This creates a WiFi network that other devices can connect to."
    echo ""
    
    read -p "Enter new hotspot SSID: " new_hotspot_ssid
    if [ -z "$new_hotspot_ssid" ]; then
        echo "SSID cannot be empty. Exiting."
        exit 1
    fi
    
    read -p "Enter hotspot password (min 8 characters): " -s new_hotspot_password
    echo ""
    
    if [ ${#new_hotspot_password} -lt 8 ]; then
        echo "Password must be at least 8 characters. Exiting."
        exit 1
    fi
    
    echo ""
    echo "Advanced options (press Enter for defaults):"
    read -p "WiFi Channel [7]: " wifi_channel
    wifi_channel=${wifi_channel:-7}
    
    read -p "Country code [US]: " hotspot_country
    hotspot_country=${hotspot_country:-US}
    
    # Update hostapd configuration
    sudo tee /etc/hostapd/hostapd.conf > /dev/null << EOF
# ReViision Test Bench WiFi Hotspot Configuration
# Interface and driver
interface=wlan1
driver=nl80211

# Network settings
ssid=$new_hotspot_ssid
hw_mode=g
channel=$wifi_channel
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0

# Security settings
wpa=2
wpa_passphrase=$new_hotspot_password
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP

# Country code
country_code=$hotspot_country
ieee80211n=1
ieee80211d=1
EOF
    
    echo -e "${GREEN}✅ Hotspot configuration updated${NC}"
fi

echo ""
echo -e "${YELLOW}Configuration Summary:${NC}"

if [ "$update_internet" = true ]; then
    echo "Internet WiFi (wlan0): $new_internet_ssid"
fi

if [ "$update_hotspot" = true ]; then
    echo "Hotspot (wlan1): $new_hotspot_ssid"
fi

echo ""

# Warning about service restart
if [ "$SSH_WARNING" = true ] && [ "$update_internet" = true ]; then
    echo -e "${RED}⚠️  WARNING: SSH DISCONNECTION IMMINENT${NC}"
    echo ""
    echo "You are connected via SSH and have updated internet WiFi settings."
    echo "Restarting network services will disconnect your SSH session."
    echo ""
    echo "After disconnection:"
    echo "1. Wait 1-2 minutes for services to restart"
    echo "2. Connect to the new WiFi network: $new_internet_ssid"
    echo "3. Reconnect via SSH to the Pi's new IP address"
    
    if [ "$update_hotspot" = true ]; then
        echo "4. Look for the new hotspot: $new_hotspot_ssid"
    fi
    
    echo ""
    echo "Proceeding with network restart in 15 seconds..."
    echo "Press Ctrl+C to cancel and restart manually later."
    
    for i in {15..1}; do
        printf "\rRestarting network in %2d seconds... " $i
        sleep 1
    done
    echo ""
    echo ""
fi

# Restart network services
echo -e "${YELLOW}Restarting network services...${NC}"

# Determine dhcpcd service name
if systemctl list-unit-files | grep -q "^dhcpcd\.service"; then
    dhcpcd_service="dhcpcd"
elif systemctl list-unit-files | grep -q "^dhcpcd5\.service"; then
    dhcpcd_service="dhcpcd5"
else
    dhcpcd_service="dhcpcd"
fi

# Stop services
sudo systemctl stop hostapd 2>/dev/null || true
sudo systemctl stop dnsmasq 2>/dev/null || true
sudo systemctl stop wpa_supplicant@wlan0 2>/dev/null || true
sudo systemctl stop $dhcpcd_service 2>/dev/null || true

# Restart services in correct order
echo "Starting network services..."
sudo systemctl start $dhcpcd_service
sleep 2

if [ "$update_internet" = true ]; then
    sudo systemctl restart wpa_supplicant@wlan0
    sleep 3
fi

sudo systemctl start iptables-restore 2>/dev/null || true
sudo systemctl start reviision-network-prep 2>/dev/null || true
sleep 2

if [ "$update_hotspot" = true ]; then
    sudo systemctl restart hostapd
    sleep 2
    sudo systemctl restart dnsmasq
fi

echo ""
echo -e "${GREEN}🎉 WiFi credentials updated successfully!${NC}"
echo ""

# Show final status
echo "Network Service Status:"
echo "  $dhcpcd_service: $(sudo systemctl is-active $dhcpcd_service 2>/dev/null || echo 'inactive')"

if [ "$update_internet" = true ]; then
    echo "  wpa_supplicant@wlan0: $(sudo systemctl is-active wpa_supplicant@wlan0 2>/dev/null || echo 'inactive')"
fi

if [ "$update_hotspot" = true ]; then
    echo "  hostapd: $(sudo systemctl is-active hostapd 2>/dev/null || echo 'inactive')"
    echo "  dnsmasq: $(sudo systemctl is-active dnsmasq 2>/dev/null || echo 'inactive')"
fi

echo ""
echo "Updated Configuration:"

if [ "$update_internet" = true ]; then
    echo "  Internet WiFi: $new_internet_ssid"
fi

if [ "$update_hotspot" = true ]; then
    echo "  Hotspot: $new_hotspot_ssid"
fi

echo ""
echo "To verify the setup is working:"
echo "  bash scripts/verify_network_setup.sh"
echo ""

if [ "$SSH_WARNING" = true ] && [ "$update_internet" = true ]; then
    echo -e "${YELLOW}Your SSH session may disconnect shortly.${NC}"
    echo "Reconnect after the Pi connects to the new WiFi network."
fi 