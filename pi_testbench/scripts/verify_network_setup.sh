#!/bin/bash

# ReViision Pi Network Setup Verification Script
# ==============================================
# Run this after network setup to verify everything is working

echo "ReViision Pi Network Setup Verification"
echo "========================================"
echo ""

# Check if services are running
echo "🔍 Checking Network Services:"

# Determine dhcpcd service name
if systemctl list-unit-files | grep -q "^dhcpcd\.service"; then
    dhcpcd_service="dhcpcd"
elif systemctl list-unit-files | grep -q "^dhcpcd5\.service"; then
    dhcpcd_service="dhcpcd5"
else
    dhcpcd_service="dhcpcd"
fi

services=("$dhcpcd_service" "wpa_supplicant@wlan0" "hostapd" "dnsmasq" "iptables-restore" "reviision-network-prep")

all_running=true
for service in "${services[@]}"; do
    status=$(sudo systemctl is-active $service 2>/dev/null || echo "inactive")
    if [ "$status" = "active" ]; then
        echo "  ✅ $service: $status"
    else
        echo "  ❌ $service: $status"
        all_running=false
    fi
done

echo ""

# Check network interfaces
echo "🌐 Network Interface Status:"
echo "wlan0 (Internet connection):"
wlan0_ip=$(ip addr show wlan0 2>/dev/null | grep 'inet ' | awk '{print $2}' | head -1)
if [ -n "$wlan0_ip" ]; then
    echo "  ✅ IP Address: $wlan0_ip"
else
    echo "  ❌ No IP address assigned"
    all_running=false
fi

echo ""
echo "wlan1 (Hotspot interface):"
wlan1_ip=$(ip addr show wlan1 2>/dev/null | grep 'inet ' | awk '{print $2}' | head -1)
if [ "$wlan1_ip" = "192.168.4.1/24" ]; then
    echo "  ✅ IP Address: $wlan1_ip"
else
    echo "  ❌ Expected 192.168.4.1/24, got: ${wlan1_ip:-'none'}"
    all_running=false
fi

echo ""

# Check WiFi hotspot
echo "📶 WiFi Hotspot Status:"
hotspot_info=$(iwconfig wlan1 2>/dev/null | grep -E "(ESSID|Mode)")
if echo "$hotspot_info" | grep -q "Mode:Master"; then
    essid=$(echo "$hotspot_info" | grep "ESSID" | sed 's/.*ESSID:"\([^"]*\)".*/\1/')
    echo "  ✅ Hotspot active: $essid"
else
    echo "  ❌ Hotspot not active or not in AP mode"
    all_running=false
fi

echo ""

# Check internet connectivity
echo "🌍 Internet Connectivity:"
if ping -c 1 -W 5 8.8.8.8 >/dev/null 2>&1; then
    echo "  ✅ Internet connection working"
else
    echo "  ❌ No internet connection"
    all_running=false
fi

echo ""

# Check NAT forwarding
echo "🔄 NAT Forwarding Rules:"
if sudo iptables -t nat -L POSTROUTING | grep -q "MASQUERADE.*wlan0"; then
    echo "  ✅ NAT MASQUERADE rule active"
else
    echo "  ❌ NAT MASQUERADE rule missing"
    all_running=false
fi

if sudo iptables -L FORWARD | grep -q "wlan0.*wlan1.*RELATED,ESTABLISHED"; then
    echo "  ✅ Forward rules active"
else
    echo "  ❌ Forward rules missing"
    all_running=false
fi

echo ""

# Check DHCP clients (if any)
echo "📱 Connected Devices:"
lease_file="/var/lib/dhcp/dhcpcd.leases"
if [ -f "$lease_file" ] && [ -s "$lease_file" ]; then
    echo "  📋 DHCP leases found:"
    cat "$lease_file" | grep "lease" | tail -5
else
    echo "  ℹ️  No DHCP clients currently connected"
fi

echo ""

# Overall status
if [ "$all_running" = true ]; then
    echo "🎉 All systems operational!"
    echo ""
    echo "Network Summary:"
    echo "  • Internet: wlan0 ($wlan0_ip)"
    echo "  • Hotspot: wlan1 (192.168.4.1) - '$essid'"
    echo "  • NAT forwarding: Active (wlan1 → wlan0 → Internet)"
    echo ""
    echo "You can now:"
    echo "  1. Connect devices to the '$essid' hotspot (password: testbench2024)"
    echo "  2. Access the Pi via 192.168.4.1 from hotspot devices"
    echo "  3. Connected devices will have internet access"
else
    echo "❌ Some issues detected!"
    echo ""
    echo "Troubleshooting steps:"
    echo "  1. Check service logs: sudo journalctl -u <service-name> -n 20"
    echo "  2. Restart services: sudo systemctl restart <service-name>"
    echo "  3. Check interface status: ip addr show"
    echo "  4. Verify config files in /etc/"
    echo ""
    echo "Service restart command:"
    echo "  sudo systemctl restart $dhcpcd_service wpa_supplicant@wlan0 iptables-restore reviision-network-prep hostapd dnsmasq"
fi

echo ""
echo "For detailed status: sudo systemctl status hostapd dnsmasq dhcpcd"
echo "For network interfaces: ip addr show && iwconfig" 