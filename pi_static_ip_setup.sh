#!/bin/bash
# ReViision Pi Static IP Setup Script
# Sets up static IP for phone hotspot demonstration

set -e  # Exit on any error

echo "🎯 ReViision Pi Static IP Setup for Phone Hotspot Demo"
echo "======================================================"

# Default values
DEFAULT_STATIC_IP="192.168.43.100"
DEFAULT_GATEWAY="192.168.43.1"
DEFAULT_DNS="8.8.8.8,8.8.4.4"

# Get user input with defaults
echo ""
echo "📋 Network Configuration:"
read -p "Static IP for Pi [$DEFAULT_STATIC_IP]: " STATIC_IP
STATIC_IP=${STATIC_IP:-$DEFAULT_STATIC_IP}

read -p "Gateway (usually phone IP) [$DEFAULT_GATEWAY]: " GATEWAY
GATEWAY=${GATEWAY:-$DEFAULT_GATEWAY}

read -p "DNS servers [$DEFAULT_DNS]: " DNS
DNS=${DNS:-$DEFAULT_DNS}

# Extract network from static IP (assumes /24)
NETWORK=$(echo $STATIC_IP | cut -d'.' -f1-3).0/24

echo ""
echo "📝 Configuration Summary:"
echo "   Static IP: $STATIC_IP/24"
echo "   Gateway: $GATEWAY"
echo "   Network: $NETWORK"
echo "   DNS: $DNS"
echo ""

read -p "Proceed with configuration? (y/N): " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "❌ Setup cancelled."
    exit 1
fi

echo ""
echo "🔧 Configuring static IP..."

# Backup current dhcpcd.conf
echo "📦 Creating backup of dhcpcd.conf..."
sudo cp /etc/dhcpcd.conf /etc/dhcpcd.conf.backup.$(date +%Y%m%d_%H%M%S)

# Create new dhcpcd configuration
echo "⚙️  Updating dhcpcd configuration..."

# Remove any existing static wlan0 configuration
sudo sed -i '/# Static IP for phone hotspot demo/,/^$/d' /etc/dhcpcd.conf
sudo sed -i '/interface wlan0/,/^$/d' /etc/dhcpcd.conf

# Add new static configuration
sudo tee -a /etc/dhcpcd.conf > /dev/null << EOF

# Static IP for phone hotspot demo
# Added by ReViision setup script $(date)
interface wlan0
static ip_address=$STATIC_IP/24
static routers=$GATEWAY
static domain_name_servers=$DNS

# Fallback profile for phone hotspot
profile phone_hotspot
static ip_address=$STATIC_IP/24
static routers=$GATEWAY
static domain_name_servers=$DNS
EOF

echo "✅ dhcpcd.conf updated successfully"

# Create network test script
echo "📋 Creating network test script..."
cat > ~/test_network.sh << 'EOF'
#!/bin/bash
echo "🔍 Network Status Check"
echo "======================"
echo "Current IP address:"
ip addr show wlan0 | grep "inet " | awk '{print $2}'
echo ""
echo "Gateway connectivity:"
ping -c 1 $(ip route | grep default | awk '{print $3}') && echo "✅ Gateway reachable" || echo "❌ Gateway unreachable"
echo ""
echo "Internet connectivity:"
ping -c 1 8.8.8.8 && echo "✅ Internet reachable" || echo "❌ Internet unreachable"
echo ""
echo "ReViision service status:"
if systemctl is-active --quiet reviision 2>/dev/null; then
    echo "✅ ReViision service is running"
else
    echo "ℹ️  ReViision service not found (this is normal if not installed)"
fi
EOF

chmod +x ~/test_network.sh

echo ""
echo "🔄 Network configuration complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Restart networking: sudo systemctl restart dhcpcd"
echo "2. OR reboot Pi: sudo reboot"
echo "3. Connect to your phone hotspot"
echo "4. Test network: ~/test_network.sh"
echo "5. Run ReViision setup: python setup_reviision.py phone_demo"
echo ""

read -p "Restart networking now? (y/N): " RESTART
if [[ "$RESTART" =~ ^[Yy]$ ]]; then
    echo "🔄 Restarting dhcpcd service..."
    sudo systemctl restart dhcpcd
    sleep 3
    
    echo "🔍 Testing network configuration..."
    ./~/test_network.sh
else
    echo "ℹ️  Remember to restart networking or reboot before testing"
fi

echo ""
echo "🎉 Static IP setup complete!"
echo ""
echo "📚 Troubleshooting:"
echo "   - If static IP doesn't work, check phone hotspot IP range"
echo "   - Common ranges: 192.168.43.x (Android), 172.20.10.x (iPhone)"
echo "   - Restore backup: sudo cp /etc/dhcpcd.conf.backup.* /etc/dhcpcd.conf"
echo "   - Check status: systemctl status dhcpcd"
echo ""
echo "📖 See DEMO_SETUP_GUIDE.md for complete demo instructions" 