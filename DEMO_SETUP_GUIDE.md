# ReViision Demo Setup Guide
## Phone Hotspot Demonstration Environment

This guide helps you set up ReViision for client demonstrations using your phone's hotspot.

## 📱 **Network Topology for Demo**

```
[Phone Hotspot] → [Computer] + [Raspberry Pi] → [Pi Hotspot] → [Camera]
  192.168.43.x      .auto      .100 (static)    192.168.4.x    .31
```

### **Advantages of Phone Hotspot Demo:**
- ✅ **Simplified Network**: No corporate network dependencies
- ✅ **Portable**: Works anywhere with cellular coverage
- ✅ **Controlled Environment**: You control the entire network
- ✅ **No IT Restrictions**: Bypass corporate firewall/security policies
- ✅ **Easy Client Access**: Client can connect to same hotspot

## 🔧 **Setup Instructions**

### **Step 1: Configure Your Phone Hotspot**
1. **Enable Mobile Hotspot** on your phone
2. **Set Network Name**: `YourPhone-Hotspot` (or any name)
3. **Set Password**: Use a secure password
4. **Note the IP Range**: Usually `192.168.43.x` or `192.168.137.x`

### **Step 2: Configure Raspberry Pi Static IP**

**Option A: Using dhcpcd (Recommended)**

SSH into your Raspberry Pi and edit the dhcpcd configuration:

```bash
sudo nano /etc/dhcpcd.conf
```

Add these lines at the end (adjust IP range to match your phone):

```bash
# Static IP for phone hotspot demo
interface wlan0
static ip_address=192.168.43.100/24
static routers=192.168.43.1
static domain_name_servers=8.8.8.8 8.8.4.4

# Fallback to DHCP if static fails
profile static_eth0
static ip_address=192.168.43.100/24
static routers=192.168.43.1
static domain_name_servers=8.8.8.8

# Fallback profile
profile phone_hotspot
static ip_address=192.168.43.100/24
static routers=192.168.43.1
static domain_name_servers=8.8.8.8
```

**Option B: Using Network Manager (Alternative)**

```bash
# Create static connection for phone hotspot
sudo nmcli con add type wifi ifname wlan0 con-name "phone-demo" ssid "YourPhone-Hotspot"
sudo nmcli con modify "phone-demo" wifi-sec.key-mgmt wpa-psk
sudo nmcli con modify "phone-demo" wifi-sec.psk "your-phone-password"
sudo nmcli con modify "phone-demo" ipv4.method manual
sudo nmcli con modify "phone-demo" ipv4.addresses 192.168.43.100/24
sudo nmcli con modify "phone-demo" ipv4.gateway 192.168.43.1
sudo nmcli con modify "phone-demo" ipv4.dns "8.8.8.8,8.8.4.4"
```

### **Step 3: Restart Pi Networking**

```bash
sudo systemctl restart dhcpcd
# OR reboot for complete refresh
sudo reboot
```

### **Step 4: Connect Computer to Phone Hotspot**

1. Connect your **Windows development machine** to the same phone hotspot
2. Note your computer's IP (should be something like `192.168.43.101`)

### **Step 5: Run ReViision Setup**

From your computer, run the demo setup:

```bash
python setup_reviision.py phone_demo
```

This will configure:
- Pi IP: `192.168.43.100` (static)
- Camera network: `192.168.4.31` (isolated)
- Port forwarding: `192.168.43.100:8554` → `192.168.4.31:554`
- Demo-optimized settings

## 🎯 **Demo Network Configuration**

### **After Setup, Your Network Will Look Like:**

```
Phone Hotspot Network (192.168.43.x):
├── Phone: 192.168.43.1 (gateway)
├── Your Computer: 192.168.43.101 (auto-assigned)
├── Client Devices: 192.168.43.102+ (if connected)
└── Raspberry Pi: 192.168.43.100 (static)
    
Pi Hotspot Network (192.168.4.x):
├── Pi: 192.168.4.1 (gateway)
└── Camera: 192.168.4.31 (isolated)
```

### **Access Points:**
- **ReViision Web Interface**: `http://192.168.43.100:5000`
- **Camera Stream**: `rtsp://192.168.43.100:8554/stream1`
- **Pi SSH**: `ssh pi@192.168.43.100`

## 🔒 **Security for Demo**

Even in demo mode, security is maintained:
- Camera remains isolated on `192.168.4.x` network
- Only Pi can access camera directly
- All web traffic encrypted with TLS
- Credentials protected with Fernet encryption

## 📋 **Demo Checklist**

**Before Client Arrives:**
- [ ] Phone hotspot enabled and tested
- [ ] Pi connected with static IP `192.168.43.100`
- [ ] Computer connected to same hotspot
- [ ] Camera connected to Pi hotspot
- [ ] ReViision running and accessible at `http://192.168.43.100:5000`
- [ ] Test video stream working

**During Demo:**
- [ ] Connect client devices to your phone hotspot
- [ ] Share ReViision URL: `http://192.168.43.100:5000`
- [ ] Demonstrate real-time analytics
- [ ] Show different views (demographics, heatmap, etc.)

## 🛠 **Troubleshooting Demo Setup**

### **Pi Can't Get Static IP**
```bash
# Check current IP
ip addr show wlan0

# Check dhcpcd status
sudo systemctl status dhcpcd

# Check network connections
nmcli con show

# Reset network if needed
sudo systemctl restart networking
```

### **Computer Can't Access Pi**
```bash
# From computer, test Pi connectivity
ping 192.168.43.100
telnet 192.168.43.100 5000

# Check if ReViision is running on Pi
curl http://192.168.43.100:5000
```

### **Camera Not Accessible**
```bash
# On Pi, test camera connection
telnet 192.168.4.31 554

# Check port forwarding
sudo iptables -t nat -L PREROUTING -n | grep 8554

# Test external access
telnet 192.168.43.100 8554
```

## 📱 **Phone Hotspot Tips**

### **Optimize Phone Settings:**
- **Unlimited Data Plan**: Ensure sufficient data allowance
- **5GHz Band**: Use 5GHz if available for better performance
- **Maximum Connections**: Set to allow multiple devices
- **Battery**: Keep phone plugged in during demo

### **Common Phone Hotspot IP Ranges:**
- **Android**: Usually `192.168.43.x`
- **iPhone**: Usually `172.20.10.x`
- **Adjust setup_config.yaml** if your phone uses different range

### **Alternative Setup for iPhone Hotspot:**
If using iPhone (172.20.10.x range):

```yaml
phone_demo:
  network:
    pi_ip: "172.20.10.100"
    phone_hotspot_network: "172.20.10.0/24"
    # ... rest same
```

## 🎉 **Demo Presentation Flow**

1. **Setup (5 min)**: Connect everyone to hotspot, access web interface
2. **Overview (10 min)**: Show dashboard, explain real-time analytics
3. **Features (15 min)**: Demonstrate demographics, heatmaps, path tracking
4. **Technical (10 min)**: Show configuration, security features
5. **Q&A (10 min)**: Answer questions, discuss deployment options

**Total Demo Time: ~50 minutes**

This phone hotspot setup provides a **professional, portable demonstration environment** that works anywhere and gives you complete control over the network and demo experience! 