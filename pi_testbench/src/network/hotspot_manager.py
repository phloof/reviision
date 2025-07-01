"""
WiFi Hotspot Manager for Raspberry Pi
Manages WiFi access point functionality and network sharing
"""

import logging
import subprocess
import time
import socket
import netifaces
from typing import Dict, Any, List, Optional
import os
import signal

class HotspotManager:
    """Manages WiFi hotspot functionality with internet forwarding"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('reviision_pi.hotspot_manager')
        
        # Hotspot configuration
        hotspot_config = config.get('hotspot', {})
        self.interface = hotspot_config.get('interface', 'wlan0')
        self.ssid = hotspot_config.get('ssid', 'ReViision-TestBench')
        self.password = hotspot_config.get('password', 'testbench2024')
        self.gateway = hotspot_config.get('gateway', '192.168.4.1')
        self.ip_range = hotspot_config.get('ip_range', '192.168.4.0/24')
        self.dhcp_range = hotspot_config.get('dhcp_range', '192.168.4.2,192.168.4.20')
        
        # Internet connection configuration
        internet_config = config.get('internet', {})
        self.internet_method = internet_config.get('method', 'ethernet')
        self.internet_interface = None
        
        if self.internet_method == 'ethernet':
            self.internet_interface = internet_config.get('ethernet', {}).get('interface', 'eth0')
        elif self.internet_method == 'wifi':
            self.internet_interface = internet_config.get('wifi', {}).get('interface', 'wlan1')
        
        # Forwarding configuration
        forwarding_config = internet_config.get('forwarding', {})
        self.forwarding_enabled = forwarding_config.get('enabled', True)
        self.masquerade_enabled = forwarding_config.get('masquerade', True)
        self.allowed_ports = forwarding_config.get('allowed_ports', [80, 443, 53, 123, 554, 1935])
        self.blocked_domains = forwarding_config.get('blocked_domains', [])
        self.blocked_ips = forwarding_config.get('blocked_ips', [])
        
        # Process tracking
        self.hostapd_process = None
        self.dnsmasq_process = None
        
        # Status
        self.is_running = False
        self.connected_clients = []
        self.internet_connected = False
        
    def start_hotspot(self) -> bool:
        """
        Start the WiFi hotspot
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Starting WiFi hotspot...")
            
            # Check if services are already running
            if self.is_hotspot_running():
                self.logger.info("Hotspot is already running")
                return True
            
            # Configure network interface
            if not self._configure_interface():
                return False
            
            # Enable IP forwarding
            if not self._enable_ip_forwarding():
                return False
            
            # Configure iptables for NAT
            if not self._configure_nat():
                return False
            
            # Start hostapd
            if not self._start_hostapd():
                return False
            
            # Start dnsmasq
            if not self._start_dnsmasq():
                return False
            
            self.is_running = True
            self.logger.info("WiFi hotspot started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting hotspot: {e}")
            self.stop_hotspot()
            return False
    
    def stop_hotspot(self) -> bool:
        """
        Stop the WiFi hotspot
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Stopping WiFi hotspot...")
            
            # Stop processes
            self._stop_hostapd()
            self._stop_dnsmasq()
            
            # Remove NAT rules
            self._remove_nat_rules()
            
            # Reset interface
            self._reset_interface()
            
            self.is_running = False
            self.connected_clients = []
            
            self.logger.info("WiFi hotspot stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping hotspot: {e}")
            return False
    
    def restart_hotspot(self) -> bool:
        """Restart the WiFi hotspot"""
        self.logger.info("Restarting WiFi hotspot...")
        self.stop_hotspot()
        time.sleep(2)
        return self.start_hotspot()
    
    def is_hotspot_running(self) -> bool:
        """Check if the hotspot is currently running"""
        try:
            # Check if hostapd is running
            result = subprocess.run(['pgrep', 'hostapd'], capture_output=True)
            hostapd_running = result.returncode == 0
            
            # Check if dnsmasq is running
            result = subprocess.run(['pgrep', 'dnsmasq'], capture_output=True)
            dnsmasq_running = result.returncode == 0
            
            # Check if interface has correct IP
            interface_configured = self._check_interface_config()
            
            return hostapd_running and dnsmasq_running and interface_configured
            
        except Exception as e:
            self.logger.error(f"Error checking hotspot status: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get hotspot status and statistics"""
        try:
            # Check internet connectivity
            self.internet_connected = self.check_internet_connectivity()
            
            status = {
                'running': self.is_hotspot_running(),
                'interface': self.interface,
                'ssid': self.ssid,
                'gateway': self.gateway,
                'connected_clients': self.get_connected_clients(),
                'network_stats': self._get_network_stats(),
                'internet': {
                    'method': self.internet_method,
                    'interface': self.internet_interface,
                    'connected': self.internet_connected,
                    'forwarding_enabled': self.forwarding_enabled
                }
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting hotspot status: {e}")
            return {
                'running': False,
                'error': str(e)
            }
    
    def get_connected_clients(self) -> List[Dict[str, Any]]:
        """Get list of connected clients"""
        try:
            clients = []
            
            # Parse DHCP leases file
            dhcp_leases_file = '/var/lib/dhcp/dhcpd.leases'
            if os.path.exists(dhcp_leases_file):
                clients.extend(self._parse_dhcp_leases(dhcp_leases_file))
            
            # Alternative: parse dnsmasq leases
            dnsmasq_leases_file = '/var/lib/dhcpcd5/dhcpcd.leases'
            if os.path.exists(dnsmasq_leases_file):
                clients.extend(self._parse_dnsmasq_leases(dnsmasq_leases_file))
            
            # Get ARP table for additional info
            clients = self._enrich_with_arp_info(clients)
            
            self.connected_clients = clients
            return clients
            
        except Exception as e:
            self.logger.error(f"Error getting connected clients: {e}")
            return []
    
    def _configure_interface(self) -> bool:
        """Configure the wireless interface for hotspot mode"""
        try:
            # Bring interface down
            subprocess.run(['sudo', 'ip', 'link', 'set', self.interface, 'down'], 
                         check=True, capture_output=True)
            
            # Set IP address
            subprocess.run(['sudo', 'ip', 'addr', 'flush', 'dev', self.interface], 
                         check=True, capture_output=True)
            subprocess.run(['sudo', 'ip', 'addr', 'add', f'{self.gateway}/24', 'dev', self.interface], 
                         check=True, capture_output=True)
            
            # Bring interface up
            subprocess.run(['sudo', 'ip', 'link', 'set', self.interface, 'up'], 
                         check=True, capture_output=True)
            
            self.logger.debug(f"Interface {self.interface} configured with IP {self.gateway}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error configuring interface: {e}")
            return False
    
    def _check_interface_config(self) -> bool:
        """Check if interface is properly configured"""
        try:
            # Get interface info
            interfaces = netifaces.interfaces()
            if self.interface not in interfaces:
                return False
            
            addresses = netifaces.ifaddresses(self.interface)
            if netifaces.AF_INET not in addresses:
                return False
            
            # Check if our gateway IP is assigned
            for addr_info in addresses[netifaces.AF_INET]:
                if addr_info.get('addr') == self.gateway:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking interface config: {e}")
            return False
    
    def _enable_ip_forwarding(self) -> bool:
        """Enable IP forwarding for internet sharing"""
        try:
            # Enable IPv4 forwarding
            subprocess.run(['sudo', 'sysctl', 'net.ipv4.ip_forward=1'], 
                         check=True, capture_output=True)
            
            # Make it persistent
            sysctl_conf = '/etc/sysctl.conf'
            if os.path.exists(sysctl_conf):
                with open(sysctl_conf, 'r') as f:
                    content = f.read()
                
                if 'net.ipv4.ip_forward=1' not in content:
                    subprocess.run(['sudo', 'bash', '-c', 
                                  f'echo "net.ipv4.ip_forward=1" >> {sysctl_conf}'], 
                                 check=True)
            
            self.logger.debug("IP forwarding enabled")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error enabling IP forwarding: {e}")
            return False
    
    def _configure_nat(self) -> bool:
        """Configure NAT and firewall for internet sharing"""
        try:
            if not self.forwarding_enabled or not self.internet_interface:
                self.logger.info("Internet forwarding disabled or no internet interface")
                return True
            
            # Clear existing rules
            subprocess.run(['sudo', 'iptables', '-t', 'nat', '-F'], 
                         capture_output=True)
            subprocess.run(['sudo', 'iptables', '-F'], 
                         capture_output=True)
            subprocess.run(['sudo', 'iptables', '-X'], 
                         capture_output=True)
            
            # Create custom chains for filtering
            subprocess.run(['sudo', 'iptables', '-N', 'REVIISION_FORWARD'], 
                         capture_output=True)
            subprocess.run(['sudo', 'iptables', '-N', 'REVIISION_OUTPUT'], 
                         capture_output=True)
            
            # Set up masquerading for outgoing traffic if enabled
            if self.masquerade_enabled:
                subprocess.run(['sudo', 'iptables', '-t', 'nat', '-A', 'POSTROUTING', 
                              '-o', self.internet_interface, '-j', 'MASQUERADE'], 
                             check=True, capture_output=True)
                self.logger.debug(f"NAT masquerading enabled for {self.internet_interface}")
            
            # Allow loopback traffic
            subprocess.run(['sudo', 'iptables', '-A', 'INPUT', '-i', 'lo', '-j', 'ACCEPT'], 
                         check=True, capture_output=True)
            subprocess.run(['sudo', 'iptables', '-A', 'OUTPUT', '-o', 'lo', '-j', 'ACCEPT'], 
                         check=True, capture_output=True)
            
            # Allow established and related connections
            subprocess.run(['sudo', 'iptables', '-A', 'INPUT', 
                          '-m', 'state', '--state', 'ESTABLISHED,RELATED', '-j', 'ACCEPT'], 
                         check=True, capture_output=True)
            subprocess.run(['sudo', 'iptables', '-A', 'FORWARD', 
                          '-m', 'state', '--state', 'ESTABLISHED,RELATED', '-j', 'ACCEPT'], 
                         check=True, capture_output=True)
            
            # Allow hotspot network to access internet through specific ports
            for port in self.allowed_ports:
                # TCP traffic
                subprocess.run(['sudo', 'iptables', '-A', 'REVIISION_FORWARD',
                              '-i', self.interface, '-o', self.internet_interface,
                              '-p', 'tcp', '--dport', str(port), '-j', 'ACCEPT'], 
                             check=True, capture_output=True)
                # UDP traffic (for DNS, NTP, etc.)
                if port in [53, 123]:  # DNS and NTP
                    subprocess.run(['sudo', 'iptables', '-A', 'REVIISION_FORWARD',
                                  '-i', self.interface, '-o', self.internet_interface,
                                  '-p', 'udp', '--dport', str(port), '-j', 'ACCEPT'], 
                                 check=True, capture_output=True)
            
            # Allow ICMP (ping) for connectivity testing
            subprocess.run(['sudo', 'iptables', '-A', 'REVIISION_FORWARD',
                          '-i', self.interface, '-o', self.internet_interface,
                          '-p', 'icmp', '-j', 'ACCEPT'], 
                         check=True, capture_output=True)
            
            # Apply the custom forward chain
            subprocess.run(['sudo', 'iptables', '-A', 'FORWARD', '-j', 'REVIISION_FORWARD'], 
                         check=True, capture_output=True)
            
            # Allow traffic from internet interface to hotspot (return traffic)
            subprocess.run(['sudo', 'iptables', '-A', 'FORWARD', 
                          '-i', self.internet_interface, '-o', self.interface, 
                          '-m', 'state', '--state', 'RELATED,ESTABLISHED', '-j', 'ACCEPT'], 
                         check=True, capture_output=True)
            
            # Allow SSH access to Pi from hotspot network
            subprocess.run(['sudo', 'iptables', '-A', 'INPUT', 
                          '-i', self.interface, '-p', 'tcp', '--dport', '22', '-j', 'ACCEPT'], 
                         check=True, capture_output=True)
            
            # Allow DHCP traffic
            subprocess.run(['sudo', 'iptables', '-A', 'INPUT', 
                          '-i', self.interface, '-p', 'udp', '--dport', '67', '-j', 'ACCEPT'], 
                         check=True, capture_output=True)
            subprocess.run(['sudo', 'iptables', '-A', 'INPUT', 
                          '-i', self.interface, '-p', 'udp', '--dport', '68', '-j', 'ACCEPT'], 
                         check=True, capture_output=True)
            
            # Allow DNS traffic to Pi
            subprocess.run(['sudo', 'iptables', '-A', 'INPUT', 
                          '-i', self.interface, '-p', 'udp', '--dport', '53', '-j', 'ACCEPT'], 
                         check=True, capture_output=True)
            subprocess.run(['sudo', 'iptables', '-A', 'INPUT', 
                          '-i', self.interface, '-p', 'tcp', '--dport', '53', '-j', 'ACCEPT'], 
                         check=True, capture_output=True)
            
            # Block any other forwarding by default
            subprocess.run(['sudo', 'iptables', '-A', 'FORWARD', '-j', 'DROP'], 
                         check=True, capture_output=True)
            
            # Apply blocked IPs if configured
            for blocked_ip in self.blocked_ips:
                subprocess.run(['sudo', 'iptables', '-A', 'REVIISION_FORWARD',
                              '-d', blocked_ip, '-j', 'DROP'], 
                             capture_output=True)
            
            self.logger.debug(f"Enhanced NAT and firewall configured for {self.internet_interface}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error configuring NAT: {e}")
            return False
    
    def _start_hostapd(self) -> bool:
        """Start hostapd service"""
        try:
            # Check if hostapd config exists
            hostapd_conf = '/etc/hostapd/hostapd.conf'
            if not os.path.exists(hostapd_conf):
                self.logger.error(f"hostapd configuration not found: {hostapd_conf}")
                return False
            
            # Start hostapd as a background process
            self.hostapd_process = subprocess.Popen(
                ['sudo', 'hostapd', hostapd_conf],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a moment and check if it started successfully
            time.sleep(2)
            if self.hostapd_process.poll() is not None:
                stdout, stderr = self.hostapd_process.communicate()
                self.logger.error(f"hostapd failed to start: {stderr.decode()}")
                return False
            
            self.logger.debug("hostapd started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting hostapd: {e}")
            return False
    
    def _start_dnsmasq(self) -> bool:
        """Start dnsmasq service"""
        try:
            # Check if dnsmasq config exists
            dnsmasq_conf = '/etc/dnsmasq.conf'
            if not os.path.exists(dnsmasq_conf):
                self.logger.error(f"dnsmasq configuration not found: {dnsmasq_conf}")
                return False
            
            # Start dnsmasq
            result = subprocess.run(['sudo', 'systemctl', 'start', 'dnsmasq'], 
                                  check=True, capture_output=True)
            
            self.logger.debug("dnsmasq started")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error starting dnsmasq: {e}")
            return False
    
    def _stop_hostapd(self):
        """Stop hostapd process"""
        try:
            if self.hostapd_process:
                self.hostapd_process.terminate()
                try:
                    self.hostapd_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.hostapd_process.kill()
                self.hostapd_process = None
            
            # Also kill any running hostapd processes
            subprocess.run(['sudo', 'pkill', 'hostapd'], capture_output=True)
            
        except Exception as e:
            self.logger.error(f"Error stopping hostapd: {e}")
    
    def _stop_dnsmasq(self):
        """Stop dnsmasq service"""
        try:
            subprocess.run(['sudo', 'systemctl', 'stop', 'dnsmasq'], 
                         capture_output=True)
        except Exception as e:
            self.logger.error(f"Error stopping dnsmasq: {e}")
    
    def _remove_nat_rules(self):
        """Remove NAT iptables rules"""
        try:
            subprocess.run(['sudo', 'iptables', '-t', 'nat', '-F'], 
                         capture_output=True)
            subprocess.run(['sudo', 'iptables', '-F'], 
                         capture_output=True)
        except Exception as e:
            self.logger.error(f"Error removing NAT rules: {e}")
    
    def _reset_interface(self):
        """Reset the wireless interface"""
        try:
            subprocess.run(['sudo', 'ip', 'addr', 'flush', 'dev', self.interface], 
                         capture_output=True)
            subprocess.run(['sudo', 'ip', 'link', 'set', self.interface, 'down'], 
                         capture_output=True)
        except Exception as e:
            self.logger.error(f"Error resetting interface: {e}")
    
    def _parse_dhcp_leases(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse DHCP leases file"""
        clients = []
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse DHCP lease format
            # Implementation depends on DHCP server format
            
        except Exception as e:
            self.logger.debug(f"Error parsing DHCP leases: {e}")
        
        return clients
    
    def _parse_dnsmasq_leases(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse dnsmasq leases file"""
        clients = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        clients.append({
                            'timestamp': parts[0],
                            'mac': parts[1],
                            'ip': parts[2],
                            'hostname': parts[3] if len(parts) > 3 else 'unknown'
                        })
        except Exception as e:
            self.logger.debug(f"Error parsing dnsmasq leases: {e}")
        
        return clients
    
    def _enrich_with_arp_info(self, clients: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich client info with ARP table data"""
        try:
            # Get ARP table
            result = subprocess.run(['arp', '-a'], capture_output=True, text=True)
            if result.returncode == 0:
                arp_lines = result.stdout.strip().split('\n')
                arp_table = {}
                
                for line in arp_lines:
                    # Parse ARP entries
                    if '(' in line and ')' in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            ip = parts[1].strip('()')
                            mac = parts[3]
                            arp_table[ip] = mac
                
                # Match with clients
                for client in clients:
                    if client.get('ip') in arp_table:
                        client['arp_mac'] = arp_table[client['ip']]
            
        except Exception as e:
            self.logger.debug(f"Error enriching with ARP info: {e}")
        
        return clients
    
    def check_internet_connectivity(self) -> bool:
        """Check if internet connectivity is available"""
        try:
            if not self.internet_interface:
                return False
            
            # Check if internet interface is up and has an IP
            interfaces = netifaces.interfaces()
            if self.internet_interface not in interfaces:
                return False
            
            addresses = netifaces.ifaddresses(self.internet_interface)
            if netifaces.AF_INET not in addresses:
                return False
            
            # Try to ping a reliable host
            test_hosts = ['8.8.8.8', '1.1.1.1']
            for host in test_hosts:
                try:
                    result = subprocess.run(
                        ['ping', '-c', '1', '-W', '3', host],
                        capture_output=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        return True
                except subprocess.TimeoutExpired:
                    continue
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Error checking internet connectivity: {e}")
            return False
    
    def configure_wifi_internet(self, ssid: str, password: str = None, enterprise_config: Dict[str, Any] = None) -> bool:
        """Configure WiFi for internet connection (requires secondary WiFi interface)"""
        try:
            if self.internet_method != 'wifi' or not self.internet_interface:
                self.logger.error("WiFi internet method not configured")
                return False

            # Create wpa_supplicant configuration
            wpa_config = f"""country=US
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

"""
            
            # Check if this is WPA2 Enterprise configuration
            if enterprise_config and enterprise_config.get('enabled', False):
                # WPA2 Enterprise configuration
                network_config = f"""network={{
    ssid="{ssid}"
    key_mgmt=WPA-EAP
    eap={enterprise_config.get('eap_method', 'PEAP').upper()}
    identity="{enterprise_config.get('identity', '')}"
"""
                
                # Add anonymous identity if provided
                if enterprise_config.get('anonymous_identity'):
                    network_config += f'    anonymous_identity="{enterprise_config["anonymous_identity"]}"\n'
                
                # Add password for PEAP/TTLS
                if enterprise_config.get('password') and enterprise_config.get('eap_method', '').lower() in ['peap', 'ttls']:
                    network_config += f'    password="{enterprise_config["password"]}"\n'
                
                # Add phase 2 authentication
                if enterprise_config.get('phase2_auth'):
                    phase2 = enterprise_config['phase2_auth'].upper()
                    if enterprise_config.get('eap_method', '').lower() == 'peap':
                        network_config += f'    phase2="auth={phase2}"\n'
                    elif enterprise_config.get('eap_method', '').lower() == 'ttls':
                        network_config += f'    phase2="autheap={phase2}"\n'
                
                # Add certificates
                certificates = enterprise_config.get('certificates', {})
                if certificates.get('ca_cert'):
                    network_config += f'    ca_cert="{certificates["ca_cert"]}"\n'
                if certificates.get('client_cert'):
                    network_config += f'    client_cert="{certificates["client_cert"]}"\n'
                if certificates.get('private_key'):
                    network_config += f'    private_key="{certificates["private_key"]}"\n'
                if certificates.get('private_key_passwd'):
                    network_config += f'    private_key_passwd="{certificates["private_key_passwd"]}"\n'
                
                # Add security settings
                security = enterprise_config.get('security', {})
                if security.get('domain_suffix_match'):
                    network_config += f'    domain_suffix_match="{security["domain_suffix_match"]}"\n'
                if security.get('subject_match'):
                    network_config += f'    subject_match="{security["subject_match"]}"\n'
                if security.get('ca_cert_verification', True):
                    # Enable certificate verification by default
                    pass
                else:
                    network_config += '    ca_cert=""\n'  # Disable certificate verification (not recommended)
                
                # Close network block
                network_config += "}\n"
                
                self.logger.info(f"Configuring WPA2 Enterprise WiFi for {ssid}")
                
            else:
                # Standard WPA2-PSK configuration
                if not password:
                    self.logger.error("Password required for standard WiFi configuration")
                    return False
                
                network_config = f"""network={{
    ssid="{ssid}"
    psk="{password}"
    key_mgmt=WPA-PSK
}}
"""
                self.logger.info(f"Configuring standard WiFi for {ssid}")
            
            # Combine configuration
            wpa_config += network_config
            
            config_file = f"/etc/wpa_supplicant/wpa_supplicant-{self.internet_interface}.conf"
            
            # Write configuration
            with open('/tmp/wpa_supplicant_temp.conf', 'w') as f:
                f.write(wpa_config)
            
            subprocess.run(['sudo', 'cp', '/tmp/wpa_supplicant_temp.conf', config_file],
                         check=True)
            subprocess.run(['sudo', 'chmod', '600', config_file],
                         check=True)
            subprocess.run(['rm', '/tmp/wpa_supplicant_temp.conf'],
                         check=True)
            
            # Stop existing wpa_supplicant service
            subprocess.run(['sudo', 'systemctl', 'stop', f'wpa_supplicant@{self.internet_interface}'],
                         capture_output=True)
            
            # Start wpa_supplicant for the interface
            subprocess.run(['sudo', 'systemctl', 'enable', f'wpa_supplicant@{self.internet_interface}'],
                         check=True)
            subprocess.run(['sudo', 'systemctl', 'start', f'wpa_supplicant@{self.internet_interface}'],
                         check=True)
            
            # Wait for connection
            time.sleep(5)
            
            # Verify connection
            if self.check_internet_connectivity():
                self.logger.info(f"WiFi internet successfully configured for {self.internet_interface}")
                return True
            else:
                self.logger.warning("WiFi configuration completed but internet connectivity test failed")
                return True  # Still return True as config was applied
            
        except Exception as e:
            self.logger.error(f"Error configuring WiFi internet: {e}")
            return False
    
    def get_internet_status(self) -> Dict[str, Any]:
        """Get detailed internet connection status"""
        try:
            status = {
                'method': self.internet_method,
                'interface': self.internet_interface,
                'connected': False,
                'ip_address': None,
                'gateway': None,
                'dns_servers': [],
                'speed_test': None
            }
            
            if not self.internet_interface:
                return status
            
            # Get interface information
            interfaces = netifaces.interfaces()
            if self.internet_interface in interfaces:
                addresses = netifaces.ifaddresses(self.internet_interface)
                
                if netifaces.AF_INET in addresses:
                    inet_info = addresses[netifaces.AF_INET][0]
                    status['ip_address'] = inet_info.get('addr')
                    status['connected'] = True
                
                # Get gateway information
                gateways = netifaces.gateways()
                if 'default' in gateways and netifaces.AF_INET in gateways['default']:
                    default_gw = gateways['default'][netifaces.AF_INET]
                    if default_gw[1] == self.internet_interface:
                        status['gateway'] = default_gw[0]
            
            # Check actual connectivity
            if status['connected']:
                status['connected'] = self.check_internet_connectivity()
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting internet status: {e}")
            return {'error': str(e)}
    
    def get_bandwidth_usage(self) -> Dict[str, Any]:
        """Get bandwidth usage statistics"""
        try:
            import psutil
            
            # Get network I/O statistics
            net_io = psutil.net_io_counters(pernic=True)
            
            usage_stats = {}
            for interface in [self.interface, self.internet_interface]:
                if interface and interface in net_io:
                    stats = net_io[interface]
                    usage_stats[interface] = {
                        'bytes_sent': stats.bytes_sent,
                        'bytes_recv': stats.bytes_recv,
                        'packets_sent': stats.packets_sent,
                        'packets_recv': stats.packets_recv,
                        'errors_in': stats.errin,
                        'errors_out': stats.errout,
                        'drops_in': stats.dropin,
                        'drops_out': stats.dropout
                    }
            
            return usage_stats
            
        except Exception as e:
            self.logger.error(f"Error getting bandwidth usage: {e}")
            return {}
    
    def _get_network_stats(self) -> Dict[str, Any]:
        """Get network interface statistics"""
        try:
            interfaces = netifaces.interfaces()
            stats = {}
            
            # Include both hotspot and internet interfaces
            check_interfaces = [self.interface]
            if self.internet_interface:
                check_interfaces.append(self.internet_interface)
            
            for interface in check_interfaces:
                if interface in interfaces:
                    addresses = netifaces.ifaddresses(interface)
                    stats[interface] = {
                        'addresses': addresses,
                        'is_up': interface in netifaces.interfaces()
                    }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting network stats: {e}")
            return {} 