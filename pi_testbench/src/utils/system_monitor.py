"""
System monitoring utility for Raspberry Pi
Monitors CPU, memory, temperature, and other system metrics
"""

import asyncio
import psutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time

class SystemMonitor:
    """System health monitoring for Raspberry Pi"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('reviision_pi.system_monitor')
        
        # Performance thresholds
        self.max_cpu_usage = config.get('max_cpu_usage', 80)
        self.max_memory_usage = config.get('max_memory_usage', 80)
        
        # Temperature thresholds
        temp_config = config.get('temperature', {})
        self.temp_warning = temp_config.get('warning_threshold', 70)
        self.temp_critical = temp_config.get('critical_threshold', 80)
        
        # Cache for performance
        self.last_check = 0
        self.cache_duration = 30  # seconds
        self.cached_status = None
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Check overall system health
        
        Returns:
            Dict containing health status and metrics
        """
        current_time = time.time()
        
        # Return cached result if recent
        if (self.cached_status and 
            current_time - self.last_check < self.cache_duration):
            return self.cached_status
        
        try:
            # Get system metrics
            cpu_percent = await self.get_cpu_usage()
            memory_info = await self.get_memory_info()
            disk_info = await self.get_disk_info()
            temperature = await self.get_temperature()
            network_info = await self.get_network_info()
            
            # Determine overall status
            status = "healthy"
            messages = []
            
            # Check CPU usage
            if cpu_percent > self.max_cpu_usage:
                if cpu_percent > 95:
                    status = "critical"
                    messages.append(f"CPU usage critical: {cpu_percent:.1f}%")
                elif status != "critical":
                    status = "warning"
                    messages.append(f"CPU usage high: {cpu_percent:.1f}%")
            
            # Check memory usage
            memory_percent = memory_info['percent']
            if memory_percent > self.max_memory_usage:
                if memory_percent > 95:
                    status = "critical"
                    messages.append(f"Memory usage critical: {memory_percent:.1f}%")
                elif status != "critical":
                    status = "warning"
                    messages.append(f"Memory usage high: {memory_percent:.1f}%")
            
            # Check temperature
            if temperature:
                if temperature > self.temp_critical:
                    status = "critical"
                    messages.append(f"Temperature critical: {temperature:.1f}°C")
                elif temperature > self.temp_warning and status != "critical":
                    status = "warning"
                    messages.append(f"Temperature high: {temperature:.1f}°C")
            
            # Check disk space
            if disk_info['percent'] > 90:
                if disk_info['percent'] > 95:
                    status = "critical"
                    messages.append(f"Disk space critical: {disk_info['percent']:.1f}%")
                elif status != "critical":
                    status = "warning"
                    messages.append(f"Disk space low: {disk_info['percent']:.1f}%")
            
            health_status = {
                'status': status,
                'message': "; ".join(messages) if messages else "System operating normally",
                'metrics': {
                    'cpu_percent': cpu_percent,
                    'memory': memory_info,
                    'disk': disk_info,
                    'temperature': temperature,
                    'network': network_info,
                    'uptime': self.get_uptime()
                },
                'timestamp': current_time
            }
            
            # Cache the result
            self.cached_status = health_status
            self.last_check = current_time
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
            return {
                'status': 'error',
                'message': f"Health check failed: {str(e)}",
                'metrics': {},
                'timestamp': current_time
            }
    
    async def get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        # Use interval for more accurate reading
        return psutil.cpu_percent(interval=1)
    
    async def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent
        }
    
    async def get_disk_info(self) -> Dict[str, Any]:
        """Get disk usage information"""
        disk = psutil.disk_usage('/')
        return {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent': (disk.used / disk.total) * 100
        }
    
    async def get_temperature(self) -> Optional[float]:
        """Get CPU temperature (Raspberry Pi specific)"""
        try:
            # Try thermal zone method first
            thermal_file = Path('/sys/class/thermal/thermal_zone0/temp')
            if thermal_file.exists():
                temp_str = thermal_file.read_text().strip()
                return float(temp_str) / 1000.0
            
            # Fallback to vcgencmd if available
            import subprocess
            result = subprocess.run(
                ['vcgencmd', 'measure_temp'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                temp_str = result.stdout.strip()
                # Parse "temp=45.1'C" format
                if 'temp=' in temp_str:
                    temp_value = temp_str.split('=')[1].split("'")[0]
                    return float(temp_value)
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Could not read temperature: {e}")
            return None
    
    async def get_network_info(self) -> Dict[str, Any]:
        """Get network interface information"""
        try:
            network_stats = psutil.net_io_counters(pernic=True)
            interfaces = {}
            
            for interface, stats in network_stats.items():
                interfaces[interface] = {
                    'bytes_sent': stats.bytes_sent,
                    'bytes_recv': stats.bytes_recv,
                    'packets_sent': stats.packets_sent,
                    'packets_recv': stats.packets_recv,
                    'errors_in': stats.errin,
                    'errors_out': stats.errout,
                    'drops_in': stats.dropin,
                    'drops_out': stats.dropout
                }
            
            return interfaces
            
        except Exception as e:
            self.logger.error(f"Error getting network info: {e}")
            return {}
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.readline().split()[0])
            return uptime_seconds
        except Exception:
            # Fallback using psutil
            return time.time() - psutil.boot_time()
    
    async def get_process_info(self) -> Dict[str, Any]:
        """Get information about running processes"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    proc_info = proc.info
                    if proc_info['cpu_percent'] > 0 or proc_info['memory_percent'] > 1:
                        processes.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Sort by CPU usage
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            
            return {
                'total_processes': len(list(psutil.process_iter())),
                'top_processes': processes[:10]  # Top 10 processes
            }
            
        except Exception as e:
            self.logger.error(f"Error getting process info: {e}")
            return {}
    
    async def get_network_connections(self) -> Dict[str, Any]:
        """Get active network connections"""
        try:
            connections = psutil.net_connections(kind='inet')
            
            listening = []
            established = []
            
            for conn in connections:
                if conn.status == psutil.CONN_LISTEN:
                    listening.append({
                        'address': conn.laddr,
                        'pid': conn.pid
                    })
                elif conn.status == psutil.CONN_ESTABLISHED:
                    established.append({
                        'local': conn.laddr,
                        'remote': conn.raddr,
                        'pid': conn.pid
                    })
            
            return {
                'listening': listening,
                'established': established,
                'total_connections': len(connections)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting network connections: {e}")
            return {} 