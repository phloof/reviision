#!/usr/bin/env python3
"""
ReViision Raspberry Pi Test Bench
Main application entry point for the Pi-based test bench system.

This application provides:
- WiFi hotspot management
- E-paper display for real-time analytics
- Data collection from ReViision server
- System monitoring and health checks
"""

import asyncio
import logging
import signal
import sys
import os
from pathlib import Path
import yaml
import time
from typing import Optional

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.display.epaper_manager import EpaperManager
from src.network.hotspot_manager import HotspotManager
from src.data.reviision_client import ReViisionClient
from src.utils.config_manager import ConfigManager
from src.utils.system_monitor import SystemMonitor
from src.utils.logger import setup_logging

class ReViisionPiTestBench:
    """Main application class for the Pi test bench"""
    
    def __init__(self, config_path: str = "config/pi_config.yaml"):
        self.config_path = config_path
        self.config = None
        self.logger = None
        self.running = False
        
        # Component managers
        self.epaper_manager: Optional[EpaperManager] = None
        self.hotspot_manager: Optional[HotspotManager] = None
        self.reviision_client: Optional[ReViisionClient] = None
        self.system_monitor: Optional[SystemMonitor] = None
        
        # Async tasks
        self.tasks = []
        
    def load_config(self) -> bool:
        """Load configuration from YAML file"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                print(f"Configuration file not found: {config_file}")
                return False
                
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            return True
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def setup_logging(self) -> bool:
        """Setup logging configuration"""
        try:
            log_config = self.config.get('logging', {})
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            
            self.logger = setup_logging(
                level=log_config.get('level', 'INFO'),
                log_file=log_config.get('log_file', 'logs/pi_testbench.log'),
                max_size=log_config.get('max_size', '10MB'),
                backup_count=log_config.get('backup_count', 5),
                format_str=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            
            self.logger.info("ReViision Pi Test Bench starting up...")
            self.logger.info(f"Configuration loaded from: {self.config_path}")
            return True
        except Exception as e:
            print(f"Error setting up logging: {e}")
            return False
    
    def initialize_components(self) -> bool:
        """Initialize all system components"""
        try:
            self.logger.info("Initializing system components...")
            
            # Initialize system monitor
            self.system_monitor = SystemMonitor(self.config.get('system', {}))
            
            # Initialize hotspot manager
            hotspot_config = self.config.get('network', {}).get('hotspot', {})
            self.hotspot_manager = HotspotManager(hotspot_config)
            
            # Initialize ReViision client
            server_config = self.config.get('network', {}).get('server', {})
            self.reviision_client = ReViisionClient(server_config)
            
            # Initialize e-paper display manager
            display_config = self.config.get('display', {})
            self.epaper_manager = EpaperManager(display_config)
            
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            return False
    
    async def start_background_tasks(self):
        """Start all background tasks"""
        try:
            self.logger.info("Starting background tasks...")
            
            # Data collection task
            data_config = self.config.get('data', {})
            self.tasks.append(asyncio.create_task(
                self.data_collection_loop(data_config)
            ))
            
            # Display update task
            self.tasks.append(asyncio.create_task(
                self.display_update_loop()
            ))
            
            # System monitoring task
            system_config = self.config.get('system', {})
            if system_config.get('health_check', {}).get('enabled', True):
                self.tasks.append(asyncio.create_task(
                    self.system_monitoring_loop(system_config)
                ))
            
            # Network monitoring task
            self.tasks.append(asyncio.create_task(
                self.network_monitoring_loop()
            ))
            
            self.logger.info(f"Started {len(self.tasks)} background tasks")
            
        except Exception as e:
            self.logger.error(f"Error starting background tasks: {e}")
            raise
    
    async def data_collection_loop(self, config: dict):
        """Main data collection loop"""
        polling_config = config.get('polling', {})
        analytics_interval = polling_config.get('analytics', 15)
        
        self.logger.info(f"Starting data collection loop (interval: {analytics_interval}s)")
        
        while self.running:
            try:
                # Collect analytics data
                analytics_data = await self.reviision_client.get_analytics_data()
                if analytics_data:
                    # Update display data
                    self.epaper_manager.update_analytics_data(analytics_data)
                    self.logger.debug("Analytics data updated")
                
                await asyncio.sleep(analytics_interval)
                
            except Exception as e:
                self.logger.error(f"Error in data collection loop: {e}")
                await asyncio.sleep(analytics_interval)
    
    async def display_update_loop(self):
        """Display update loop"""
        display_config = self.config.get('display', {})
        refresh_interval = display_config.get('refresh_interval', 30)
        
        self.logger.info(f"Starting display update loop (interval: {refresh_interval}s)")
        
        while self.running:
            try:
                # Update the e-paper display
                await self.epaper_manager.update_display()
                self.logger.debug("Display updated")
                
                await asyncio.sleep(refresh_interval)
                
            except Exception as e:
                self.logger.error(f"Error in display update loop: {e}")
                await asyncio.sleep(refresh_interval)
    
    async def system_monitoring_loop(self, config: dict):
        """System health monitoring loop"""
        health_config = config.get('health_check', {})
        interval = health_config.get('interval', 300)
        
        self.logger.info(f"Starting system monitoring loop (interval: {interval}s)")
        
        while self.running:
            try:
                # Check system health
                health_status = await self.system_monitor.check_health()
                
                # Update display with system status
                self.epaper_manager.update_system_status(health_status)
                
                # Log warnings or critical issues
                if health_status.get('status') == 'warning':
                    self.logger.warning(f"System health warning: {health_status.get('message')}")
                elif health_status.get('status') == 'critical':
                    self.logger.error(f"System health critical: {health_status.get('message')}")
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    async def network_monitoring_loop(self):
        """Network status monitoring loop"""
        data_config = self.config.get('data', {})
        polling_config = data_config.get('polling', {})
        network_interval = polling_config.get('network_status', 120)
        
        self.logger.info(f"Starting network monitoring loop (interval: {network_interval}s)")
        
        while self.running:
            try:
                # Check hotspot status
                hotspot_status = self.hotspot_manager.get_status()
                
                # Check ReViision server connectivity
                server_status = await self.reviision_client.check_connectivity()
                
                # Update display with network info
                network_info = {
                    'hotspot': hotspot_status,
                    'server': server_status,
                    'timestamp': time.time()
                }
                self.epaper_manager.update_network_info(network_info)
                
                await asyncio.sleep(network_interval)
                
            except Exception as e:
                self.logger.error(f"Error in network monitoring loop: {e}")
                await asyncio.sleep(network_interval)
    
    async def shutdown(self):
        """Graceful shutdown procedure"""
        self.logger.info("Initiating graceful shutdown...")
        self.running = False
        
        # Cancel all background tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Shutdown components
        if self.epaper_manager:
            await self.epaper_manager.shutdown()
        
        if self.reviision_client:
            await self.reviision_client.close()
        
        self.logger.info("Shutdown complete")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self):
        """Main application run method"""
        try:
            # Load configuration
            if not self.load_config():
                return 1
            
            # Setup logging
            if not self.setup_logging():
                return 1
            
            # Initialize components
            if not self.initialize_components():
                return 1
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Wait for startup delay
            startup_delay = self.config.get('system', {}).get('startup_delay', 30)
            if startup_delay > 0:
                self.logger.info(f"Waiting {startup_delay} seconds for system stabilization...")
                await asyncio.sleep(startup_delay)
            
            # Start the application
            self.running = True
            self.logger.info("ReViision Pi Test Bench is now running")
            
            # Start background tasks
            await self.start_background_tasks()
            
            # Keep running until shutdown signal
            while self.running:
                await asyncio.sleep(1)
            
            # Graceful shutdown
            await self.shutdown()
            
            return 0
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Fatal error in main application: {e}")
            else:
                print(f"Fatal error: {e}")
            return 1

def main():
    """Main entry point"""
    # Create and run the application
    app = ReViisionPiTestBench()
    
    try:
        # Run the async application
        return asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, shutting down...")
        return 0
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 