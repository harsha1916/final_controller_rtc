"""
DS3232 RTC Module for MaxPark RFID Access Control System
Provides accurate time even during power outages and extended offline periods.
"""

import time
import logging
import os
from datetime import datetime, timezone
from typing import Optional, Tuple
import subprocess
import re

class DS3232RTC:
    """
    DS3232 Real-Time Clock integration for Raspberry Pi.
    Provides accurate timestamps even during power outages.
    """
    
    def __init__(self, enable_rtc: bool = True, i2c_bus: int = 1, i2c_address: int = 0x68):
        """
        Initialize DS3232 RTC module.
        
        Args:
            enable_rtc: Enable RTC functionality (default: True)
            i2c_bus: I2C bus number (default: 1)
            i2c_address: I2C address of DS3232 (default: 0x68)
        """
        self.enable_rtc = enable_rtc
        self.i2c_bus = i2c_bus
        self.i2c_address = i2c_address
        self.logger = logging.getLogger(__name__)
        self.rtc_available = False
        
        if self.enable_rtc:
            self._initialize_rtc()
    
    def _initialize_rtc(self):
        """Initialize RTC and check availability."""
        try:
            # Check if RTC is available via i2c
            if self._check_rtc_presence():
                self.rtc_available = True
                self.logger.info("DS3232 RTC module detected and initialized")
                
                # Sync system time with RTC on startup
                self._sync_system_time_from_rtc()
            else:
                self.rtc_available = False
                self.logger.warning("DS3232 RTC module not detected, falling back to system time")
                
        except Exception as e:
            self.rtc_available = False
            self.logger.error(f"Error initializing DS3232 RTC: {e}")
    
    def _check_rtc_presence(self) -> bool:
        """Check if DS3232 RTC is present on I2C bus."""
        try:
            # Use i2cdetect to check for RTC at address 0x68
            result = subprocess.run(
                ['i2cdetect', '-y', str(self.i2c_bus)],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            self.logger.info(f"i2cdetect output: {result.stdout}")
            self.logger.info(f"i2cdetect stderr: {result.stderr}")
            self.logger.info(f"Looking for address: {self.i2c_address:02x}")
            
            if result.returncode == 0:
                # Check if our address (0x68) is present in the output
                address_found = f"{self.i2c_address:02x}" in result.stdout.lower()
                self.logger.info(f"RTC address {self.i2c_address:02x} found: {address_found}")
                return address_found
            else:
                self.logger.error(f"i2cdetect failed with return code {result.returncode}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("i2cdetect timeout")
            return False
        except FileNotFoundError:
            self.logger.error("i2cdetect command not found. Make sure i2c-tools is installed.")
            return False
        except Exception as e:
            self.logger.error(f"Error checking RTC presence: {e}")
            return False
    
    def _sync_system_time_from_rtc(self):
        """Sync system time from RTC module."""
        try:
            rtc_time = self._read_rtc_time()
            if rtc_time:
                # Convert to system time format
                system_time = rtc_time.strftime('%Y-%m-%d %H:%M:%S')
                
                # Set system time (requires sudo)
                result = subprocess.run(
                    ['sudo', 'date', '-s', system_time],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    self.logger.info(f"System time synced from RTC: {system_time}")
                else:
                    self.logger.warning(f"Failed to sync system time from RTC: {result.stderr}")
                    
        except Exception as e:
            self.logger.error(f"Error syncing system time from RTC: {e}")
    
    def _read_rtc_time(self) -> Optional[datetime]:
        """Read time from DS3232 RTC module."""
        try:
            # Use hwclock to read RTC time
            result = subprocess.run(
                ['sudo', 'hwclock', '--show', '--utc'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Parse the output: "2024-01-15 14:30:25.123456+00:00"
                time_str = result.stdout.strip()
                # Extract just the date and time part
                time_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', time_str)
                if time_match:
                    time_str = time_match.group(1)
                    return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            
            return None
            
        except subprocess.TimeoutExpired:
            self.logger.error("hwclock timeout")
            return None
        except Exception as e:
            self.logger.error(f"Error reading RTC time: {e}")
            return None
    
    def get_current_time(self) -> datetime:
        """
        Get current time from RTC if available, otherwise fall back to system time.
        
        Returns:
            datetime: Current timestamp
        """
        if self.rtc_available and self.enable_rtc:
            try:
                rtc_time = self._read_rtc_time()
                if rtc_time:
                    return rtc_time
                else:
                    self.logger.warning("RTC read failed, falling back to system time")
            except Exception as e:
                self.logger.error(f"RTC error, falling back to system time: {e}")
        
        # Fallback to system time
        return datetime.now()
    
    def get_timestamp(self) -> int:
        """
        Get current Unix timestamp from RTC if available, otherwise system time.
        
        Returns:
            int: Unix timestamp
        """
        return int(self.get_current_time().timestamp())
    
    def get_iso_timestamp(self) -> str:
        """
        Get current ISO format timestamp from RTC if available, otherwise system time.
        
        Returns:
            str: ISO format timestamp
        """
        return self.get_current_time().isoformat()
    
    def sync_rtc_from_system(self):
        """Sync RTC time from system time (useful when system time is more accurate)."""
        if not self.rtc_available:
            self.logger.warning("RTC not available, cannot sync")
            return
        
        try:
            # Write system time to RTC
            result = subprocess.run(
                ['sudo', 'hwclock', '--systohc'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.logger.info("RTC synced from system time")
            else:
                self.logger.error(f"Failed to sync RTC from system time: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error syncing RTC from system time: {e}")
    
    def get_rtc_status(self) -> dict:
        """
        Get RTC status and information.
        
        Returns:
            dict: RTC status information
        """
        status = {
            "rtc_enabled": self.enable_rtc,
            "rtc_available": self.rtc_available,
            "i2c_bus": self.i2c_bus,
            "i2c_address": f"0x{self.i2c_address:02x}",
            "current_time": self.get_iso_timestamp(),
            "time_source": "RTC" if (self.rtc_available and self.enable_rtc) else "System"
        }
        
        if self.rtc_available:
            try:
                rtc_time = self._read_rtc_time()
                if rtc_time:
                    status["rtc_time"] = rtc_time.isoformat()
                    status["system_time"] = datetime.now().isoformat()
                    
                    # Calculate time difference
                    time_diff = abs((rtc_time - datetime.now()).total_seconds())
                    status["time_difference_seconds"] = time_diff
            except Exception as e:
                status["rtc_error"] = str(e)
        
        return status
    
    def setup_rtc_systemd(self):
        """
        Setup systemd service to sync RTC on boot.
        This should be run once during system setup.
        """
        try:
            # Create systemd service file for RTC sync
            service_content = f"""[Unit]
Description=DS3232 RTC Time Sync
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/sbin/hwclock --hctosys
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
"""
            
            service_file = "/etc/systemd/system/ds3232-rtc-sync.service"
            
            # Write service file (requires sudo)
            with open(service_file, 'w') as f:
                f.write(service_content)
            
            # Enable the service
            subprocess.run(['sudo', 'systemctl', 'enable', 'ds3232-rtc-sync.service'], 
                         check=True, timeout=10)
            
            self.logger.info("DS3232 RTC systemd service configured")
            
        except Exception as e:
            self.logger.error(f"Error setting up RTC systemd service: {e}")

# Global RTC instance
_rtc_instance = None

def get_rtc_instance() -> DS3232RTC:
    """Get global RTC instance (singleton pattern)."""
    global _rtc_instance
    if _rtc_instance is None:
        enable_rtc = os.environ.get('RTC_ENABLED', 'true').lower() == 'true'
        i2c_bus = int(os.environ.get('RTC_I2C_BUS', '1'))
        i2c_address = int(os.environ.get('RTC_I2C_ADDRESS', '0x68'), 16)
        
        _rtc_instance = DS3232RTC(
            enable_rtc=enable_rtc,
            i2c_bus=i2c_bus,
            i2c_address=i2c_address
        )
    return _rtc_instance

def get_accurate_timestamp() -> int:
    """
    Get accurate timestamp from RTC if available, otherwise system time.
    This is the main function to use throughout the application.
    
    Returns:
        int: Unix timestamp
    """
    rtc = get_rtc_instance()
    return rtc.get_timestamp()

def get_accurate_datetime() -> datetime:
    """
    Get accurate datetime from RTC if available, otherwise system time.
    
    Returns:
        datetime: Current datetime
    """
    rtc = get_rtc_instance()
    return rtc.get_current_time()

def get_accurate_iso_timestamp() -> str:
    """
    Get accurate ISO timestamp from RTC if available, otherwise system time.
    
    Returns:
        str: ISO format timestamp
    """
    rtc = get_rtc_instance()
    return rtc.get_iso_timestamp()
