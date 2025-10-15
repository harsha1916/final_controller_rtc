#!/usr/bin/env python3
"""
RTC Availability Check Script for MaxPark RFID System
Checks if DS3231/DS3232 RTC module is properly connected and configured.
"""
import os
import subprocess
import time

def check_dev_rtc():
    """Check if /dev/rtc device exists."""
    try:
        rtc_devices = [f for f in os.listdir("/dev") if f.startswith("rtc")]
        if rtc_devices:
            print(f"✅ RTC device(s) found: {', '.join(rtc_devices)}")
            return True
        else:
            print("❌ No /dev/rtc device found.")
            return False
    except Exception as e:
        print(f"❌ Error checking /dev/rtc: {e}")
        return False

def check_hwclock():
    """Check if hwclock can read the RTC."""
    try:
        output = subprocess.check_output(["sudo", "hwclock", "-r"], stderr=subprocess.STDOUT)
        print("✅ hwclock read successful:")
        print(f"   {output.decode().strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ hwclock failed:")
        print(f"   {e.output.decode().strip()}")
        return False
    except FileNotFoundError:
        print("❌ hwclock command not found")
        return False
    except Exception as e:
        print(f"❌ hwclock error: {e}")
        return False

def check_i2c_modules():
    """Check if I2C kernel modules are loaded."""
    try:
        output = subprocess.check_output(["lsmod"], stderr=subprocess.STDOUT)
        output_str = output.decode()
        
        i2c_modules = []
        for line in output_str.split('\n'):
            if 'i2c' in line.lower():
                i2c_modules.append(line.split()[0])
        
        if i2c_modules:
            print(f"✅ I2C modules loaded: {', '.join(i2c_modules)}")
            return True
        else:
            print("❌ No I2C modules loaded")
            print("   Run: sudo raspi-config → Interface Options → I2C → Enable")
            return False
    except Exception as e:
        print(f"❌ Error checking I2C modules: {e}")
        return False

def check_rtc_driver():
    """Check if RTC driver is loaded."""
    try:
        output = subprocess.check_output(["lsmod"], stderr=subprocess.STDOUT)
        output_str = output.decode()
        
        if 'rtc_ds1307' in output_str or 'rtc_ds3231' in output_str:
            print("✅ RTC driver loaded (rtc_ds1307 or rtc_ds3231)")
            return True
        else:
            print("⚠️  RTC driver not loaded")
            print("   To load driver, run:")
            print("   echo ds3231 0x68 | sudo tee /sys/bus/i2c/devices/i2c-1/new_device")
            return False
    except Exception as e:
        print(f"❌ Error checking RTC driver: {e}")
        return False

def check_system_time():
    """Check current system time."""
    try:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"✅ System time: {current_time}")
        return True
    except Exception as e:
        print(f"❌ Error getting system time: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("RTC AVAILABILITY CHECK FOR MAXPARK RFID SYSTEM")
    print("=" * 50)
    print()
    
    print("1. Checking /dev/rtc device...")
    dev_ok = check_dev_rtc()
    print()
    
    print("2. Checking hwclock...")
    hwclock_ok = check_hwclock()
    print()
    
    print("3. Checking I2C modules...")
    i2c_ok = check_i2c_modules()
    print()
    
    print("4. Checking RTC driver...")
    driver_ok = check_rtc_driver()
    print()
    
    print("5. Checking system time...")
    time_ok = check_system_time()
    print()
    
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if dev_ok and hwclock_ok:
        print("✅ RTC hardware clock is available and working correctly.")
        print("   Your RTC module is properly configured!")
    elif i2c_ok and not dev_ok:
        print("⚠️  I2C is working but RTC driver not loaded.")
        print("   Run this command to load the driver:")
        print("   echo ds3231 0x68 | sudo tee /sys/bus/i2c/devices/i2c-1/new_device")
    elif not i2c_ok:
        print("❌ I2C is not enabled or not working.")
        print("   Enable I2C:")
        print("   1. Run: sudo raspi-config")
        print("   2. Go to: Interface Options → I2C → Enable")
        print("   3. Reboot: sudo reboot")
    else:
        print("❌ RTC not detected or not configured properly.")
        print("   Check hardware connections:")
        print("   - VCC → 3.3V or 5V")
        print("   - GND → Ground")
        print("   - SDA → GPIO 2 (Pin 3)")
        print("   - SCL → GPIO 3 (Pin 5)")
    
    print()
    print("For more help, see: DS3232_RTC_GUIDE.md")
    print("=" * 50)

