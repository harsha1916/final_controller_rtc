#!/usr/bin/env python3
"""
RFID Access Control System Restart Script
This script provides a simple way to restart the RFID system
"""

import os
import sys
import time
import subprocess
import signal
import psutil

def find_rfid_processes():
    """Find running RFID system processes."""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any('integrated_access_camera.py' in arg for arg in cmdline):
                processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes

def restart_rfid_system():
    """Restart the RFID access control system."""
    print("🔄 RFID Access Control System Restart")
    print("=" * 40)
    
    # Find current processes
    processes = find_rfid_processes()
    
    if processes:
        print(f"📋 Found {len(processes)} running RFID process(es)")
        
        # Stop existing processes
        for proc in processes:
            try:
                print(f"⏹️  Stopping process {proc.pid}...")
                proc.terminate()
                proc.wait(timeout=5)
                print(f"✅ Process {proc.pid} stopped")
            except psutil.TimeoutExpired:
                print(f"⚠️  Process {proc.pid} didn't stop gracefully, forcing...")
                proc.kill()
            except Exception as e:
                print(f"❌ Error stopping process {proc.pid}: {e}")
    else:
        print("ℹ️  No running RFID processes found")
    
    # Wait a moment
    print("⏳ Waiting 2 seconds...")
    time.sleep(2)
    
    # Start new process
    script_path = os.path.join(os.path.dirname(__file__), 'integrated_access_camera.py')
    
    if not os.path.exists(script_path):
        print(f"❌ Error: Script not found at {script_path}")
        return False
    
    try:
        print("🚀 Starting new RFID system process...")
        subprocess.Popen([sys.executable, script_path], 
                        cwd=os.path.dirname(script_path),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
        
        # Wait a moment and check if it started
        time.sleep(3)
        
        new_processes = find_rfid_processes()
        if new_processes:
            print(f"✅ RFID system restarted successfully! (PID: {new_processes[0].pid})")
            return True
        else:
            print("❌ Failed to start new RFID system process")
            return False
            
    except Exception as e:
        print(f"❌ Error starting new process: {e}")
        return False

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("RFID Access Control System Restart Script")
        print("Usage: python3 restart_rfid.py")
        print("")
        print("This script will:")
        print("1. Find and stop any running RFID system processes")
        print("2. Start a new instance of the RFID system")
        print("3. Verify the new process is running")
        return
    
    try:
        success = restart_rfid_system()
        if success:
            print("\n🎉 Restart completed successfully!")
            print("📱 You can now access the web interface")
        else:
            print("\n💥 Restart failed!")
            print("🔧 Please check the logs and try again")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Restart cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
