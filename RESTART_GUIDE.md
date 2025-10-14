# RFID Access Control System - Restart Guide

## 🔄 System Restart Options

The RFID Access Control System provides multiple ways to restart the application, depending on your setup and needs.

## 🌐 Web Interface Reset (Recommended)

### How to Use
1. **Login** to the web interface at `http://your-ip:5001`
2. **Navigate** to Dashboard or Configuration tab
3. **Click** the red "Reset System" button
4. **Confirm** the reset action
5. **Wait** for the countdown and automatic restart

### What Happens
- System gracefully stops current processes
- Starts new instance automatically
- No manual intervention required
- Works with or without systemd service

## 🖥️ Command Line Restart Options

### Option 1: Python Restart Script (Standalone)
```bash
# Restart using Python script (no service required)
python3 restart_rfid.py

# Or using the management script
./start_rfid_system.sh restart-python
```

### Option 2: Systemd Service (If Installed)
```bash
# Restart systemd service
sudo systemctl restart rfid-access-control

# Or using the management script
./start_rfid_system.sh restart
```

### Option 3: Manual Process Management
```bash
# Find and stop RFID processes
pkill -f integrated_access_camera.py

# Start new instance
python3 integrated_access_camera.py
```

## 🛠️ Setup Options

### Standalone Mode (No Service)
- Run directly: `python3 integrated_access_camera.py`
- Use web interface reset button
- Use Python restart script: `python3 restart_rfid.py`

### Service Mode (Recommended for Production)
```bash
# Install as system service
./start_rfid_system.sh install

# Start service
./start_rfid_system.sh start

# Check status
./start_rfid_system.sh status
```

## 🔧 Troubleshooting

### Reset Button Not Working
1. **Check API Key**: Ensure API key is configured in Configuration tab
2. **Check Logs**: Look for error messages in the logs
3. **Manual Restart**: Use command line options as fallback

### Process Not Restarting
1. **Check Permissions**: Ensure script has execute permissions
2. **Check Dependencies**: Verify all Python packages are installed
3. **Check Port**: Ensure port 5001 is not blocked

### Service Not Starting
1. **Check Service Status**: `sudo systemctl status rfid-access-control`
2. **Check Logs**: `sudo journalctl -u rfid-access-control -f`
3. **Reinstall Service**: `./start_rfid_system.sh uninstall && ./start_rfid_system.sh install`

## 📋 File Structure

```
rfid-access-control/
├── integrated_access_camera.py    # Main application
├── restart_rfid.py               # Python restart script
├── start_rfid_system.sh          # Management script
├── rfid-access-control.service   # Systemd service file
└── RESTART_GUIDE.md             # This guide
```

## 🎯 Best Practices

### For Development
- Use standalone mode: `python3 integrated_access_camera.py`
- Use web interface reset button for quick restarts
- Use Python restart script for clean restarts

### For Production
- Install as systemd service for automatic startup
- Use web interface reset button for maintenance
- Monitor logs for any issues

### For Troubleshooting
- Always check logs first
- Try different restart methods
- Verify network connectivity
- Check system resources

## 🚨 Emergency Restart

If the system becomes completely unresponsive:

1. **SSH into the Pi**
2. **Kill all processes**: `pkill -f integrated_access_camera.py`
3. **Start fresh**: `python3 integrated_access_camera.py`
4. **Check web interface**: `http://your-ip:5001`

## 📞 Support

If you continue to have issues:
1. Check the system logs
2. Verify all dependencies are installed
3. Ensure proper file permissions
4. Check network connectivity
5. Verify API key configuration

The restart functionality is designed to be robust and work in various scenarios, but if you encounter persistent issues, the manual restart methods provide reliable fallback options.
