#!/bin/bash

# RFID Access Control System Startup Script
# This script helps manage the RFID system with automatic restart capability

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="rfid-access-control"
PYTHON_SCRIPT="integrated_access_camera.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}RFID Access Control System Manager${NC}"
echo "=================================="

# Function to check if service is running
check_service() {
    if systemctl is-active --quiet $SERVICE_NAME; then
        echo -e "${GREEN}✓ Service is running${NC}"
        return 0
    else
        echo -e "${RED}✗ Service is not running${NC}"
        return 1
    fi
}

# Function to start service
start_service() {
    echo -e "${YELLOW}Starting RFID Access Control System...${NC}"
    sudo systemctl start $SERVICE_NAME
    sleep 2
    check_service
}

# Function to stop service
stop_service() {
    echo -e "${YELLOW}Stopping RFID Access Control System...${NC}"
    sudo systemctl stop $SERVICE_NAME
    sleep 2
    check_service
}

# Function to restart service
restart_service() {
    echo -e "${YELLOW}Restarting RFID Access Control System...${NC}"
    sudo systemctl restart $SERVICE_NAME
    sleep 2
    check_service
}

# Function to show service status
show_status() {
    echo -e "${BLUE}Service Status:${NC}"
    sudo systemctl status $SERVICE_NAME --no-pager
    echo ""
    echo -e "${BLUE}Recent Logs:${NC}"
    sudo journalctl -u $SERVICE_NAME -n 20 --no-pager
}

# Function to install service
install_service() {
    echo -e "${YELLOW}Installing systemd service...${NC}"
    
    # Copy service file
    sudo cp $SCRIPT_DIR/$SERVICE_NAME.service /etc/systemd/system/
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    # Enable service
    sudo systemctl enable $SERVICE_NAME
    
    echo -e "${GREEN}✓ Service installed and enabled${NC}"
    echo -e "${BLUE}Use 'sudo systemctl start $SERVICE_NAME' to start the service${NC}"
}

# Function to uninstall service
uninstall_service() {
    echo -e "${YELLOW}Uninstalling systemd service...${NC}"
    
    # Stop and disable service
    sudo systemctl stop $SERVICE_NAME
    sudo systemctl disable $SERVICE_NAME
    
    # Remove service file
    sudo rm /etc/systemd/system/$SERVICE_NAME.service
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    echo -e "${GREEN}✓ Service uninstalled${NC}"
}

# Function to run directly (for development)
run_direct() {
    echo -e "${YELLOW}Running RFID system directly (development mode)...${NC}"
    cd $SCRIPT_DIR
    python3 $PYTHON_SCRIPT
}

# Function to restart using Python script
restart_python() {
    echo -e "${YELLOW}Restarting RFID system using Python script...${NC}"
    cd $SCRIPT_DIR
    python3 restart_rfid.py
}

# Main menu
case "$1" in
    "start")
        start_service
        ;;
    "stop")
        stop_service
        ;;
    "restart")
        restart_service
        ;;
    "status")
        show_status
        ;;
    "install")
        install_service
        ;;
    "uninstall")
        uninstall_service
        ;;
    "run")
        run_direct
        ;;
    "restart-python")
        restart_python
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|install|uninstall|run|restart-python}"
        echo ""
        echo "Commands:"
        echo "  start          - Start the RFID system service"
        echo "  stop           - Stop the RFID system service"
        echo "  restart        - Restart the RFID system service"
        echo "  restart-python - Restart using Python script (no service required)"
        echo "  status         - Show service status and recent logs"
        echo "  install        - Install systemd service for auto-start"
        echo "  uninstall      - Remove systemd service"
        echo "  run            - Run directly (development mode)"
        echo ""
        echo "Examples:"
        echo "  $0 install         # Install as system service"
        echo "  $0 start           # Start the service"
        echo "  $0 restart-python  # Restart without service (standalone)"
        echo "  $0 status          # Check status"
        echo "  $0 run             # Run directly for testing"
        ;;
esac
