#!/bin/bash

# DS3232 RTC Setup Script for MaxPark RFID Access Control System
# This script configures the DS3232 RTC module for accurate timestamps

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}DS3232 RTC Setup for MaxPark RFID System${NC}"
echo "=============================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root (use sudo)${NC}"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install package
install_package() {
    if ! command_exists "$1"; then
        echo -e "${YELLOW}Installing $1...${NC}"
        apt-get update && apt-get install -y "$1"
    else
        echo -e "${GREEN}✓ $1 is already installed${NC}"
    fi
}

echo -e "${BLUE}Step 1: Installing required packages${NC}"
install_package "i2c-tools"
install_package "python3-smbus"

echo -e "${BLUE}Step 2: Enabling I2C interface${NC}"
# Enable I2C interface
if ! grep -q "dtparam=i2c_arm=on" /boot/config.txt; then
    echo "dtparam=i2c_arm=on" >> /boot/config.txt
    echo -e "${GREEN}✓ I2C interface enabled in /boot/config.txt${NC}"
else
    echo -e "${GREEN}✓ I2C interface already enabled${NC}"
fi

# Enable I2C kernel module
if ! grep -q "i2c-dev" /etc/modules; then
    echo "i2c-dev" >> /etc/modules
    echo -e "${GREEN}✓ i2c-dev module added to /etc/modules${NC}"
else
    echo -e "${GREEN}✓ i2c-dev module already enabled${NC}"
fi

echo -e "${BLUE}Step 3: Configuring DS3232 RTC module${NC}"
# Add DS3232 RTC to device tree
if ! grep -q "dtoverlay=i2c-rtc,ds3231" /boot/config.txt; then
    echo "dtoverlay=i2c-rtc,ds3231" >> /boot/config.txt
    echo -e "${GREEN}✓ DS3232 RTC overlay added to /boot/config.txt${NC}"
else
    echo -e "${GREEN}✓ DS3232 RTC overlay already configured${NC}"
fi

echo -e "${BLUE}Step 4: Creating systemd service for RTC sync${NC}"
# Create systemd service for RTC sync
cat > /etc/systemd/system/ds3232-rtc-sync.service << 'EOF'
[Unit]
Description=DS3232 RTC Time Sync
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/sbin/hwclock --hctosys
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

# Enable the service
systemctl enable ds3232-rtc-sync.service
echo -e "${GREEN}✓ DS3232 RTC sync service created and enabled${NC}"

echo -e "${BLUE}Step 5: Testing I2C communication${NC}"
# Test I2C communication
if command_exists i2cdetect; then
    echo -e "${YELLOW}Scanning I2C bus for DS3232...${NC}"
    i2cdetect -y 1 | grep -q "68"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ DS3232 RTC detected at address 0x68${NC}"
    else
        echo -e "${RED}✗ DS3232 RTC not detected at address 0x68${NC}"
        echo -e "${YELLOW}Please check your wiring and connections${NC}"
    fi
else
    echo -e "${RED}✗ i2cdetect not found${NC}"
fi

echo -e "${BLUE}Step 6: Setting up RTC time sync${NC}"
# Set system time from RTC
if command_exists hwclock; then
    echo -e "${YELLOW}Syncing system time from RTC...${NC}"
    hwclock --hctosys
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ System time synced from RTC${NC}"
    else
        echo -e "${RED}✗ Failed to sync system time from RTC${NC}"
    fi
else
    echo -e "${RED}✗ hwclock not found${NC}"
fi

echo -e "${BLUE}Step 7: Configuring environment variables${NC}"
# Create or update .env file with RTC configuration
ENV_FILE="/home/maxpark/.env"
if [ -f "$ENV_FILE" ]; then
    # Update existing .env file
    if ! grep -q "RTC_ENABLED" "$ENV_FILE"; then
        echo "" >> "$ENV_FILE"
        echo "# DS3232 RTC Configuration" >> "$ENV_FILE"
        echo "RTC_ENABLED=true" >> "$ENV_FILE"
        echo "RTC_I2C_BUS=1" >> "$ENV_FILE"
        echo "RTC_I2C_ADDRESS=0x68" >> "$ENV_FILE"
        echo -e "${GREEN}✓ RTC configuration added to $ENV_FILE${NC}"
    else
        echo -e "${GREEN}✓ RTC configuration already exists in $ENV_FILE${NC}"
    fi
else
    echo -e "${YELLOW}Creating $ENV_FILE with RTC configuration...${NC}"
    mkdir -p /home/maxpark
    cat > "$ENV_FILE" << 'EOF'
# DS3232 RTC Configuration
RTC_ENABLED=true
RTC_I2C_BUS=1
RTC_I2C_ADDRESS=0x68
EOF
    chown maxpark:maxpark "$ENV_FILE"
    echo -e "${GREEN}✓ RTC configuration created in $ENV_FILE${NC}"
fi

echo -e "${BLUE}Step 8: Testing RTC functionality${NC}"
# Test RTC functionality
if command_exists hwclock; then
    echo -e "${YELLOW}Current RTC time:${NC}"
    hwclock --show
    echo -e "${YELLOW}Current system time:${NC}"
    date
else
    echo -e "${RED}✗ hwclock not available for testing${NC}"
fi

echo ""
echo -e "${GREEN}DS3232 RTC Setup Complete!${NC}"
echo "================================"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Reboot your Raspberry Pi: ${BLUE}sudo reboot${NC}"
echo "2. After reboot, verify RTC is working: ${BLUE}sudo hwclock --show${NC}"
echo "3. Start the RFID system: ${BLUE}./start_rfid_system.sh start${NC}"
echo "4. Check RTC status via API: ${BLUE}curl http://localhost:5001/rtc_status${NC}"
echo ""
echo -e "${YELLOW}Troubleshooting:${NC}"
echo "- If RTC not detected: Check wiring (SDA to GPIO 2, SCL to GPIO 3, VCC to 3.3V, GND to GND)"
echo "- If time sync fails: Ensure DS3232 has battery backup"
echo "- Check logs: ${BLUE}sudo journalctl -u ds3232-rtc-sync${NC}"
echo ""
echo -e "${GREEN}Setup completed successfully!${NC}"
