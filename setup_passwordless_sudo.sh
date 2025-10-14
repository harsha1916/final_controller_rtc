#!/bin/bash
# Setup script to enable passwordless sudo for network configuration
# Run this script once to avoid password prompts during network changes

echo "Setting up passwordless sudo for MaxPark RFID System network configuration..."

# Create a sudoers file for the maxpark user (or pi user)
SUDOERS_FILE="/etc/sudoers.d/maxpark-network"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root (use sudo)"
    exit 1
fi

# Create sudoers file for network configuration commands
cat > "$SUDOERS_FILE" << 'EOF'
# MaxPark RFID System - Passwordless sudo for network configuration
# This allows the RFID system to modify network settings without password prompts

# Allow network configuration commands
%maxpark ALL=(ALL) NOPASSWD: /bin/cp /etc/dhcpcd.conf*
%maxpark ALL=(ALL) NOPASSWD: /bin/sed -i * /etc/dhcpcd.conf
%maxpark ALL=(ALL) NOPASSWD: /bin/tee -a /etc/dhcpcd.conf
%maxpark ALL=(ALL) NOPASSWD: /bin/systemctl restart dhcpcd
%maxpark ALL=(ALL) NOPASSWD: /bin/systemctl restart networking
%maxpark ALL=(ALL) NOPASSWD: /bin/systemctl restart rfid-access-control
%maxpark ALL=(ALL) NOPASSWD: /bin/systemctl status dhcpcd
%maxpark ALL=(ALL) NOPASSWD: /bin/systemctl status networking
%maxpark ALL=(ALL) NOPASSWD: /bin/systemctl status rfid-access-control

# Also allow for pi user (common Raspberry Pi default)
%pi ALL=(ALL) NOPASSWD: /bin/cp /etc/dhcpcd.conf*
%pi ALL=(ALL) NOPASSWD: /bin/sed -i * /etc/dhcpcd.conf
%pi ALL=(ALL) NOPASSWD: /bin/tee -a /etc/dhcpcd.conf
%pi ALL=(ALL) NOPASSWD: /bin/systemctl restart dhcpcd
%pi ALL=(ALL) NOPASSWD: /bin/systemctl restart networking
%pi ALL=(ALL) NOPASSWD: /bin/systemctl restart rfid-access-control
%pi ALL=(ALL) NOPASSWD: /bin/systemctl status dhcpcd
%pi ALL=(ALL) NOPASSWD: /bin/systemctl status networking
%pi ALL=(ALL) NOPASSWD: /bin/systemctl status rfid-access-control
EOF

# Set proper permissions
chmod 440 "$SUDOERS_FILE"

echo "Passwordless sudo configuration created at: $SUDOERS_FILE"
echo ""
echo "This allows the following users to run network configuration commands without password:"
echo "- Users in 'maxpark' group"
echo "- Users in 'pi' group"
echo ""
echo "Commands allowed without password:"
echo "- Copy dhcpcd.conf files"
echo "- Edit dhcpcd.conf files"
echo "- Restart network services"
echo "- Restart RFID system"
echo ""
echo "To test, run: sudo -l"
echo "To remove this configuration, run: sudo rm $SUDOERS_FILE"
echo ""
echo "Setup complete! Network configuration should now work without password prompts."
