# Password Management Guide

## 🔐 **Default Login Credentials**

### **Initial Access:**
- **Username**: `admin`
- **Password**: `admin123`

## 🚨 **Important Security Notes**

### **First Login:**
1. **Change Default Password**: Immediately change the default password after first login
2. **Use Strong Password**: Choose a password with at least 8 characters, numbers, and symbols
3. **Keep It Secure**: Don't share the password with unauthorized users

## 🔧 **Password Management Features**

### **1. Change Password (Normal)**
- Go to **Configuration** → **Change Password** tab
- Enter current password
- Enter new password (minimum 6 characters)
- Confirm new password
- Click **Change Password**

### **2. Emergency Password Reset**
- Go to **Configuration** → **Change Password** tab
- Scroll down to **Emergency Password Reset** section
- Enter new password (or leave empty for default: `admin123`)
- Click **Reset Password**
- **Warning**: This bypasses current password verification

## 🆘 **If You're Locked Out**

### **Method 1: Emergency Reset (If You Have API Access)**
```bash
# If you have the API key, you can reset via API
curl -X POST http://[device-ip]:5001/reset_password \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"new_password": "your-new-password"}'
```

### **Method 2: Physical Access (Raspberry Pi)**
```bash
# Connect monitor and keyboard to Raspberry Pi
# Login with: pi / raspberry (default Pi user)

# Edit the .env file
sudo nano /home/maxpark/.env

# Find the line: ADMIN_PASSWORD_HASH=...
# Generate new hash for your password:
python3 -c "import hashlib; print(hashlib.sha256('your-new-password'.encode()).hexdigest())"

# Replace the hash in the .env file
# Save and restart the service
sudo systemctl restart rfid-access-control
```

### **Method 3: SD Card Access**
1. **Remove SD card** from Raspberry Pi
2. **Insert into computer** with SD card reader
3. **Navigate to**: `/home/maxpark/.env`
4. **Edit the file** and update `ADMIN_PASSWORD_HASH`
5. **Reinsert SD card** and boot

## 🔑 **Generating Password Hashes**

### **Using Python:**
```python
import hashlib
password = "your-password"
hash_value = hashlib.sha256(password.encode()).hexdigest()
print(hash_value)
```

### **Using Command Line:**
```bash
# On Raspberry Pi or Linux
echo -n "your-password" | sha256sum
```

## 📋 **Password Best Practices**

### **Strong Password Requirements:**
- ✅ **Minimum 8 characters**
- ✅ **Mix of uppercase and lowercase**
- ✅ **Include numbers**
- ✅ **Include special characters**
- ✅ **Avoid common words**
- ✅ **Don't reuse passwords**

### **Examples of Strong Passwords:**
- `MaxPark2024!`
- `RFID@Secure#123`
- `Admin$2024&System`

### **Examples of Weak Passwords:**
- ❌ `admin123` (default)
- ❌ `password`
- ❌ `123456`
- ❌ `admin`

## 🔒 **Security Features**

### **Session Management:**
- **Session Duration**: 24 hours
- **Auto-logout**: After 24 hours of inactivity
- **Token-based**: Secure session tokens
- **API Protection**: All sensitive operations require API key

### **Password Security:**
- **SHA-256 Hashing**: Passwords are hashed, not stored in plain text
- **Environment Variables**: Password hash stored in `.env` file
- **API Authentication**: Password operations require API key
- **Logging**: Password changes are logged for security

## 🛠️ **Troubleshooting**

### **Issue: "Invalid username or password"**
- **Check**: Username is exactly `admin`
- **Check**: Password is correct (case-sensitive)
- **Try**: Default password `admin123`
- **Solution**: Use emergency reset if needed

### **Issue: "Session expired"**
- **Cause**: 24-hour session limit reached
- **Solution**: Login again with credentials

### **Issue: "API key required"**
- **Cause**: Trying to access protected endpoint
- **Solution**: Use API key in request headers

### **Issue: "Password change failed"**
- **Check**: Current password is correct
- **Check**: New password meets requirements (6+ characters)
- **Check**: New passwords match
- **Solution**: Try emergency reset if locked out

## 📞 **Emergency Recovery**

### **Complete Password Reset:**
```bash
# On Raspberry Pi (physical access)
sudo systemctl stop rfid-access-control
sudo nano /home/maxpark/.env

# Replace ADMIN_PASSWORD_HASH with:
ADMIN_PASSWORD_HASH=ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f

# This resets to default password: admin123
sudo systemctl start rfid-access-control
```

### **Reset to Factory Defaults:**
```bash
# Complete system reset (use with caution)
sudo systemctl stop rfid-access-control
sudo rm /home/maxpark/.env
sudo systemctl start rfid-access-control
# System will use default credentials: admin / admin123
```

## ✅ **Quick Reference**

| Action | Username | Password | Notes |
|--------|----------|----------|-------|
| **First Login** | `admin` | `admin123` | Change immediately |
| **After Reset** | `admin` | `admin123` | Default after reset |
| **Custom** | `admin` | `your-password` | After change |

### **Important URLs:**
- **Login**: `http://[device-ip]:5001/login`
- **Dashboard**: `http://[device-ip]:5001/`
- **API**: `http://[device-ip]:5001/[endpoint]`

---

**Remember**: Always change the default password after first login and keep your credentials secure! 🔐
