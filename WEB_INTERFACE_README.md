# RFID Access Control System - Complete Web Interface

## Overview

The web interface provides a comprehensive management system for RFID access control, including user management, configuration, monitoring, and image management.

## Features

### 🎛️ **Three Main Tabs:**

#### 1. **📊 Dashboard Tab**
- **Statistics Cards**: Total scans, granted/denied counts, image counts
- **Recent Scans Display**: Real-time RFID transaction history
- **Image Gallery**: Browse captured images with upload status
- **Auto-refresh**: Updates every 5 seconds

#### 2. **👥 User Management Tab**
- **Add Users**: Complete form with card number, user ID, name, reference ID
- **Delete Users**: Remove users by card number
- **Block/Unblock Users**: Manage user access permissions
- **Search Users**: Find users by ID
- **User List**: View all users with quick action buttons

#### 3. **⚙️ Configuration Tab**
- **Camera Configuration**: Set RTSP URLs, IP addresses, credentials
- **S3 Configuration**: Configure upload endpoints and retry settings
- **System Configuration**: Server settings, API keys, ports
- **Real-time RTSP URL Generation**: Auto-updates when camera settings change

### 📸 **Image Management**
- **Display Limit**: 100 most recent images (bucket limit for viewing)
- **Unlimited Storage**: All images stored locally
- **Upload Status Tracking**:
  - ✅ **Uploaded**: Successfully uploaded to S3
  - ⏳ **Pending**: Waiting to be uploaded
  - ❌ **Failed**: Upload failed (will retry)
- **Interactive Features**:
  - Click images to view full-size in modal
  - Download images directly from modal
  - Delete images with confirmation
- **Image Details**: Card number, timestamp, upload status, S3 location

### 🔄 **Real-time Features**
- Auto-refresh every 5 seconds
- Manual refresh button (floating action button)
- Connection status indicator
- Offline/online detection
- Live notifications for all actions

## API Endpoints

### **Dashboard & Monitoring**
- `GET /` - Main web interface with tabs
- `GET /get_transactions` - Recent RFID transactions (10 latest)
- `GET /get_images` - Image list with upload status (100 limit for display)
- `GET /image/<filename>` - Serve image files
- `GET /status` - System status and health check

### **User Management**
- `GET /add_user` - Add new user (requires API key)
- `GET /delete_user` - Delete user (requires API key)
- `GET /block_user` - Block user access (requires API key)
- `GET /unblock_user` - Unblock user access (requires API key)
- `GET /search_user` - Search user by ID
- `GET /get_users` - Get all users list

### **Configuration Management**
- `GET /get_config` - Get current system configuration
- `POST /update_config` - Update configuration (requires API key)

### **Image Management**
- `DELETE /delete_image/<filename>` - Delete image (requires API key)

### **System Control**
- `GET /relay` - Control door relays (requires API key)

## Image Storage Details

### File Naming Convention
Images are saved with the format: `{CARD_NUMBER}_{TIMESTAMP}.jpg`

Example: `123456789_1703123456.jpg`

### Upload Status Tracking
- Each image has a corresponding `.uploaded.json` sidecar file
- Contains upload metadata:
  ```json
  {
    "uploaded_at": 1703123456,
    "s3_location": "https://s3.amazonaws.com/bucket/path/image.jpg"
  }
  ```

### Storage Limits
- **Display Limit**: 100 images shown in interface
- **Storage Limit**: No limit - all images are stored locally
- **Auto-cleanup**: Manual deletion via API (future enhancement)

## Keyboard Shortcuts

- **Ctrl+R** or **F5**: Refresh data
- **Ctrl+A**: Toggle auto-refresh
- **Escape**: Close modals

## Browser Compatibility

- Modern browsers with ES6+ support
- Bootstrap 5.1.3
- Font Awesome 6.0.0
- Responsive design for mobile devices

## Security Features

- File type validation (jpg/jpeg only)
- Directory traversal protection
- API key authentication for sensitive operations
- Input sanitization

## Usage

1. **Access the Interface**: Navigate to `http://your-server:5001/`
2. **Monitor Scans**: View real-time RFID access attempts
3. **Browse Images**: Click on images to view full-size
4. **Check Status**: Monitor upload status of captured images
5. **Download Images**: Click on images in modal to download

## Configuration

The interface uses the same configuration as the main application:
- `IMAGES_DIR`: Directory for storing images (default: "images")
- `API_KEY`: Authentication key for sensitive operations
- `FLASK_HOST`: Server host (default: 0.0.0.0)
- `FLASK_PORT`: Server port (default: 5001)

## Troubleshooting

### Images Not Loading
- Check if `IMAGES_DIR` exists and is readable
- Verify image files are in correct format (jpg/jpeg)
- Check file permissions

### Upload Status Issues
- Verify `.uploaded.json` sidecar files exist
- Check S3 API configuration
- Review application logs for upload errors

### Performance Issues
- Reduce display limit if needed
- Check available disk space
- Monitor memory usage with large image collections
