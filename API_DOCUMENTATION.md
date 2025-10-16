# MaxPark RFID Access Control System - API Documentation

## Overview
This document provides comprehensive API documentation for the MaxPark RFID Access Control System. The system provides both web interface routes and REST API endpoints for managing RFID access control, user management, camera operations, and system administration.

## Base URL
```
http://your-raspberry-pi-ip:5001
```

## Authentication Methods

### 1. Session Authentication (Web Interface)
- **Method**: Session-based authentication
- **Usage**: Web interface login/logout
- **Headers**: `Cookie: session=session_token`

### 2. API Key Authentication (REST API)
- **Method**: API Key in header
- **Header**: `X-API-Key: your_api_key`
- **Usage**: REST API endpoints marked with `@require_api_key`

### 3. Session Token Authentication (Internal)
- **Method**: Session token validation
- **Usage**: Internal web interface operations
- **Headers**: `X-Session-Token: session_token`

---

## Web Interface Routes

### 1. Home Page
- **URL**: `GET /`
- **Description**: Redirects to login page
- **Authentication**: None
- **Response**: Redirect to `/login`

### 2. Login Page
- **URL**: `GET /login`
- **Description**: Display login form
- **Authentication**: None
- **Response**: HTML login page

### 3. Login Authentication
- **URL**: `POST /login`
- **Description**: Authenticate user credentials
- **Authentication**: None
- **Request Body**:
  ```json
  {
    "username": "admin",
    "password": "password"
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Login successful"
  }
  ```

### 4. Logout
- **URL**: `POST /logout`
- **Description**: Logout current user
- **Authentication**: Session required
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Logged out successfully"
  }
  ```

### 5. Dashboard
- **URL**: `GET /dashboard`
- **Description**: Main system dashboard
- **Authentication**: Session required
- **Response**: HTML dashboard page

### 6. Change Password
- **URL**: `POST /change_password`
- **Description**: Change user password
- **Authentication**: Session required
- **Request Body**:
  ```json
  {
    "current_password": "old_password",
    "new_password": "new_password"
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Password changed successfully"
  }
  ```

---

## System Status & Health APIs

### 7. System Status
- **URL**: `GET /status`
- **Description**: Get overall system status
- **Authentication**: None
- **Response**:
  ```json
  {
    "status": "running",
    "timestamp": "2024-01-01T12:00:00Z",
    "uptime": "2 days, 5 hours",
    "active_readers": 2,
    "total_transactions": 1500
  }
  ```

### 8. Health Check
- **URL**: `GET /health_check`
- **Description**: Comprehensive system health check
- **Authentication**: None
- **Response**:
  ```json
  {
    "internet": true,
    "firebase": true,
    "cameras": {
      "camera_1": {
        "status": "online",
        "enabled": true,
        "last_check": "2024-01-01T12:00:00Z"
      },
      "camera_2": {
        "status": "offline",
        "enabled": true,
        "last_check": "2024-01-01T12:00:00Z"
      }
    },
    "storage": {
      "total_gb": 32.0,
      "free_gb": 15.2,
      "used_gb": 16.8
    }
  }
  ```

---

## User Management APIs

### 9. Get All Users
- **URL**: `GET /get_users`
- **Description**: Retrieve all users from local storage
- **Authentication**: None
- **Response**:
  ```json
  {
    "users": [
      {
        "card_id": "1234567890",
        "name": "John Doe",
        "department": "IT",
        "blocked": false,
        "added_date": "2024-01-01T10:00:00Z"
      }
    ],
    "blocked_users": [
      {
        "card_id": "0987654321",
        "name": "Jane Smith",
        "department": "HR",
        "blocked": true,
        "blocked_date": "2024-01-01T11:00:00Z"
      }
    ]
  }
  ```

### 10. Add User
- **URL**: `GET /add_user`
- **Description**: Add new user to local storage
- **Authentication**: API Key required
- **Query Parameters**:
  - `card_id`: RFID card ID
  - `name`: User name
  - `department`: User department
- **Response**:
  ```json
  {
    "status": "success",
    "message": "User added successfully"
  }
  ```

### 11. Delete User
- **URL**: `GET /delete_user`
- **Description**: Delete user from local storage
- **Authentication**: API Key required
- **Query Parameters**:
  - `card_id`: RFID card ID to delete
- **Response**:
  ```json
  {
    "status": "success",
    "message": "User deleted successfully"
  }
  ```

### 12. Block User
- **URL**: `GET /block_user`
- **Description**: Block user access
- **Authentication**: API Key required
- **Query Parameters**:
  - `card_id`: RFID card ID to block
- **Response**:
  ```json
  {
    "status": "success",
    "message": "User blocked successfully"
  }
  ```

### 13. Unblock User
- **URL**: `GET /unblock_user`
- **Description**: Unblock user access
- **Authentication**: API Key required
- **Query Parameters**:
  - `card_id`: RFID card ID to unblock
- **Response**:
  ```json
  {
    "status": "success",
    "message": "User unblocked successfully"
  }
  ```

### 14. Search User
- **URL**: `GET /search_user`
- **Description**: Search for user by card ID
- **Authentication**: None
- **Query Parameters**:
  - `card_id`: RFID card ID to search
- **Response**:
  ```json
  {
    "status": "success",
    "user": {
      "card_id": "1234567890",
      "name": "John Doe",
      "department": "IT",
      "blocked": false,
      "added_date": "2024-01-01T10:00:00Z"
    }
  }
  ```

---

## Transaction & Image APIs

### 15. Get Transactions
- **URL**: `GET /get_transactions`
- **Description**: Retrieve recent transactions
- **Authentication**: None
- **Query Parameters**:
  - `limit`: Number of transactions to return (default: 100)
- **Response**:
  ```json
  {
    "transactions": [
      {
        "timestamp": "2024-01-01T12:00:00Z",
        "card_id": "1234567890",
        "reader_id": 1,
        "access_granted": true,
        "user_name": "John Doe",
        "image_filename": "1234567890_r1_20240101120000.jpg"
      }
    ],
    "total": 1500
  }
  ```

### 16. Get Images
- **URL**: `GET /get_images`
- **Description**: Retrieve recent captured images
- **Authentication**: None
- **Query Parameters**:
  - `limit`: Number of images to return (default: 100)
- **Response**:
  ```json
  {
    "images": [
      {
        "filename": "1234567890_r1_20240101120000.jpg",
        "timestamp": "2024-01-01T12:00:00Z",
        "card_id": "1234567890",
        "reader_id": 1,
        "uploaded": true,
        "upload_timestamp": "2024-01-01T12:01:00Z"
      }
    ],
    "total": 500
  }
  ```

### 17. Serve Image
- **URL**: `GET /serve_image/<filename>`
- **Description**: Serve image file
- **Authentication**: None
- **Response**: Binary image data (JPEG)

### 18. Delete Image
- **URL**: `DELETE /delete_image/<filename>`
- **Description**: Delete specific image file
- **Authentication**: API Key required
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Image deleted successfully"
  }
  ```

---

## Offline Images Management

### 19. Get Offline Images
- **URL**: `GET /get_offline_images`
- **Description**: Retrieve all locally stored images with metadata
- **Authentication**: None
- **Query Parameters**:
  - `limit`: Number of images to return (default: 50)
  - `offset`: Offset for pagination (default: 0)
  - `entry_type`: Filter by entry type ("in", "out", "all")
  - `upload_status`: Filter by upload status ("uploaded", "pending", "all")
  - `date_filter`: Filter by date (YYYY-MM-DD format)
  - `card_filter`: Filter by card ID
- **Response**:
  ```json
  {
    "images": [
      {
        "filename": "1234567890_r1_20240101120000.jpg",
        "card_id": "1234567890",
        "reader_id": 1,
        "entry_type": "in",
        "timestamp": "2024-01-01T12:00:00Z",
        "uploaded": true,
        "upload_timestamp": "2024-01-01T12:01:00Z",
        "file_size": 245760
      }
    ],
    "total": 1000,
    "stats": {
      "total": 1000,
      "in": 600,
      "out": 400,
      "pending_uploads": 50
    }
  }
  ```

### 20. Clear All Offline Images
- **URL**: `POST /clear_all_offline_images`
- **Description**: Delete all locally stored images
- **Authentication**: API Key required
- **Response**:
  ```json
  {
    "status": "success",
    "message": "All offline images cleared successfully",
    "deleted_count": 1000
  }
  ```

---

## Storage Management APIs

### 21. Get Storage Info
- **URL**: `GET /get_storage_info`
- **Description**: Get current storage usage and limits
- **Authentication**: None
- **Response**:
  ```json
  {
    "image_usage_gb": 5.2,
    "max_storage_gb": 18.0,
    "cleanup_threshold_gb": 5.4,
    "disk_info": {
      "total_gb": 32.0,
      "free_gb": 15.2,
      "used_gb": 16.8
    },
    "allocation_percent": 60,
    "cleanup_percent": 30
  }
  ```

### 22. Trigger Storage Cleanup
- **URL**: `POST /trigger_storage_cleanup`
- **Description**: Manually trigger storage cleanup
- **Authentication**: API Key required
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Storage cleanup completed",
    "deleted_count": 150,
    "freed_space_gb": 2.1
  }
  ```

### 23. Get Storage Stats
- **URL**: `GET /get_storage_stats`
- **Description**: Get storage statistics and daily access data
- **Authentication**: None
- **Response**:
  ```json
  {
    "storage_usage": {
      "total_images": 1000,
      "total_size_gb": 5.2,
      "oldest_image": "2024-01-01T10:00:00Z",
      "newest_image": "2024-01-01T12:00:00Z"
    },
    "daily_stats": [
      {
        "date": "2024-01-01",
        "valid_entries": 45,
        "invalid_entries": 5,
        "total_entries": 50
      }
    ]
  }
  ```

### 24. Cleanup Old Images
- **URL**: `POST /cleanup_old_images`
- **Description**: Clean up old images based on storage limits
- **Authentication**: Session required
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Cleanup completed",
    "deleted_count": 100
  }
  ```

---

## Configuration APIs

### 25. Get Configuration
- **URL**: `GET /get_config`
- **Description**: Get current system configuration
- **Authentication**: None
- **Response**:
  ```json
  {
    "cameras": {
      "camera_1": {
        "enabled": true,
        "rtsp_url": "rtsp://192.168.1.100:554/stream1"
      },
      "camera_2": {
        "enabled": true,
        "rtsp_url": "rtsp://192.168.1.101:554/stream1"
      }
    },
    "s3": {
      "endpoint": "https://s3.amazonaws.com",
      "bucket": "my-bucket",
      "access_key": "***",
      "secret_key": "***"
    },
    "system": {
      "scan_rate_limit": 2
    }
  }
  ```

### 26. Update Configuration
- **URL**: `POST /update_config`
- **Description**: Update system configuration
- **Authentication**: API Key required
- **Request Body**:
  ```json
  {
    "config_type": "system",
    "scan_rate_limit": 3
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Configuration updated successfully"
  }
  ```

---

## Firebase Sync APIs

### 27. Sync Transactions
- **URL**: `POST /sync_transactions`
- **Description**: Manually sync cached transactions to Firebase
- **Authentication**: API Key required
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Transaction sync completed",
    "synced_count": 25,
    "failed_count": 0
  }
  ```

### 28. Transaction Cache Status
- **URL**: `GET /transaction_cache_status`
- **Description**: Get status of cached transactions
- **Authentication**: None
- **Response**:
  ```json
  {
    "cached_count": 25,
    "last_sync": "2024-01-01T11:30:00Z",
    "sync_in_progress": false,
    "last_error": null
  }
  ```

---

## Hardware Control APIs

### 29. Relay Control
- **URL**: `GET /relay`
- **Description**: Control relay operations
- **Authentication**: API Key required
- **Query Parameters**:
  - `action`: Relay action ("open_hold", "close_hold", "normal", "test_pulse")
  - `relay`: Relay number (1 or 2)
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Relay 1 opened and held for 3 seconds"
  }
  ```

---

## System Administration APIs

### 30. System Reset
- **URL**: `POST /system_reset`
- **Description**: Restart the entire application
- **Authentication**: API Key required
- **Response**:
  ```json
  {
    "status": "success",
    "message": "System restart initiated. The application will restart in a few seconds."
  }
  ```

### 31. Cleanup Old Stats
- **URL**: `POST /cleanup_old_stats`
- **Description**: Clean up old daily statistics
- **Authentication**: Session required
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Old statistics cleaned up"
  }
  ```

### 32. Clear All Stats
- **URL**: `POST /clear_all_stats`
- **Description**: Clear all daily statistics
- **Authentication**: Session required
- **Response**:
  ```json
  {
    "status": "success",
    "message": "All statistics cleared"
  }
  ```

---

## Static File Serving

### 33. Serve Static Files
- **URL**: `GET /static/<filename>`
- **Description**: Serve static files (company images, etc.)
- **Authentication**: None
- **Response**: Binary file data
- **Supported Files**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.svg`

---

## Error Responses

All API endpoints return consistent error responses:

```json
{
  "status": "error",
  "message": "Error description",
  "error_code": "ERROR_CODE"
}
```

### Common HTTP Status Codes:
- `200`: Success
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `500`: Internal Server Error

---

## Rate Limiting

- **Scan Rate Limiting**: Configurable delay between RFID scans (default: 2 seconds)
- **API Rate Limiting**: No built-in rate limiting (can be added if needed)

---

## Environment Variables

Key environment variables used by the system:

```bash
# Storage Configuration
IMAGES_DIR=images
MAX_STORAGE_GB=20
CLEANUP_THRESHOLD_GB=10
STORAGE_CHECK_INTERVAL=300

# GPIO Configuration
D0_PIN_1=18
D1_PIN_1=23
D0_PIN_2=19
D1_PIN_2=24

# Camera Configuration
CAMERA_1_ENABLED=true
CAMERA_2_ENABLED=true

# S3 Configuration
S3_ENDPOINT=https://s3.amazonaws.com
S3_BUCKET=your-bucket
S3_ACCESS_KEY=your-access-key
S3_SECRET_KEY=your-secret-key

# Firebase Configuration
FIREBASE_CREDENTIALS_PATH=path/to/credentials.json
FIREBASE_PROJECT_ID=your-project-id

# System Configuration
API_KEY=your-secure-api-key
ADMIN_PASSWORD=your-admin-password
```

---

## Security Considerations

1. **API Key Protection**: Store API keys securely and rotate regularly
2. **Session Management**: Sessions expire after inactivity
3. **File Upload Security**: Only image files are allowed for static serving
4. **Path Traversal Protection**: All file operations are protected against directory traversal
5. **Input Validation**: All user inputs are validated and sanitized

---

## Usage Examples

### Python Example (using requests):
```python
import requests

# Set up session
session = requests.Session()
base_url = "http://192.168.1.100:5001"
api_key = "your-api-key"

# Login
login_response = session.post(f"{base_url}/login", json={
    "username": "admin",
    "password": "password"
})

# Get users
users_response = session.get(f"{base_url}/get_users")
users = users_response.json()

# Add user (requires API key)
add_user_response = requests.get(f"{base_url}/add_user", 
    params={"card_id": "1234567890", "name": "John Doe", "department": "IT"},
    headers={"X-API-Key": api_key}
)

# Control relay
relay_response = requests.get(f"{base_url}/relay",
    params={"action": "open_hold", "relay": 1},
    headers={"X-API-Key": api_key}
)
```

### JavaScript Example (using fetch):
```javascript
const baseUrl = 'http://192.168.1.100:5001';
const apiKey = 'your-api-key';

// Get system status
fetch(`${baseUrl}/status`)
    .then(response => response.json())
    .then(data => console.log(data));

// Add user
fetch(`${baseUrl}/add_user?card_id=1234567890&name=John Doe&department=IT`, {
    headers: {
        'X-API-Key': apiKey
    }
})
.then(response => response.json())
.then(data => console.log(data));

// Control relay
fetch(`${baseUrl}/relay?action=open_hold&relay=1`, {
    headers: {
        'X-API-Key': apiKey
    }
})
.then(response => response.json())
.then(data => console.log(data));
```

---

This documentation covers all 33 API endpoints available in the MaxPark RFID Access Control System. Each endpoint is designed to work seamlessly with the web interface while also providing programmatic access for integration with other systems.
