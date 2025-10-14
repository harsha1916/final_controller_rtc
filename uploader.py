import os
import time
import requests
import logging
from typing import Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import S3_API_URL, MAX_RETRIES, RETRY_DELAY

class ImageUploader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Create optimized session with connection pooling
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["POST"]
        )
        
        # Configure adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,  # Number of connection pools
            pool_maxsize=20,      # Maximum number of connections in pool
            pool_block=False      # Don't block when pool is full
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set optimized headers
        self.session.headers.update({
            'User-Agent': 'MaxPark-RFID-System/1.0',
            'Connection': 'keep-alive',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        })

    def upload(self, filepath: str) -> Optional[str]:
        """Upload image file to S3-compatible API with optimized settings."""
        if not os.path.exists(filepath):
            self.logger.error(f"File does not exist: {filepath}")
            return None
            
        if not os.path.isfile(filepath):
            self.logger.error(f"Path is not a file: {filepath}")
            return None
            
        # Check file size (limit to 15MB - increased for better quality images)
        file_size = os.path.getsize(filepath)
        if file_size > 15 * 1024 * 1024:  # 15MB
            self.logger.error(f"File too large: {filepath} ({file_size} bytes)")
            return None
            
        try:
            with open(filepath, "rb") as image_file:
                files = {
                    "singleFile": (os.path.basename(filepath), image_file, "image/jpeg")
                }
                # Use optimized session with connection pooling
                response = self.session.post(
                    S3_API_URL, 
                    files=files, 
                    timeout=45,  # Increased from 30 to 45 seconds
                    stream=False  # Disable streaming for better performance
                )

                if response.status_code == 200:
                    self.logger.info(f"Successfully uploaded: {filepath}")
                    try:
                        response_json = response.json()
                        location = response_json.get("Location")
                        if location:
                            self.logger.debug(f"S3 Response: {response_json}")
                            # Don't remove file - keep for gallery display
                            return location
                        else:
                            self.logger.error(f"No Location in response: {response_json}")
                    except ValueError as e:
                        self.logger.error(f"Invalid JSON response: {e}")
                        self.logger.error(f"Response content: {response.text}")
                elif response.status_code == 413:
                    self.logger.error(f"File too large for upload: {filepath}")
                    return None  # Don't retry for file size errors
                else:
                    self.logger.error(f"Upload failed {filepath}: {response.status_code} - {response.text}")
                    return None  # Let the retry strategy handle retries

        except requests.exceptions.Timeout:
            self.logger.warning(f"Upload timeout for {filepath}")
            return None
        except requests.exceptions.ConnectionError:
            self.logger.warning(f"Connection error for {filepath}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Upload error for {filepath}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during upload of {filepath}: {e}")
            return None
