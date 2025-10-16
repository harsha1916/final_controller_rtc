import cv2
import time
import datetime
import os
import threading
import logging
import requests
from typing import Optional
from config import RTSP_CAMERAS, MAX_RETRIES, RETRY_DELAY
from uploader import ImageUploader

class CameraService:
    def __init__(self):
        self.uploader = ImageUploader()
        self.logger = logging.getLogger(__name__)
        
        # Ensure images directory exists
        os.makedirs("images", exist_ok=True)
    
    def check_internet_connection(self) -> bool:
        """Check if internet connection is available."""
        try:
            response = requests.get('https://www.google.com', timeout=5)
            return response.status_code == 200
        except:
            return False

    def _capture_image(self, camera_key: str) -> Optional[str]:
        """Capture image from specified camera and upload it."""
        if camera_key not in RTSP_CAMERAS:
            self.logger.error(f"Invalid camera key: {camera_key}")
            return None
            
        rtsp_url = RTSP_CAMERAS.get(camera_key)
        retries = 0

        while retries < MAX_RETRIES:
            cap = None
            try:
                cap = cv2.VideoCapture(rtsp_url)

                if not cap.isOpened():
                    self.logger.warning(f"{camera_key}: Camera not available. Retrying ({retries + 1}/{MAX_RETRIES})...")
                    time.sleep(RETRY_DELAY)
                    retries += 1
                    continue

                ret, frame = cap.read()

                if ret:
                    timestamp = int(time.time())
                    filename = f"{timestamp}_{camera_key}.jpg"
                    filepath = os.path.join("images", filename)
                    
                    if cv2.imwrite(filepath, frame):
                        self.logger.info(f"{camera_key}: Image captured -> {filename}")
                        
                        # Always return the local file path for now
                        # Upload will be handled by the web app or background service
                        self.logger.info(f"{camera_key}: Image saved locally, upload will be handled separately")
                        return f"local:{filepath}"
                    else:
                        self.logger.error(f"{camera_key}: Failed to save image to {filepath}")
                        retries += 1
                        continue

                self.logger.warning(f"{camera_key}: Failed to capture frame. Retrying...")
                retries += 1
                time.sleep(RETRY_DELAY)
                
            except Exception as e:
                self.logger.error(f"{camera_key}: Error during capture: {e}")
                retries += 1
                time.sleep(RETRY_DELAY)
            finally:
                if cap is not None:
                    cap.release()

        self.logger.error(f"{camera_key}: Max retries reached. Skipping.")
        return None

    def capture_camera_1(self) -> Optional[str]:
        """Capture image from camera 1."""
        return self._capture_image("camera_1")

    def capture_camera_2(self) -> Optional[str]:
        """Capture image from camera 2."""
        return self._capture_image("camera_2")
