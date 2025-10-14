#!/usr/bin/env python3
"""
Upload Performance Testing Script for MaxPark RFID System
Tests the optimized upload system performance.
"""

import os
import time
import logging
import requests
from uploader import ImageUploader
from concurrent.futures import ThreadPoolExecutor
import tempfile
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_image(filename: str, size_mb: float = 1.0):
    """Create a test image file of specified size in MB."""
    # Create a simple test image
    width, height = 1920, 1080
    color = (100, 150, 200)
    
    # Calculate iterations needed for target size
    target_bytes = size_mb * 1024 * 1024
    # Rough estimate: each pixel ~3 bytes for RGB
    pixels_needed = target_bytes // 3
    
    # Adjust dimensions to get close to target size
    if pixels_needed > width * height:
        width = int((pixels_needed / height) ** 0.5)
        height = int(pixels_needed / width)
    
    # Create image
    image = Image.new('RGB', (width, height), color)
    
    # Add some variation to make it more realistic
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    for i in range(0, width, 100):
        for j in range(0, height, 100):
            draw.rectangle([i, j, i+50, j+50], fill=(i % 255, j % 255, (i+j) % 255))
    
    image.save(filename, 'JPEG', quality=85)
    
    actual_size = os.path.getsize(filename) / (1024 * 1024)
    logger.info(f"Created test image: {filename} ({actual_size:.2f} MB)")
    return actual_size

def test_single_upload(uploader: ImageUploader, image_path: str) -> dict:
    """Test single image upload and return performance metrics."""
    start_time = time.time()
    
    try:
        result = uploader.upload(image_path)
        end_time = time.time()
        
        if result:
            return {
                'success': True,
                'duration': end_time - start_time,
                'size_mb': os.path.getsize(image_path) / (1024 * 1024),
                'speed_mbps': (os.path.getsize(image_path) / (1024 * 1024)) / (end_time - start_time)
            }
        else:
            return {
                'success': False,
                'duration': end_time - start_time,
                'error': 'Upload failed'
            }
    except Exception as e:
        end_time = time.time()
        return {
            'success': False,
            'duration': end_time - start_time,
            'error': str(e)
        }

def test_concurrent_uploads(num_images: int = 5, num_workers: int = 3, image_size_mb: float = 1.0):
    """Test concurrent uploads with multiple workers."""
    logger.info(f"Testing {num_images} concurrent uploads with {num_workers} workers")
    
    # Create test images
    test_images = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        for i in range(num_images):
            image_path = os.path.join(temp_dir, f"test_image_{i}.jpg")
            create_test_image(image_path, image_size_mb)
            test_images.append(image_path)
        
        # Test concurrent uploads
        uploader = ImageUploader()
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(test_single_upload, uploader, img) for img in test_images]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Analyze results
        successful_uploads = [r for r in results if r['success']]
        failed_uploads = [r for r in results if not r['success']]
        
        if successful_uploads:
            avg_upload_time = sum(r['duration'] for r in successful_uploads) / len(successful_uploads)
            avg_speed = sum(r['speed_mbps'] for r in successful_uploads) / len(successful_uploads)
            total_data_mb = sum(r['size_mb'] for r in successful_uploads)
            overall_speed = total_data_mb / total_duration
        else:
            avg_upload_time = 0
            avg_speed = 0
            total_data_mb = 0
            overall_speed = 0
        
        logger.info("=" * 60)
        logger.info("UPLOAD PERFORMANCE TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Images: {num_images}")
        logger.info(f"Successful Uploads: {len(successful_uploads)}")
        logger.info(f"Failed Uploads: {len(failed_uploads)}")
        logger.info(f"Total Duration: {total_duration:.2f} seconds")
        logger.info(f"Average Upload Time: {avg_upload_time:.2f} seconds")
        logger.info(f"Average Speed per Upload: {avg_speed:.2f} MB/s")
        logger.info(f"Overall Throughput: {overall_speed:.2f} MB/s")
        logger.info(f"Total Data Transferred: {total_data_mb:.2f} MB")
        
        if failed_uploads:
            logger.warning("Failed uploads:")
            for i, result in enumerate(failed_uploads):
                logger.warning(f"  Image {i}: {result.get('error', 'Unknown error')}")
        
        return {
            'total_images': num_images,
            'successful': len(successful_uploads),
            'failed': len(failed_uploads),
            'total_duration': total_duration,
            'avg_upload_time': avg_upload_time,
            'avg_speed': avg_speed,
            'overall_speed': overall_speed,
            'total_data_mb': total_data_mb
        }
        
    finally:
        # Cleanup test images
        for image_path in test_images:
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass

def test_upload_queue_performance():
    """Test the upload queue system performance."""
    logger.info("Testing upload queue system performance...")
    
    # This would require the full system to be running
    # For now, we'll simulate the queue behavior
    from queue import Queue
    import threading
    
    image_queue = Queue()
    results = []
    
    def worker():
        uploader = ImageUploader()
        while True:
            try:
                image_path = image_queue.get(timeout=1)
                result = test_single_upload(uploader, image_path)
                results.append(result)
                image_queue.task_done()
            except:
                break
    
    # Start worker threads
    num_workers = 5
    threads = []
    for _ in range(num_workers):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)
    
    # Add test images to queue
    temp_dir = tempfile.mkdtemp()
    test_images = []
    
    try:
        for i in range(10):
            image_path = os.path.join(temp_dir, f"queue_test_{i}.jpg")
            create_test_image(image_path, 0.5)  # Smaller images for queue test
            test_images.append(image_path)
            image_queue.put(image_path)
        
        # Wait for all uploads to complete
        image_queue.join()
        
        # Stop worker threads
        for _ in range(num_workers):
            image_queue.put(None)
        
        for t in threads:
            t.join(timeout=5)
        
        successful = len([r for r in results if r['success']])
        total_time = sum(r['duration'] for r in results)
        
        logger.info(f"Queue Test Results: {successful}/{len(results)} successful in {total_time:.2f}s")
        
    finally:
        # Cleanup
        for image_path in test_images:
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass

if __name__ == "__main__":
    logger.info("MaxPark RFID Upload Performance Test")
    logger.info("=" * 50)
    
    # Test 1: Single upload
    logger.info("Test 1: Single Image Upload")
    temp_dir = tempfile.mkdtemp()
    test_image = os.path.join(temp_dir, "single_test.jpg")
    
    try:
        create_test_image(test_image, 2.0)
        uploader = ImageUploader()
        result = test_single_upload(uploader, test_image)
        
        if result['success']:
            logger.info(f"Single upload: {result['duration']:.2f}s, {result['speed_mbps']:.2f} MB/s")
        else:
            logger.error(f"Single upload failed: {result.get('error', 'Unknown error')}")
    finally:
        if os.path.exists(test_image):
            os.remove(test_image)
        os.rmdir(temp_dir)
    
    # Test 2: Concurrent uploads
    logger.info("\nTest 2: Concurrent Uploads")
    test_concurrent_uploads(num_images=5, num_workers=3, image_size_mb=1.0)
    
    # Test 3: Queue system
    logger.info("\nTest 3: Upload Queue System")
    test_upload_queue_performance()
    
    logger.info("\nPerformance testing completed!")
