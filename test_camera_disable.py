#!/usr/bin/env python3
"""
Test script to verify camera disable functionality works correctly.
This script tests that disabled cameras don't capture images when RFID cards are scanned.
"""

import os
import sys
import time
import logging
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_camera_enabled_status():
    """Test the camera enabled status function."""
    from integrated_access_camera import is_camera_enabled
    
    # Load environment variables
    load_dotenv()
    
    logger.info("Testing camera enabled status...")
    
    # Test each camera
    for reader_id in [1, 2, 3]:
        enabled = is_camera_enabled(reader_id)
        logger.info(f"Camera {reader_id} enabled: {enabled}")
        
        # Check the environment variable directly
        env_key = f"CAMERA_{reader_id}_ENABLED"
        env_value = os.getenv(env_key, "true")
        logger.info(f"  Environment variable {env_key}: {env_value}")
    
    return True

def test_camera_disable_enable():
    """Test disabling and enabling cameras."""
    logger.info("Testing camera disable/enable functionality...")
    
    # Test disabling camera 1
    logger.info("Disabling Camera 1...")
    os.environ["CAMERA_1_ENABLED"] = "false"
    
    from integrated_access_camera import is_camera_enabled
    camera_1_enabled = is_camera_enabled(1)
    logger.info(f"Camera 1 enabled after disable: {camera_1_enabled}")
    
    if camera_1_enabled:
        logger.error("❌ Camera 1 should be disabled but is still enabled!")
        return False
    else:
        logger.info("✅ Camera 1 successfully disabled")
    
    # Test enabling camera 1
    logger.info("Enabling Camera 1...")
    os.environ["CAMERA_1_ENABLED"] = "true"
    
    camera_1_enabled = is_camera_enabled(1)
    logger.info(f"Camera 1 enabled after enable: {camera_1_enabled}")
    
    if not camera_1_enabled:
        logger.error("❌ Camera 1 should be enabled but is still disabled!")
        return False
    else:
        logger.info("✅ Camera 1 successfully enabled")
    
    return True

def test_camera_capture_logic():
    """Test the camera capture logic without actually capturing images."""
    logger.info("Testing camera capture logic...")
    
    # Mock the capture function behavior
    def mock_capture_for_reader_async(reader_id: int, card_int: int):
        from integrated_access_camera import is_camera_enabled
        
        if not is_camera_enabled(reader_id):
            logger.info(f"✅ Camera {reader_id} is disabled, skipping image capture for card {card_int}")
            return "SKIPPED"
        else:
            logger.info(f"📸 Camera {reader_id} is enabled, would capture image for card {card_int}")
            return "CAPTURED"
    
    # Test with all cameras enabled
    logger.info("Testing with all cameras enabled...")
    results = []
    for reader_id in [1, 2, 3]:
        result = mock_capture_for_reader_async(reader_id, 12345)
        results.append((reader_id, result))
    
    # Test with camera 2 disabled
    logger.info("Testing with Camera 2 disabled...")
    os.environ["CAMERA_2_ENABLED"] = "false"
    
    results_disabled = []
    for reader_id in [1, 2, 3]:
        result = mock_capture_for_reader_async(reader_id, 67890)
        results_disabled.append((reader_id, result))
    
    # Verify results
    expected_enabled = ["CAPTURED", "CAPTURED", "CAPTURED"]
    expected_disabled = ["CAPTURED", "SKIPPED", "CAPTURED"]
    
    actual_enabled = [result for _, result in results]
    actual_disabled = [result for _, result in results_disabled]
    
    if actual_enabled == expected_enabled:
        logger.info("✅ All cameras captured when enabled")
    else:
        logger.error(f"❌ Expected {expected_enabled}, got {actual_enabled}")
        return False
    
    if actual_disabled == expected_disabled:
        logger.info("✅ Camera 2 correctly skipped when disabled")
    else:
        logger.error(f"❌ Expected {expected_disabled}, got {actual_disabled}")
        return False
    
    # Re-enable camera 2
    os.environ["CAMERA_2_ENABLED"] = "true"
    
    return True

def test_environment_variable_loading():
    """Test that environment variables are loaded correctly."""
    logger.info("Testing environment variable loading...")
    
    # Test different formats
    test_cases = [
        ("true", True),
        ("TRUE", True),
        ("True", True),
        ("false", False),
        ("FALSE", False),
        ("False", False),
        ("", True),  # Default should be true
        (None, True)  # Default should be true
    ]
    
    for value, expected in test_cases:
        os.environ["CAMERA_1_ENABLED"] = value if value is not None else ""
        if value is None:
            del os.environ["CAMERA_1_ENABLED"]
        
        from integrated_access_camera import is_camera_enabled
        result = is_camera_enabled(1)
        
        if result == expected:
            logger.info(f"✅ Value '{value}' correctly parsed as {result}")
        else:
            logger.error(f"❌ Value '{value}' should be {expected}, got {result}")
            return False
    
    return True

def main():
    """Run all tests."""
    logger.info("🧪 Starting Camera Disable Functionality Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Environment Variable Loading", test_environment_variable_loading),
        ("Camera Enabled Status", test_camera_enabled_status),
        ("Camera Disable/Enable", test_camera_disable_enable),
        ("Camera Capture Logic", test_camera_capture_logic)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running: {test_name}")
        logger.info("-" * 40)
        
        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Camera disable functionality is working correctly.")
        return True
    else:
        logger.error("💥 Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
