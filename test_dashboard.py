#!/usr/bin/env python3
"""
Dashboard API Test Script
Tests if all API endpoints are responding correctly.
"""
import requests
import json

# Configuration
BASE_URL = "http://localhost:5001"
API_KEY = "your-api-key-change-this"  # Update this with your actual API key

def test_endpoint(name, url, method="GET", headers=None, data=None):
    """Test a single endpoint."""
    try:
        full_url = f"{BASE_URL}{url}"
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"URL: {full_url}")
        print(f"Method: {method}")
        
        if method == "GET":
            response = requests.get(full_url, headers=headers, timeout=5)
        elif method == "POST":
            response = requests.post(full_url, headers=headers, json=data, timeout=5)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS")
            try:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)[:200]}...")
            except:
                print(f"Response: {response.text[:200]}...")
        else:
            print(f"❌ FAILED")
            print(f"Response: {response.text[:200]}")
        
        return response.status_code == 200
        
    except requests.exceptions.ConnectionError:
        print(f"❌ CONNECTION ERROR - Is the server running?")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    print("="*60)
    print("MAXPARK RFID DASHBOARD API TEST")
    print("="*60)
    
    results = {}
    
    # Test public endpoints (no auth)
    print("\n" + "="*60)
    print("PUBLIC ENDPOINTS (No Authentication)")
    print("="*60)
    
    results['status'] = test_endpoint("System Status", "/status")
    results['health'] = test_endpoint("Health Check", "/health_check")
    results['transactions'] = test_endpoint("Get Transactions", "/get_transactions")
    results['recent'] = test_endpoint("Recent Transactions", "/get_recent_transactions")
    results['users'] = test_endpoint("Get Users", "/get_users")
    results['config'] = test_endpoint("Get Config", "/get_config")
    
    # Test authenticated endpoints
    print("\n" + "="*60)
    print("AUTHENTICATED ENDPOINTS (API Key Required)")
    print("="*60)
    
    auth_headers = {'X-API-Key': API_KEY}
    
    results['entity_config'] = test_endpoint(
        "Get Entity Config", 
        "/get_entity_config",
        headers=auth_headers
    )
    
    results['rtc_debug'] = test_endpoint(
        "RTC Debug", 
        "/rtc_debug",
        headers=auth_headers
    )
    
    results['cache_status'] = test_endpoint(
        "Transaction Cache Status", 
        "/transaction_cache_status"
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! Dashboard should be working.")
    else:
        print("❌ Some tests failed. Check the errors above.")
        print("\nCommon issues:")
        print("1. Server not running: sudo systemctl status rfid-access-control")
        print("2. Wrong port: Check if server is running on port 5001")
        print("3. Wrong API key: Update API_KEY in this script")
        print("4. Firewall: Check if port 5001 is open")
    
    print("="*60)

if __name__ == "__main__":
    main()

