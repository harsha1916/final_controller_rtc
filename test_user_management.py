#!/usr/bin/env python3
"""
Test script for User Management functionality
"""

import os
import json
import requests
import time

def test_user_management():
    """Test the user management functionality"""
    
    base_url = "http://localhost:5001"
    api_key = "your-api-key-change-this"
    
    print("🧪 Testing User Management Functionality")
    print("=" * 50)
    
    # Test 1: Get users (should be empty initially)
    print("\n1. Testing get users...")
    try:
        response = requests.get(f"{base_url}/get_users", timeout=10)
        if response.status_code == 200:
            users = response.json()
            print(f"✅ Get users working - {len(users)} users found")
            for user in users:
                print(f"   - {user['name']} (Card: {user['card_number']}) - {'Blocked' if user['blocked'] else 'Active'}")
        else:
            print(f"❌ Get users failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Get users error: {e}")
    
    # Test 2: Add a test user
    print("\n2. Testing add user...")
    try:
        params = {
            'api_key': api_key,
            'card_number': '123456789',
            'id': 'test001',
            'name': 'Test User',
            'ref_id': 'TEST001'
        }
        response = requests.get(f"{base_url}/add_user", params=params, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Add user working: {result['message']}")
        else:
            print(f"❌ Add user failed: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Add user error: {e}")
    
    # Test 3: Get users again (should have 1 user now)
    print("\n3. Testing get users after add...")
    try:
        response = requests.get(f"{base_url}/get_users", timeout=10)
        if response.status_code == 200:
            users = response.json()
            print(f"✅ Get users working - {len(users)} users found")
            for user in users:
                print(f"   - {user['name']} (Card: {user['card_number']}) - {'Blocked' if user['blocked'] else 'Active'}")
        else:
            print(f"❌ Get users failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Get users error: {e}")
    
    # Test 4: Block user
    print("\n4. Testing block user...")
    try:
        params = {
            'api_key': api_key,
            'card_number': '123456789'
        }
        response = requests.get(f"{base_url}/block_user", params=params, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Block user working: {result['message']}")
        else:
            print(f"❌ Block user failed: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Block user error: {e}")
    
    # Test 5: Get users again (should show blocked status)
    print("\n5. Testing get users after block...")
    try:
        response = requests.get(f"{base_url}/get_users", timeout=10)
        if response.status_code == 200:
            users = response.json()
            print(f"✅ Get users working - {len(users)} users found")
            for user in users:
                print(f"   - {user['name']} (Card: {user['card_number']}) - {'Blocked' if user['blocked'] else 'Active'}")
        else:
            print(f"❌ Get users failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Get users error: {e}")
    
    # Test 6: Unblock user
    print("\n6. Testing unblock user...")
    try:
        params = {
            'api_key': api_key,
            'card_number': '123456789'
        }
        response = requests.get(f"{base_url}/unblock_user", params=params, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Unblock user working: {result['message']}")
        else:
            print(f"❌ Unblock user failed: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Unblock user error: {e}")
    
    # Test 7: Search user
    print("\n7. Testing search user...")
    try:
        params = {
            'id': 'test001'
        }
        response = requests.get(f"{base_url}/search_user", params=params, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result['status'] == 'success':
                user = result['users'][0]
                print(f"✅ Search user working: Found {user['name']} (Card: {user['card_number']})")
            else:
                print(f"❌ Search user failed: {result['message']}")
        else:
            print(f"❌ Search user failed: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Search user error: {e}")
    
    # Test 8: Delete user
    print("\n8. Testing delete user...")
    try:
        params = {
            'api_key': api_key,
            'card_number': '123456789'
        }
        response = requests.get(f"{base_url}/delete_user", params=params, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Delete user working: {result['message']}")
        else:
            print(f"❌ Delete user failed: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Delete user error: {e}")
    
    # Test 9: Get users again (should be empty now)
    print("\n9. Testing get users after delete...")
    try:
        response = requests.get(f"{base_url}/get_users", timeout=10)
        if response.status_code == 200:
            users = response.json()
            print(f"✅ Get users working - {len(users)} users found")
            for user in users:
                print(f"   - {user['name']} (Card: {user['card_number']}) - {'Blocked' if user['blocked'] else 'Active'}")
        else:
            print(f"❌ Get users failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Get users error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 User Management testing completed!")

if __name__ == "__main__":
    print("🚀 RFID Access Control System - User Management Test")
    print("Make sure the Flask application is running on localhost:5001")
    print()
    
    test_user_management()
    
    print("\n💡 To test the web interface, open: http://localhost:5001/")
    print("💡 Go to the 'User Management' tab to test the interface")
