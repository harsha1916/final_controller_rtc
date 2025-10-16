#!/usr/bin/env python3
"""
Test script to verify offline capability and Firestore structure consistency
"""

import json
import os
import time
from datetime import datetime

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_success(message):
    print(f"{GREEN}✅ {message}{RESET}")

def print_error(message):
    print(f"{RED}❌ {message}{RESET}")

def print_warning(message):
    print(f"{YELLOW}⚠️  {message}{RESET}")

def print_info(message):
    print(f"{BLUE}ℹ️  {message}{RESET}")

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print_success(f"{description} exists: {filepath}")
        return True
    else:
        print_error(f"{description} NOT found: {filepath}")
        return False

def check_cache_structure(cache_file):
    """Verify transaction cache structure"""
    print_info(f"\nChecking cache structure: {cache_file}")
    
    if not os.path.exists(cache_file):
        print_warning("Cache file doesn't exist yet (no transactions)")
        return True
    
    try:
        with open(cache_file, 'r') as f:
            transactions = json.load(f)
        
        if not isinstance(transactions, list):
            print_error("Cache is not a list")
            return False
        
        print_success(f"Cache contains {len(transactions)} transactions")
        
        if len(transactions) > 0:
            # Check structure of first transaction
            tx = transactions[0]
            required_fields = ['name', 'card', 'reader', 'status', 'timestamp', 'entity_id']
            
            missing_fields = []
            for field in required_fields:
                if field not in tx:
                    missing_fields.append(field)
            
            if missing_fields:
                print_error(f"Transaction missing fields: {missing_fields}")
                print_info(f"Transaction structure: {json.dumps(tx, indent=2)}")
                return False
            else:
                print_success("Transaction structure is correct")
                print_info(f"Sample transaction: {json.dumps(tx, indent=2)}")
        
        return True
        
    except json.JSONDecodeError as e:
        print_error(f"Cache file is not valid JSON: {e}")
        return False
    except Exception as e:
        print_error(f"Error reading cache: {e}")
        return False

def check_firestore_code_consistency():
    """Check if integrated_access_camera.py has consistent Firestore structure"""
    print_info("\nChecking Firestore code consistency...")
    
    try:
        with open('integrated_access_camera.py', 'r') as f:
            content = f.read()
        
        # Check for old nested structure (should NOT exist)
        old_structure_patterns = [
            'db.collection("entities").document(ENTITY_ID).collection("transactions")',
        ]
        
        issues = []
        for pattern in old_structure_patterns:
            if pattern in content:
                issues.append(f"Found old structure: {pattern}")
        
        if issues:
            for issue in issues:
                print_error(issue)
            return False
        
        # Check for correct flat structure
        if 'db.collection("transactions").add(' in content:
            print_success("Using flat structure: db.collection('transactions').add()")
        else:
            print_warning("Could not verify flat structure usage")
        
        # Check for entity_id filter
        if 'FieldFilter("entity_id", "==", ENTITY_ID)' in content:
            print_success("Using entity_id filter in queries")
        else:
            print_warning("Could not verify entity_id filter usage")
        
        # Check if cache is deleted after sync
        if 'os.remove(TRANSACTION_CACHE_FILE)' in content:
            print_error("Code deletes cache file after sync (should NOT delete)")
            return False
        else:
            print_success("Cache file is preserved (not deleted)")
        
        # Check if cache is always called
        if 'cache_transaction(transaction)' in content:
            print_success("Cache function is used")
        else:
            print_error("Cache function not found")
            return False
        
        return True
        
    except FileNotFoundError:
        print_error("integrated_access_camera.py not found")
        return False
    except Exception as e:
        print_error(f"Error reading code: {e}")
        return False

def simulate_offline_scenario():
    """Simulate offline operation"""
    print_info("\n" + "="*60)
    print_info("SIMULATING OFFLINE SCENARIO")
    print_info("="*60)
    
    # Create a test transaction
    test_transaction = {
        "name": "Test User",
        "card": "1234567890",
        "reader": 1,
        "status": "granted",
        "timestamp": int(time.time()),
        "entity_id": "test_entity"
    }
    
    cache_file = "transactions_cache.json"
    
    # Read existing cache
    existing_transactions = []
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                existing_transactions = json.load(f)
            print_info(f"Loaded {len(existing_transactions)} existing transactions")
        except:
            pass
    
    # Add test transaction
    existing_transactions.append(test_transaction)
    
    # Write back to cache
    try:
        with open(cache_file, 'w') as f:
            json.dump(existing_transactions, f, indent=2)
        print_success(f"Added test transaction to cache")
        print_info(f"Total cached transactions: {len(existing_transactions)}")
        return True
    except Exception as e:
        print_error(f"Failed to write cache: {e}")
        return False

def check_env_config():
    """Check environment configuration"""
    print_info("\nChecking environment configuration...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    entity_id = os.getenv('ENTITY_ID', 'default_entity')
    print_success(f"ENTITY_ID: {entity_id}")
    
    base_dir = os.getenv('BASE_DIR', '/home/maxpark')
    print_info(f"BASE_DIR: {base_dir}")
    
    return True

def main():
    """Main test function"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}OFFLINE CAPABILITY & FIRESTORE STRUCTURE TEST{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    results = []
    
    # Test 1: Check environment config
    print_info("\n[TEST 1] Environment Configuration")
    results.append(("Environment Config", check_env_config()))
    
    # Test 2: Check required files
    print_info("\n[TEST 2] Required Files")
    files_ok = all([
        check_file_exists("integrated_access_camera.py", "Main application"),
        check_file_exists("config.py", "Configuration"),
        check_file_exists("uploader.py", "Image uploader"),
        check_file_exists(".env", "Environment file"),
    ])
    results.append(("Required Files", files_ok))
    
    # Test 3: Check cache structure
    print_info("\n[TEST 3] Cache Structure")
    base_dir = os.getenv('BASE_DIR', '/home/maxpark')
    cache_file = os.path.join(base_dir, "transactions_cache.json")
    results.append(("Cache Structure", check_cache_structure(cache_file)))
    
    # Test 4: Check code consistency
    print_info("\n[TEST 4] Firestore Code Consistency")
    results.append(("Code Consistency", check_firestore_code_consistency()))
    
    # Test 5: Simulate offline operation
    print_info("\n[TEST 5] Offline Simulation")
    results.append(("Offline Simulation", simulate_offline_scenario()))
    
    # Print summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}TEST SUMMARY{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        if result:
            print_success(f"{test_name}: PASSED")
            passed += 1
        else:
            print_error(f"{test_name}: FAILED")
            failed += 1
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{GREEN}Passed: {passed}{RESET} | {RED}Failed: {failed}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    if failed == 0:
        print_success("🎉 ALL TESTS PASSED! System is ready for offline operation.")
        return 0
    else:
        print_error(f"⚠️  {failed} test(s) failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())

