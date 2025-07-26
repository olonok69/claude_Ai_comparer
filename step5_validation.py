#!/usr/bin/env python3
"""
Step 5 Validation Script

This script validates that all Step 5 components are properly implemented and working.
It checks file existence, configuration validity, and runs basic tests.
"""

import os
import sys
from pathlib import Path
import importlib.util
import yaml

def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ MISSING {description}: {filepath}")
        return False

def check_import(module_name, filepath):
    """Check if a Python module can be imported."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None:
            print(f"❌ Cannot load module spec for {module_name}")
            return False
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"✅ Successfully imported {module_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return False

def check_config_section(config_path, section_path, description):
    """Check if a configuration section exists."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Navigate to the section using dot notation
        current = config
        for key in section_path.split('.'):
            if key in current:
                current = current[key]
            else:
                print(f"❌ Missing config section '{section_path}' in {config_path}")
                return False
        
        print(f"✅ Found config section '{section_path}' in {config_path}")
        return True
    except Exception as e:
        print(f"❌ Error checking config section '{section_path}' in {config_path}: {e}")
        return False

def validate_step5_implementation():
    """Validate all Step 5 implementation components."""
    print("🔍 Validating Step 5 Implementation")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Check main processor file
    print("\n📁 Checking Core Files:")
    all_checks_passed &= check_file_exists("neo4j_session_processor.py", "Neo4j Session Processor")
    all_checks_passed &= check_file_exists("old_neo4j_session_processor.py", "Old Neo4j Session Processor")
    
    # Check comparison tools
    print("\n🔧 Checking Comparison Tools:")
    all_checks_passed &= check_file_exists("comparers/5_compare_processors.py", "Comparison Tool")
    all_checks_passed &= check_file_exists("comparers/5_simple_test_runner.py", "Simple Test Runner")
    
    # Check configuration files
    print("\n⚙️  Checking Configuration Files:")
    all_checks_passed &= check_file_exists("config/config_vet.yaml", "BVA Config")
    all_checks_passed &= check_file_exists("config/config_ecomm.yaml", "ECOMM Config")
    all_checks_passed &= check_file_exists("config/config.yaml", "Legacy Config")
    
    # Check configuration sections
    print("\n📋 Checking Configuration Sections:")
    all_checks_passed &= check_config_section("config/config_vet.yaml", "event", "Event Config in BVA")
    all_checks_passed &= check_config_section("config/config_ecomm.yaml", "event", "Event Config in ECOMM")
    all_checks_passed &= check_config_section("config/config_vet.yaml", "session_files", "Session Files in BVA")
    all_checks_passed &= check_config_section("config/config_ecomm.yaml", "session_files", "Session Files in ECOMM")
    
    # Check required utility files
    print("\n🛠️  Checking Utility Files:")
    all_checks_passed &= check_file_exists("utils/config_utils.py", "Config Utils")
    all_checks_passed &= check_file_exists("utils/logging_utils.py", "Logging Utils")
    all_checks_passed &= check_file_exists("pipeline.py", "Pipeline")
    all_checks_passed &= check_file_exists("main.py", "Main Script")
    
    # Check Python imports
    print("\n🐍 Checking Python Imports:")
    if os.path.exists("neo4j_session_processor.py"):
        all_checks_passed &= check_import("neo4j_session_processor", "neo4j_session_processor.py")
    
    if os.path.exists("comparers/5_compare_processors.py"):
        all_checks_passed &= check_import("compare_processors_5", "comparers/5_compare_processors.py")
    
    # Check Neo4j dependency
    print("\n🗄️  Checking Dependencies:")
    try:
        import neo4j
        print("✅ Neo4j driver available")
    except ImportError:
        print("❌ Neo4j driver not available - run: pip install neo4j")
        all_checks_passed = False
    
    try:
        import pandas
        print("✅ Pandas available")
    except ImportError:
        print("❌ Pandas not available - run: pip install pandas")
        all_checks_passed = False
    
    try:
        from dotenv import load_dotenv
        print("✅ Python-dotenv available")
    except ImportError:
        print("❌ Python-dotenv not available - run: pip install python-dotenv")
        all_checks_passed = False
    
    # Check directories
    print("\n📂 Checking Directory Structure:")
    required_dirs = [
        "comparers",
        "config",
        "utils",
        "logs",
        "data"
    ]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory exists: {dir_name}")
        else:
            print(f"⚠️  Directory missing (may be created later): {dir_name}")
            if dir_name in ["comparers", "config", "utils"]:
                all_checks_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ Step 5 implementation is ready for testing.")
        print("\nNext steps:")
        print("1. Run: python comparers/5_simple_test_runner.py")
        print("2. Run: python comparers/5_compare_processors.py")
        print("3. If tests pass, enable neo4j_session_processing in config")
        return True
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("⚠️  Please fix the issues above before proceeding.")
        print("\nCommon fixes:")
        print("- Install missing dependencies: pip install neo4j pandas python-dotenv")
        print("- Create missing configuration sections")
        print("- Ensure all files are in correct locations")
        return False

def main():
    """Main validation function."""
    success = validate_step5_implementation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()