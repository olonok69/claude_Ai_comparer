#!/usr/bin/env python3
"""
Step 6 Validation Script
Validates that the generic Neo4j Job Stream Processor implementation is complete and working.
"""

import os
import sys
from pathlib import Path


def check_file_exists(file_path, description):
    """Check if a file exists and print result."""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - NOT FOUND")
        return False


def check_import(module_name, file_path):
    """Check if a Python module can be imported."""
    try:
        if file_path:
            sys.path.insert(0, os.path.dirname(file_path))
        __import__(module_name)
        print(f"✅ Python import: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ Python import: {module_name} - {e}")
        return False


def check_config_section(config_path, section_path, description):
    """Check if a config section exists."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Navigate to the section
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


def validate_neo4j_connection():
    """Validate Neo4j connection and database state."""
    try:
        from neo4j import GraphDatabase
        from dotenv import load_dotenv
        
        # Load environment
        load_dotenv("keys/.env")
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME") 
        password = os.getenv("NEO4J_PASSWORD")
        
        if not all([uri, username, password]):
            print("❌ Neo4j credentials missing in .env file")
            return False
        
        # Test connection
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session() as session:
            # Check for visitors with show attribute
            result = session.run("""
                MATCH (v:Visitor_this_year) 
                RETURN v.show, count(v) as count 
                ORDER BY v.show
            """)
            visitors = list(result)
            
            if not visitors:
                print("❌ No visitors found in Neo4j")
                return False
            
            print("✅ Neo4j connection successful")
            for record in visitors:
                print(f"  - {record['v.show']} show: {record['count']} visitors")
            
            # Check for streams with show attribute
            result = session.run("""
                MATCH (s:Stream) 
                RETURN s.show, count(s) as count 
                ORDER BY s.show
            """)
            streams = list(result)
            
            for record in streams:
                print(f"  - {record['s.show']} show: {record['count']} streams")
            
            # Check current job stream relationships
            result = session.run("""
                MATCH (v:Visitor_this_year)-[r:job_to_stream]->(s:Stream)
                RETURN v.show, count(r) as relationships
                ORDER BY v.show
            """)
            relationships = list(result)
            
            if relationships:
                print("📊 Existing job stream relationships:")
                for record in relationships:
                    print(f"  - {record['v.show']} show: {record['relationships']} relationships")
            else:
                print("📊 No existing job stream relationships (expected for Step 6)")
        
        driver.close()
        return True
        
    except Exception as e:
        print(f"❌ Neo4j validation failed: {e}")
        return False


def validate_job_to_stream_csv():
    """Validate the job_to_stream.csv file structure."""
    print("\n📋 Checking job_to_stream.csv file:")
    
    # Check for the file in common locations
    possible_paths = [
        "job_to_stream.csv",
        "data/bva/job_to_stream.csv",
        "data/ecomm/job_to_stream.csv"
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if not csv_path:
        print("❌ job_to_stream.csv file not found in expected locations")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        if "Job Role" not in df.columns:
            print("❌ job_to_stream.csv missing 'Job Role' column")
            return False
        
        print(f"✅ Found job_to_stream.csv at: {csv_path}")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {len(df.columns)}")
        print(f"  - Job roles: {', '.join(df['Job Role'].head(3).tolist())}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading job_to_stream.csv: {e}")
        return False


def validate_step6_implementation():
    """Validate all Step 6 implementation components."""
    print("🔍 Validating Step 6 Implementation")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Check main processor file
    print("\n📁 Checking Core Files:")
    all_checks_passed &= check_file_exists("neo4j_job_stream_processor.py", "Neo4j Job Stream Processor")
    all_checks_passed &= check_file_exists("old_neo4j_job_stream_processor.py", "Old Neo4j Job Stream Processor")
    
    # Check comparison tools
    print("\n🔧 Checking Comparison Tools:")
    all_checks_passed &= check_file_exists("comparers/6_compare_processors.py", "Comparison Tool")
    all_checks_passed &= check_file_exists("comparers/6_simple_test_runner.py", "Simple Test Runner")
    
    # Check configuration files
    print("\n⚙️  Checking Configuration Files:")
    all_checks_passed &= check_file_exists("config/config_vet.yaml", "BVA Config")
    all_checks_passed &= check_file_exists("config/config_ecomm.yaml", "ECOMM Config")
    all_checks_passed &= check_file_exists("config/config.yaml", "Legacy Config")
    
    # Check configuration sections
    print("\n📋 Checking Configuration Sections:")
    all_checks_passed &= check_config_section("config/config_vet.yaml", "neo4j.show_name", "Show Name in BVA")
    all_checks_passed &= check_config_section("config/config_ecomm.yaml", "neo4j.show_name", "Show Name in ECOMM")
    all_checks_passed &= check_config_section("config/config_vet.yaml", "neo4j.job_stream_mapping", "Job Stream Config in BVA")
    all_checks_passed &= check_config_section("config/config_ecomm.yaml", "neo4j.job_stream_mapping", "Job Stream Config in ECOMM")
    
    # Check required utility files
    print("\n🛠️  Checking Utility Files:")
    all_checks_passed &= check_file_exists("utils/config_utils.py", "Config Utils")
    all_checks_passed &= check_file_exists("utils/logging_utils.py", "Logging Utils")
    all_checks_passed &= check_file_exists("utils/summary_utils.py", "Summary Utils")
    all_checks_passed &= check_file_exists("pipeline.py", "Pipeline")
    all_checks_passed &= check_file_exists("main.py", "Main Script")
    
    # Check Python imports
    print("\n🐍 Checking Python Imports:")
    if os.path.exists("neo4j_job_stream_processor.py"):
        all_checks_passed &= check_import("neo4j_job_stream_processor", "neo4j_job_stream_processor.py")
    
    if os.path.exists("comparers/6_compare_processors.py"):
        sys.path.insert(0, "comparers")
        all_checks_passed &= check_import("6_compare_processors", "comparers/6_compare_processors.py")
    
    # Check Neo4j dependency
    print("\n🗄️  Checking Dependencies:")
    try:
        import neo4j
        print("✅ Neo4j driver installed")
    except ImportError:
        print("❌ Neo4j driver not installed - run: pip install neo4j")
        all_checks_passed = False
    
    try:
        import pandas
        print("✅ Pandas installed")
    except ImportError:
        print("❌ Pandas not installed - run: pip install pandas")
        all_checks_passed = False
    
    try:
        import yaml
        print("✅ PyYAML installed")
    except ImportError:
        print("❌ PyYAML not installed - run: pip install pyyaml")
        all_checks_passed = False
    
    # Check CSV file
    all_checks_passed &= validate_job_to_stream_csv()
    
    # Check Neo4j connection and database state
    print("\n🗄️  Checking Neo4j Database:")
    all_checks_passed &= validate_neo4j_connection()
    
    # Final result
    print(f"\n{'='*50}")
    if all_checks_passed:
        print("🎉 SUCCESS: Step 6 implementation is complete and ready!")
        print("\nNext steps:")
        print("1. Run: cd comparers && python 6_simple_test_runner.py")
        print("2. Run: cd comparers && python 6_compare_processors.py")
        print("3. Test: python main.py --config config/config_vet.yaml --only-steps 6")
        print("4. Test: python main.py --config config/config_ecomm.yaml --only-steps 6")
    else:
        print("❌ FAILURE: Some components are missing or invalid!")
        print("\nPlease address the issues above before proceeding.")
    
    return all_checks_passed


if __name__ == "__main__":
    success = validate_step6_implementation()
    sys.exit(0 if success else 1)