#!/usr/bin/env python3
"""
Simple Test Runner for Registration Processors

This script runs both processors and does a quick comparison of key outputs.
FIXED VERSION: Uses pipeline functions to ensure vet-specific functions are applied.
"""

import os
import sys
import pandas as pd
from pathlib import Path
import tempfile
import shutil

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    from old_registration_processor import RegistrationProcessor as OldRegistrationProcessor
    from pipeline import run_registration_processing  # Import pipeline function
    from utils.config_utils import load_config
    from utils.logging_utils import setup_logging
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nMake sure you have:")
    print("1. old_registration_processor.py")
    print("2. registration_processor.py")
    print("3. pipeline.py")
    print("4. config/config.yaml")
    print("5. config/config_vet.yaml")
    sys.exit(1)


def compare_key_outputs(old_processor, new_processor):
    """Compare the key dataframes from both processors."""
    print("\n📊 Comparing Key Dataframes...")
    
    comparisons = []
    
    # Define the key dataframes to compare - both processors should have the same variable names
    key_dfs = [
        ('df_reg_demo_this', 'df_reg_demo_this'),
        ('df_reg_demo_last_bva', 'df_reg_demo_last_bva'),
        ('df_reg_demo_last_lva', 'df_reg_demo_last_lva')
    ]
    
    for old_attr, new_attr in key_dfs:
        print(f"\n🔍 Comparing {old_attr}...")
        
        if not hasattr(old_processor, old_attr):
            print(f"❌ Old processor missing: {old_attr}")
            comparisons.append(False)
            continue
            
        if not hasattr(new_processor, new_attr):
            print(f"❌ New processor missing: {new_attr}")
            comparisons.append(False)
            continue
        
        old_df = getattr(old_processor, old_attr)
        new_df = getattr(new_processor, new_attr)
        
        if old_df is None or new_df is None:
            print(f"❌ One or both dataframes are None")
            comparisons.append(False)
            continue
        
        # Compare shapes
        if old_df.shape != new_df.shape:
            print(f"❌ Shape mismatch: {old_df.shape} vs {new_df.shape}")
            comparisons.append(False)
            continue
        
        # Compare columns
        old_cols = set(old_df.columns)
        new_cols = set(new_df.columns)
        if old_cols != new_cols:
            print(f"❌ Column mismatch:")
            print(f"   Missing in new: {old_cols - new_cols}")
            print(f"   Extra in new: {new_cols - old_cols}")
            comparisons.append(False)
            continue
        
        # Compare data (sample check)
        try:
            # Sort both dataframes for consistent comparison
            sort_cols = [col for col in old_df.columns if col in ['BadgeId', 'Email']]
            if sort_cols:
                old_sorted = old_df.sort_values(sort_cols[0]).reset_index(drop=True)
                new_sorted = new_df.sort_values(sort_cols[0]).reset_index(drop=True)
            else:
                old_sorted = old_df.reset_index(drop=True)
                new_sorted = new_df.reset_index(drop=True)
            
            # Check if dataframes are equal
            if old_sorted.equals(new_sorted):
                print(f"✅ {old_attr}: IDENTICAL")
                comparisons.append(True)
            else:
                print(f"❌ {old_attr}: Data differences found")
                
                # Show a few differences
                differences = []
                for col in old_sorted.columns:
                    if not old_sorted[col].equals(new_sorted[col]):
                        if old_sorted[col].dtype == 'object':
                            diff_count = (old_sorted[col] != new_sorted[col]).sum()
                        else:
                            diff_count = ((old_sorted[col] != new_sorted[col]) & 
                                        ~(pd.isna(old_sorted[col]) & pd.isna(new_sorted[col]))).sum()
                        differences.append(f"{col}: {diff_count} differences")
                
                for diff in differences[:5]:  # Show first 5 differences
                    print(f"   - {diff}")
                
                comparisons.append(False)
                
        except Exception as e:
            print(f"❌ Error comparing data: {e}")
            comparisons.append(False)
    
    return all(comparisons)


def compare_output_files(old_output, new_output):
    """Compare the output files from both processors."""
    print("\n📁 Comparing Output Files...")
    
    # Get all CSV files from both directories
    old_csvs = {f.name: f for f in old_output.glob("**/*.csv")}
    new_csvs = {f.name: f for f in new_output.glob("**/*.csv")}
    
    # Get all JSON files from both directories
    old_jsons = {f.name: f for f in old_output.glob("**/*.json")}
    new_jsons = {f.name: f for f in new_output.glob("**/*.json")}
    
    all_identical = True
    
    # Compare CSV files
    all_csv_files = set(old_csvs.keys()) | set(new_csvs.keys())
    for filename in all_csv_files:
        if filename not in old_csvs:
            print(f"❌ Missing in old: {filename}")
            all_identical = False
            continue
        if filename not in new_csvs:
            print(f"❌ Missing in new: {filename}")
            all_identical = False
            continue
        
        try:
            old_df = pd.read_csv(old_csvs[filename])
            new_df = pd.read_csv(new_csvs[filename])
            
            if old_df.shape != new_df.shape:
                print(f"❌ {filename}: Shape mismatch {old_df.shape} vs {new_df.shape}")
                all_identical = False
            elif not old_df.equals(new_df):
                print(f"❌ {filename}: Content differs")
                all_identical = False
            else:
                print(f"✅ {filename}: IDENTICAL")
                
        except Exception as e:
            print(f"❌ Error comparing {filename}: {e}")
            all_identical = False
    
    # Compare JSON files
    all_json_files = set(old_jsons.keys()) | set(new_jsons.keys())
    for filename in all_json_files:
        if filename not in old_jsons:
            print(f"❌ Missing in old: {filename}")
            all_identical = False
            continue
        if filename not in new_jsons:
            print(f"❌ Missing in new: {filename}")
            all_identical = False
            continue
        
        try:
            import json
            with open(old_jsons[filename], 'r') as f:
                old_data = json.load(f)
            with open(new_jsons[filename], 'r') as f:
                new_data = json.load(f)
            
            # Compare JSON content
            old_str = json.dumps(old_data, sort_keys=True)
            new_str = json.dumps(new_data, sort_keys=True)
            
            if old_str == new_str:
                print(f"✅ {filename}: IDENTICAL")
            else:
                print(f"❌ {filename}: Content differs")
                all_identical = False
                
        except Exception as e:
            print(f"❌ Error comparing {filename}: {e}")
            all_identical = False
    
    return all_identical


def run_test():
    """Run the test comparison."""
    print("🚀 Starting Registration Processor Comparison Test")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging(log_file="logs/simple_test.log")
    
    # Create temporary directories for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        old_output = Path(temp_dir) / "old_output"
        new_output = Path(temp_dir) / "new_output"
        old_output.mkdir()
        new_output.mkdir()
        
        try:
            # Load configurations
            print("📁 Loading configurations...")
            old_config = load_config("config/config.yaml")
            new_config = load_config("config/config_vet.yaml")
            
            # Update output directories
            old_config["output_dir"] = str(old_output)
            new_config["output_dir"] = str(new_output)
            
            # Run old processor (direct instantiation since it doesn't need vet-specific functions)
            print("\n🔄 Running OLD processor...")
            old_processor = OldRegistrationProcessor(old_config)
            old_processor.process()
            print("✅ Old processor completed")
            
            # Run new processor (using pipeline to ensure vet-specific functions are applied)
            print("\n🔄 Running NEW processor via pipeline...")
            new_processor = run_registration_processing(new_config)
            print("✅ New processor completed")
            
            # Check if vet-specific functions were applied
            if hasattr(new_processor, '_vet_specific_active') and new_processor._vet_specific_active:
                print("✅ Vet-specific functions were applied to new processor")
            else:
                print("⚠️  Vet-specific functions were NOT applied to new processor")
            
            # Compare dataframes
            dataframes_identical = compare_key_outputs(old_processor, new_processor)
            
            # Compare output files
            files_identical = compare_output_files(old_output, new_output)
            
            # Final result
            print("\n" + "=" * 60)
            if dataframes_identical and files_identical:
                print("🎉 SUCCESS: All outputs are IDENTICAL!")
                print("✅ The new processor produces the same results as the old one.")
                return True
            else:
                print("❌ FAILURE: Outputs are DIFFERENT!")
                print("⚠️  The new processor produces different results.")
                if not dataframes_identical:
                    print("   - Key dataframes differ")
                if not files_identical:
                    print("   - Output files differ")
                return False
                
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            logger.error(f"Test failed: {e}", exc_info=True)
            return False


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)