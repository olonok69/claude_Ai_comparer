#!/usr/bin/env python3
"""
Simple Test Runner for Session Processors

This script runs both session processors and does a quick comparison of key outputs.
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
    from old_session_processor import SessionProcessor as OldSessionProcessor
    from session_processor import SessionProcessor as NewSessionProcessor
    from utils.config_utils import load_config
    from utils.logging_utils import setup_logging
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nMake sure you have:")
    print("1. old_session_processor.py")
    print("2. session_processor.py")
    print("3. config/config.yaml")
    print("4. config/config_vet.yaml")
    sys.exit(1)


def compare_key_outputs(old_processor, new_processor):
    """Compare the key dataframes from both processors."""
    print("\n📊 Comparing Key Dataframes...")
    
    comparisons = []
    
    # Define the key dataframes to compare - EXCLUDE streams_catalog due to LLM non-determinism
    key_dfs = [
        ('session_this_filtered_valid_cols', 'session_this_filtered_valid_cols'),
        ('session_last_filtered_valid_cols_bva', 'session_last_filtered_valid_cols_bva'),
        ('session_last_filtered_valid_cols_lva', 'session_last_filtered_valid_cols_lva'),
        ('unique_streams', 'unique_streams'),
        # Excluding streams_catalog - LLM descriptions are non-deterministic
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
        
        old_val = getattr(old_processor, old_attr)
        new_val = getattr(new_processor, new_attr)
        
        if old_val is None or new_val is None:
            print(f"❌ One or both values are None")
            comparisons.append(False)
            continue
        
        # Handle different data types
        if isinstance(old_val, pd.DataFrame) and isinstance(new_val, pd.DataFrame):
            # Compare dataframes
            if old_val.shape != new_val.shape:
                print(f"❌ Shape mismatch: {old_val.shape} vs {new_val.shape}")
                comparisons.append(False)
                continue
            
            # Check if columns match
            if list(old_val.columns) != list(new_val.columns):
                print(f"❌ Column mismatch")
                print(f"   Old: {list(old_val.columns)}")
                print(f"   New: {list(new_val.columns)}")
                comparisons.append(False)
                continue
            
            # Compare data (handle NaN properly)
            try:
                old_sorted = old_val.sort_values(by=old_val.columns.tolist()).reset_index(drop=True)
                new_sorted = new_val.sort_values(by=new_val.columns.tolist()).reset_index(drop=True)
                
                if old_sorted.equals(new_sorted):
                    print(f"✅ DataFrames are identical ({old_val.shape})")
                    comparisons.append(True)
                else:
                    print(f"❌ DataFrames have different data")
                    comparisons.append(False)
            except Exception as e:
                print(f"❌ Error comparing dataframes: {e}")
                comparisons.append(False)
                
        elif isinstance(old_val, list) and isinstance(new_val, list):
            # Compare lists
            if sorted(old_val) == sorted(new_val):
                print(f"✅ Lists are identical ({len(old_val)} items)")
                comparisons.append(True)
            else:
                print(f"❌ Lists differ")
                print(f"   Old: {len(old_val)} items")
                print(f"   New: {len(new_val)} items")
                comparisons.append(False)
                
        elif isinstance(old_val, dict) and isinstance(new_val, dict):
            # Compare dictionaries
            if old_val == new_val:
                print(f"✅ Dictionaries are identical ({len(old_val)} items)")
                comparisons.append(True)
            else:
                print(f"❌ Dictionaries differ")
                print(f"   Old: {len(old_val)} items")
                print(f"   New: {len(new_val)} items")
                comparisons.append(False)
        else:
            # Direct comparison for other types
            if old_val == new_val:
                print(f"✅ Values are identical")
                comparisons.append(True)
            else:
                print(f"❌ Values differ")
                comparisons.append(False)
    
    # Add note about excluded comparisons
    print(f"\n📝 Note: Skipped streams_catalog comparison due to LLM non-determinism")
    
    return all(comparisons)


def compare_output_files(old_output: Path, new_output: Path):
    """Compare the CSV and JSON output files."""
    print("\n📁 Comparing Output Files...")
    
    comparisons = []
    
    # Files to compare - EXCLUDE streams.json due to LLM non-determinism
    files_to_compare = [
        "output/session_this_filtered_valid_cols.csv",
        "output/session_last_filtered_valid_cols_bva.csv", 
        "output/session_last_filtered_valid_cols_lva.csv",
        # Excluding streams.json - LLM descriptions are non-deterministic
    ]
    
    for file_path in files_to_compare:
        print(f"\n🔍 Comparing {file_path}...")
        
        old_file = old_output / file_path
        new_file = new_output / file_path
        
        if not old_file.exists():
            print(f"❌ Old file missing: {file_path}")
            comparisons.append(False)
            continue
            
        if not new_file.exists():
            print(f"❌ New file missing: {file_path}")
            comparisons.append(False)
            continue
        
        try:
            if file_path.endswith('.csv'):
                # Compare CSV files
                old_df = pd.read_csv(old_file)
                new_df = pd.read_csv(new_file)
                
                if old_df.shape != new_df.shape:
                    print(f"❌ Shape mismatch: {old_df.shape} vs {new_df.shape}")
                    comparisons.append(False)
                    continue
                
                if list(old_df.columns) != list(new_df.columns):
                    print(f"❌ Column mismatch")
                    comparisons.append(False)
                    continue
                
                # Sort and compare
                old_sorted = old_df.sort_values(by=old_df.columns.tolist()).reset_index(drop=True)
                new_sorted = new_df.sort_values(by=new_df.columns.tolist()).reset_index(drop=True)
                
                if old_sorted.equals(new_sorted):
                    print(f"✅ CSV files are identical ({old_df.shape})")
                    comparisons.append(True)
                else:
                    print(f"❌ CSV files have different data")
                    comparisons.append(False)
                    
            elif file_path.endswith('.json'):
                # Compare JSON files
                import json
                with open(old_file, 'r') as f:
                    old_json = json.load(f)
                with open(new_file, 'r') as f:
                    new_json = json.load(f)
                
                if old_json == new_json:
                    print(f"✅ JSON files are identical")
                    comparisons.append(True)
                else:
                    print(f"❌ JSON files differ")
                    comparisons.append(False)
                    
        except Exception as e:
            print(f"❌ Error comparing {file_path}: {e}")
            comparisons.append(False)
    
    return all(comparisons)


def run_test():
    """Run the simple comparison test."""
    print("🚀 Starting Session Processor Comparison Test")
    print("=" * 60)
    print("ℹ️  This test assumes you've already run:")
    print("   python main.py --config config/config.yaml --only-steps 1,2")
    print("   python main.py --config config/config_vet.yaml --only-steps 1,2")
    print()
    
    # Setup logging
    logger = setup_logging(log_file="logs/simple_session_test.log")
    
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
            
            # Run old processor
            print("\n🔄 Running OLD session processor...")
            old_processor = OldSessionProcessor(old_config)
            old_processor.process()
            print("✅ Old processor completed")
            
            # Run new processor
            print("\n🔄 Running NEW session processor...")
            new_processor = NewSessionProcessor(new_config)
            new_processor.process()
            print("✅ New processor completed")
            
            # Compare dataframes
            dataframes_identical = compare_key_outputs(old_processor, new_processor)
            
            # Compare output files
            files_identical = compare_output_files(old_output, new_output)
            
            # Final result
            print("\n" + "=" * 60)
            if dataframes_identical and files_identical:
                print("🎉 SUCCESS: All outputs are IDENTICAL!")
                print("✅ The new session processor produces the same results as the old one.")
                return True
            else:
                print("❌ FAILURE: Outputs are DIFFERENT!")
                print("⚠️  The new session processor produces different results.")
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