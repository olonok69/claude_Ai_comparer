#!/usr/bin/env python3
"""
Session Processor Comparison Script

This script compares the outputs of the old session processor (old_session_processor.py)
and the new generic processor (session_processor.py) to ensure they produce identical results.

Usage:
    python 3_compare_processors.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import hashlib
from typing import Dict, List, Tuple, Any

# Add the current directory to the path to import processors
sys.path.insert(0, os.getcwd())

# Import both processors
try:
    from old_session_processor import SessionProcessor as OldSessionProcessor
    from session_processor import SessionProcessor as NewSessionProcessor
    from utils.config_utils import load_config
    from utils.logging_utils import setup_logging
except ImportError as e:
    print(f"Error importing processors: {e}")
    print("Make sure you have:")
    print("1. old_session_processor.py (renamed from original)")
    print("2. session_processor.py (new generic version)")
    print("3. Both config files in the config/ directory")
    sys.exit(1)


class SessionProcessorComparison:
    """Compare outputs between old and new session processors."""
    
    def __init__(self):
        self.logger = setup_logging(log_file="logs/session_processor_comparison.log")
        self.comparison_results = {}
        self.differences_found = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comparison output directory
        self.comparison_dir = Path("comparison_output_session")
        self.comparison_dir.mkdir(exist_ok=True)
        
        # Create separate output directories for each processor
        self.old_output_dir = self.comparison_dir / "old_processor_output"
        self.new_output_dir = self.comparison_dir / "new_processor_output"
        self.old_output_dir.mkdir(exist_ok=True)
        self.new_output_dir.mkdir(exist_ok=True)
        
        self.logger.info("SessionProcessorComparison initialized")
    
    def run_old_processor_with_config(self, config: Dict) -> OldSessionProcessor:
        """Run the old session processor with provided config."""
        self.logger.info("Running old session processor...")
        
        # Update config to use our comparison output directory
        config_copy = config.copy()
        config_copy["output_dir"] = str(self.old_output_dir)
        
        old_processor = OldSessionProcessor(config_copy)
        old_processor.process()
        self.logger.info("Old session processor completed successfully")
        return old_processor
    
    def run_new_processor_with_config(self, config: Dict) -> NewSessionProcessor:
        """Run the new session processor with provided config."""
        self.logger.info("Running new session processor...")
        
        # Update config to use our comparison output directory
        config_copy = config.copy()
        config_copy["output_dir"] = str(self.new_output_dir)
        
        new_processor = NewSessionProcessor(config_copy)
        new_processor.process()
        self.logger.info("New session processor completed successfully")
        return new_processor
    
    def compare_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame, name: str) -> Dict:
        """Compare two dataframes and return comparison results."""
        self.logger.info(f"Comparing dataframes: {name}")
        
        comparison = {
            "name": name,
            "identical": False,
            "shape_match": False,
            "columns_match": False,
            "data_match": False,
            "differences": []
        }
        
        try:
            # Compare shapes
            if df1.shape == df2.shape:
                comparison["shape_match"] = True
                self.logger.info(f"{name}: Shapes match ({df1.shape})")
            else:
                diff = f"Shape mismatch - Old: {df1.shape}, New: {df2.shape}"
                comparison["differences"].append(diff)
                self.logger.warning(f"{name}: {diff}")
                return comparison
            
            # Compare columns
            if list(df1.columns) == list(df2.columns):
                comparison["columns_match"] = True
                self.logger.info(f"{name}: Columns match")
            else:
                old_cols = set(df1.columns)
                new_cols = set(df2.columns)
                missing_in_new = old_cols - new_cols
                extra_in_new = new_cols - old_cols
                
                if missing_in_new:
                    diff = f"Columns missing in new: {missing_in_new}"
                    comparison["differences"].append(diff)
                    self.logger.warning(f"{name}: {diff}")
                
                if extra_in_new:
                    diff = f"Extra columns in new: {extra_in_new}"
                    comparison["differences"].append(diff)
                    self.logger.warning(f"{name}: {diff}")
                
                return comparison
            
            # Compare data (handling NaN values properly)
            try:
                # Align dataframes by sorting
                df1_sorted = df1.sort_values(by=df1.columns.tolist()).reset_index(drop=True)
                df2_sorted = df2.sort_values(by=df2.columns.tolist()).reset_index(drop=True)
                
                # Use pandas equals which handles NaN properly
                if df1_sorted.equals(df2_sorted):
                    comparison["data_match"] = True
                    comparison["identical"] = True
                    self.logger.info(f"{name}: Data matches perfectly")
                else:
                    # Find specific differences
                    differences = []
                    for col in df1.columns:
                        if not df1_sorted[col].equals(df2_sorted[col]):
                            differences.append(f"Column '{col}' differs")
                    
                    comparison["differences"].extend(differences)
                    self.logger.warning(f"{name}: Data differs in columns: {differences}")
                    
            except Exception as e:
                comparison["differences"].append(f"Error comparing data: {str(e)}")
                self.logger.error(f"{name}: Error comparing data: {e}")
        
        except Exception as e:
            comparison["differences"].append(f"Comparison error: {str(e)}")
            self.logger.error(f"Error comparing {name}: {e}")
        
        return comparison
    
    def compare_files(self, old_dir: Path, new_dir: Path) -> Dict:
        """Compare CSV and JSON files between directories."""
        self.logger.info("Comparing output files...")
        
        file_comparisons = {
            "csv_files": [],
            "json_files": [],
            "missing_files": {"old_missing": [], "new_missing": []}
        }
        
        # Get all files from both directories
        old_files = {f.name: f for f in old_dir.rglob("*") if f.is_file()}
        new_files = {f.name: f for f in new_dir.rglob("*") if f.is_file()}
        
        # Check for missing files
        old_file_names = set(old_files.keys())
        new_file_names = set(new_files.keys())
        
        missing_in_new = old_file_names - new_file_names
        missing_in_old = new_file_names - old_file_names
        
        if missing_in_new:
            file_comparisons["missing_files"]["new_missing"] = list(missing_in_new)
            self.logger.warning(f"Files missing in new processor output: {missing_in_new}")
        
        if missing_in_old:
            file_comparisons["missing_files"]["old_missing"] = list(missing_in_old)
            self.logger.warning(f"Files missing in old processor output: {missing_in_old}")
        
        # Compare common files
        common_files = old_file_names & new_file_names
        
        for filename in common_files:
            old_file = old_files[filename]
            new_file = new_files[filename]
            
            # Skip streams.json due to LLM non-determinism
            if filename == 'streams.json':
                self.logger.info(f"Skipping {filename} comparison due to LLM non-determinism")
                continue
            
            if filename.endswith('.csv'):
                try:
                    old_df = pd.read_csv(old_file)
                    new_df = pd.read_csv(new_file)
                    comparison = self.compare_dataframes(old_df, new_df, filename)
                    file_comparisons["csv_files"].append(comparison)
                except Exception as e:
                    self.logger.error(f"Error comparing CSV {filename}: {e}")
                    file_comparisons["csv_files"].append({
                        "name": filename,
                        "identical": False,
                        "differences": [f"Error reading file: {str(e)}"]
                    })
            
            elif filename.endswith('.json'):
                try:
                    with open(old_file, 'r') as f:
                        old_json = json.load(f)
                    with open(new_file, 'r') as f:
                        new_json = json.load(f)
                    
                    if old_json == new_json:
                        file_comparisons["json_files"].append({
                            "name": filename,
                            "identical": True,
                            "differences": []
                        })
                        self.logger.info(f"JSON file {filename}: Identical")
                    else:
                        file_comparisons["json_files"].append({
                            "name": filename,
                            "identical": False,
                            "differences": ["JSON content differs"]
                        })
                        self.logger.warning(f"JSON file {filename}: Content differs")
                        
                except Exception as e:
                    self.logger.error(f"Error comparing JSON {filename}: {e}")
                    file_comparisons["json_files"].append({
                        "name": filename,
                        "identical": False,
                        "differences": [f"Error reading file: {str(e)}"]
                    })
        
        return file_comparisons
    
    def compare_processor_attributes(self, old_processor, new_processor) -> Dict:
        """Compare key attributes between processors."""
        self.logger.info("Comparing processor attributes...")
        
        attr_comparisons = []
        
        # Define key dataframe attributes to compare
        key_dataframes = [
            ('session_this_filtered_valid_cols', 'session_this_filtered_valid_cols'),
            ('session_last_filtered_valid_cols_bva', 'session_last_filtered_valid_cols_bva'),
            ('session_last_filtered_valid_cols_lva', 'session_last_filtered_valid_cols_lva'),
        ]
        
        for old_attr, new_attr in key_dataframes:
            if hasattr(old_processor, old_attr) and hasattr(new_processor, new_attr):
                old_df = getattr(old_processor, old_attr)
                new_df = getattr(new_processor, new_attr)
                comparison = self.compare_dataframes(old_df, new_df, f"attr_{old_attr}")
                attr_comparisons.append(comparison)
            else:
                attr_comparisons.append({
                    "name": f"attr_{old_attr}",
                    "identical": False,
                    "differences": ["Attribute missing in one processor"]
                })
        
        # Compare other key attributes - EXCLUDE streams_catalog due to LLM non-determinism
        other_attrs = ['unique_streams']  # Removed 'streams_catalog'
        for attr in other_attrs:
            if hasattr(old_processor, attr) and hasattr(new_processor, attr):
                old_val = getattr(old_processor, attr)
                new_val = getattr(new_processor, attr)
                
                if old_val == new_val:
                    attr_comparisons.append({
                        "name": f"attr_{attr}",
                        "identical": True,
                        "differences": []
                    })
                    self.logger.info(f"Attribute {attr}: Identical")
                else:
                    attr_comparisons.append({
                        "name": f"attr_{attr}",
                        "identical": False,
                        "differences": ["Attribute values differ"]
                    })
                    self.logger.warning(f"Attribute {attr}: Values differ")
            else:
                attr_comparisons.append({
                    "name": f"attr_{attr}",
                    "identical": False,
                    "differences": ["Attribute missing in one processor"]
                })
        
        # Add note about excluded comparisons
        self.logger.info("Skipped streams_catalog comparison due to LLM non-determinism")
        
        return attr_comparisons
    
    def run_comparison(self, old_config_path: str, new_config_path: str) -> bool:
        """Run the complete comparison between processors."""
        try:
            self.logger.info("Starting session processor comparison")
            print("🚀 Starting Session Processor Comparison")
            print("=" * 60)
            
            # Load configurations
            print("📁 Loading configurations...")
            old_config = load_config(old_config_path)
            new_config = load_config(new_config_path)
            
            # Run processors
            print("🔄 Running old processor...")
            old_processor = self.run_old_processor_with_config(old_config)
            
            print("🔄 Running new processor...")
            new_processor = self.run_new_processor_with_config(new_config)
            
            # Compare outputs
            print("📊 Comparing outputs...")
            
            # Compare files
            file_comparisons = self.compare_files(self.old_output_dir, self.new_output_dir)
            
            # Compare processor attributes
            attr_comparisons = self.compare_processor_attributes(old_processor, new_processor)
            
            # Compile results
            comparison_data = {
                "timestamp": self.timestamp,
                "old_config": old_config_path,
                "new_config": new_config_path,
                "file_comparisons": file_comparisons,
                "attribute_comparisons": attr_comparisons,
                "summary": {
                    "csv_files_total": len(file_comparisons["csv_files"]),
                    "csv_files_identical": sum(1 for f in file_comparisons["csv_files"] if f["identical"]),
                    "json_files_total": len(file_comparisons["json_files"]),
                    "json_files_identical": sum(1 for f in file_comparisons["json_files"] if f["identical"]),
                    "attr_total": len(attr_comparisons),
                    "attr_identical": sum(1 for a in attr_comparisons if a["identical"])
                }
            }
            
            # Save detailed comparison results
            comparison_data_path = self.comparison_dir / f"session_comparison_data_{self.timestamp}.json"
            with open(comparison_data_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, indent=2, default=str, ensure_ascii=False)
            
            # Generate readable report
            report_path = self.comparison_dir / f"session_comparison_report_{self.timestamp}.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("SESSION PROCESSOR COMPARISON REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Timestamp: {self.timestamp}\n")
                f.write(f"Old Config: {old_config_path}\n")
                f.write(f"New Config: {new_config_path}\n\n")
                
                f.write("SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(f"CSV Files: {comparison_data['summary']['csv_files_identical']}/{comparison_data['summary']['csv_files_total']} identical\n")
                f.write(f"JSON Files: {comparison_data['summary']['json_files_identical']}/{comparison_data['summary']['json_files_total']} identical\n")
                f.write(f"Processor Attributes: {comparison_data['summary']['attr_identical']}/{comparison_data['summary']['attr_total']} identical\n\n")
                
                # Categorize missing files
                missing_old = file_comparisons["missing_files"]["old_missing"]
                missing_new = file_comparisons["missing_files"]["new_missing"]
                critical_missing_old = [f for f in missing_old if not f.endswith('_cache.json')]
                critical_missing_new = [f for f in missing_new if not f.endswith('_cache.json')]
                non_critical_missing_old = [f for f in missing_old if f.endswith('_cache.json')]
                non_critical_missing_new = [f for f in missing_new if f.endswith('_cache.json')]
                
                if critical_missing_old:
                    f.write("CRITICAL FILES MISSING IN OLD PROCESSOR OUTPUT:\n")
                    for file in critical_missing_old:
                        f.write(f"  - {file}\n")
                    f.write("\n")
                
                if critical_missing_new:
                    f.write("CRITICAL FILES MISSING IN NEW PROCESSOR OUTPUT:\n")
                    for file in critical_missing_new:
                        f.write(f"  - {file}\n")
                    f.write("\n")
                
                if non_critical_missing_old or non_critical_missing_new:
                    f.write("NON-CRITICAL FILES (e.g., cache files) - DIFFERENCES EXPECTED:\n")
                    for file in non_critical_missing_old:
                        f.write(f"  - Missing in old: {file}\n")
                    for file in non_critical_missing_new:
                        f.write(f"  - Missing in new: {file}\n")
                    f.write("\n")
                
                f.write("CSV FILE COMPARISONS\n")
                f.write("-" * 20 + "\n")
                for comp in file_comparisons["csv_files"]:
                    status = "IDENTICAL" if comp["identical"] else "DIFFERENT"
                    f.write(f"{comp['name']}: {status}\n")
                    if comp.get("differences"):
                        for diff in comp["differences"]:
                            f.write(f"    - {diff}\n")
                f.write("\n")
                
                f.write("JSON FILE COMPARISONS\n")
                f.write("-" * 20 + "\n")
                for comp in file_comparisons["json_files"]:
                    status = "IDENTICAL" if comp["identical"] else "DIFFERENT"
                    f.write(f"{comp['name']}: {status}\n")
                    if comp.get("differences"):
                        for diff in comp["differences"]:
                            f.write(f"    - {diff}\n")
                f.write("\n")
                
                f.write("PROCESSOR ATTRIBUTE COMPARISONS\n")
                f.write("-" * 35 + "\n")
                for comp in attr_comparisons:
                    status = "IDENTICAL" if comp["identical"] else "DIFFERENT"
                    f.write(f"{comp['name']}: {status}\n")
                    if comp.get("differences"):
                        for diff in comp["differences"]:
                            f.write(f"    - {diff}\n")
                
                f.write("\nNOTE: streams_catalog and streams.json comparisons were skipped due to LLM non-determinism.\n")
            
            # Print summary
            print("\nSESSION PROCESSOR COMPARISON SUMMARY")
            print(f"{'='*60}")
            print(f"CSV Files: {comparison_data['summary']['csv_files_identical']}/{comparison_data['summary']['csv_files_total']} identical")
            print(f"JSON Files: {comparison_data['summary']['json_files_identical']}/{comparison_data['summary']['json_files_total']} identical")
            print(f"Processor Attributes: {comparison_data['summary']['attr_identical']}/{comparison_data['summary']['attr_total']} identical")
            print(f"\nDetailed report saved to: {report_path}")
            print(f"Comparison data saved to: {comparison_data_path}")
            
            # Return True if everything important is identical
            # Ignore missing cache files and LLM-generated content
            missing_old = file_comparisons["missing_files"]["old_missing"]
            missing_new = file_comparisons["missing_files"]["new_missing"]
            
            # Filter out non-critical missing files (cache files, etc.)
            critical_missing_old = [f for f in missing_old if not f.endswith('_cache.json')]
            critical_missing_new = [f for f in missing_new if not f.endswith('_cache.json')]
            
            all_identical = (
                comparison_data['summary']['csv_files_identical'] == comparison_data['summary']['csv_files_total'] and
                comparison_data['summary']['json_files_identical'] == comparison_data['summary']['json_files_total'] and
                comparison_data['summary']['attr_identical'] == comparison_data['summary']['attr_total'] and
                not critical_missing_old and
                not critical_missing_new
            )
            
            if all_identical:
                print("SUCCESS: All critical outputs are identical!")
                if missing_old or missing_new:
                    print("Note: Some non-critical files (like cache files) differ, but this is expected.")
                self.logger.info("Comparison completed successfully - all critical outputs identical")
            else:
                print("DIFFERENCES FOUND: Check the report for details")
                self.logger.warning("Comparison completed - differences found")
            
            return all_identical
            
        except Exception as e:
            self.logger.error(f"Error during comparison: {str(e)}", exc_info=True)
            print(f"ERROR: {str(e)}")
            return False


def main():
    """Main function to run the comparison."""
    comparison = SessionProcessorComparison()
    
    # Use the same config files as previous steps
    old_config_path = "config/config.yaml"
    new_config_path = "config/config_vet.yaml"
    
    success = comparison.run_comparison(old_config_path, new_config_path)
    
    if success:
        print("\n🎉 SUCCESS: Session processors produce identical results!")
        return True
    else:
        print("\n❌ FAILURE: Session processors produce different results!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)