#!/usr/bin/env python3
"""
Processor Comparison Script

This script compares the outputs of the old registration processor (old_registration_processor.py)
and the new event-agnostic processor (registration_processor.py) to ensure they produce identical results.

FIXED VERSION: Uses pipeline functions to ensure vet-specific functions are applied.

Usage:
    python compare_processors.py
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

# Import both processors and pipeline
try:
    from old_registration_processor import RegistrationProcessor as OldRegistrationProcessor
    from pipeline import run_registration_processing  # Import pipeline function
    from utils.config_utils import load_config
    from utils.logging_utils import setup_logging
except ImportError as e:
    print(f"Error importing processors: {e}")
    print("Make sure you have:")
    print("1. old_registration_processor.py (renamed from original)")
    print("2. registration_processor.py (new event-agnostic version)")
    print("3. pipeline.py")
    print("4. Both config files in the config/ directory")
    sys.exit(1)


class ProcessorComparison:
    """Compare outputs between old and new registration processors."""
    
    def __init__(self):
        self.logger = setup_logging(log_file="logs/processor_comparison.log")
        self.comparison_results = {}
        self.differences_found = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comparison output directory
        self.comparison_dir = Path("comparison_output")
        self.comparison_dir.mkdir(exist_ok=True)
        
        # Create separate output directories for each processor
        self.old_output_dir = self.comparison_dir / "old_processor_output"
        self.new_output_dir = self.comparison_dir / "new_processor_output"
        self.old_output_dir.mkdir(exist_ok=True)
        self.new_output_dir.mkdir(exist_ok=True)
        
        self.logger.info("ProcessorComparison initialized")
    
    def run_old_processor(self, config_path: str) -> OldRegistrationProcessor:
        """Run the old registration processor."""
        self.logger.info("Running old registration processor...")
        
        # Load old config
        old_config = load_config(config_path)
        
        # Update output directory to avoid conflicts
        old_config["output_dir"] = str(self.old_output_dir)
        
        # Create and run old processor (direct instantiation)
        old_processor = OldRegistrationProcessor(old_config)
        old_processor.process()
        
        self.logger.info("Old processor completed successfully")
        return old_processor
    
    def run_new_processor(self, config_path: str):
        """Run the new registration processor via pipeline."""
        self.logger.info("Running new registration processor via pipeline...")
        
        # Load new config
        new_config = load_config(config_path)
        
        # Update output directory to avoid conflicts
        new_config["output_dir"] = str(self.new_output_dir)
        
        # Run new processor via pipeline to ensure vet-specific functions are applied
        new_processor = run_registration_processing(new_config)
        
        # Check if vet-specific functions were applied
        if hasattr(new_processor, '_vet_specific_active') and new_processor._vet_specific_active:
            self.logger.info("Vet-specific functions were successfully applied to new processor")
        else:
            self.logger.warning("Vet-specific functions were NOT applied to new processor")
        
        self.logger.info("New processor completed successfully")
        return new_processor
    
    def compare_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame, name: str) -> Dict[str, Any]:
        """Compare two dataframes and return comparison results."""
        self.logger.info(f"Comparing dataframes: {name}")
        
        comparison = {
            "name": name,
            "identical": True,
            "differences": [],
            "old_shape": df1.shape,
            "new_shape": df2.shape,
            "old_columns": sorted(df1.columns.tolist()),
            "new_columns": sorted(df2.columns.tolist()),
            "old_dtypes": df1.dtypes.to_dict(),
            "new_dtypes": df2.dtypes.to_dict(),
        }
        
        # Compare shapes
        if df1.shape != df2.shape:
            comparison["identical"] = False
            comparison["differences"].append(f"Shape mismatch: {df1.shape} vs {df2.shape}")
            self.logger.warning(f"{name}: Shape mismatch - {df1.shape} vs {df2.shape}")
        
        # Compare columns
        if set(df1.columns) != set(df2.columns):
            comparison["identical"] = False
            missing_in_new = set(df1.columns) - set(df2.columns)
            missing_in_old = set(df2.columns) - set(df1.columns)
            if missing_in_new:
                comparison["differences"].append(f"Columns missing in new: {missing_in_new}")
            if missing_in_old:
                comparison["differences"].append(f"Columns missing in old: {missing_in_old}")
            self.logger.warning(f"{name}: Column mismatch")
        
        # Compare data for common columns
        common_columns = set(df1.columns) & set(df2.columns)
        if common_columns and df1.shape[0] == df2.shape[0]:
            try:
                # Sort both dataframes by all common columns for consistent comparison
                sort_columns = [col for col in common_columns if col in df1.columns and col in df2.columns]
                if sort_columns:
                    df1_sorted = df1[list(common_columns)].sort_values(sort_columns).reset_index(drop=True)
                    df2_sorted = df2[list(common_columns)].sort_values(sort_columns).reset_index(drop=True)
                    
                    # Compare each column
                    for col in common_columns:
                        if not df1_sorted[col].equals(df2_sorted[col]):
                            comparison["identical"] = False
                            
                            # Count differences
                            if df1_sorted[col].dtype == 'object':
                                # For string columns, compare directly
                                diff_count = (df1_sorted[col] != df2_sorted[col]).sum()
                            else:
                                # For numeric columns, handle NaN values
                                diff_count = ((df1_sorted[col] != df2_sorted[col]) & 
                                            ~(pd.isna(df1_sorted[col]) & pd.isna(df2_sorted[col]))).sum()
                            
                            comparison["differences"].append(f"Column '{col}': {diff_count} different values")
                            self.logger.warning(f"{name}: Column '{col}' has {diff_count} different values")
                            
                            # Sample some differences for analysis
                            if diff_count > 0:
                                mask = df1_sorted[col] != df2_sorted[col]
                                if df1_sorted[col].dtype == 'object':
                                    # Don't include NaN comparisons for objects
                                    mask = mask & pd.notna(df1_sorted[col]) & pd.notna(df2_sorted[col])
                                
                                sample_diffs = []
                                for idx in df1_sorted[mask].head(3).index:
                                    old_val = df1_sorted.loc[idx, col]
                                    new_val = df2_sorted.loc[idx, col]
                                    sample_diffs.append(f"Row {idx}: '{old_val}' vs '{new_val}'")
                                
                                comparison["differences"].append(f"Sample differences in '{col}': {sample_diffs}")
                
            except Exception as e:
                comparison["differences"].append(f"Error comparing data: {str(e)}")
                self.logger.error(f"{name}: Error comparing data - {str(e)}")
        
        return comparison
    
    def compare_files(self, old_dir: Path, new_dir: Path) -> Dict[str, Any]:
        """Compare files in both output directories."""
        self.logger.info("Comparing output files...")
        
        # Get all CSV files from both directories
        old_csvs = {f.name: f for f in old_dir.glob("**/*.csv")}
        new_csvs = {f.name: f for f in new_dir.glob("**/*.csv")}
        
        # Get all JSON files from both directories
        old_jsons = {f.name: f for f in old_dir.glob("**/*.json")}
        new_jsons = {f.name: f for f in new_dir.glob("**/*.json")}
        
        file_comparisons = {
            "csv_files": {},
            "json_files": {},
            "missing_files": {
                "old_missing": [],
                "new_missing": []
            }
        }
        
        # Compare CSV files
        all_csv_files = set(old_csvs.keys()) | set(new_csvs.keys())
        for filename in all_csv_files:
            if filename not in old_csvs:
                file_comparisons["missing_files"]["old_missing"].append(filename)
                continue
            if filename not in new_csvs:
                file_comparisons["missing_files"]["new_missing"].append(filename)
                continue
            
            try:
                old_df = pd.read_csv(old_csvs[filename])
                new_df = pd.read_csv(new_csvs[filename])
                file_comparisons["csv_files"][filename] = self.compare_dataframes(
                    old_df, new_df, filename
                )
            except Exception as e:
                self.logger.error(f"Error comparing {filename}: {str(e)}")
                file_comparisons["csv_files"][filename] = {
                    "name": filename,
                    "identical": False,
                    "differences": [f"Error reading files: {str(e)}"]
                }
        
        # Compare JSON files
        all_json_files = set(old_jsons.keys()) | set(new_jsons.keys())
        for filename in all_json_files:
            if filename not in old_jsons:
                file_comparisons["missing_files"]["old_missing"].append(filename)
                continue
            if filename not in new_jsons:
                file_comparisons["missing_files"]["new_missing"].append(filename)
                continue
            
            try:
                with open(old_jsons[filename], 'r', encoding='utf-8') as f:
                    old_data = json.load(f)
                with open(new_jsons[filename], 'r', encoding='utf-8') as f:
                    new_data = json.load(f)
                
                # Compare JSON content
                old_str = json.dumps(old_data, sort_keys=True)
                new_str = json.dumps(new_data, sort_keys=True)
                
                file_comparisons["json_files"][filename] = {
                    "name": filename,
                    "identical": old_str == new_str,
                    "old_size": len(old_data) if isinstance(old_data, (list, dict)) else 1,
                    "new_size": len(new_data) if isinstance(new_data, (list, dict)) else 1,
                    "differences": [] if old_str == new_str else ["Content differs"]
                }
                
            except Exception as e:
                self.logger.error(f"Error comparing {filename}: {str(e)}")
                file_comparisons["json_files"][filename] = {
                    "name": filename,
                    "identical": False,
                    "differences": [f"Error reading files: {str(e)}"]
                }
        
        return file_comparisons
    
    def compare_processor_attributes(self, old_processor: OldRegistrationProcessor, 
                                   new_processor) -> Dict[str, Any]:
        """Compare key attributes of both processors."""
        self.logger.info("Comparing processor attributes...")
        
        attr_comparisons = {}
        
        # List of dataframe attributes to compare
        dataframe_attrs = [
            'df_reg_demo_this',
            'df_reg_demo_last_bva',
            'df_reg_demo_last_lva'
        ]
        
        # Both processors should have the same attribute names now
        for attr in dataframe_attrs:
            if hasattr(old_processor, attr) and hasattr(new_processor, attr):
                old_df = getattr(old_processor, attr)
                new_df = getattr(new_processor, attr)
                
                if isinstance(old_df, pd.DataFrame) and isinstance(new_df, pd.DataFrame):
                    attr_comparisons[attr] = self.compare_dataframes(
                        old_df, new_df, f"processor_attr_{attr}"
                    )
                else:
                    attr_comparisons[attr] = {
                        "name": attr,
                        "identical": False,
                        "differences": ["Attribute is not a DataFrame"]
                    }
            else:
                missing_attrs = []
                if not hasattr(old_processor, attr):
                    missing_attrs.append(f"old_processor.{attr}")
                if not hasattr(new_processor, attr):
                    missing_attrs.append(f"new_processor.{attr}")
                
                attr_comparisons[attr] = {
                    "name": attr,
                    "identical": False,
                    "differences": [f"Missing attributes: {missing_attrs}"]
                }
        
        return attr_comparisons
    
    def generate_report(self, file_comparisons: Dict[str, Any], 
                       attr_comparisons: Dict[str, Any]) -> str:
        """Generate a detailed comparison report."""
        self.logger.info("Generating comparison report...")
        
        report = f"""
# Processor Comparison Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
"""
        
        # Count identical vs different files
        csv_identical = sum(1 for comp in file_comparisons["csv_files"].values() if comp["identical"])
        csv_total = len(file_comparisons["csv_files"])
        json_identical = sum(1 for comp in file_comparisons["json_files"].values() if comp["identical"])
        json_total = len(file_comparisons["json_files"])
        attr_identical = sum(1 for comp in attr_comparisons.values() if comp["identical"])
        attr_total = len(attr_comparisons)
        
        report += f"""
- CSV Files: {csv_identical}/{csv_total} identical
- JSON Files: {json_identical}/{json_total} identical
- Processor Attributes: {attr_identical}/{attr_total} identical
"""
        
        # Missing files
        if file_comparisons["missing_files"]["old_missing"] or file_comparisons["missing_files"]["new_missing"]:
            report += "\n## Missing Files\n"
            if file_comparisons["missing_files"]["old_missing"]:
                report += f"- Missing in old processor: {file_comparisons['missing_files']['old_missing']}\n"
            if file_comparisons["missing_files"]["new_missing"]:
                report += f"- Missing in new processor: {file_comparisons['missing_files']['new_missing']}\n"
        
        # CSV file details
        report += "\n## CSV File Comparisons\n"
        for filename, comp in file_comparisons["csv_files"].items():
            status = "IDENTICAL" if comp["identical"] else "DIFFERENT"
            report += f"\n### {filename} {status}\n"
            report += f"- Old shape: {comp.get('old_shape', 'N/A')}\n"
            report += f"- New shape: {comp.get('new_shape', 'N/A')}\n"
            
            if not comp["identical"]:
                report += "- Differences:\n"
                for diff in comp["differences"]:
                    report += f"  - {diff}\n"
        
        # JSON file details
        if file_comparisons["json_files"]:
            report += "\n## JSON File Comparisons\n"
            for filename, comp in file_comparisons["json_files"].items():
                status = "IDENTICAL" if comp["identical"] else "DIFFERENT"
                report += f"\n### {filename} {status}\n"
                report += f"- Old size: {comp.get('old_size', 'N/A')}\n"
                report += f"- New size: {comp.get('new_size', 'N/A')}\n"
                
                if not comp["identical"]:
                    report += "- Differences:\n"
                    for diff in comp["differences"]:
                        report += f"  - {diff}\n"
        
        # Processor attribute details
        report += "\n## Processor Attribute Comparisons\n"
        for attr_name, comp in attr_comparisons.items():
            status = "IDENTICAL" if comp["identical"] else "DIFFERENT"
            report += f"\n### {attr_name} {status}\n"
            report += f"- Old shape: {comp.get('old_shape', 'N/A')}\n"
            report += f"- New shape: {comp.get('new_shape', 'N/A')}\n"
            
            if not comp["identical"]:
                report += "- Differences:\n"
                for diff in comp["differences"]:
                    report += f"  - {diff}\n"
        
        return report
    
    def run_comparison(self, old_config_path: str = "config/config.yaml", 
                      new_config_path: str = "config/config_vet.yaml") -> bool:
        """Run the complete comparison between old and new processors."""
        self.logger.info("Starting processor comparison...")
        
        try:
            # Run both processors
            old_processor = self.run_old_processor(old_config_path)
            new_processor = self.run_new_processor(new_config_path)
            
            # Compare output files
            file_comparisons = self.compare_files(self.old_output_dir, self.new_output_dir)
            
            # Compare processor attributes
            attr_comparisons = self.compare_processor_attributes(old_processor, new_processor)
            
            # Generate report
            report = self.generate_report(file_comparisons, attr_comparisons)
            
            # Save report with UTF-8 encoding
            report_path = self.comparison_dir / f"comparison_report_{self.timestamp}.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # Save detailed comparison data with UTF-8 encoding
            comparison_data = {
                "timestamp": self.timestamp,
                "file_comparisons": file_comparisons,
                "attr_comparisons": attr_comparisons,
                "summary": {
                    "csv_files_identical": sum(1 for comp in file_comparisons["csv_files"].values() if comp["identical"]),
                    "csv_files_total": len(file_comparisons["csv_files"]),
                    "json_files_identical": sum(1 for comp in file_comparisons["json_files"].values() if comp["identical"]),
                    "json_files_total": len(file_comparisons["json_files"]),
                    "attr_identical": sum(1 for comp in attr_comparisons.values() if comp["identical"]),
                    "attr_total": len(attr_comparisons)
                }
            }
            
            comparison_data_path = self.comparison_dir / f"comparison_data_{self.timestamp}.json"
            with open(comparison_data_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, indent=2, default=str, ensure_ascii=False)
            
            # Print summary
            print(f"\n{'='*60}")
            print("PROCESSOR COMPARISON SUMMARY")
            print(f"{'='*60}")
            print(f"CSV Files: {comparison_data['summary']['csv_files_identical']}/{comparison_data['summary']['csv_files_total']} identical")
            print(f"JSON Files: {comparison_data['summary']['json_files_identical']}/{comparison_data['summary']['json_files_total']} identical")
            print(f"Processor Attributes: {comparison_data['summary']['attr_identical']}/{comparison_data['summary']['attr_total']} identical")
            print(f"\nDetailed report saved to: {report_path}")
            print(f"Comparison data saved to: {comparison_data_path}")
            
            # Check if vet-specific functions were applied
            if hasattr(new_processor, '_vet_specific_active') and new_processor._vet_specific_active:
                print("✅ Vet-specific functions were applied to new processor")
            else:
                print("⚠️  Vet-specific functions were NOT applied to new processor")
            
            # Return True if everything is identical
            all_identical = (
                comparison_data['summary']['csv_files_identical'] == comparison_data['summary']['csv_files_total'] and
                comparison_data['summary']['json_files_identical'] == comparison_data['summary']['json_files_total'] and
                comparison_data['summary']['attr_identical'] == comparison_data['summary']['attr_total'] and
                not file_comparisons["missing_files"]["old_missing"] and
                not file_comparisons["missing_files"]["new_missing"]
            )
            
            if all_identical:
                print("SUCCESS: All outputs are identical!")
                self.logger.info("Comparison completed successfully - all outputs identical")
            else:
                print("DIFFERENCES FOUND: Check the report for details")
                self.logger.warning("Comparison completed - differences found")
            
            return all_identical
            
        except Exception as e:
            self.logger.error(f"Error during comparison: {str(e)}", exc_info=True)
            print(f"ERROR: {str(e)}")
            return False


def main():
    """Main function to run the processor comparison."""
    print("Starting Processor Comparison...")
    print("This will compare outputs between old and new registration processors.")
    print("The new processor will be run via pipeline to ensure vet-specific functions are applied.")
    print()
    
    # Create comparison instance
    comparison = ProcessorComparison()
    
    # Run comparison
    success = comparison.run_comparison()
    
    if success:
        print("\nProcessors produce identical outputs!")
        sys.exit(0)
    else:
        print("\nProcessors produce different outputs. Check the report for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()