#!/usr/bin/env python3
"""
Scan Processor Comparison Script

This script compares the outputs of the old scan processor (old_scan_processor.py)
and the new generic processor (scan_processor.py) to ensure they produce identical results.

Usage:
    python compare_scan_processors.py
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
    from old_scan_processor import ScanProcessor as OldScanProcessor
    from scan_processor import ScanProcessor as NewScanProcessor
    from utils.config_utils import load_config
    from utils.logging_utils import setup_logging
except ImportError as e:
    print(f"Error importing processors: {e}")
    print("Make sure you have:")
    print("1. old_scan_processor.py (renamed from original)")
    print("2. scan_processor.py (new generic version)")
    print("3. Both config files in the config/ directory")
    sys.exit(1)


class ScanProcessorComparison:
    """Compare outputs between old and new scan processors."""
    
    def __init__(self):
        self.logger = setup_logging(log_file="logs/scan_processor_comparison.log")
        self.comparison_results = {}
        self.differences_found = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comparison output directory
        self.comparison_dir = Path("comparison_output_scan")
        self.comparison_dir.mkdir(exist_ok=True)
        
        # Create separate output directories for each processor
        self.old_output_dir = self.comparison_dir / "old_processor_output"
        self.new_output_dir = self.comparison_dir / "new_processor_output"
        self.old_output_dir.mkdir(exist_ok=True)
        self.new_output_dir.mkdir(exist_ok=True)
        
        self.logger.info("ScanProcessorComparison initialized")
    
    def run_old_processor_with_config(self, config: Dict, registration_output_dir: str) -> OldScanProcessor:
        """Run the old scan processor with provided config."""
        self.logger.info("Running old scan processor...")
        old_processor = OldScanProcessorWithCustomPaths(config, registration_output_dir)
        old_processor.process()
        self.logger.info("Old scan processor completed successfully")
        return old_processor
    
    def run_new_processor_with_config(self, config: Dict, registration_output_dir: str) -> NewScanProcessor:
        """Run the new scan processor with provided config."""
        self.logger.info("Running new scan processor...")
        new_processor = NewScanProcessorWithCustomPaths(config, registration_output_dir)
        new_processor.process()
        self.logger.info("New scan processor completed successfully")
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
        
        file_comparisons = {
            "csv_files": {},
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
        
        return file_comparisons
    
    def compare_processor_attributes(self, old_processor: OldScanProcessor, 
                                   new_processor: NewScanProcessor) -> Dict[str, Any]:
        """Compare key attributes of both processors."""
        self.logger.info("Comparing processor attributes...")
        
        attr_comparisons = {}
        
        # List of dataframe attributes to compare
        dataframe_attrs = [
            'seminars_scans_past_enhanced_bva',
            'seminars_scans_past_enhanced_lva',
            'enhanced_seminars_df_bva',
            'enhanced_seminars_df_lva'
        ]
        
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
# Scan Processor Comparison Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
"""
        
        # Count identical vs different files
        csv_identical = sum(1 for comp in file_comparisons["csv_files"].values() if comp["identical"])
        csv_total = len(file_comparisons["csv_files"])
        attr_identical = sum(1 for comp in attr_comparisons.values() if comp["identical"])
        attr_total = len(attr_comparisons)
        
        report += f"""
- CSV Files: {csv_identical}/{csv_total} identical
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
        self.logger.info("Starting scan processor comparison...")
        
        print("ℹ️  This comparison assumes you've already run:")
        print("   python main.py --config config/config.yaml --only-steps 1")
        print("   python main.py --config config/config_vet.yaml --only-steps 1")
        print()
        
        try:
            # Load configurations
            old_config = load_config(old_config_path)
            new_config = load_config(new_config_path)
            
            # Get the production output directories where step 1 files exist
            production_old_output = old_config.get("output_dir", "output")
            production_new_output = new_config.get("output_dir", "data/bva")
            
            print(f"📂 Production old output directory: {production_old_output}")
            print(f"📂 Production new output directory: {production_new_output}")
            
            # Check if required files exist
            required_files_old = [
                f"{production_old_output}/output/df_reg_demo_last_bva.csv",
                f"{production_old_output}/output/df_reg_demo_last_lva.csv",
                f"{production_old_output}/output/Registration_data_with_demographicdata_bva_last.csv",
                f"{production_old_output}/output/Registration_data_with_demographicdata_lva_last.csv"
            ]
            
            required_files_new = [
                f"{production_new_output}/output/df_reg_demo_last_bva.csv",
                f"{production_new_output}/output/df_reg_demo_last_lva.csv",
                f"{production_new_output}/output/Registration_data_with_demographicdata_bva_last.csv",
                f"{production_new_output}/output/Registration_data_with_demographicdata_lva_last.csv"
            ]
            
            # Check if files exist
            missing_files = []
            for file_path in required_files_old + required_files_new:
                if not Path(file_path).exists():
                    missing_files.append(file_path)
            
            if missing_files:
                print("❌ Missing required files from registration processor:")
                for file_path in missing_files:
                    print(f"   - {file_path}")
                print("\n💡 Please run the registration processor first:")
                print("   python main.py --config config/config.yaml --only-steps 1")
                print("   python main.py --config config/config_vet.yaml --only-steps 1")
                return False
            
            print("✅ All required registration files found!")
            
            # Update configs to use comparison directories for output
            old_config["output_dir"] = str(self.old_output_dir)
            new_config["output_dir"] = str(self.new_output_dir)
            
            # Run both processors with custom paths
            old_processor = self.run_old_processor_with_config(old_config, production_old_output)
            new_processor = self.run_new_processor_with_config(new_config, production_new_output)
            
            # Compare output files
            file_comparisons = self.compare_files(self.old_output_dir, self.new_output_dir)
            
            # Compare processor attributes
            attr_comparisons = self.compare_processor_attributes(old_processor, new_processor)
            
            # Generate report
            report = self.generate_report(file_comparisons, attr_comparisons)
            
            # Save report with UTF-8 encoding
            report_path = self.comparison_dir / f"scan_comparison_report_{self.timestamp}.md"
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
                    "attr_identical": sum(1 for comp in attr_comparisons.values() if comp["identical"]),
                    "attr_total": len(attr_comparisons)
                }
            }
            
            comparison_data_path = self.comparison_dir / f"scan_comparison_data_{self.timestamp}.json"
            with open(comparison_data_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, indent=2, default=str, ensure_ascii=False)
            
            # Print summary
            print(f"\n{'='*60}")
            print("SCAN PROCESSOR COMPARISON SUMMARY")
            print(f"{'='*60}")
            print(f"CSV Files: {comparison_data['summary']['csv_files_identical']}/{comparison_data['summary']['csv_files_total']} identical")
            print(f"Processor Attributes: {comparison_data['summary']['attr_identical']}/{comparison_data['summary']['attr_total']} identical")
            print(f"\nDetailed report saved to: {report_path}")
            print(f"Comparison data saved to: {comparison_data_path}")
            
            # Return True if everything is identical
            all_identical = (
                comparison_data['summary']['csv_files_identical'] == comparison_data['summary']['csv_files_total'] and
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


# Custom processor classes that know where to find registration files
class OldScanProcessorWithCustomPaths(OldScanProcessor):
    def __init__(self, config, registration_output_dir):
        super().__init__(config)
        self.registration_output_dir = registration_output_dir
    
    def load_registration_data(self):
        """Load processed registration data from the production location."""
        try:
            # Load registration data with demographics from production location
            reg_data_last_bva_path = os.path.join(
                self.registration_output_dir, "output", "df_reg_demo_last_bva.csv"
            )
            reg_data_last_lva_path = os.path.join(
                self.registration_output_dir, "output", "df_reg_demo_last_lva.csv"
            )

            detailed_reg_data_last_bva_path = os.path.join(
                self.registration_output_dir,
                "output",
                "Registration_data_with_demographicdata_bva_last.csv",
            )
            detailed_reg_data_last_lva_path = os.path.join(
                self.registration_output_dir,
                "output",
                "Registration_data_with_demographicdata_lva_last.csv",
            )

            # Read registration data
            self.reg_data_last_bva = pd.read_csv(reg_data_last_bva_path)
            self.reg_data_last_lva = pd.read_csv(reg_data_last_lva_path)

            self.df_reg_24_wdemo_data_bva = pd.read_csv(detailed_reg_data_last_bva_path)
            self.df_reg_24_wdemo_data_lva = pd.read_csv(detailed_reg_data_last_lva_path)

            self.logger.info(
                f"Loaded registration data: {len(self.reg_data_last_bva)} BVA records, "
                f"{len(self.reg_data_last_lva)} LVA records"
            )

            self.logger.info(
                f"Loaded detailed registration data: {len(self.df_reg_24_wdemo_data_bva)} BVA records, "
                f"{len(self.df_reg_24_wdemo_data_lva)} LVA records"
            )

        except Exception as e:
            self.logger.error(f"Error loading registration data: {e}", exc_info=True)
            raise

class NewScanProcessorWithCustomPaths(NewScanProcessor):
    def __init__(self, config, registration_output_dir):
        super().__init__(config)
        self.registration_output_dir = registration_output_dir
    
    def load_registration_data(self):
        """Load processed registration data from the production location."""
        try:
            # Get output file names from config with fallback to default names
            combined_output_files = self.config.get("output_files", {}).get("combined_demographic_registration", {})
            
            # Load registration data with demographics from production location
            reg_data_last_main_path = os.path.join(
                self.registration_output_dir, "output", 
                combined_output_files.get("last_year_main", "df_reg_demo_last_bva.csv")
            )
            reg_data_last_secondary_path = os.path.join(
                self.registration_output_dir, "output", 
                combined_output_files.get("last_year_secondary", "df_reg_demo_last_lva.csv")
            )

            # Get registration with demographic file names
            reg_with_demo_files = self.config.get("output_files", {}).get("registration_with_demographic", {})
            
            detailed_reg_data_last_main_path = os.path.join(
                self.registration_output_dir,
                "output",
                reg_with_demo_files.get("last_year_main", "Registration_data_with_demographicdata_bva_last.csv"),
            )
            detailed_reg_data_last_secondary_path = os.path.join(
                self.registration_output_dir,
                "output",
                reg_with_demo_files.get("last_year_secondary", "Registration_data_with_demographicdata_lva_last.csv"),
            )

            # Read registration data
            self.reg_data_last_main = pd.read_csv(reg_data_last_main_path)
            self.reg_data_last_secondary = pd.read_csv(reg_data_last_secondary_path)

            self.df_reg_24_wdemo_data_main = pd.read_csv(detailed_reg_data_last_main_path)
            self.df_reg_24_wdemo_data_secondary = pd.read_csv(detailed_reg_data_last_secondary_path)
            
            # Keep old names for backward compatibility
            self.reg_data_last_bva = self.reg_data_last_main
            self.reg_data_last_lva = self.reg_data_last_secondary
            self.df_reg_24_wdemo_data_bva = self.df_reg_24_wdemo_data_main
            self.df_reg_24_wdemo_data_lva = self.df_reg_24_wdemo_data_secondary

            self.logger.info(
                f"Loaded registration data: {len(self.reg_data_last_main)} {self.main_event_name} records, "
                f"{len(self.reg_data_last_secondary)} {self.secondary_event_name} records"
            )

            self.logger.info(
                f"Loaded detailed registration data: {len(self.df_reg_24_wdemo_data_main)} {self.main_event_name} records, "
                f"{len(self.df_reg_24_wdemo_data_secondary)} {self.secondary_event_name} records"
            )

        except Exception as e:
            self.logger.error(f"Error loading registration data: {e}", exc_info=True)
            raise


def main():
    """Main function to run the scan processor comparison."""
    print("Starting Scan Processor Comparison...")
    print("This will compare outputs between old and new scan processors.")
    print()
    
    # Create comparison instance
    comparison = ScanProcessorComparison()
    
    # Run comparison
    success = comparison.run_comparison()
    
    if success:
        print("\nScan processors produce identical outputs!")
        sys.exit(0)
    else:
        print("\nScan processors produce different outputs. Check the report for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()