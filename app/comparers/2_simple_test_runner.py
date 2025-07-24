#!/usr/bin/env python3
"""
Simple Test Runner for Scan Processors

This script runs both scan processors and does a quick comparison of key outputs.
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
    from old_scan_processor import ScanProcessor as OldScanProcessor
    from scan_processor import ScanProcessor as NewScanProcessor
    from utils.config_utils import load_config
    from utils.logging_utils import setup_logging
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nMake sure you have:")
    print("1. old_scan_processor.py")
    print("2. scan_processor.py")
    print("3. config/config.yaml")
    print("4. config/config_vet.yaml")
    sys.exit(1)


def compare_key_outputs(old_processor, new_processor):
    """Compare the key dataframes from both processors."""
    print("\n📊 Comparing Key Dataframes...")
    
    comparisons = []
    
    # Define the key dataframes to compare - both processors should have the same variable names
    key_dfs = [
        ('seminars_scans_past_enhanced_bva', 'seminars_scans_past_enhanced_bva'),
        ('seminars_scans_past_enhanced_lva', 'seminars_scans_past_enhanced_lva'),
        ('enhanced_seminars_df_bva', 'enhanced_seminars_df_bva'),
        ('enhanced_seminars_df_lva', 'enhanced_seminars_df_lva')
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
            sort_cols = [col for col in old_df.columns if col in ['Badge Id', 'Short Name', 'Seminar Name']]
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
    
    return all_identical


def run_test():
    """Run the test comparison."""
    print("🚀 Starting Scan Processor Comparison Test")
    print("=" * 60)
    print("ℹ️  This test assumes you've already run:")
    print("   python main.py --config config/config.yaml --only-steps 1")
    print("   python main.py --config config/config_vet.yaml --only-steps 1")
    print()
    
    # Setup logging
    logger = setup_logging(log_file="logs/simple_scan_test.log")
    
    try:
        # Load configurations
        print("📁 Loading configurations...")
        old_config = load_config("config/config.yaml")
        new_config = load_config("config/config_vet.yaml")
        
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
        
        # Create comparison directories within the production directories
        old_comparison_dir = Path(production_old_output) / "scan_comparison" / "old_output"
        new_comparison_dir = Path(production_new_output) / "scan_comparison" / "new_output"
        old_comparison_dir.mkdir(parents=True, exist_ok=True)
        new_comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure processors to read from production location but write to comparison dirs
        old_config_copy = old_config.copy()
        new_config_copy = new_config.copy()
        
        # The scan processor will read registration files from the original output_dir
        # but write its own outputs to the comparison directories
        old_config_copy["scan_output_dir"] = str(old_comparison_dir)
        new_config_copy["scan_output_dir"] = str(new_comparison_dir)
        
        # Run processors with modified configs
        success = run_processors_and_compare(old_config_copy, new_config_copy, 
                                           old_comparison_dir, new_comparison_dir,
                                           production_old_output, production_new_output)
        return success
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

def run_processors_and_compare(old_config, new_config, old_output, new_output, 
                              prod_old_dir, prod_new_dir):
    """Run both scan processors and compare their outputs."""
    try:
        # Create custom scan processors that know where to find registration files
        print("\n🔄 Running OLD scan processor...")
        old_processor = OldScanProcessorWithCustomPaths(old_config, prod_old_dir)
        old_processor.process()
        print("✅ Old processor completed")
        
        print("\n🔄 Running NEW scan processor...")
        new_processor = NewScanProcessorWithCustomPaths(new_config, prod_new_dir)
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
            print("✅ The new scan processor produces the same results as the old one.")
            return True
        else:
            print("❌ FAILURE: Outputs are DIFFERENT!")
            print("⚠️  The new scan processor produces different results.")
            if not dataframes_identical:
                print("   - Key dataframes differ")
            if not files_identical:
                print("   - Output files differ")
            return False
            
    except Exception as e:
        print(f"❌ Error running processors: {e}")
        raise

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


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
            
        
    # Compare dataframes
    dataframes_identical = compare_key_outputs(old_processor, new_processor)
    
    # Compare output files
    files_identical = compare_output_files(old_output, new_output)
    
    # Final result
    print("\n" + "=" * 60)
    if dataframes_identical and files_identical:
        print("🎉 SUCCESS: All outputs are IDENTICAL!")
        print("✅ The new scan processor produces the same results as the old one.")
        
    else:
        print("❌ FAILURE: Outputs are DIFFERENT!")
        print("⚠️  The new scan processor produces different results.")
        if not dataframes_identical:
            print("   - Key dataframes differ")
        if not files_identical:
            print("   - Output files differ")
        
        


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)