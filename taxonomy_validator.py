#!/usr/bin/env python3
"""
Taxonomy Validator Module
Validates company classifications against the official taxonomy.
"""

import csv
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set
from datetime import datetime
import json


class TaxonomyValidator:
    """Validates company classifications against official taxonomy."""
    
    def __init__(self, taxonomy_path: str = None):
        """Initialize with taxonomy file path."""
        if taxonomy_path is None:
            # Try multiple possible paths for the taxonomy file
            possible_paths = [
                Path("client/mcp_servers/company_tagging/categories/classes.csv"),
                Path("classes.csv"),
                Path(__file__).parent / "client" / "mcp_servers" / "company_tagging" / "categories" / "classes.csv",
            ]
            
            for path in possible_paths:
                if path.exists():
                    taxonomy_path = path
                    break
            
            if taxonomy_path is None:
                raise FileNotFoundError(
                    f"Could not find taxonomy file. Tried paths: {[str(p) for p in possible_paths]}"
                )
        
        self.taxonomy_path = Path(taxonomy_path)
        self.valid_pairs = self.load_taxonomy()
        self.validation_stats = {
            "total_rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "blank_rows": 0,
            "partial_blank_rows": 0,
            "invalid_pairs": []
        }
    
    def load_taxonomy(self) -> Set[Tuple[str, str]]:
        """Load valid taxonomy pairs from CSV."""
        valid_pairs = set()
        
        if not self.taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {self.taxonomy_path}")
        
        try:
            with open(self.taxonomy_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Debug: check headers
                if reader.fieldnames:
                    print(f"📋 Taxonomy CSV headers: {reader.fieldnames}")
                
                for row in reader:
                    industry = row.get('Industry', '').strip()
                    product = row.get('Product', '').strip()
                    if industry and product:
                        valid_pairs.add((industry, product))
            
            print(f"✅ Loaded {len(valid_pairs)} valid taxonomy pairs")
            
            # Debug: show some examples
            if valid_pairs:
                print("📋 Example taxonomy pairs:")
                for i, (ind, prod) in enumerate(sorted(valid_pairs)[:5]):
                    print(f"   {i+1}. {ind} | {prod}")
                
                # Check if specific pairs exist
                test_pairs = [
                    ("Software & Services", "Other Services"),
                    ("Platforms & Software", "Big Data & Analytics Tools"),
                    ("Data Management", "Data Analytics & Integration")
                ]
                print("\n🔍 Checking existence of pairs from your output:")
                for ind, prod in test_pairs:
                    exists = (ind, prod) in valid_pairs
                    print(f"   {ind} | {prod}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
            
            return valid_pairs
            
        except Exception as e:
            raise ValueError(f"Error loading taxonomy: {e}")
    
    def is_valid_pair(self, industry: str, product: str) -> bool:
        """Check if an industry-product pair exists in taxonomy."""
        if not industry or not product:
            return True  # Empty pairs are considered valid (not flagged)
        
        # Ensure we're working with strings
        industry = str(industry).strip()
        product = str(product).strip()
        
        # Check exact match
        return (industry, product) in self.valid_pairs
    
    def validate_row(self, row: Dict[str, str]) -> Tuple[bool, List[str], bool]:
        """
        Validate a single row of company data.
        
        Returns:
            - is_valid: True if all pairs are valid
            - invalid_pairs: List of invalid pair descriptions
            - is_all_blank: True if all tech fields are blank
        """
        invalid_pairs = []
        blank_count = 0
        total_pairs = 4
        
        # Check each of the 4 pairs
        for i in range(1, 5):
            industry_col = f"Tech Industry {i}"
            product_col = f"Tech Product {i}"
            
            # Handle None values and convert to string
            industry_val = row.get(industry_col)
            product_val = row.get(product_col)
            
            # Convert None to empty string and strip
            industry = str(industry_val).strip() if industry_val is not None else ''
            product = str(product_val).strip() if product_val is not None else ''
            
            # Check if both are blank
            if not industry and not product:
                blank_count += 1
                continue
            
            # Check if only one is blank (invalid state)
            if bool(industry) != bool(product):
                invalid_pairs.append(f"Pair {i}: Incomplete (Industry='{industry}', Product='{product}')")
                continue
            
            # Check if pair exists in taxonomy
            if not self.is_valid_pair(industry, product):
                invalid_pairs.append(f"Pair {i}: Not in taxonomy (Industry='{industry}', Product='{product}')")
        
        is_all_blank = blank_count == total_pairs
        is_valid = len(invalid_pairs) == 0 and not is_all_blank
        
        return is_valid, invalid_pairs, is_all_blank
    
    def validate_csv(self, csv_path: str) -> Dict[str, any]:
        """
        Validate entire CSV file against taxonomy.
        
        Returns dictionary with:
            - flagged_rows: List of row indices that need re-evaluation
            - validation_report: Detailed validation statistics
            - invalid_details: Details of invalid rows
        """
        flagged_rows = []
        invalid_details = []
        
        self.validation_stats = {
            "total_rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "blank_rows": 0,
            "partial_blank_rows": 0,
            "invalid_pairs": []
        }
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Check if required columns exist
                if reader.fieldnames:
                    required_cols = [f"Tech Industry {i}" for i in range(1, 5)] + \
                                  [f"Tech Product {i}" for i in range(1, 5)]
                    missing_cols = [col for col in required_cols if col not in reader.fieldnames]
                    if missing_cols:
                        print(f"⚠️  Warning: Missing columns: {missing_cols}")
                        print(f"   Available columns: {reader.fieldnames}")
                
                for idx, row in enumerate(reader):
                    self.validation_stats["total_rows"] += 1
                    
                    try:
                        is_valid, invalid_pairs, is_all_blank = self.validate_row(row)
                        
                        if is_all_blank:
                            self.validation_stats["blank_rows"] += 1
                            flagged_rows.append(idx)
                            invalid_details.append({
                                "row_index": idx,
                                "company_name": row.get('Company Name', ''),
                                "reason": "All tech fields are blank"
                            })
                        elif not is_valid:
                            self.validation_stats["invalid_rows"] += 1
                            flagged_rows.append(idx)
                            invalid_details.append({
                                "row_index": idx,
                                "company_name": row.get('Company Name', ''),
                                "reason": "; ".join(invalid_pairs)
                            })
                            
                            # Track unique invalid pairs
                            for pair_desc in invalid_pairs:
                                if "Not in taxonomy" in pair_desc:
                                    self.validation_stats["invalid_pairs"].append(pair_desc)
                        else:
                            self.validation_stats["valid_rows"] += 1
                            
                    except Exception as row_error:
                        print(f"⚠️  Error processing row {idx}: {row_error}")
                        print(f"   Row data: {row}")
                        raise
            
            return {
                "flagged_rows": flagged_rows,
                "validation_report": self.validation_stats,
                "invalid_details": invalid_details
            }
            
        except Exception as e:
            raise ValueError(f"Error validating CSV: {e}")
    
    def create_backup(self, file_path: str) -> str:
        """Create timestamped backup of file."""
        file_path = Path(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.parent / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
        
        try:
            shutil.copy2(file_path, backup_path)
            print(f"📋 Created backup: {backup_path}")
            return str(backup_path)
        except Exception as e:
            raise ValueError(f"Error creating backup: {e}")
    
    def get_row_data(self, csv_path: str, row_indices: List[int]) -> List[Dict[str, str]]:
        """Extract specific rows from CSV file."""
        rows_data = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for idx, row in enumerate(reader):
                    if idx in row_indices:
                        # Ensure all values are strings, not None
                        cleaned_row = {}
                        for k, v in row.items():
                            if k is not None:
                                cleaned_row[str(k)] = str(v) if v is not None else ''
                        rows_data.append(cleaned_row)
            
            return rows_data
            
        except Exception as e:
            raise ValueError(f"Error reading CSV rows: {e}")
    
    def update_csv_rows(self, csv_path: str, updates: Dict[int, Dict[str, str]]):
        """Update specific rows in CSV file."""
        # Read all data
        all_rows = []
        headers = None
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                
                # Clean headers - remove None values
                if headers:
                    headers = [h for h in headers if h is not None]
                
                for idx, row in enumerate(reader):
                    if idx in updates:
                        # Update this row with new data
                        updated_row = updates[idx]
                        # Preserve non-tech fields
                        for key in row:
                            # Check if key is not None and is a string
                            if key is not None and isinstance(key, str) and not key.startswith('Tech'):
                                updated_row[key] = row[key]
                        all_rows.append(updated_row)
                    else:
                        # Clean the row to remove None keys
                        cleaned_row = {}
                        for k, v in row.items():
                            if k is not None:
                                cleaned_row[k] = v
                        all_rows.append(cleaned_row)
            
            # Write updated data
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(all_rows)
            
            print(f"✅ Updated {len(updates)} rows in {csv_path}")
            
        except Exception as e:
            raise ValueError(f"Error updating CSV: {e}")
    
    def generate_validation_report(self, validation_results: Dict, output_path: str):
        """Generate detailed validation report."""
        report_lines = [
            "# Taxonomy Validation Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- Total rows: {validation_results['validation_report']['total_rows']}",
            f"- Valid rows: {validation_results['validation_report']['valid_rows']}",
            f"- Invalid rows: {validation_results['validation_report']['invalid_rows']}",
            f"- Blank rows: {validation_results['validation_report']['blank_rows']}",
            f"- Rows flagged for re-evaluation: {len(validation_results['flagged_rows'])}",
            "",
            "## Invalid Details",
            ""
        ]
        
        if validation_results['invalid_details']:
            for detail in validation_results['invalid_details'][:50]:  # Limit to first 50
                company_name = detail.get('company_name', 'Unknown')
                report_lines.append(f"### Row {detail['row_index'] + 1}: {company_name}")
                report_lines.append(f"- Reason: {detail['reason']}")
                report_lines.append("")
        
        # Save report
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            print(f"📊 Validation report saved to: {output_path}")
        except Exception as e:
            print(f"⚠️ Error saving report: {e}")
    
    def export_valid_pairs_reference(self, output_path: str):
        """Export all valid taxonomy pairs for reference."""
        sorted_pairs = sorted(self.valid_pairs)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# Valid Taxonomy Pairs Reference\n\n")
                f.write("| Industry | Product |\n")
                f.write("|----------|----------|\n")
                
                for industry, product in sorted_pairs:
                    f.write(f"| {industry} | {product} |\n")
            
            print(f"📋 Valid pairs reference saved to: {output_path}")
        except Exception as e:
            print(f"⚠️ Error saving reference: {e}")