#!/usr/bin/env python3
"""
Clean CSV file by removing None headers and fixing structure
"""

import csv
import sys
import shutil
from datetime import datetime

def clean_csv_file(input_path, output_path=None):
    """Clean CSV file by removing None headers."""
    if output_path is None:
        # Create backup and use original filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{input_path}.backup_{timestamp}"
        shutil.copy2(input_path, backup_path)
        print(f"Created backup: {backup_path}")
        output_path = input_path
    
    print(f"Cleaning CSV: {input_path}")
    
    try:
        # Read the CSV
        all_rows = []
        clean_headers = None
        
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Get headers and clean them
            original_headers = reader.fieldnames
            clean_headers = [h for h in original_headers if h is not None]
            
            print(f"Original headers: {len(original_headers)}")
            print(f"Clean headers: {len(clean_headers)}")
            
            # Read all rows, keeping only clean columns
            for row in reader:
                clean_row = {}
                for header in clean_headers:
                    clean_row[header] = row.get(header, '')
                all_rows.append(clean_row)
        
        # Write cleaned CSV
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=clean_headers)
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"✅ Cleaned CSV saved to: {output_path}")
        print(f"   Rows: {len(all_rows)}")
        print(f"   Columns: {len(clean_headers)}")
        
        # Show the clean headers
        print("\nClean headers:")
        for i, header in enumerate(clean_headers):
            print(f"  [{i}] '{header}'")
            
    except Exception as e:
        print(f"❌ Error cleaning CSV: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_csv_headers.py <csv_file> [output_file]")
        print("If output_file is not specified, the input file will be cleaned in place (with backup)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    clean_csv_file(input_file, output_file)