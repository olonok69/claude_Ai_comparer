#!/usr/bin/env python3
"""
Debug script to test CSV reading and identify None value issues
"""

import csv
import sys
from pathlib import Path

def debug_csv(csv_path):
    """Debug CSV file to find None values and other issues."""
    print(f"Debugging CSV: {csv_path}")
    print("=" * 80)
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Read first few lines raw
            print("First 3 lines (raw):")
            f.seek(0)
            for i in range(3):
                line = f.readline()
                print(f"Line {i+1}: {repr(line[:100])}")
            
            # Reset and use CSV reader
            f.seek(0)
            reader = csv.DictReader(f)
            
            print(f"\nHeaders: {reader.fieldnames}")
            print(f"Number of columns: {len(reader.fieldnames) if reader.fieldnames else 0}")
            
            # Check first 5 rows
            print("\nFirst 5 rows analysis:")
            for idx, row in enumerate(reader):
                if idx >= 5:
                    break
                    
                print(f"\nRow {idx + 1}:")
                print(f"  Company Name: {repr(row.get('Company Name'))}")
                
                # Check tech fields
                for i in range(1, 5):
                    industry = row.get(f'Tech Industry {i}')
                    product = row.get(f'Tech Product {i}')
                    print(f"  Tech Pair {i}:")
                    print(f"    Industry: {repr(industry)} (type: {type(industry).__name__})")
                    print(f"    Product: {repr(product)} (type: {type(product).__name__})")
                    
                    if industry is None or product is None:
                        print(f"    ⚠️  WARNING: None value detected!")
                
    except Exception as e:
        print(f"Error reading CSV: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_csv_test.py <csv_file>")
        sys.exit(1)
    
    debug_csv(sys.argv[1])