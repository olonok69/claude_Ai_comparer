#!/usr/bin/env python3
"""
Check CSV headers for None values and other issues
"""

import csv
import sys

def check_csv_headers(csv_path):
    """Check CSV file headers for issues."""
    print(f"Checking CSV headers in: {csv_path}")
    print("=" * 80)
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Read raw header line
            first_line = f.readline()
            print(f"Raw header line: {repr(first_line[:200])}")
            
            # Reset and use CSV reader
            f.seek(0)
            reader = csv.reader(f)
            headers_list = next(reader)
            
            print(f"\nHeaders as list ({len(headers_list)} columns):")
            for i, header in enumerate(headers_list):
                if header is None:
                    print(f"  [{i}] None (WARNING: This will cause issues!)")
                elif header == '':
                    print(f"  [{i}] '' (empty string)")
                else:
                    print(f"  [{i}] '{header}'")
            
            # Now check with DictReader
            f.seek(0)
            dict_reader = csv.DictReader(f)
            dict_headers = dict_reader.fieldnames
            
            print(f"\nDictReader fieldnames ({len(dict_headers)} columns):")
            for i, header in enumerate(dict_headers):
                if header is None:
                    print(f"  [{i}] None (WARNING: This will cause issues!)")
                elif header == '':
                    print(f"  [{i}] '' (empty string)")
                else:
                    print(f"  [{i}] '{header}'")
            
            # Check for None keys in first row
            f.seek(0)
            dict_reader = csv.DictReader(f)
            first_row = next(dict_reader)
            
            print(f"\nFirst row keys:")
            for key in first_row.keys():
                if key is None:
                    print(f"  None: '{first_row[key]}' (WARNING: None key!)")
                else:
                    print(f"  '{key}': '{first_row[key][:50] if first_row[key] else ''}'")
                    
    except Exception as e:
        print(f"Error reading CSV: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_csv_headers.py <csv_file>")
        sys.exit(1)
    
    check_csv_headers(sys.argv[1])