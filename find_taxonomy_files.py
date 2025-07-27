#!/usr/bin/env python3
"""
Find and analyze potential taxonomy files
"""

import os
import csv
from pathlib import Path

def find_csv_files(start_path="."):
    """Find all CSV files in the project."""
    csv_files = []
    for root, dirs, files in os.walk(start_path):
        # Skip hidden directories and common non-relevant paths
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env']]
        
        for file in files:
            if file.endswith('.csv') and 'output' not in file.lower():
                csv_files.append(Path(root) / file)
    
    return csv_files

def analyze_csv(csv_path):
    """Analyze a CSV file to see if it might be a taxonomy file."""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Read first few lines to check structure
            lines = f.readlines()[:5]
            
            # Reset and use CSV reader
            f.seek(0)
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            
            # Count rows
            row_count = sum(1 for _ in reader)
            
            # Check if it looks like a taxonomy file
            is_taxonomy = False
            if headers and any(h in str(headers) for h in ['Industry', 'Product', 'Category', 'Show']):
                is_taxonomy = True
            
            return {
                'path': csv_path,
                'headers': headers,
                'row_count': row_count,
                'is_taxonomy': is_taxonomy,
                'first_line': lines[0].strip() if lines else ''
            }
    except Exception as e:
        return {
            'path': csv_path,
            'error': str(e)
        }

def main():
    """Find and analyze all CSV files."""
    print("🔍 Searching for CSV files...")
    
    csv_files = find_csv_files()
    
    print(f"\n📋 Found {len(csv_files)} CSV files\n")
    
    taxonomy_candidates = []
    
    for csv_file in csv_files:
        info = analyze_csv(csv_file)
        
        if 'error' not in info:
            print(f"📄 {info['path']}")
            print(f"   Headers: {info['headers']}")
            print(f"   Rows: {info['row_count']}")
            
            if info['is_taxonomy']:
                print(f"   ✅ Possible taxonomy file!")
                taxonomy_candidates.append(info)
            print()
    
    if taxonomy_candidates:
        print("\n🎯 Likely taxonomy files:")
        for candidate in taxonomy_candidates:
            print(f"\n📋 {candidate['path']}")
            print(f"   Headers: {candidate['headers']}")
            print(f"   Rows: {candidate['row_count']}")
            
            # Show sample data
            try:
                with open(candidate['path'], 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    print("   Sample rows:")
                    for i, row in enumerate(reader):
                        if i >= 3:
                            break
                        print(f"      {i+1}. {dict(row)}")
            except:
                pass

if __name__ == "__main__":
    main()