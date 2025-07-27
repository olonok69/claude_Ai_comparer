#!/usr/bin/env python3
"""
Recover validation results from a failed run
This script helps save the re-evaluated data when the final update step fails
"""

import csv
import json
import pickle
import sys
from pathlib import Path
from datetime import datetime

def find_latest_backup(output_csv_path):
    """Find the latest backup file."""
    output_path = Path(output_csv_path)
    backup_pattern = f"{output_path.stem}_backup_*.csv"
    backups = list(output_path.parent.glob(backup_pattern))
    
    if backups:
        # Sort by modification time, get the latest
        latest_backup = max(backups, key=lambda p: p.stat().st_mtime)
        return latest_backup
    return None

def save_validation_results_json(results_dict, output_path):
    """Save validation results to JSON for recovery."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved validation results to: {output_path}")

def apply_validation_results_safe(csv_path, validation_json_path, output_path=None):
    """Safely apply validation results to CSV."""
    # Load validation results
    with open(validation_json_path, 'r', encoding='utf-8') as f:
        updates = json.load(f)
    
    # Convert string keys back to integers
    updates = {int(k): v for k, v in updates.items()}
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{Path(csv_path).stem}_recovered_{timestamp}.csv"
    
    print(f"Applying {len(updates)} validation results...")
    
    # Read the CSV and apply updates
    all_rows = []
    headers = None
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Get and clean headers
        original_headers = reader.fieldnames
        headers = [h for h in original_headers if h is not None]
        
        print(f"Headers: {headers}")
        
        for idx, row in enumerate(reader):
            if idx in updates:
                # This row was re-evaluated
                updated_row = updates[idx]
                
                # Preserve non-tech fields from original
                clean_row = {}
                for key in headers:
                    if key and key.startswith('Tech'):
                        # Use updated value for Tech fields
                        clean_row[key] = updated_row.get(key, '')
                    else:
                        # Preserve original value for non-Tech fields
                        clean_row[key] = row.get(key, '')
                
                all_rows.append(clean_row)
                print(f"✅ Updated row {idx + 1}: {clean_row.get('Company Name', '')}")
            else:
                # Keep original row, but clean it
                clean_row = {}
                for key in headers:
                    clean_row[key] = row.get(key, '')
                all_rows.append(clean_row)
    
    # Write the recovered CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"✅ Recovered CSV saved to: {output_path}")
    print(f"   Total rows: {len(all_rows)}")
    print(f"   Updated rows: {len(updates)}")
    
    return output_path

def create_recovery_instructions():
    """Create instructions for manual recovery."""
    instructions = """
RECOVERY INSTRUCTIONS
====================

If the validation process completed but failed to save, you can recover the results:

1. **Check for backup files**:
   ls -la output_0_8000_p_backup_*.csv
   
2. **Look for temp files**:
   ls -la temp_reeval*
   ls -la *.pkl
   ls -la *.json

3. **If you have the Python session still open**:
   In the Python interpreter where the error occurred, try:
   
   # Save the updates dictionary
   import json
   with open('validation_updates.json', 'w') as f:
       json.dump(updates, f)
   
4. **To apply saved validation results**:
   python recover_validation_results.py apply output_0_8000_p.csv validation_updates.json

5. **To continue from where it stopped**:
   Since 1214 rows were successfully re-evaluated, you can:
   - Use the backup file as the starting point
   - Run validation again with --report-only to see what's left
   - Re-run only the remaining rows

PREVENTION FOR NEXT TIME:
- Clean the CSV headers first: python clean_csv_headers.py output.csv
- Run with smaller batches if memory is an issue
- Save intermediate results: Add periodic JSON dumps in the validator
"""
    print(instructions)

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python recover_validation_results.py instructions")
        print("  python recover_validation_results.py find-backup <output_csv>")
        print("  python recover_validation_results.py apply <csv_file> <validation_json>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "instructions":
        create_recovery_instructions()
    
    elif command == "find-backup":
        if len(sys.argv) < 3:
            print("Usage: python recover_validation_results.py find-backup <output_csv>")
            sys.exit(1)
        
        backup = find_latest_backup(sys.argv[2])
        if backup:
            print(f"✅ Found latest backup: {backup}")
            print(f"   Created: {datetime.fromtimestamp(backup.stat().st_mtime)}")
        else:
            print("❌ No backup files found")
    
    elif command == "apply":
        if len(sys.argv) < 4:
            print("Usage: python recover_validation_results.py apply <csv_file> <validation_json>")
            sys.exit(1)
        
        csv_file = sys.argv[2]
        json_file = sys.argv[3]
        output_file = sys.argv[4] if len(sys.argv) > 4 else None
        
        apply_validation_results_safe(csv_file, json_file, output_file)
    
    else:
        print(f"Unknown command: {command}")
        create_recovery_instructions()

if __name__ == "__main__":
    main()