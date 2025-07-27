#!/usr/bin/env python3
"""
Simple validation script to check output against taxonomy
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

def load_taxonomy(taxonomy_path):
    """Load taxonomy pairs from CSV."""
    valid_pairs = set()
    
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        print(f"Taxonomy headers: {reader.fieldnames}")
        
        for row in reader:
            # Try different possible column names
            industry = (row.get('Industry') or row.get('industry') or '').strip()
            product = (row.get('Product') or row.get('product') or '').strip()
            
            if industry and product:
                valid_pairs.add((industry, product))
    
    return valid_pairs

def validate_output(output_path, valid_pairs):
    """Validate output file against taxonomy."""
    stats = {
        'total': 0,
        'valid': 0,
        'invalid': 0,
        'blank': 0,
        'invalid_pairs': defaultdict(int)
    }
    
    invalid_companies = []
    
    with open(output_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for idx, row in enumerate(reader):
            stats['total'] += 1
            company_name = row.get('Company Name', '')
            
            all_blank = True
            has_invalid = False
            invalid_in_row = []
            
            # Check each pair
            for i in range(1, 5):
                industry = (row.get(f'Tech Industry {i}') or '').strip()
                product = (row.get(f'Tech Product {i}') or '').strip()
                
                if industry or product:
                    all_blank = False
                    
                    # Both must be present
                    if bool(industry) != bool(product):
                        has_invalid = True
                        invalid_in_row.append(f"Incomplete pair {i}")
                    elif industry and product:
                        # Check if pair exists in taxonomy
                        if (industry, product) not in valid_pairs:
                            has_invalid = True
                            invalid_in_row.append(f"Invalid pair {i}: {industry} | {product}")
                            stats['invalid_pairs'][(industry, product)] += 1
            
            if all_blank:
                stats['blank'] += 1
            elif has_invalid:
                stats['invalid'] += 1
                if len(invalid_companies) < 10:  # Only keep first 10
                    invalid_companies.append({
                        'row': idx + 1,
                        'company': company_name,
                        'issues': invalid_in_row
                    })
            else:
                stats['valid'] += 1
    
    return stats, invalid_companies

def main():
    if len(sys.argv) < 3:
        print("Usage: python validate_simple.py <output_csv> <taxonomy_csv>")
        print("Example: python validate_simple.py output_0_8000_p.csv client/mcp_servers/company_tagging/categories/classes.csv")
        sys.exit(1)
    
    output_path = sys.argv[1]
    taxonomy_path = sys.argv[2]
    
    print(f"📋 Loading taxonomy from: {taxonomy_path}")
    valid_pairs = load_taxonomy(taxonomy_path)
    print(f"✅ Loaded {len(valid_pairs)} valid pairs")
    
    # Show some example pairs
    print("\n📋 Example valid pairs:")
    for i, (ind, prod) in enumerate(sorted(valid_pairs)[:10]):
        print(f"   {i+1}. {ind} | {prod}")
    
    print(f"\n🔍 Validating: {output_path}")
    stats, invalid_companies = validate_output(output_path, valid_pairs)
    
    print(f"\n📊 Validation Results:")
    print(f"   Total rows: {stats['total']}")
    print(f"   Valid rows: {stats['valid']} ({stats['valid']/max(1,stats['total'])*100:.1f}%)")
    print(f"   Invalid rows: {stats['invalid']} ({stats['invalid']/max(1,stats['total'])*100:.1f}%)")
    print(f"   Blank rows: {stats['blank']} ({stats['blank']/max(1,stats['total'])*100:.1f}%)")
    print(f"   Total to fix: {stats['invalid'] + stats['blank']}")
    
    if invalid_companies:
        print(f"\n❌ Example invalid entries:")
        for comp in invalid_companies:
            print(f"\n   Row {comp['row']}: {comp['company']}")
            for issue in comp['issues']:
                print(f"      - {issue}")
    
    if stats['invalid_pairs']:
        print(f"\n🔍 Most common invalid pairs:")
        sorted_pairs = sorted(stats['invalid_pairs'].items(), key=lambda x: x[1], reverse=True)
        for (ind, prod), count in sorted_pairs[:10]:
            print(f"   {count}x: {ind} | {prod}")

if __name__ == "__main__":
    main()