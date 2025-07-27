#!/usr/bin/env python3
"""
Company Classification Validator and Re-evaluator
Validates existing classifications and re-processes invalid entries.
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

# Add the client directory to the path
sys.path.insert(0, str(Path(__file__).parent / "client"))

from dotenv import load_dotenv
from taxonomy_validator import TaxonomyValidator

# Import the main CLI tool to reuse its functionality
from company_cli import EnhancedCompanyClassificationCLI

# Load environment variables
load_dotenv()


class CompanyValidatorCLI:
    """Validates and re-evaluates company classifications."""
    
    def __init__(self, input_csv: str, output_csv: str, taxonomy_path: Optional[str] = None, 
                 servers: str = "perplexity", batch_size: int = 1):
        self.input_csv = Path(input_csv)
        self.output_csv = Path(output_csv)
        
        # Initialize validator
        self.validator = TaxonomyValidator(taxonomy_path)
        
        # Re-evaluation settings
        self.batch_size = batch_size
        
        # Parse server selection
        if servers == "google":
            self.servers = {"google", "company_tagging"}
        elif servers == "perplexity":
            self.servers = {"perplexity", "company_tagging"}
        elif servers == "both":
            self.servers = {"google", "perplexity", "company_tagging"}
        else:
            # Default to perplexity
            self.servers = {"perplexity", "company_tagging"}
        
        print(f"🔍 Company Validator initialized")
        print(f"   Input CSV: {self.input_csv}")
        print(f"   Output CSV: {self.output_csv}")
        print(f"   Taxonomy pairs loaded: {len(self.validator.valid_pairs)}")
        print(f"   Servers for re-evaluation: {', '.join(sorted(self.servers))}")
        print(f"   Batch size for re-evaluation: {self.batch_size}")
    
    def validate_and_flag(self) -> Dict[str, any]:
        """Validate output CSV and identify rows needing re-evaluation."""
        print("\n📋 Step 1: Validating existing classifications...")
        
        if not self.output_csv.exists():
            raise FileNotFoundError(f"Output CSV not found: {self.output_csv}")
        
        # Validate the output CSV
        validation_results = self.validator.validate_csv(str(self.output_csv))
        
        # Print summary
        report = validation_results['validation_report']
        print(f"\n📊 Validation Summary:")
        print(f"   Total rows: {report['total_rows']}")
        print(f"   Valid rows: {report['valid_rows']} ({report['valid_rows']/max(1, report['total_rows'])*100:.1f}%)")
        print(f"   Invalid rows: {report['invalid_rows']}")
        print(f"   Blank rows: {report['blank_rows']}")
        print(f"   Rows to re-evaluate: {len(validation_results['flagged_rows'])}")
        
        # Show some examples of invalid entries
        if validation_results['invalid_details']:
            print(f"\n❌ Example invalid entries:")
            for detail in validation_results['invalid_details'][:5]:
                print(f"   Row {detail['row_index'] + 1}: {detail['company_name']}")
                print(f"      Reason: {detail['reason']}")
        
        return validation_results
    
    def create_backup(self):
        """Create backup of output CSV."""
        print(f"\n💾 Step 2: Creating backup...")
        backup_path = self.validator.create_backup(str(self.output_csv))
        return backup_path
    
    async def re_evaluate_flagged_rows(self, flagged_rows: List[int]) -> Dict[int, Dict[str, str]]:
        """Re-evaluate flagged rows using the classification CLI."""
        if not flagged_rows:
            print("✅ No rows to re-evaluate!")
            return {}
        
        print(f"\n🔄 Step 3: Re-evaluating {len(flagged_rows)} flagged rows...")
        print(f"   Using batch size: {self.batch_size}")
        
        # Get the original data for flagged rows from input CSV
        original_data = self._get_original_data_for_rows(flagged_rows)
        
        if not original_data:
            print("⚠️ Could not find original data for flagged rows")
            return {}
        
        # Initialize the classification CLI with stricter settings
        cli = EnhancedCompanyClassificationCLI(
            batch_size=self.batch_size,
            enabled_servers=self.servers,
            output_base=str(self.output_csv.parent / "temp_reeval")
        )
        
        try:
            # Setup connections
            await cli.setup_connections()
            
            # Process companies
            updated_rows = {}
            
            # Convert dict to list for batch processing
            items = list(original_data.items())
            
            # Save intermediate results periodically
            save_interval = 50  # Save every 50 companies
            
            # Process in batches
            for i in range(0, len(items), self.batch_size):
                batch_items = items[i:i + self.batch_size]
                batch_companies = [item[1] for item in batch_items]
                batch_indices = [item[0] for item in batch_items]
                
                batch_num = i // self.batch_size + 1
                total_batches = (len(items) + self.batch_size - 1) // self.batch_size
                
                print(f"\n🔍 Re-evaluating batch {batch_num}/{total_batches} ({len(batch_companies)} companies)")
                for idx, company in zip(batch_indices, batch_companies):
                    print(f"   - Row {idx + 1}: {company.get('Account Name', '')}")
                
                # Create enhanced prompt with stricter instructions
                prompt = self._create_strict_prompt(batch_companies)
                
                # Process with the agent
                try:
                    from langchain_core.messages import HumanMessage
                    conversation_memory = [HumanMessage(content=prompt)]
                    
                    response = await asyncio.wait_for(
                        cli.agent.ainvoke({"messages": conversation_memory}),
                        timeout=300 * len(batch_companies)  # 5 minutes per company
                    )
                    
                    # Extract and validate the response
                    results = self._extract_validated_results(response, batch_companies)
                    
                    if results:
                        # Map results back to indices
                        for idx, result in zip(batch_indices, results):
                            if result:
                                updated_rows[idx] = result
                                print(f"   ✅ Row {idx + 1} successfully re-evaluated with valid taxonomy pairs")
                            else:
                                print(f"   ❌ Row {idx + 1} re-evaluation did not produce valid results")
                                # Keep the row blank rather than invalid
                                updated_rows[idx] = self._create_blank_result(batch_companies[batch_indices.index(idx)])
                    else:
                        print(f"   ❌ Batch {batch_num} did not produce valid results")
                        # Keep all rows blank
                        for idx, company in zip(batch_indices, batch_companies):
                            updated_rows[idx] = self._create_blank_result(company)
                        
                except Exception as e:
                    print(f"   ❌ Error during batch {batch_num} re-evaluation: {str(e)}")
                    # Keep all rows in batch blank on error
                    for idx, company in zip(batch_indices, batch_companies):
                        updated_rows[idx] = self._create_blank_result(company)
                
                # Save intermediate results periodically
                if len(updated_rows) % save_interval == 0 or batch_num == total_batches:
                    intermediate_path = self.output_csv.parent / f"validation_intermediate_{len(updated_rows)}.json"
                    try:
                        with open(intermediate_path, 'w', encoding='utf-8') as f:
                            json.dump(updated_rows, f, indent=2, ensure_ascii=False)
                        print(f"   💾 Saved intermediate results: {len(updated_rows)} rows to {intermediate_path}")
                    except Exception as e:
                        print(f"   ⚠️ Failed to save intermediate results: {e}")
            
            # Save final results before returning
            final_results_path = self.output_csv.parent / "validation_final_results.json"
            try:
                with open(final_results_path, 'w', encoding='utf-8') as f:
                    json.dump(updated_rows, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Saved final validation results: {len(updated_rows)} rows to {final_results_path}")
            except Exception as e:
                print(f"⚠️ Failed to save final results: {e}")
            
            return updated_rows
            
        finally:
            # Cleanup
            await cli.cleanup()
    
    def _get_original_data_for_rows(self, row_indices: List[int]) -> Dict[int, Dict[str, str]]:
        """Get original company data from input CSV for specific rows."""
        original_data = {}
        
        # First, get company names from output CSV
        output_companies = self.validator.get_row_data(str(self.output_csv), row_indices)
        company_names = {}
        for idx, row in zip(row_indices, output_companies):
            company_name = row.get('Company Name', '')
            if company_name:
                company_names[company_name] = idx
        
        # Now find matching rows in input CSV
        try:
            with open(self.input_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    account_name = row.get('Account Name', '')
                    if account_name and account_name in company_names:
                        # Clean the row data
                        cleaned_row = {}
                        for k, v in row.items():
                            if k is not None:
                                cleaned_row[str(k)] = str(v) if v is not None else ''
                        original_data[company_names[account_name]] = cleaned_row
            
            return original_data
            
        except Exception as e:
            print(f"⚠️ Error reading input CSV: {e}")
            return {}
    
    def _create_strict_prompt(self, companies: List[Dict]) -> str:
        """Create a prompt with very strict taxonomy enforcement."""
        company_data = self._format_company_data(companies)
        
        return f"""You are a STRICT data analyst who MUST follow taxonomy rules EXACTLY.

IMPORTANT CONTEXT: All companies in this dataset are TECHNOLOGY COMPANIES that exhibit at tech trade shows. They all operate in the technology sector, even if their names might not immediately suggest it.

COMPANY DATA TO ANALYZE:
{company_data}

CRITICAL RULES - ABSOLUTELY MANDATORY:

1. **FIRST STEP**: Use search_show_categories tool to get the COMPLETE taxonomy list

2. **VALIDATION REQUIREMENT**: You can ONLY use industry|product pairs that EXACTLY match the taxonomy
   - If you cannot find an exact match, leave the fields BLANK
   - NEVER create new industry or product names
   - NEVER modify existing names (not even changing & to "and" or adding/removing spaces)

3. **Research the company** using available search tools
   - Remember: These are ALL tech companies, so look for their technology products/services
   - Even logistics, finance, or retail companies here have tech offerings
   - Focus on their SOFTWARE, PLATFORMS, AI, DATA, CLOUD, or INFRASTRUCTURE offerings
   - **SEARCH QUERY HINT**: Add "technology software platform" to your searches
   - Example: Instead of "Epost Express", search "Epost Express technology software platform"

4. **STRICT MATCHING**: 
   - Only use pairs that exist in the taxonomy as exact matches
   - The industry and product MUST be from the SAME ROW in the taxonomy
   - Better to leave fields blank than to use non-existent pairs

5. **Output ONLY data rows** (no header) in this exact format:
   | Company Name | Trading Name | Tech Industry 1 | Tech Product 1 | Tech Industry 2 | Tech Product 2 | Tech Industry 3 | Tech Product 3 | Tech Industry 4 | Tech Product 4 |

IMPORTANT CONTEXT FOR MATCHING:
- These companies exhibit at: Cloud and AI Infrastructure (CAI), DevOps Live (DOL), Data Centre World (DCW), Cloud and Cyber Security Expo (CCSE), Big Data and AI World (BDAIW)
- Look for their TECHNOLOGY offerings, not their general business
- A logistics company might offer "Platforms & Software | Big Data & Analytics Tools" for supply chain
- A finance company might offer "Platforms & Software | AI Applications" for fintech
- A retail company might offer "Platforms & Software | Cloud Security Solutions" for e-commerce

SEARCH STRATEGY:
- For domain searches: "site:[domain] technology platform software cloud AI data"
- For company searches: "[Company Name] technology software platform services"
- Always append tech keywords to find their tech offerings

IMPORTANT: If you cannot find valid taxonomy matches, output the row with ALL tech fields BLANK rather than making up values.

Example of CORRECT behavior:
- If a company does AI but "Artificial Intelligence" is not in the taxonomy, leave it blank
- If the taxonomy has "Platforms & Software | AI Applications", use that EXACTLY
- Do NOT create "AI & Software | Machine Learning" if it's not in the taxonomy

Begin the analysis now with search_show_categories."""
    
    def _format_company_data(self, companies: List[Dict]) -> str:
        """Format company data for prompt."""
        lines = []
        for company in companies:
            lines.append(f"Account Name = {company.get('Account Name', '')}")
            lines.append(f"Trading Name = {company.get('Trading Name', '')}")
            lines.append(f"Domain = {company.get('Domain', '')}")
            lines.append(f"Industry = {company.get('Industry', '')}")
            lines.append(f"Product/Service Type = {company.get('Product/Service Type', '')}")
            lines.append(f"Event = {company.get('Event', '')}")
        return '\n'.join(lines)
    
    def _extract_validated_results(self, response: any, batch_companies: List[Dict]) -> List[Optional[Dict[str, str]]]:
        """Extract and validate results from agent response for a batch."""
        from langchain_core.messages import AIMessage
        
        # Extract the response content
        assistant_response = None
        if "messages" in response:
            for msg in response["messages"]:
                if isinstance(msg, AIMessage) and hasattr(msg, "content") and msg.content:
                    assistant_response = str(msg.content)
                    break
        
        if not assistant_response or "|" not in assistant_response:
            return [None] * len(batch_companies)
        
        # Parse the table rows
        results = []
        lines = assistant_response.strip().split('\n')
        data_rows = []
        
        for line in lines:
            if "|" in line and not line.strip().startswith("|---"):
                # Parse columns
                columns = [col.strip() for col in line.split('|')]
                if columns and not columns[0]:
                    columns = columns[1:]
                if columns and not columns[-1]:
                    columns = columns[:-1]
                
                if len(columns) >= 10:  # Should have 10 columns
                    data_rows.append(columns)
        
        # Match results to batch companies
        for i, company in enumerate(batch_companies):
            if i < len(data_rows):
                columns = data_rows[i]
                result = {
                    'Company Name': columns[0],
                    'Trading Name': columns[1],
                    'Tech Industry 1': columns[2],
                    'Tech Product 1': columns[3],
                    'Tech Industry 2': columns[4],
                    'Tech Product 2': columns[5],
                    'Tech Industry 3': columns[6],
                    'Tech Product 3': columns[7],
                    'Tech Industry 4': columns[8],
                    'Tech Product 4': columns[9]
                }
                
                # Validate all pairs
                all_valid = True
                for j in range(1, 5):
                    industry = result[f'Tech Industry {j}'].strip()
                    product = result[f'Tech Product {j}'].strip()
                    
                    if industry and product:
                        if not self.validator.is_valid_pair(industry, product):
                            print(f"      ❌ Invalid pair {j}: {industry} | {product}")
                            all_valid = False
                
                if all_valid:
                    results.append(result)
                else:
                    results.append(None)
            else:
                results.append(None)
        
        return results
    
    def _extract_validated_result(self, response: any, original_data: Dict) -> Optional[Dict[str, str]]:
        """Extract and validate result from agent response for a single company."""
        # Use the batch method for consistency
        results = self._extract_validated_results(response, [original_data])
        return results[0] if results else None
    
    def _create_blank_result(self, original_data: Dict) -> Dict[str, str]:
        """Create a result with all tech fields blank."""
        return {
            'Company Name': original_data.get('Account Name', ''),
            'Trading Name': original_data.get('Trading Name', ''),
            'Tech Industry 1': '',
            'Tech Product 1': '',
            'Tech Industry 2': '',
            'Tech Product 2': '',
            'Tech Industry 3': '',
            'Tech Product 3': '',
            'Tech Industry 4': '',
            'Tech Product 4': ''
        }
    
    def update_output_file(self, updates: Dict[int, Dict[str, str]]):
        """Update the output file with validated results."""
        if not updates:
            print("\n✅ No updates needed!")
            return
        
        print(f"\n📝 Step 4: Updating output file with {len(updates)} validated rows...")
        
        # Update the CSV
        self.validator.update_csv_rows(str(self.output_csv), updates)
        
        print("✅ Output file updated successfully!")
    
    def generate_final_report(self, initial_validation: Dict, updates: Dict):
        """Generate final validation report."""
        print("\n📊 Step 5: Generating final report...")
        
        # Re-validate the updated file
        final_validation = self.validator.validate_csv(str(self.output_csv))
        
        # Generate report
        report_path = self.output_csv.parent / f"{self.output_csv.stem}_validation_report.md"
        self.validator.generate_validation_report(final_validation, str(report_path))
        
        # Print improvement summary
        initial_invalid = initial_validation['validation_report']['invalid_rows']
        initial_blank = initial_validation['validation_report']['blank_rows']
        final_invalid = final_validation['validation_report']['invalid_rows']
        final_blank = final_validation['validation_report']['blank_rows']
        
        print(f"\n📈 Improvement Summary:")
        print(f"   Invalid rows: {initial_invalid} → {final_invalid}")
        print(f"   Blank rows: {initial_blank} → {final_blank}")
        print(f"   Total issues: {initial_invalid + initial_blank} → {final_invalid + final_blank}")
        
        if final_invalid == 0 and final_blank == 0:
            print("\n🎉 Perfect! All rows now have valid taxonomy classifications!")
        elif final_invalid == 0:
            print(f"\n✅ All non-blank rows have valid taxonomy classifications!")
            print(f"   {final_blank} rows remain blank (no suitable taxonomy matches found)")


async def main():
    """Main entry point for the validator CLI."""
    parser = argparse.ArgumentParser(
        description="Validate and re-evaluate company classifications against taxonomy"
    )
    
    parser.add_argument("--input", "-i", required=True, 
                       help="Path to original input CSV file")
    parser.add_argument("--output", "-o", required=True, 
                       help="Path to output CSV file to validate and update")
    parser.add_argument("--taxonomy", "-t", 
                       help="Path to taxonomy CSV file (default: auto-detect)")
    parser.add_argument("--servers", "-s", type=str, default="perplexity",
                       choices=["google", "perplexity", "both"],
                       help="MCP servers to use for re-evaluation (default: perplexity)")
    parser.add_argument("--batch-size", "-b", type=int, default=1,
                       help="Batch size for re-evaluation (default: 1, recommended for accuracy)")
    parser.add_argument("--report-only", action="store_true",
                       help="Only generate validation report without re-evaluation")
    parser.add_argument("--export-taxonomy", 
                       help="Export valid taxonomy pairs to specified file")
    
    args = parser.parse_args()
    
    try:
        # Initialize validator with server selection and batch size
        validator_cli = CompanyValidatorCLI(
            args.input, 
            args.output, 
            args.taxonomy, 
            args.servers,
            args.batch_size
        )
        
        # Export taxonomy reference if requested
        if args.export_taxonomy:
            validator_cli.validator.export_valid_pairs_reference(args.export_taxonomy)
            return 0
        
        # Step 1: Validate and flag invalid rows
        validation_results = validator_cli.validate_and_flag()
        
        if args.report_only:
            # Generate report only
            report_path = Path(args.output).parent / f"{Path(args.output).stem}_validation_report.md"
            validator_cli.validator.generate_validation_report(validation_results, str(report_path))
            return 0
        
        # Check if re-evaluation is needed
        if not validation_results['flagged_rows']:
            print("\n✅ All rows are valid! No re-evaluation needed.")
            return 0
        
        # Step 2: Create backup
        backup_path = validator_cli.create_backup()
        
        # Step 3: Re-evaluate flagged rows
        updates = await validator_cli.re_evaluate_flagged_rows(validation_results['flagged_rows'])
        
        # Step 4: Update output file
        validator_cli.update_output_file(updates)
        
        # Step 5: Generate final report
        validator_cli.generate_final_report(validation_results, updates)
        
        print(f"\n✅ Validation and re-evaluation completed!")
        print(f"   Backup saved to: {backup_path}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))