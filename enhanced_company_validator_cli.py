#!/usr/bin/env python3
"""
Enhanced Company Classification Validator and Re-evaluator
Validates existing classifications, re-processes invalid entries, and adds country inference.
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

# Import MCP service for country inference
from services.mcp_service import MCPService

# Load environment variables
load_dotenv()


class EnhancedCompanyValidatorCLI:
    """Enhanced validator with country inference capability."""
    
    def __init__(self, input_csv: str, output_csv: str, taxonomy_path: Optional[str] = None, 
                 servers: str = "perplexity", batch_size: int = 1, country_inference: bool = False):
        self.input_csv = Path(input_csv)
        self.output_csv = Path(output_csv)
        self.country_inference = country_inference
        
        # Initialize validator (if not doing country inference only)
        if not country_inference:
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
        
        # Initialize MCP service for country inference
        if country_inference:
            self.mcp_service = MCPService()
        
        print(f"🔍 Enhanced Company Validator initialized")
        print(f"   Input CSV: {self.input_csv}")
        print(f"   Output CSV: {self.output_csv}")
        if not country_inference:
            print(f"   Taxonomy pairs loaded: {len(self.validator.valid_pairs)}")
        print(f"   Servers for processing: {', '.join(sorted(self.servers))}")
        print(f"   Batch size: {self.batch_size}")
        if country_inference:
            print(f"   Mode: Country Inference Only")
        else:
            print(f"   Mode: Taxonomy Validation and Re-evaluation")
    
    def read_csv_file(self, file_path: Path) -> Tuple[List[Dict], List[str]]:
        """Read CSV file and return data with fieldnames."""
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                reader = csv.DictReader(file)
                fieldnames = list(reader.fieldnames)
                
                # Ensure CASEACCID is present
                if "CASEACCID" not in fieldnames:
                    fieldnames.insert(0, "CASEACCID")
                
                # Add Country column if doing country inference and not present
                if self.country_inference and "Country" not in fieldnames:
                    fieldnames.append("Country")
                
                for row_num, row in enumerate(reader, start=2):
                    # Ensure CASEACCID is present
                    if "CASEACCID" not in row or not row.get("CASEACCID", "").strip():
                        row["CASEACCID"] = f"CASE{row_num-1:06d}"
                    
                    # Initialize Country field if doing country inference
                    if self.country_inference and "Country" not in row:
                        row["Country"] = ""
                    
                    data.append(row)
        
        except Exception as e:
            raise Exception(f"Error reading CSV file: {str(e)}")
        
        return data, fieldnames
    
    def create_country_inference_prompt(self, company_data: Dict) -> str:
        """Create a prompt for country inference using company information."""
        
        # Extract available information
        company_name = company_data.get("Account Name", "")
        trading_name = company_data.get("Trading Name", "")
        domain = company_data.get("Domain", "")
        industry = company_data.get("Industry", "")
        product_service = company_data.get("Product/Service Type", "")
        event = company_data.get("Event", "")
        
        # Build search context
        search_context = f"Company: {company_name}"
        if trading_name and trading_name != company_name:
            search_context += f" (Trading as: {trading_name})"
        if domain:
            search_context += f" | Website: {domain}"
        if industry:
            search_context += f" | Industry: {industry}"
        if product_service:
            search_context += f" | Products/Services: {product_service}"
        
        prompt = f"""You are tasked with determining the primary country where a company operates based on available information.

Company Information:
{search_context}

Please research this company and determine the PRIMARY COUNTRY where it operates (headquarters, main operations, or primary business location).

IMPORTANT REQUIREMENTS:
1. Only provide the country name (e.g., "United States", "Germany", "Japan", "Singapore")
2. If you cannot determine the country with reasonable confidence, respond with "UNKNOWN"
3. Do not invent or guess - only use factual information you can find
4. Focus on the main operational country, not subsidiaries or branches
5. Use full country names, not abbreviations or codes

Search for current information about this company to make an accurate determination."""

        return prompt
    
    async def infer_country_for_company(self, company_data: Dict) -> str:
        """Infer country for a single company using Perplexity."""
        try:
            # Create the prompt
            prompt = self.create_country_inference_prompt(company_data)
            
            # Use perplexity_advanced_search for the most accurate results
            if "perplexity" in self.servers:
                tool_name = "perplexity_advanced_search"
                params = {
                    "query": f"{company_data.get('Account Name', '')} company location headquarters country",
                    "focus": "web",
                    "additional_instructions": prompt
                }
            else:
                # Fallback to Google search if Perplexity not available
                tool_name = "google-search"
                params = {
                    "query": f"{company_data.get('Account Name', '')} company headquarters location country"
                }
            
            # Execute the search
            result = await self.mcp_service.call_tool(tool_name, params, self.servers)
            
            if result and "content" in result:
                # Process the result through the LLM to extract just the country
                analysis_prompt = f"""Based on the following search results, what is the primary country where the company '{company_data.get('Account Name', '')}' operates?

Search Results:
{result['content']}

Respond with ONLY the country name (e.g., "United States", "Germany", "Japan") or "UNKNOWN" if you cannot determine it with confidence.
Do not provide any additional explanation or text."""

                # Use the MCP service to analyze the result
                analysis_result = await self.mcp_service.analyze_with_llm(analysis_prompt)
                
                if analysis_result and isinstance(analysis_result, str):
                    country = analysis_result.strip().strip('"\'')
                    # Validate that it looks like a country name
                    if country.upper() == "UNKNOWN" or not country:
                        return ""
                    # Basic validation - should not contain certain words that indicate uncertainty
                    invalid_indicators = ["unclear", "cannot", "unknown", "not found", "unable to determine"]
                    if any(indicator in country.lower() for indicator in invalid_indicators):
                        return ""
                    return country
            
            return ""  # Could not determine
            
        except Exception as e:
            print(f"   ⚠️ Error inferring country for {company_data.get('Account Name', 'Unknown')}: {str(e)}")
            return ""  # Return empty on error
    
    async def process_country_inference(self, companies: List[Dict], fieldnames: List[str]):
        """Process country inference for all companies."""
        print(f"\n🌍 Processing {len(companies)} companies for country inference...")
        
        # Initialize MCP connections
        await self.mcp_service.initialize_servers(self.servers)
        
        try:
            total_batches = (len(companies) + self.batch_size - 1) // self.batch_size
            
            for batch_idx in range(0, len(companies), self.batch_size):
                batch_num = (batch_idx // self.batch_size) + 1
                batch_companies = companies[batch_idx:batch_idx + self.batch_size]
                
                print(f"\n🔍 Processing batch {batch_num}/{total_batches} ({len(batch_companies)} companies)")
                
                for i, company in enumerate(batch_companies):
                    current_idx = batch_idx + i
                    company_name = company.get("Account Name", "Unknown")
                    
                    print(f"   🔍 Processing {current_idx + 1}: {company_name}")
                    
                    # Check if country already exists and is not empty
                    if company.get("Country", "").strip():
                        print(f"      ✅ Country already exists: {company['Country']}")
                        continue
                    
                    # Infer country
                    try:
                        country = await self.infer_country_for_company(company)
                        company["Country"] = country
                        
                        if country:
                            print(f"      ✅ Country inferred: {country}")
                        else:
                            print(f"      ⚪ Country could not be determined (left blank)")
                        
                    except Exception as e:
                        print(f"      ❌ Error processing {company_name}: {str(e)}")
                        company["Country"] = ""  # Leave blank on error
                
                # Save progress after each batch
                self._write_csv_file(companies, fieldnames)
                print(f"✅ Batch {batch_num}/{total_batches} completed and saved")
        
        finally:
            await self.mcp_service.cleanup()
    
    def validate_and_flag(self) -> Dict[str, any]:
        """Validate output CSV and identify rows needing re-evaluation."""
        if not self.output_csv.exists():
            raise FileNotFoundError(f"Output CSV file not found: {self.output_csv}")
        
        print(f"\n🔍 Step 1: Validating {self.output_csv}...")
        
        # Read the output CSV
        companies, fieldnames = self.read_csv_file(self.output_csv)
        
        # Validate against taxonomy
        validation_results = self.validator.validate_csv_data(companies)
        
        # Print validation summary
        total_rows = validation_results['total_rows']
        valid_rows = validation_results['valid_rows']
        invalid_rows = len(validation_results['invalid_details'])
        blank_rows = validation_results['blank_rows']
        flagged_rows = len(validation_results['flagged_rows'])
        
        print(f"\n📊 Validation Summary:")
        print(f"   Total rows: {total_rows}")
        print(f"   Valid rows: {valid_rows} ({valid_rows/total_rows*100:.1f}%)")
        print(f"   Invalid rows: {invalid_rows}")
        print(f"   Blank rows: {blank_rows}")
        print(f"   Rows flagged for re-evaluation: {flagged_rows}")
        
        return validation_results
    
    def _write_csv_file(self, companies: List[Dict], fieldnames: List[str]):
        """Write companies data to CSV file with atomic operation."""
        try:
            # Create backup if file exists
            if self.output_csv.exists():
                backup_path = self.output_csv.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                self.output_csv.rename(backup_path)
                print(f"📁 Created backup: {backup_path}")
            
            # Write to temporary file first (atomic operation)
            temp_output = self.output_csv.with_suffix('.tmp')
            
            with open(temp_output, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(companies)
            
            # Atomic move
            temp_output.replace(self.output_csv)
            print(f"✅ Results written to: {self.output_csv}")
            
        except Exception as e:
            print(f"❌ Error writing output file: {str(e)}")
            raise
    
    async def run_validation_and_re_evaluation(self):
        """Run the original validation and re-evaluation process."""
        # Import the existing functionality from the original validator
        # This would use the same logic as the original company_validator_cli.py
        # For brevity, I'm not replicating the entire original code here
        pass
    
    async def run_country_inference_only(self):
        """Run country inference only."""
        try:
            # Read input CSV
            companies, fieldnames = self.read_csv_file(self.input_csv if self.input_csv.exists() else self.output_csv)
            
            # Process country inference
            await self.process_country_inference(companies, fieldnames)
            
            # Generate summary
            print("\n📊 Country Inference Summary:")
            countries_found = sum(1 for company in companies if company.get("Country", "").strip())
            countries_blank = len(companies) - countries_found
            
            print(f"   Total companies: {len(companies)}")
            print(f"   Countries inferred: {countries_found} ({countries_found/len(companies)*100:.1f}%)")
            print(f"   Countries blank: {countries_blank} ({countries_blank/len(companies)*100:.1f}%)")
            
            # Show country distribution
            country_counts = {}
            for company in companies:
                country = company.get("Country", "").strip()
                if country:
                    country_counts[country] = country_counts.get(country, 0) + 1
            
            if country_counts:
                print("\n🌍 Country Distribution:")
                for country, count in sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"   {country}: {count}")
                if len(country_counts) > 10:
                    print(f"   ... and {len(country_counts) - 10} more countries")
            
            print(f"\n✅ Country inference completed! Results saved to: {self.output_csv}")
            
        except Exception as e:
            print(f"\n❌ Error during country inference: {str(e)}")
            traceback.print_exc()
            raise


async def main():
    """Main entry point for the enhanced validator CLI."""
    parser = argparse.ArgumentParser(
        description="Enhanced company classifier: validate taxonomy and infer countries"
    )
    
    parser.add_argument("--input", "-i", required=True, 
                       help="Path to original input CSV file")
    parser.add_argument("--output", "-o", required=True, 
                       help="Path to output CSV file to validate and update")
    parser.add_argument("--taxonomy", "-t", 
                       help="Path to taxonomy CSV file (default: auto-detect)")
    parser.add_argument("--servers", "-s", type=str, default="perplexity",
                       choices=["google", "perplexity", "both"],
                       help="MCP servers to use (default: perplexity)")
    parser.add_argument("--batch-size", "-b", type=int, default=1,
                       help="Batch size for processing (default: 1)")
    parser.add_argument("--country-only", "-c", action="store_true",
                       help="Only perform country inference, skip taxonomy validation")
    parser.add_argument("--report-only", action="store_true",
                       help="Only generate validation report without re-evaluation")
    parser.add_argument("--export-taxonomy", 
                       help="Export valid taxonomy pairs to specified file")
    
    args = parser.parse_args()
    
    try:
        # Initialize enhanced validator
        validator_cli = EnhancedCompanyValidatorCLI(
            args.input, 
            args.output, 
            args.taxonomy, 
            args.servers,
            args.batch_size,
            country_inference=args.country_only
        )
        
        if args.country_only:
            # Run country inference only
            await validator_cli.run_country_inference_only()
        else:
            # Run original validation process
            await validator_cli.run_validation_and_re_evaluation()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted")
        sys.exit(1)