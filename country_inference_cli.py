#!/usr/bin/env python3
"""
Simple Country Addition CLI
Just adds CASEACCID and Country columns to existing classification output.
NO re-evaluation - keeps all existing classifications exactly as they are.
"""

import argparse
import asyncio
import csv
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add the client directory to the path
sys.path.insert(0, str(Path(__file__).parent / "client"))

from dotenv import load_dotenv

# Only import what works
try:
    from services.ai_service import create_llm_model
    from services.mcp_service import setup_mcp_client, get_tools_from_client
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

load_dotenv()


class SimpleCountryAdder:
    """Just adds Country and CASEACCID columns - no re-evaluation."""
    
    def __init__(self, input_csv: str, output_csv: str, servers: str = "perplexity", batch_size: int = 1, resume: bool = False):
        self.input_csv = Path(input_csv)  # Original data for company info
        self.output_csv = Path(output_csv)  # Classification file to enhance
        self.batch_size = batch_size
        self.servers = servers
        self.resume = resume
        
        # MCP components
        self.client = None
        self.tools = []
        
        print(f"🌍 Simple Country Adder")
        print(f"   Input (for company info): {self.input_csv}")  
        print(f"   Output (to enhance): {self.output_csv}")
        print(f"   Resume mode: {'ON' if resume else 'OFF'}")
        print(f"   Will add: CASEACCID + Country columns only")
    
    def load_company_info(self) -> Dict[str, Dict]:
        """Load company info from input CSV for country search."""
        companies = {}
        
        try:
            with open(self.input_csv, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row_num, row in enumerate(reader, start=2):
                    account_name = row.get('Account Name', '').strip()
                    if account_name:
                        # Ensure CASEACCID
                        if not row.get("CASEACCID", "").strip():
                            row["CASEACCID"] = f"CASE{row_num-1:06d}"
                        companies[account_name] = row
        
        except Exception as e:
            raise Exception(f"Error reading input CSV: {e}")
        
        print(f"📊 Loaded company info for {len(companies)} companies")
        return companies
    
    def find_latest_progress_file(self) -> Optional[Path]:
        """Find the latest progress file in temp folder."""
        temp_dir = Path("temp")
        if not temp_dir.exists():
            return None
        
        # Look for both old and new progress file patterns
        old_pattern = f"{self.output_csv.stem}_progress_batch_*.csv"
        new_pattern = f"{self.output_csv.stem}_progress_company_*.csv"
        
        old_files = list(temp_dir.glob(old_pattern))
        new_files = list(temp_dir.glob(new_pattern))
        
        all_progress_files = old_files + new_files
        
        if not all_progress_files:
            return None
        
        # Sort by file modification time (most recent first)
        all_progress_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        latest_file = all_progress_files[0]
        
        print(f"🔍 Latest progress file found: {latest_file}")
        
        # Also show all available progress files for debugging
        print(f"📁 Available progress files:")
        for pfile in all_progress_files[:5]:  # Show top 5
            mtime = datetime.fromtimestamp(pfile.stat().st_mtime)
            print(f"   {pfile.name} (modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
        
        return latest_file
    
    def load_progress_file(self, progress_file: Path) -> Tuple[List[Dict], List[str], int]:
        """Load progress from file and determine where to continue."""
        try:
            classifications = []
            
            with open(progress_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                fieldnames = list(reader.fieldnames)
                
                for row in reader:
                    classifications.append(row)
            
            # Count how many are actually processed by checking for both CASEACCID AND Country
            processed_count = 0
            for i, classification in enumerate(classifications):
                caseaccid = classification.get("CASEACCID", "").strip()
                country = classification.get("Country", "").strip()
                
                # A row is processed if it has CASEACCID (even if Country is blank - that means we tried)
                # We count blank Country as processed too, since we attempted the search
                if caseaccid:
                    processed_count = i + 1  # +1 because index is 0-based
                else:
                    break  # Stop at first unprocessed row
            
            print(f"📊 Loaded progress: {len(classifications)} total rows")
            print(f"📊 Actually processed: {processed_count} companies")
            
            # Show some stats about what we found
            countries_found = sum(1 for c in classifications[:processed_count] if c.get("Country", "").strip())
            countries_blank = processed_count - countries_found
            
            print(f"📊 Countries found: {countries_found}, blank: {countries_blank}")
            
            return classifications, fieldnames, processed_count
            
        except Exception as e:
            print(f"⚠️ Error loading progress file: {e}")
            return [], [], 0
    def load_classifications(self) -> Tuple[List[Dict], List[str], int]:
        """Load existing classifications - with resume support."""
        
        # If resume mode, try to load from progress file first
        if self.resume:
            print("🔍 Scanning temp folder for progress files...")
            latest_progress = self.find_latest_progress_file()
            if latest_progress:
                print(f"🔄 Resuming from progress file...")
                return self.load_progress_file(latest_progress)
            else:
                print(f"⚠️ No progress file found in temp folder, starting from beginning")
        
        # Load from original output file
        classifications = []
        processed_count = 0
        
        try:
            with open(self.output_csv, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                fieldnames = list(reader.fieldnames)
                
                # Add new columns if not present
                if "CASEACCID" not in fieldnames:
                    fieldnames.insert(0, "CASEACCID")
                if "Country" not in fieldnames:
                    fieldnames.append("Country")
                
                for row in reader:
                    # Keep all existing data exactly as is
                    if "CASEACCID" not in row:
                        row["CASEACCID"] = ""
                    if "Country" not in row:
                        row["Country"] = ""
                    classifications.append(row)
        
        except Exception as e:
            raise Exception(f"Error reading classification CSV: {e}")
        
        print(f"📊 Loaded {len(classifications)} existing classifications")
        return classifications, fieldnames, processed_count
    
    async def setup_perplexity(self):
        """Setup Perplexity connection."""
        try:
            if not os.getenv("PERPLEXITY_API_KEY"):
                raise ValueError("PERPLEXITY_API_KEY required")
            
            config = {
                "Perplexity Search": {
                    "transport": "sse",
                    "url": "http://localhost:8001/sse", 
                    "timeout": 600
                }
            }
            
            self.client = await setup_mcp_client(config)
            self.tools = await get_tools_from_client(self.client)
            print("✅ Perplexity connected")
            
        except Exception as e:
            print(f"❌ Error setting up Perplexity: {e}")
            raise
    
    async def find_country(self, company_info: Dict) -> str:
        """Find country using Perplexity advanced search."""
        try:
            company_name = company_info.get("Account Name", "")
            domain = company_info.get("Domain", "")
            
            # Build search query
            search_query = f"{company_name} company headquarters location country"
            if domain:
                search_query += f" {domain}"
            
            # Find perplexity tool
            perplexity_tool = None
            for tool in self.tools:
                if "perplexity" in tool.name.lower():
                    perplexity_tool = tool
                    break
            
            if perplexity_tool:
                result = await perplexity_tool.ainvoke({"query": search_query})
                
                if result:
                    content = ""
                    if hasattr(result, 'content'):
                        content = result.content
                    elif isinstance(result, str):
                        content = result
                    
                    if content:
                        content_lower = content.lower()
                        
                        # Simple country detection
                        countries = {
                            "united states": "United States", "usa": "United States",
                            "germany": "Germany", "singapore": "Singapore", 
                            "japan": "Japan", "china": "China",
                            "france": "France", "united kingdom": "United Kingdom",
                            "uk": "United Kingdom", "canada": "Canada",
                            "australia": "Australia", "netherlands": "Netherlands",
                            "sweden": "Sweden", "switzerland": "Switzerland",
                            "india": "India", "israel": "Israel",
                            "south korea": "South Korea", "taiwan": "Taiwan",
                            "hong kong": "Hong Kong", "thailand": "Thailand",
                            "malaysia": "Malaysia", "indonesia": "Indonesia",
                            "philippines": "Philippines", "vietnam": "Vietnam",
                            "italy": "Italy", "spain": "Spain", "brazil": "Brazil"
                        }
                        
                        for pattern, country_name in countries.items():
                            if pattern in content_lower:
                                return country_name
            
            return ""
            
        except Exception as e:
            print(f"         ⚠️ Error finding country: {e}")
            return ""
    
    async def enhance_classifications(self, classifications: List[Dict], company_info: Dict[str, Dict], fieldnames: List[str], start_from: int = 0):
        """Add CASEACCID and Country to existing classifications."""
        print(f"\n🔄 Enhancing {len(classifications)} classifications...")
        print(f"   Starting from record: {start_from + 1}")
        print(f"   Saving progress every 50 companies to temp folder")
        
        total_batches = (len(classifications) + self.batch_size - 1) // self.batch_size
        start_batch = start_from // self.batch_size
        
        for batch_idx in range(start_batch * self.batch_size, len(classifications), self.batch_size):
            batch_num = (batch_idx // self.batch_size) + 1
            batch_classifications = classifications[batch_idx:batch_idx + self.batch_size]
            
            print(f"\n🔍 Processing batch {batch_num}/{total_batches} ({len(batch_classifications)} companies)")
            
            for i, classification in enumerate(batch_classifications):
                current_idx = batch_idx + i + 1
                
                # Skip if we're resuming and this record was already processed
                if current_idx <= start_from:
                    continue
                    
                company_name = classification.get("Company Name", "Unknown")
                print(f"   {current_idx}. {company_name}")
                
                # Add CASEACCID if not present
                if not classification.get("CASEACCID", "").strip():
                    if company_name in company_info:
                        classification["CASEACCID"] = company_info[company_name].get("CASEACCID", "")
                    else:
                        classification["CASEACCID"] = f"CASE{current_idx:06d}"
                    print(f"      ✅ Added CASEACCID: {classification['CASEACCID']}")
                
                # Add Country if not present
                if not classification.get("Country", "").strip():
                    if company_name in company_info:
                        country = await self.find_country(company_info[company_name])
                        classification["Country"] = country
                        
                        if country:
                            print(f"      ✅ Found country: {country}")
                        else:
                            print(f"      ⚪ Country not determined")
                    else:
                        print(f"      ⚠️ Company info not found in input CSV")
                        classification["Country"] = ""
                else:
                    print(f"      ✅ Country already exists: {classification['Country']}")
            
            # Save progress every 50 companies to temp folder
            if current_idx % 50 == 0 or batch_num == total_batches:
                self.save_progress_to_temp(classifications, fieldnames, current_idx, len(classifications))
                print(f"💾 Progress saved at company {current_idx}/{len(classifications)}")
        
        # Final save to actual output file
        print(f"\n💾 Saving final results...")
        self.save_enhanced_classifications(classifications, fieldnames)
    
    def save_progress_to_temp(self, classifications: List[Dict], fieldnames: List[str], current_idx: int, total_count: int):
        """Save progress to temp folder every 50 companies."""
        try:
            # Ensure temp folder exists
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            # Save progress file in temp
            progress_file = temp_dir / f"{self.output_csv.stem}_progress_company_{current_idx}.csv"
            
            with open(progress_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(classifications)
            
            print(f"      📁 Progress saved to: {progress_file}")
            
        except Exception as e:
            print(f"      ⚠️ Error saving progress: {e}")
    
    def save_enhanced_classifications(self, classifications: List[Dict], fieldnames: List[str]):
        """Save final enhanced classifications with atomic operation."""
        try:
            # Create backup if file exists
            if self.output_csv.exists():
                backup_path = self.output_csv.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                self.output_csv.rename(backup_path)
                print(f"📁 Created backup: {backup_path.name}")
            
            # Write to temp file first
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            temp_output = temp_dir / f"{self.output_csv.stem}_final.tmp"
            
            with open(temp_output, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(classifications)
            
            # Atomic move to final location
            temp_output.replace(self.output_csv)
            print(f"✅ Final results saved to: {self.output_csv}")
            
        except Exception as e:
            print(f"❌ Error saving final file: {e}")
            raise
    
    async def run(self):
        """Main execution - just add the 2 columns."""
        try:
            # Step 1: Load company info from input
            print("\n🔄 Step 1: Loading company info...")
            company_info = self.load_company_info()
            
            # Step 2: Load existing classifications 
            print("\n🔄 Step 2: Loading existing classifications...")
            classifications, fieldnames, processed_count = self.load_classifications()
            
            # Step 3: Setup Perplexity
            print("\n🔄 Step 3: Setting up Perplexity...")
            await self.setup_perplexity()
            
            # Step 4: Enhance with CASEACCID and Country
            print("\n🔄 Step 4: Adding CASEACCID and Country columns...")
            await self.enhance_classifications(classifications, company_info, fieldnames, processed_count)
            
            # Step 5: Summary
            print("\n📊 Enhancement Summary:")
            caseaccids_added = sum(1 for c in classifications if c.get("CASEACCID", "").strip())
            countries_found = sum(1 for c in classifications if c.get("Country", "").strip())
            
            print(f"   Total classifications: {len(classifications)}")
            print(f"   CASEACCIDs added: {caseaccids_added}")
            print(f"   Countries found: {countries_found}")
            
            print(f"\n✅ Enhancement completed!")
            print(f"   All existing classifications preserved")
            print(f"   Added CASEACCID and Country columns only")
            print(f"   Results saved to: {self.output_csv}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            traceback.print_exc()
            raise
        
        finally:
            if self.client:
                try:
                    await self.client.__aexit__(None, None, None)
                except:
                    pass


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Add CASEACCID and Country to existing classifications")
    
    parser.add_argument("--input", "-i", required=True, help="Input CSV (for company info)")
    parser.add_argument("--output", "-o", required=True, help="Output CSV (classifications to enhance)")
    parser.add_argument("--servers", "-s", default="perplexity", help="Servers (default: perplexity)")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from latest progress file in temp folder")
    
    args = parser.parse_args()
    
    try:
        cli = SimpleCountryAdder(args.input, args.output, args.servers, args.batch_size, args.resume)
        await cli.run()
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted")
        return 1
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted")
        sys.exit(1)