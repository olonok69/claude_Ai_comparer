#!/usr/bin/env python3
"""
Company Classification Validator and Re-evaluator
Validates existing classifications and re-processes invalid entries.
Uses MCP servers for Perplexity search and Azure OpenAI classification.
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

# Add the client directory to the path
sys.path.insert(0, str(Path(__file__).parent / "client"))

from dotenv import load_dotenv

# Import MCP-related modules
try:
    from services.mcp_service import setup_mcp_client, get_tools_from_client, prepare_server_config
    from services.ai_service import create_llm_model
    from langchain_core.messages import HumanMessage, AIMessage
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, create_react_agent
    from typing_extensions import TypedDict, Annotated
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all required modules are installed")
    sys.exit(1)

# Load environment variables
load_dotenv()


class TaxonomyValidator:
    """Validates company classifications against taxonomy."""
    
    def __init__(self, taxonomy_path: Optional[str] = None):
        self.valid_pairs = set()
        self.taxonomy_loaded = False
        # Don't load from file - will get from MCP server
        
    def load_taxonomy_from_mcp_response(self, taxonomy_text: str):
        """Load taxonomy pairs from MCP server response."""
        try:
            # Parse the response which contains exact pairs
            lines = taxonomy_text.split('\n')
            for line in lines:
                if '|' in line and not line.startswith('#'):
                    # Extract industry and product from format: "Industry | Product"
                    parts = line.split('|')
                    if len(parts) == 2:
                        # Remove any numbering like "1. " from the beginning
                        industry_part = parts[0].strip()
                        # Remove number prefix if present
                        if '. ' in industry_part:
                            industry_part = industry_part.split('. ', 1)[-1]
                        
                        industry = industry_part.strip()
                        product = parts[1].strip()
                        
                        if industry and product:
                            self.valid_pairs.add((industry, product))
            
            self.taxonomy_loaded = True
            print(f"   ✅ Loaded {len(self.valid_pairs)} taxonomy pairs from MCP server")
            
            # Show sample pairs
            if self.valid_pairs:
                sample = list(self.valid_pairs)[:3]
                for ind, prod in sample:
                    print(f"      • {ind} | {prod}")
                    
        except Exception as e:
            print(f"   ⚠️ Error parsing taxonomy from MCP: {e}")
    
    def is_valid_pair(self, industry: str, product: str) -> bool:
        """Check if an industry-product pair is valid."""
        if not self.taxonomy_loaded:
            # If taxonomy not loaded yet, assume invalid to force loading
            return False
        return (industry.strip(), product.strip()) in self.valid_pairs
    

    
    def is_valid_pair(self, industry: str, product: str) -> bool:
        """Check if an industry-product pair is valid."""
        return (industry.strip(), product.strip()) in self.valid_pairs
    
    def validate_row(self, row: Dict) -> Tuple[bool, List[str], List[Tuple[str, str]]]:
        """
        Validate a single row.
        Returns: (is_valid, invalid_reasons, valid_pairs)
        """
        invalid_reasons = []
        valid_pairs = []
        has_any_classification = False
        
        for i in range(1, 5):
            industry = row.get(f'Tech Industry {i}', '').strip()
            product = row.get(f'Tech Product {i}', '').strip()
            
            if industry or product:
                has_any_classification = True
                
                if industry and product:
                    if self.is_valid_pair(industry, product):
                        valid_pairs.append((industry, product))
                    else:
                        invalid_reasons.append(f"Invalid pair {i}: {industry} | {product}")
                elif industry and not product:
                    invalid_reasons.append(f"Pair {i}: Industry without product")
                elif product and not industry:
                    invalid_reasons.append(f"Pair {i}: Product without industry")
        
        if not has_any_classification:
            invalid_reasons.append("No classifications found")
        
        is_valid = len(invalid_reasons) == 0
        return is_valid, invalid_reasons, valid_pairs
    
    def validate_csv(self, csv_path: str) -> Dict:
        """Validate entire CSV file against taxonomy."""
        results = {
            'validation_report': {
                'total_rows': 0,
                'valid_rows': 0,
                'invalid_rows': 0,
                'blank_rows': 0
            },
            'flagged_rows': [],
            'invalid_details': [],
            'row_data': {}  # Store row data for re-evaluation
        }
        
        # Debug: Show taxonomy status
        print(f"   🔍 Validation using {len(self.valid_pairs)} taxonomy pairs")
        if len(self.valid_pairs) == 0:
            print("   ⚠️ WARNING: No taxonomy loaded! All pairs will be marked as invalid!")
            print("   ℹ️ This happens when taxonomy hasn't been loaded from MCP server yet")
        elif len(self.valid_pairs) < 10:
            print(f"   ⚠️ Only {len(self.valid_pairs)} pairs loaded - seems incomplete")
            print("   📋 Loaded pairs:", self.valid_pairs)
        
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                
                # Track unique pairs found in file
                file_pairs = set()
                
                for idx, row in enumerate(reader):
                    results['validation_report']['total_rows'] += 1
                    company_name = row.get('Company Name', '')
                    
                    is_valid, invalid_reasons, valid_pairs = self.validate_row(row)
                    
                    # Collect all pairs from file for debugging
                    for i in range(1, 5):
                        ind = row.get(f'Tech Industry {i}', '').strip()
                        prod = row.get(f'Tech Product {i}', '').strip()
                        if ind and prod:
                            file_pairs.add((ind, prod))
                    
                    if not is_valid:
                        results['flagged_rows'].append(idx)
                        results['row_data'][idx] = {
                            'row': row,
                            'valid_pairs': valid_pairs,
                            'invalid_reasons': invalid_reasons
                        }
                        
                        if "No classifications found" in invalid_reasons:
                            results['validation_report']['blank_rows'] += 1
                        else:
                            results['validation_report']['invalid_rows'] += 1
                            results['invalid_details'].append({
                                'row_index': idx,
                                'company_name': company_name,
                                'reason': '; '.join(invalid_reasons)
                            })
                    else:
                        results['validation_report']['valid_rows'] += 1
                
                # Debug: Show what pairs are in the file
                if file_pairs and len(self.valid_pairs) == 0:
                    print(f"\n   📊 Found {len(file_pairs)} unique pairs in output file")
                    print("   📋 Sample pairs from file (first 5):")
                    for pair in list(file_pairs)[:5]:
                        print(f"      • {pair[0]} | {pair[1]}")
                    print("   ⚠️ But taxonomy is empty, so all marked invalid!")
        
        except Exception as e:
            print(f"   ⚠️ Error validating CSV: {e}")
        
        return results
    
    def create_backup(self, csv_path: str) -> Path:
        """Create a backup of the CSV file."""
        csv_path = Path(csv_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = csv_path.with_suffix(f'.backup_{timestamp}.csv')
        
        import shutil
        shutil.copy2(csv_path, backup_path)
        print(f"   Backup created: {backup_path.name}")
        return backup_path


class CompanyValidatorCLI:
    """Validates and re-evaluates company classifications using MCP servers."""
    
    def __init__(self, input_csv: str, output_csv: str, taxonomy_path: Optional[str] = None, 
                 servers: str = "perplexity", batch_size: int = 1):
        self.input_csv = Path(input_csv)
        self.output_csv = Path(output_csv)
        
        # Initialize validator (taxonomy will be loaded from MCP)
        self.validator = TaxonomyValidator()
        
        # Re-evaluation settings
        self.batch_size = batch_size
        self.servers = servers
        
        # MCP components (to be initialized)
        self.agent = None
        self.llm = None
        self.client = None
        self.tools = []
        
        print(f"🔍 Company Validator initialized (12-column format with MCP)")
        print(f"   Input CSV: {self.input_csv}")
        print(f"   Output CSV: {self.output_csv}")
        print(f"   Taxonomy will be loaded from company_tagging MCP server")
        print(f"   MCP servers for re-evaluation: {servers}")
        print(f"   Batch size for re-evaluation: {self.batch_size}")
    
    def sanitize_name_field(self, value: str) -> str:
        """Remove problematic punctuation from name fields."""
        if not value:
            return value
        
        problematic_chars = [',', ';', '|', '\t', '\n', '\r']
        sanitized = value
        for char in problematic_chars:
            sanitized = sanitized.replace(char, ' ')
        
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        return sanitized
    
    async def setup_mcp_connections(self):
        """Setup MCP connections for Perplexity and company_tagging."""
        print("\n🔧 Setting up MCP connections...")
        
        try:
            # Load server configuration
            SERVER_CONFIG = {
                'mcpServers': {
                    "Company Tagging": {
                        "transport": "stdio",
                        "command": "python",
                        "args": ["-m", "client.mcp_servers.company_tagging.server"],
                        "env": {
                            "PERPLEXITY_API_KEY": "${PERPLEXITY_API_KEY}",
                            "PERPLEXITY_MODEL": "${PERPLEXITY_MODEL}"
                        },
                        "cwd": "."
                    },
                    "Perplexity Search": {
                        "transport": "sse",
                        "url": "http://localhost:8001/sse",
                        "timeout": 600,
                        "headers": None,
                        "sse_read_timeout": 900
                    },
                    "Google Search": {
                        "transport": "sse",
                        "url": "http://localhost:8002/sse",
                        "timeout": 600,
                        "headers": None,
                        "sse_read_timeout": 900
                    }
                }
            }
            
            # Check for custom config file
            config_path = "servers_config.json"
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    SERVER_CONFIG = config_data
            
            all_servers = SERVER_CONFIG.get('mcpServers', {})
            
            # Filter servers based on what we need
            filtered_servers = {}
            
            # Always include Company Tagging
            if "Company Tagging" in all_servers:
                filtered_servers["Company Tagging"] = all_servers["Company Tagging"]
            
            # Add servers based on selection
            if "perplexity" in self.servers and "Perplexity Search" in all_servers:
                filtered_servers["Perplexity Search"] = all_servers["Perplexity Search"]
            
            if "google" in self.servers and "Google Search" in all_servers:
                filtered_servers["Google Search"] = all_servers["Google Search"]
            
            print(f"   📦 Configuring {len(filtered_servers)} MCP servers...")
            for server_name in filtered_servers:
                print(f"      • {server_name}")
            
            # Prepare server configuration (expand environment variables)
            prepared_servers = prepare_server_config(filtered_servers)
            
            # Setup MCP client with ALL servers at once
            print("   🔌 Connecting to MCP servers...")
            self.client = await setup_mcp_client(prepared_servers)
            
            if not self.client:
                print("   ❌ Failed to create MCP client")
                return False
            
            print("   ✅ MCP client created")
            
            # Get tools from the client
            self.tools = await get_tools_from_client(self.client)
            
            if not self.tools:
                print("   ❌ No tools loaded from MCP servers")
                return False
            
            print(f"   📦 Loaded {len(self.tools)} tools total")
            
            # Create LLM
            print("   🤖 Creating LLM model...")
            # Determine provider based on environment variables
            if os.getenv('AZURE_API_KEY') and os.getenv('AZURE_ENDPOINT'):
                llm_provider = "Azure OpenAI"
            elif os.getenv('OPENAI_API_KEY'):
                llm_provider = "OpenAI"
            else:
                print("   ❌ No AI provider credentials found (need AZURE_API_KEY or OPENAI_API_KEY)")
                return False
            
            self.llm = create_llm_model(llm_provider)
            if not self.llm:
                print("   ❌ Failed to create LLM model. Check your API keys.")
                return False
            print(f"   ✅ LLM model initialized ({llm_provider})")
            
            # Create agent with tools using create_react_agent
            self.agent = create_react_agent(self.llm, self.tools)
            print(f"   ✅ Agent created with {len(self.tools)} tools")
            
            # Load taxonomy from MCP server
            await self._load_taxonomy_from_mcp()
            
            return True
            
        except Exception as e:
            print(f"❌ Error setting up MCP connections: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _load_taxonomy_from_mcp(self):
        """Load taxonomy using search_show_categories tool."""
        print("   📋 Loading taxonomy from company_tagging MCP server...")
        
        try:
            # Call search_show_categories to get all taxonomy pairs
            prompt = """Use the search_show_categories tool to get ALL taxonomy pairs.
            
            Call it with no filters to get the complete list.
            Return ONLY the exact pairs, no additional text."""
            
            conversation = [HumanMessage(content=prompt)]
            response = await self.agent.ainvoke({"messages": conversation})
            
            # Extract taxonomy from response
            if "messages" in response:
                for msg in response["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        content = str(msg.content)
                        if "EXACT INDUSTRY | PRODUCT PAIRS" in content or "|" in content:
                            self.validator.load_taxonomy_from_mcp_response(content)
                            break
            
            if not self.validator.taxonomy_loaded:
                print("   ⚠️ Could not load taxonomy from MCP server")
                
        except Exception as e:
            print(f"   ⚠️ Error loading taxonomy from MCP: {e}")
    

    
    def validate_and_flag(self) -> Dict[str, any]:
        """Validate output CSV and identify rows needing re-evaluation."""
        print("\n📋 Step 1: Checking output file format...")
        
        if not self.output_csv.exists():
            raise FileNotFoundError(f"Output CSV not found: {self.output_csv}")
        
        # Check format
        with open(self.output_csv, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            
            has_caseaccid = 'CASEACCID' in headers
            has_country = 'Country' in headers
            
            if has_caseaccid and has_country:
                print("   ✅ Detected 12-column format (with CASEACCID and Country)")
            elif has_caseaccid:
                print("   ⚠️ Detected 11-column format (with CASEACCID, missing Country)")
            else:
                print("   ⚠️ Detected old format (missing CASEACCID and Country)")
        
        print("\n📋 Step 2: Validating existing classifications...")
        
        # IMPORTANT: Check if taxonomy is loaded
        if not self.validator.taxonomy_loaded or len(self.validator.valid_pairs) == 0:
            print("   ⚠️ WARNING: Taxonomy not loaded yet!")
            print("   ℹ️ Will load taxonomy from MCP server before validation")
            return {
                'validation_report': {
                    'total_rows': 0,
                    'valid_rows': 0,
                    'invalid_rows': 0,
                    'blank_rows': 0
                },
                'flagged_rows': [],
                'invalid_details': [],
                'row_data': {},
                'needs_taxonomy': True  # Flag to indicate we need to load taxonomy
            }
        
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
    
    async def re_evaluate_flagged_rows(self, validation_results: Dict) -> Dict[int, Dict[str, str]]:
        """Re-evaluate flagged rows using MCP servers."""
        flagged_rows = validation_results['flagged_rows']
        row_data = validation_results['row_data']
        
        if not flagged_rows:
            print("✅ No rows to re-evaluate!")
            return {}
        
        print(f"\n🔄 Step 3: Re-evaluating {len(flagged_rows)} flagged rows...")
        print(f"   Using MCP servers: Perplexity for research + Azure OpenAI for classification")
        
        # Get original input data
        input_data = self._load_input_data()
        
        updated_rows = {}
        
        for i, row_idx in enumerate(flagged_rows):
            row_info = row_data[row_idx]
            output_row = row_info['row']
            valid_pairs = row_info['valid_pairs']
            invalid_reasons = row_info['invalid_reasons']
            
            company_name = output_row.get('Company Name', '')
            caseaccid = output_row.get('CASEACCID', '')
            
            print(f"\n🔍 Processing {i+1}/{len(flagged_rows)}: Row {row_idx + 1} - {company_name}")
            
            # Find matching input data
            input_row = self._find_input_row(input_data, caseaccid, company_name)
            if not input_row:
                print(f"   ⚠️ Could not find input data for this company")
                updated_rows[row_idx] = self._create_blank_result(output_row)
                continue
            
            # Show current issues
            if valid_pairs:
                print(f"   📌 Valid existing pairs: {len(valid_pairs)}")
                for industry, product in valid_pairs[:2]:  # Show first 2
                    print(f"      ✓ {industry} | {product}")
            
            print(f"   ❌ Issues: {'; '.join(invalid_reasons[:2])}")  # Show first 2 issues
            
            try:
                # Create prompt for re-evaluation
                prompt = self._create_reevaluation_prompt(input_row, output_row, valid_pairs)
                
                # Process with agent
                print(f"   🔄 Calling MCP servers for re-evaluation...")
                conversation = [HumanMessage(content=prompt)]
                
                response = await asyncio.wait_for(
                    self.agent.ainvoke({"messages": conversation}),
                    timeout=300  # 5 minutes timeout
                )
                
                # Extract result
                result = self._extract_classification_result(response, output_row)
                
                if result:
                    # Validate the new classification
                    is_valid = self._validate_classification_result(result)
                    
                    if is_valid:
                        updated_rows[row_idx] = result
                        print(f"   ✅ Successfully re-classified with valid taxonomy")
                    else:
                        print(f"   ⚠️ New classification still has invalid pairs, keeping original")
                        updated_rows[row_idx] = self._keep_valid_pairs_only(output_row, valid_pairs)
                else:
                    print(f"   ⚠️ Re-classification failed, keeping valid pairs only")
                    updated_rows[row_idx] = self._keep_valid_pairs_only(output_row, valid_pairs)
                    
            except asyncio.TimeoutError:
                print(f"   ⌛ Timeout during re-evaluation")
                updated_rows[row_idx] = self._keep_valid_pairs_only(output_row, valid_pairs)
            except Exception as e:
                print(f"   ❌ Error during re-evaluation: {e}")
                updated_rows[row_idx] = self._keep_valid_pairs_only(output_row, valid_pairs)
            
            # Save intermediate results
            if (i + 1) % 50 == 0 or (i + 1) == len(flagged_rows):
                temp_folder = Path("temp")
                temp_folder.mkdir(exist_ok=True)
                
                intermediate_path = temp_folder / f"validation_intermediate_{i+1}.json"
                try:
                    with open(intermediate_path, 'w', encoding='utf-8') as f:
                        json.dump(updated_rows, f, indent=2, ensure_ascii=False)
                    print(f"   💾 Saved intermediate results: {len(updated_rows)} rows to temp/{intermediate_path.name}")
                except Exception as e:
                    print(f"   ⚠️ Failed to save intermediate results: {e}")
        
        # Save final results
        temp_folder = Path("temp")
        temp_folder.mkdir(exist_ok=True)
        final_results_path = temp_folder / "validation_final_results.json"
        try:
            with open(final_results_path, 'w', encoding='utf-8') as f:
                json.dump(updated_rows, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Saved final validation results: {len(updated_rows)} rows to temp/{final_results_path.name}")
        except Exception as e:
            print(f"⚠️ Failed to save final results: {e}")
        
        return updated_rows
    
    def _load_input_data(self) -> List[Dict]:
        """Load input CSV data."""
        data = []
        try:
            with open(self.input_csv, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
        except Exception as e:
            print(f"   ⚠️ Error loading input data: {e}")
        return data
    
    def _find_input_row(self, input_data: List[Dict], caseaccid: str, company_name: str) -> Optional[Dict]:
        """Find matching row in input data."""
        # Try CASEACCID first
        if caseaccid:
            for row in input_data:
                if row.get('CASEACCID') == caseaccid:
                    return row
        
        # Try company name
        for row in input_data:
            if row.get('Account Name') == company_name:
                return row
        
        return None
    
    def _create_reevaluation_prompt(self, input_row: Dict, output_row: Dict, valid_pairs: List[Tuple[str, str]]) -> str:
        """Create prompt for re-evaluation using MCP tools."""
        company_name = input_row.get('Account Name', '')
        trading_name = input_row.get('Trading Name', '')
        domain = input_row.get('Domain', '')
        industry = input_row.get('Industry', '')
        product_service = input_row.get('Product/Service Type', '')
        caseaccid = output_row.get('CASEACCID', '')
        
        # Format valid pairs as context
        valid_context = ""
        if valid_pairs:
            valid_context = "\nCurrently valid classifications to keep:\n"
            for i, (ind, prod) in enumerate(valid_pairs, 1):
                valid_context += f"- Pair {i}: {ind} | {prod}\n"
        
        return f"""You need to re-classify this technology company using the correct taxonomy.

COMPANY INFORMATION:
- CASEACCID: {caseaccid}
- Company Name: {company_name}
- Trading Name: {trading_name}
- Domain: {domain}
- Industry Context: {industry}
- Product/Service: {product_service}
{valid_context}

REQUIRED STEPS:

1. FIRST: Use search_show_categories tool to get the COMPLETE list of valid taxonomy pairs

2. RESEARCH: Use perplexity_search or perplexity_advanced_search to research:
   - What technology products/services does {company_name} offer?
   - What is their main business focus in technology?
   - What country are they headquartered in?
   - Search query suggestions: "{company_name} technology software platform services products"

3. CLASSIFY: Based on research and ONLY using exact pairs from the taxonomy:
   - Select up to 4 relevant Industry|Product pairs
   - Keep any currently valid pairs if they still apply
   - Each pair MUST exist exactly in the taxonomy

4. OUTPUT: Provide ONLY the data row (no headers) in this exact format with 12 columns:
   {caseaccid}|{self.sanitize_name_field(company_name)}|{self.sanitize_name_field(trading_name)}|Tech Industry 1|Tech Product 1|Tech Industry 2|Tech Product 2|Tech Industry 3|Tech Product 3|Tech Industry 4|Tech Product 4|Country

CRITICAL: Only use Industry|Product pairs that exist EXACTLY in the taxonomy from search_show_categories.
If you cannot find valid matches, leave those fields blank."""
    
    def _extract_classification_result(self, response: any, original_row: Dict) -> Optional[Dict]:
        """Extract classification result from agent response."""
        try:
            # Look for the assistant's final message
            if "messages" in response:
                for msg in reversed(response["messages"]):
                    if isinstance(msg, AIMessage) and hasattr(msg, "content") and msg.content:
                        content = str(msg.content)
                        
                        # Look for pipe-separated data
                        lines = content.strip().split('\n')
                        for line in lines:
                            if "|" in line and not line.startswith("#") and not line.startswith("---"):
                                parts = [p.strip() for p in line.split('|')]
                                
                                if len(parts) >= 12:
                                    return {
                                        'CASEACCID': parts[0],
                                        'Company Name': self.sanitize_name_field(parts[1]),
                                        'Trading Name': self.sanitize_name_field(parts[2]),
                                        'Tech Industry 1': parts[3],
                                        'Tech Product 1': parts[4],
                                        'Tech Industry 2': parts[5],
                                        'Tech Product 2': parts[6],
                                        'Tech Industry 3': parts[7],
                                        'Tech Product 3': parts[8],
                                        'Tech Industry 4': parts[9],
                                        'Tech Product 4': parts[10],
                                        'Country': parts[11] if len(parts) > 11 else ''
                                    }
            
            return None
            
        except Exception as e:
            print(f"      ⚠️ Error extracting result: {e}")
            return None
    
    def _validate_classification_result(self, result: Dict) -> bool:
        """Validate that all pairs in result are valid according to taxonomy."""
        all_valid = True
        
        for i in range(1, 5):
            industry = result.get(f'Tech Industry {i}', '').strip()
            product = result.get(f'Tech Product {i}', '').strip()
            
            if industry and product:
                if not self.validator.is_valid_pair(industry, product):
                    print(f"      ❌ Invalid pair {i}: {industry} | {product}")
                    all_valid = False
                else:
                    print(f"      ✅ Valid pair {i}: {industry} | {product}")
        
        return all_valid
    
    def _keep_valid_pairs_only(self, row: Dict, valid_pairs: List[Tuple[str, str]]) -> Dict:
        """Create result keeping only valid pairs."""
        result = {
            'CASEACCID': row.get('CASEACCID', ''),
            'Company Name': self.sanitize_name_field(row.get('Company Name', '')),
            'Trading Name': self.sanitize_name_field(row.get('Trading Name', '')),
            'Country': row.get('Country', '')
        }
        
        # Add valid pairs
        for i in range(1, 5):
            if i <= len(valid_pairs):
                industry, product = valid_pairs[i-1]
                result[f'Tech Industry {i}'] = industry
                result[f'Tech Product {i}'] = product
            else:
                result[f'Tech Industry {i}'] = ''
                result[f'Tech Product {i}'] = ''
        
        return result
    
    def _create_blank_result(self, row: Dict) -> Dict[str, str]:
        """Create a result with all tech fields blank."""
        return {
            'CASEACCID': row.get('CASEACCID', ''),
            'Company Name': self.sanitize_name_field(row.get('Company Name', '')),
            'Trading Name': self.sanitize_name_field(row.get('Trading Name', '')),
            'Tech Industry 1': '',
            'Tech Product 1': '',
            'Tech Industry 2': '',
            'Tech Product 2': '',
            'Tech Industry 3': '',
            'Tech Product 3': '',
            'Tech Industry 4': '',
            'Tech Product 4': '',
            'Country': row.get('Country', '')
        }
    
    def update_output_file(self, updates: Dict[int, Dict[str, str]]):
        """Update the output file with validated results."""
        if not updates:
            print("\n✅ No updates needed!")
            return
        
        print(f"\n📝 Step 4: Updating output file with {len(updates)} validated rows...")
        
        # Read current file
        rows = []
        fieldnames = None
        
        with open(self.output_csv, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            # Ensure 12-column format
            expected_fields = [
                'CASEACCID', 'Company Name', 'Trading Name',
                'Tech Industry 1', 'Tech Product 1',
                'Tech Industry 2', 'Tech Product 2',
                'Tech Industry 3', 'Tech Product 3',
                'Tech Industry 4', 'Tech Product 4',
                'Country'
            ]
            
            if not all(field in fieldnames for field in expected_fields):
                fieldnames = expected_fields
            
            for i, row in enumerate(reader):
                if i in updates:
                    rows.append(updates[i])
                else:
                    # Sanitize existing row's name fields
                    row['Company Name'] = self.sanitize_name_field(row.get('Company Name', ''))
                    row['Trading Name'] = self.sanitize_name_field(row.get('Trading Name', ''))
                    rows.append(row)
        
        # Write updated file
        with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print("✅ Output file updated successfully!")
    
    def generate_final_report(self, initial_validation: Dict, updates: Dict):
        """Generate final validation report."""
        print("\n📊 Step 5: Generating final report...")
        
        # Re-validate the updated file
        final_validation = self.validator.validate_csv(str(self.output_csv))
        
        # Generate report
        report_path = self.output_csv.parent / f"{self.output_csv.stem}_validation_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Validation Report for {self.output_csv.name}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            initial = initial_validation['validation_report']
            final = final_validation['validation_report']
            
            f.write("| Metric | Before | After | Change |\n")
            f.write("|--------|--------|-------|--------|\n")
            f.write(f"| Total Rows | {initial['total_rows']} | {final['total_rows']} | - |\n")
            f.write(f"| Valid Rows | {initial['valid_rows']} | {final['valid_rows']} | {final['valid_rows'] - initial['valid_rows']:+d} |\n")
            f.write(f"| Invalid Rows | {initial['invalid_rows']} | {final['invalid_rows']} | {final['invalid_rows'] - initial['invalid_rows']:+d} |\n")
            f.write(f"| Blank Rows | {initial['blank_rows']} | {final['blank_rows']} | {final['blank_rows'] - initial['blank_rows']:+d} |\n")
            
            if final_validation['invalid_details']:
                f.write("\n## Remaining Invalid Entries\n\n")
                for detail in final_validation['invalid_details'][:20]:  # Show first 20
                    f.write(f"- Row {detail['row_index'] + 1}: {detail['company_name']}\n")
                    f.write(f"  - Reason: {detail['reason']}\n")
        
        print(f"   Report saved to: {report_path.name}")
        
        # Print summary
        print(f"\n📈 Improvement Summary:")
        print(f"   Invalid rows: {initial['invalid_rows']} → {final['invalid_rows']}")
        print(f"   Blank rows: {initial['blank_rows']} → {final['blank_rows']}")
        print(f"   Total issues: {initial['invalid_rows'] + initial['blank_rows']} → {final['invalid_rows'] + final['blank_rows']}")
        
        if final['invalid_rows'] == 0 and final['blank_rows'] == 0:
            print("\n🎉 Perfect! All rows now have valid taxonomy classifications!")
        elif final['invalid_rows'] == 0:
            print(f"\n✅ All non-blank rows have valid taxonomy classifications!")
            print(f"   {final['blank_rows']} rows remain blank (no suitable taxonomy matches found)")
    
    async def cleanup(self):
        """Cleanup MCP connections."""
        if hasattr(self, 'client') and self.client:
            try:
                await self.client.__aexit__(None, None, None)
                print("   🧹 Closed MCP connections")
            except Exception as e:
                print(f"   ⚠️ Error closing connections: {e}")


async def main():
    """Main entry point for the validator CLI."""
    parser = argparse.ArgumentParser(
        description="Validate and re-evaluate company classifications using MCP servers"
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
                       help="Batch size for re-evaluation (default: 1)")
    parser.add_argument("--report-only", action="store_true",
                       help="Only generate validation report without re-evaluation")
    
    args = parser.parse_args()
    
    try:
        # Initialize validator
        validator_cli = CompanyValidatorCLI(
            args.input, 
            args.output, 
            args.taxonomy, 
            args.servers,
            args.batch_size
        )
        
        # IMPORTANT: Setup MCP connections FIRST to load taxonomy
        print("\n🔄 Setting up MCP connections to load taxonomy...")
        success = await validator_cli.setup_mcp_connections()
        if not success:
            print("⚠️ Warning: Could not connect to MCP servers")
            print("ℹ️ Attempting to load taxonomy from local file...")
            
            # Try to load from local file as fallback
            taxonomy_path = args.taxonomy or 'classes.csv'
            if Path(taxonomy_path).exists():
                with open(taxonomy_path, 'r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        industry = row.get('Industry', '').strip()
                        product = row.get('Product', '').strip()
                        if industry and product:
                            validator_cli.validator.valid_pairs.add((industry, product))
                    validator_cli.validator.taxonomy_loaded = True
                    print(f"   ✅ Loaded {len(validator_cli.validator.valid_pairs)} pairs from {taxonomy_path}")
        
        # NOW validate with taxonomy loaded
        print("\n📋 Validating output file with loaded taxonomy...")
        validation_results = validator_cli.validate_and_flag()
        
        if args.report_only:
            # Generate report only
            validator_cli.generate_final_report(validation_results, {})
            return 0
        
        # Check if re-evaluation is needed
        if not validation_results.get('flagged_rows'):
            print("\n✅ All rows are valid! No re-evaluation needed.")
            return 0
        
        # Step 2: Create backup
        print("\n💾 Creating backup...")
        backup_path = validator_cli.validator.create_backup(str(validator_cli.output_csv))
        
        # Step 3: Re-evaluate flagged rows (MCP already connected)
        updates = await validator_cli.re_evaluate_flagged_rows(validation_results)
        
        # Step 4: Update output file
        validator_cli.update_output_file(updates)
        
        # Step 5: Generate final report
        validator_cli.generate_final_report(validation_results, updates)
        
        print(f"\n✅ Validation and re-evaluation completed!")
        print(f"   Backup saved to: {backup_path}")
        print(f"   Format: 12 columns (CASEACCID + Names + 4 Tech Pairs + Country)")
        
        # Cleanup
        await validator_cli.cleanup()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))