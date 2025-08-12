#!/usr/bin/env python3
"""
Enhanced Company Classification CLI Tool
Integrates validation, country inference, and taxonomy enforcement in a single workflow.
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import pickle
import traceback
import shutil
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime

# Add the client directory to the path so we can import existing modules
sys.path.insert(0, str(Path(__file__).parent / "client"))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from client.services.ai_service import create_llm_model
from client.services.mcp_service import setup_mcp_client, get_tools_from_client, prepare_server_config
from langgraph.prebuilt import create_react_agent
from client.config import SERVER_CONFIG, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS

# Load environment variables
load_dotenv()


class TaxonomyValidator:
    """Validates company classifications against official taxonomy from MCP server."""
    
    def __init__(self):
        self.valid_pairs = set()
        self.taxonomy_loaded = False
    
    def load_taxonomy_from_mcp_response(self, taxonomy_text: str):
        """Load taxonomy pairs from MCP server response."""
        try:
            lines = taxonomy_text.split('\n')
            for line in lines:
                if '|' in line and not line.startswith('#'):
                    parts = line.split('|')
                    if len(parts) == 2:
                        industry_part = parts[0].strip()
                        # Remove number prefix if present
                        if '. ' in industry_part:
                            industry_part = industry_part.split('. ', 1)[-1]
                        
                        industry = industry_part.strip()
                        product = parts[1].strip()
                        
                        if industry and product:
                            self.valid_pairs.add((industry, product))
            
            self.taxonomy_loaded = True
            print(f"✅ Loaded {len(self.valid_pairs)} taxonomy pairs from MCP server")
            
        except Exception as e:
            print(f"❌ Error parsing taxonomy from MCP: {e}")
    
    def is_valid_pair(self, industry: str, product: str) -> bool:
        """Check if an industry-product pair exists in taxonomy."""
        if not industry or not product:
            return False
        return (industry.strip(), product.strip()) in self.valid_pairs
    
    def validate_and_filter_pairs(self, classification: Dict) -> Dict:
        """Validate classification and keep only valid pairs."""
        validated = {
            'CASEACCID': classification.get('CASEACCID', ''),
            'Company Name': classification.get('Company Name', ''),
            'Trading Name': classification.get('Trading Name', ''),
            'Country': classification.get('Country', '')
        }
        
        valid_pair_count = 0
        for i in range(1, 5):
            industry = classification.get(f'Tech Industry {i}', '').strip()
            product = classification.get(f'Tech Product {i}', '').strip()
            
            if industry and product and self.is_valid_pair(industry, product):
                validated[f'Tech Industry {valid_pair_count + 1}'] = industry
                validated[f'Tech Product {valid_pair_count + 1}'] = product
                valid_pair_count += 1
        
        # Fill remaining slots with empty strings
        for i in range(valid_pair_count + 1, 5):
            validated[f'Tech Industry {i}'] = ''
            validated[f'Tech Product {i}'] = ''
        
        return validated


class ProgressTracker:
    """FIXED progress tracker with immediate saves and atomic operations."""
    
    def __init__(self, output_base: str):
        self.output_base = output_base
        self.progress_file = f"{output_base}_progress.pkl"
        self.temp_results_file = f"{output_base}_temp_results.json"
        self.error_log_file = f"{output_base}_errors.log"
        
        # Progress tracking
        self.processed_batches = set()
        self.failed_batches = set()
        self.all_data_rows = []
        self.header_row = None
        self.total_batches = 0
        self.start_time = time.time()
        self.last_save_time = None
        
        # Error tracking
        self.error_count = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3
        
        # Ensure output directory exists
        output_dir = Path(self.output_base).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set header for 12-column format
        self.header_row = [
            "CASEACCID", "Company Name", "Trading Name",
            "Tech Industry 1", "Tech Product 1",
            "Tech Industry 2", "Tech Product 2",
            "Tech Industry 3", "Tech Product 3",
            "Tech Industry 4", "Tech Product 4",
            "Country"
        ]
    
    def save_progress(self):
        """Save current progress to disk with atomic operations."""
        try:
            progress_data = {
                'processed_batches': list(self.processed_batches),
                'failed_batches': list(self.failed_batches),
                'all_data_rows': self.all_data_rows,
                'header_row': self.header_row,
                'total_batches': self.total_batches,
                'start_time': self.start_time,
                'error_count': self.error_count,
                'last_save_time': time.time()
            }
            
            # Atomic save for progress file
            temp_progress_file = f"{self.progress_file}.tmp"
            try:
                with open(temp_progress_file, 'wb') as f:
                    pickle.dump(progress_data, f)
                shutil.move(temp_progress_file, self.progress_file)
                print(f"   💾 Progress saved to: {self.progress_file}")
            except Exception as e:
                if os.path.exists(temp_progress_file):
                    os.remove(temp_progress_file)
                raise e
            
            # Atomic save for human-readable results
            if self.header_row and self.all_data_rows:
                all_results = [self.header_row] + self.all_data_rows
                temp_results_file = f"{self.temp_results_file}.tmp"
                try:
                    with open(temp_results_file, 'w', encoding='utf-8') as f:
                        json.dump(all_results, f, indent=2, ensure_ascii=False)
                    shutil.move(temp_results_file, self.temp_results_file)
                except Exception as e:
                    if os.path.exists(temp_results_file):
                        os.remove(temp_results_file)
                    raise e
            
            self.last_save_time = time.time()
            return True
            
        except Exception as e:
            print(f"❌ Error saving progress: {e}")
            return False
    
    def load_progress(self) -> bool:
        """Load existing progress if available."""
        if not os.path.exists(self.progress_file):
            return False
            
        try:
            with open(self.progress_file, 'rb') as f:
                data = pickle.load(f)
                
            self.processed_batches = set(data.get('processed_batches', []))
            self.failed_batches = set(data.get('failed_batches', []))
            self.all_data_rows = data.get('all_data_rows', [])
            self.header_row = data.get('header_row', self.header_row)
            self.total_batches = data.get('total_batches', 0)
            self.start_time = data.get('start_time', time.time())
            self.error_count = data.get('error_count', 0)
            
            print(f"📥 Progress loaded: {len(self.processed_batches)}/{self.total_batches} batches completed")
            return True
            
        except Exception as e:
            print(f"❌ Error loading progress: {e}")
            return False
    
    def mark_batch_completed(self, batch_num: int, parsed_rows: List[List[str]]):
        """Mark a batch as completed and save results immediately."""
        self.processed_batches.add(batch_num)
        self.failed_batches.discard(batch_num)
        
        # Add data rows (already validated, no header)
        if parsed_rows:
            self.all_data_rows.extend(parsed_rows)
            print(f"   📊 Added {len(parsed_rows)} rows (total: {len(self.all_data_rows)})")
        
        self.consecutive_errors = 0
        self.save_progress()
    
    def mark_batch_failed(self, batch_num: int, error: str, traceback_str: str = ""):
        """Mark a batch as failed and log the error."""
        self.failed_batches.add(batch_num)
        self.error_count += 1
        self.consecutive_errors += 1
        
        self.log_error(batch_num, error, traceback_str)
        self.save_progress()
    
    def log_error(self, batch_num: int, error: str, traceback_str: str = ""):
        """Log an error to the error log file."""
        try:
            error_entry = f"\n{'='*50}\n"
            error_entry += f"Batch {batch_num} Error - {datetime.now().isoformat()}\n"
            error_entry += f"Error: {error}\n"
            if traceback_str:
                error_entry += f"Traceback:\n{traceback_str}\n"
            
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                f.write(error_entry)
                
        except Exception as e:
            print(f"⚠️ Error writing to error log: {e}")
    
    def cleanup_temp_files(self):
        """Clean up temporary files after successful completion."""
        files_to_clean = [
            self.progress_file,
            self.temp_results_file,
            f"{self.progress_file}.tmp",
            f"{self.temp_results_file}.tmp"
        ]
        
        for file_path in files_to_clean:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass


class EnhancedCompanyClassificationCLI:
    """Enhanced CLI with integrated validation, country inference, and taxonomy enforcement."""
    
    def __init__(self, config_path: Optional[str] = None, batch_size: int = 1, 
                 enabled_servers: Set[str] = None, output_base: str = "results"):
        self.config_path = config_path or "cli_servers_config.json"
        self.batch_size = batch_size
        self.enabled_servers = enabled_servers or {"perplexity", "company_tagging"}
        self.output_base = output_base
        
        # Initialize components
        self.progress = ProgressTracker(output_base)
        self.validator = TaxonomyValidator()
        
        # MCP components
        self.client = None
        self.agent = None
        self.tools = []
        self.llm = None
        
        # Processing state
        self.companies = []
        self.retry_delay = 5
        self.max_retries = 3
    
    def sanitize_name_field(self, value: str) -> str:
        """Remove problematic punctuation from name fields to avoid CSV parsing issues."""
        if not value:
            return value
        
        # Remove problematic characters that can break CSV parsing
        problematic_chars = [',', ';', '|', '\t', '\n', '\r']
        sanitized = value
        for char in problematic_chars:
            sanitized = sanitized.replace(char, ' ')
        
        # Clean up multiple spaces
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        return sanitized
    
    def validate_server_requirements(self) -> bool:
        """Validate that required environment variables are set."""
        missing_vars = []
        
        # Check AI provider credentials (Azure OpenAI preferred)
        if self._has_azure_credentials():
            pass
        elif os.getenv("OPENAI_API_KEY"):
            pass
        else:
            missing_vars.append("AZURE_API_KEY + AZURE_ENDPOINT or OPENAI_API_KEY")
        
        # Check Perplexity credentials if enabled
        if "perplexity" in self.enabled_servers:
            if not os.getenv("PERPLEXITY_API_KEY"):
                missing_vars.append("PERPLEXITY_API_KEY")
        
        # Check Google credentials if enabled
        if "google" in self.enabled_servers:
            if not os.getenv("GOOGLE_API_KEY"):
                missing_vars.append("GOOGLE_API_KEY")
            if not os.getenv("GOOGLE_SEARCH_ENGINE_ID"):
                missing_vars.append("GOOGLE_SEARCH_ENGINE_ID")
        
        if missing_vars:
            print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            return False
        
        return True
    
    def _has_azure_credentials(self) -> bool:
        """Check if Azure OpenAI credentials are complete."""
        azure_vars = ["AZURE_API_KEY", "AZURE_ENDPOINT", "AZURE_DEPLOYMENT", "AZURE_API_VERSION"]
        return all(os.getenv(var) for var in azure_vars)
    
    def get_server_config(self) -> Dict[str, Dict]:
        """Get server configuration based on enabled servers."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                server_config = json.load(f)
            all_servers = server_config.get('mcpServers', {})
        else:
            all_servers = SERVER_CONFIG['mcpServers']
        
        filtered_servers = {}
        
        # Company Tagging is always included for taxonomy
        if "Company Tagging" in all_servers:
            filtered_servers["Company Tagging"] = all_servers["Company Tagging"]
        
        # Add search servers
        if "perplexity" in self.enabled_servers and "Perplexity Search" in all_servers:
            filtered_servers["Perplexity Search"] = all_servers["Perplexity Search"]
        
        if "google" in self.enabled_servers and "Google Search" in all_servers:
            filtered_servers["Google Search"] = all_servers["Google Search"]
        
        return filtered_servers
    
    async def setup_connections(self):
        """Set up MCP client connections and initialize the agent."""
        print("🔧 Setting up MCP connections...")
        
        if not self.validate_server_requirements():
            raise ValueError("Missing required environment variables")
        
        # Create LLM instance (Azure OpenAI preferred)
        if self._has_azure_credentials():
            llm_provider = "Azure OpenAI"
        else:
            llm_provider = "OpenAI"
        
        try:
            self.llm = create_llm_model(
                llm_provider,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS
            )
            print(f"✅ LLM initialized: {llm_provider}")
        except Exception as e:
            raise ValueError(f"Failed to initialize LLM: {e}")
        
        # Get server configuration
        try:
            servers = self.get_server_config()
            prepared_servers = prepare_server_config(servers)
            
            print(f"🔌 Connecting to {len(prepared_servers)} MCP servers...")
            
            self.client = await setup_mcp_client(prepared_servers)
            self.tools = await get_tools_from_client(self.client)
            
            # Create agent
            self.agent = create_react_agent(self.llm, self.tools)
            
            print(f"✅ Initialized with {len(self.tools)} available tools")
            
            # Load taxonomy from MCP server
            await self.load_taxonomy_from_mcp()
            
            # Clear caches
            await self.clear_search_caches()
            
        except Exception as e:
            print(f"❌ MCP Connection Error: {str(e)}")
            raise ValueError(f"Failed to setup MCP connections: {e}")
    
    async def load_taxonomy_from_mcp(self):
        """Load taxonomy using search_show_categories tool."""
        print("📋 Loading taxonomy from company_tagging MCP server...")
        
        try:
            prompt = """Use the search_show_categories tool to get ALL taxonomy pairs.
            Call it with no filters to get the complete list."""
            
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
                raise ValueError("Could not load taxonomy from MCP server")
                
        except Exception as e:
            print(f"❌ Error loading taxonomy: {e}")
            raise
    
    async def clear_search_caches(self):
        """Clear caches for search servers."""
        print("🧹 Clearing search caches...")
        
        try:
            # Find cache clearing tools
            for tool in self.tools:
                tool_name = tool.name.lower()
                if "clear" in tool_name and "cache" in tool_name:
                    print(f"   Clearing cache using: {tool.name}")
                    try:
                        await tool.ainvoke({})
                    except:
                        pass
            
            print("✅ Caches cleared")
        except Exception as e:
            print(f"⚠️ Warning: Could not clear all caches: {e}")
    
    def read_csv_file(self, csv_path: str) -> List[Dict]:
        """Read and parse the CSV file with robust delimiter detection."""
        print(f"📖 Reading CSV file: {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        companies = []
        try:
            # Use utf-8-sig to handle BOM
            with open(csv_path, 'r', encoding='utf-8-sig', newline='') as csvfile:
                # Read first few lines to detect delimiter
                first_line = csvfile.readline()
                csvfile.seek(0)
                
                # Try to detect delimiter from first line
                possible_delimiters = [',', '\t', ';', '|']
                delimiter = ','  # default
                
                # Count occurrences of each delimiter in first line
                delimiter_counts = {}
                for delim in possible_delimiters:
                    delimiter_counts[delim] = first_line.count(delim)
                
                # Use the delimiter with highest count (and at least 1 occurrence)
                if delimiter_counts:
                    max_count = max(delimiter_counts.values())
                    if max_count > 0:
                        for delim, count in delimiter_counts.items():
                            if count == max_count:
                                delimiter = delim
                                break
                
                print(f"   Detected delimiter: '{delimiter}' (repr: {repr(delimiter)})")
                
                # Now read with detected delimiter
                csvfile.seek(0)
                reader = csv.DictReader(csvfile, delimiter=delimiter)
                
                # Print detected headers for debugging
                if reader.fieldnames:
                    print(f"   Detected {len(reader.fieldnames)} columns")
                    print(f"   Headers: {', '.join(reader.fieldnames[:5])}...")  # Show first 5
                
                for row_num, row in enumerate(reader, 1):
                    # Clean row data
                    cleaned_row = {}
                    for k, v in row.items():
                        if k is not None:
                            # Strip BOM and whitespace from column names
                            key = k.strip().replace('\ufeff', '').strip()
                            # Handle None values
                            value = str(v).strip() if v is not None else ""
                            cleaned_row[key] = value
                    
                    # Check for required columns (Account Name is essential)
                    if cleaned_row.get('Account Name'):
                        # Ensure CASEACCID exists
                        if not cleaned_row.get('CASEACCID'):
                            # Generate one if missing
                            cleaned_row['CASEACCID'] = f"CASE{row_num:06d}"
                        companies.append(cleaned_row)
                    elif row_num == 1:
                        # Debug first row if it's not being captured
                        print(f"   ⚠️ First row skipped, keys: {list(cleaned_row.keys())[:5]}")
                
                print(f"✅ Successfully read {len(companies)} companies from CSV")
                
                # Show sample of first company for debugging
                if companies:
                    first = companies[0]
                    print(f"   Sample company: {first.get('Account Name', 'N/A')} (CASEACCID: {first.get('CASEACCID', 'N/A')})")
                
                return companies
                
        except Exception as e:
            # More detailed error information
            print(f"❌ Error details: {str(e)}")
            print(f"   File path: {csv_path}")
            print(f"   File exists: {os.path.exists(csv_path)}")
            
            # Try to show file content for debugging
            try:
                with open(csv_path, 'r', encoding='utf-8-sig') as f:
                    first_lines = f.read(500)
                    print(f"   First 500 chars of file:")
                    print(f"   {repr(first_lines[:200])}...")
            except:
                pass
                
            raise ValueError(f"Error reading CSV file: {e}")
    
    def parse_classification_response(self, response: str, original_company: Dict) -> Optional[Dict]:
        """Parse the classification response and ensure data integrity."""
        if not response:
            return None
        
        try:
            # Look for pipe-separated line
            lines = response.strip().split('\n')
            for line in lines:
                if '|' in line and not line.startswith('#'):
                    parts = [p.strip() for p in line.split('|')]
                    
                    if len(parts) >= 12:
                        # Create classification with original values for critical fields
                        classification = {
                            'CASEACCID': original_company.get('CASEACCID', ''),
                            'Company Name': self.sanitize_name_field(original_company.get('Account Name', '')),
                            'Trading Name': self.sanitize_name_field(original_company.get('Trading Name', '')),
                            'Tech Industry 1': parts[3] if len(parts) > 3 else '',
                            'Tech Product 1': parts[4] if len(parts) > 4 else '',
                            'Tech Industry 2': parts[5] if len(parts) > 5 else '',
                            'Tech Product 2': parts[6] if len(parts) > 6 else '',
                            'Tech Industry 3': parts[7] if len(parts) > 7 else '',
                            'Tech Product 3': parts[8] if len(parts) > 8 else '',
                            'Tech Industry 4': parts[9] if len(parts) > 9 else '',
                            'Tech Product 4': parts[10] if len(parts) > 10 else '',
                            'Country': parts[11] if len(parts) > 11 else ''
                        }
                        
                        # Validate and filter to keep only valid taxonomy pairs
                        validated = self.validator.validate_and_filter_pairs(classification)
                        
                        return validated
            
            return None
            
        except Exception as e:
            print(f"   ⚠️ Error parsing response: {e}")
            return None

    def create_integrated_prompt(self, company: Dict) -> str:
        """Create prompt that integrates search, classification, and country inference."""
        caseaccid = company.get('CASEACCID', '')
        account_name = company.get('Account Name', '')
        trading_name = company.get('Trading Name', '')
        domain = company.get('Domain', '')
        industry = company.get('Industry', '')
        product_service = company.get('Product/Service Type', '')
        
        # Sanitize name fields for output
        account_name_clean = self.sanitize_name_field(account_name)
        trading_name_clean = self.sanitize_name_field(trading_name)
        
        # Build search query components based on available data
        search_components = []
        
        # Priority 1: Domain (most specific)
        if domain and domain.strip():
            search_components.append(f"site:{domain}")
        
        # Priority 2: Company names
        if account_name and account_name.strip():
            search_components.append(account_name)
        
        if trading_name and trading_name.strip() and trading_name != account_name:
            search_components.append(f"({trading_name})")
        
        # Priority 3: Context fields that help narrow the search
        if industry and industry.strip():
            search_components.append(industry)
        
        if product_service and product_service.strip():
            search_components.append(product_service)
        
        # Add standard search terms for technology classification
        search_components.extend(["technology", "software", "platform", "services", "products", "headquarters", "location", "country"])
        
        # Build the search query
        search_query = " ".join(search_components)
        
        return f"""You need to classify this technology company using the correct taxonomy and identify its country.

    COMPANY INFORMATION:
    - Account Name: {account_name}
    - Trading Name: {trading_name}
    - Domain: {domain}
    - Industry Context: {industry}
    - Product/Service Type: {product_service}

    REQUIRED STEPS:

    1. FIRST: Use search_show_categories tool to get the COMPLETE list of valid taxonomy pairs.

    2. RESEARCH: Use perplexity_advanced_search (or perplexity_search_web if advanced not available) to research:
    - What technology products/services does this company offer?
    - What is their main business focus in technology?
    - What country are they headquartered in or primarily operate from?
    
    Search query to use: "{search_query}"
    
    Key search priorities:
    - If domain exists, prioritize site-specific search: site:{domain if domain else 'no domain'}
    - Include all available context: company names, industry, product/service type
    - Focus on current technology offerings and headquarters location
    
    If perplexity is not available, use google-search with the same query.

    3. CLASSIFY: Based on research and ONLY using exact pairs from the taxonomy:
    - Select up to 4 relevant Industry|Product pairs
    - Each pair MUST exist exactly in the taxonomy
    - If no valid matches found, leave those fields blank

    4. OUTPUT: Provide ONLY the data row (no headers) in this exact format with 12 columns separated by pipes:
    {caseaccid}|{account_name_clean}|{trading_name_clean}|Tech Industry 1|Tech Product 1|Tech Industry 2|Tech Product 2|Tech Industry 3|Tech Product 3|Tech Industry 4|Tech Product 4|Country

    CRITICAL RULES:
    - Use EXACT values from input for CASEACCID ('{caseaccid}'), Account Name (as Company Name), and Trading Name
    - The CASEACCID '{caseaccid}' must appear exactly as shown in the output
    - Only use Industry|Product pairs that exist EXACTLY in the taxonomy
    - Extract country from research results (full name like "United States", not codes)
    - If country cannot be determined, leave it blank
    - Output ONLY the pipe-separated data row, no additional text"""
    
    async def process_batch_with_retry(self, batch_companies: List[Dict], 
                                     batch_num: int, total_batches: int) -> List[List[str]]:
        """Process a batch with retry logic and validation."""
        for attempt in range(self.max_retries):
            try:
                print(f"🔄 Processing batch {batch_num}/{total_batches} (attempt {attempt + 1}/{self.max_retries})")
                
                batch_results = []
                
                for company in batch_companies:
                    company_name = company.get('Account Name', 'Unknown')
                    print(f"   Processing: {company_name}")
                    
                    # Create integrated prompt
                    prompt = self.create_integrated_prompt(company)
                    conversation = [HumanMessage(content=prompt)]
                    
                    # Process with agent
                    response = await asyncio.wait_for(
                        self.agent.ainvoke({"messages": conversation}),
                        timeout=300  # 5 minutes timeout per company
                    )
                    
                    # Extract classification
                    assistant_response = None
                    if "messages" in response:
                        for msg in reversed(response["messages"]):
                            if isinstance(msg, AIMessage) and hasattr(msg, "content") and msg.content:
                                assistant_response = str(msg.content)
                                break
                    
                    if assistant_response:
                        classification = self.parse_classification_response(assistant_response, company)
                        
                        if classification:
                            # Convert to list format for progress tracker
                            row = [
                                classification['CASEACCID'],
                                classification['Company Name'],
                                classification['Trading Name'],
                                classification['Tech Industry 1'],
                                classification['Tech Product 1'],
                                classification['Tech Industry 2'],
                                classification['Tech Product 2'],
                                classification['Tech Industry 3'],
                                classification['Tech Product 3'],
                                classification['Tech Industry 4'],
                                classification['Tech Product 4'],
                                classification['Country']
                            ]
                            batch_results.append(row)
                            print(f"      ✅ Classified with {sum(1 for i in range(1,5) if classification.get(f'Tech Industry {i}'))} valid pairs")
                        else:
                            # Create blank result with original data
                            row = [
                                company.get('CASEACCID', ''),
                                self.sanitize_name_field(company.get('Account Name', '')),
                                self.sanitize_name_field(company.get('Trading Name', '')),
                                '', '', '', '', '', '', '', '', ''
                            ]
                            batch_results.append(row)
                            print(f"      ⚠️ No valid classification found")
                
                if batch_results:
                    print(f"   ✅ Batch {batch_num} completed: {len(batch_results)} companies processed")
                    return batch_results
                    
            except asyncio.TimeoutError:
                print(f"   ⏰ Batch {batch_num} timeout")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                    
            except Exception as e:
                print(f"   ❌ Batch {batch_num} error: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
        
        return []
    
    async def classify_companies_batched(self, companies: List[Dict]) -> Tuple[str, List[List[str]]]:
        """Classify companies using batched processing with validation."""
        
        # Split companies into batches
        batches = [companies[i:i + self.batch_size] for i in range(0, len(companies), self.batch_size)]
        total_batches = len(batches)
        
        print(f"📊 Processing {len(companies)} companies in {total_batches} batches")
        
        # Load existing progress if available
        self.progress.load_progress()
        
        # Set total batches
        self.progress.total_batches = total_batches
        
        # Get remaining batches to process
        remaining_batches = [i for i in range(1, total_batches + 1) 
                           if i not in self.progress.processed_batches]
        
        if not remaining_batches:
            print("✅ All batches already completed!")
        else:
            print(f"📊 Processing {len(remaining_batches)} remaining batches")
        
        # Process remaining batches
        for batch_idx in remaining_batches:
            batch_num = batch_idx
            batch_companies = batches[batch_idx - 1]
            
            # Process batch
            batch_results = await self.process_batch_with_retry(batch_companies, batch_num, total_batches)
            
            if batch_results:
                self.progress.mark_batch_completed(batch_num, batch_results)
            else:
                self.progress.mark_batch_failed(batch_num, "Processing failed", "")
                
                if self.progress.consecutive_errors >= self.progress.max_consecutive_errors:
                    print(f"❌ Stopping due to {self.progress.consecutive_errors} consecutive errors")
                    break
            
            # Print progress
            completed = len([b for b in self.progress.processed_batches if b <= total_batches])
            print(f"📊 Progress: {completed}/{total_batches} batches completed")
        
        # Generate final results
        final_results = []
        if self.progress.header_row:
            final_results.append(self.progress.header_row)
        final_results.extend(self.progress.all_data_rows)
        
        # Generate markdown
        markdown_lines = []
        if final_results:
            header = final_results[0]
            data_rows = final_results[1:] if len(final_results) > 1 else []
            
            markdown_lines.append('|' + '|'.join(header) + '|')
            markdown_lines.append('|' + '|'.join(['---'] * len(header)) + '|')
            
            for row in data_rows:
                if len(row) >= len(header):
                    markdown_lines.append('|' + '|'.join(row[:len(header)]) + '|')
        
        markdown_content = '\n'.join(markdown_lines) if markdown_lines else "No results"
        
        return markdown_content, final_results
    
    def save_csv_results(self, results: List[List[str]], csv_path: str):
        """Save results to CSV file."""
        if not results:
            print("⚠️ No results to save")
            return
            
        try:
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(results)
            print(f"💾 CSV results saved to: {csv_path}")
        except Exception as e:
            raise ValueError(f"Error saving CSV results: {e}")
    
    async def save_results(self, markdown_content: str, csv_results: List[List[str]], base_output_path: str):
        """Save the classification results to both MD and CSV files."""
        base_path = Path(base_output_path)
        md_path = base_path.with_suffix('.md')
        csv_path = base_path.with_suffix('.csv')
        
        try:
            # Save markdown file
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"💾 Markdown results saved to: {md_path}")
            
            # Save CSV file
            if csv_results:
                self.save_csv_results(csv_results, str(csv_path))
            
            # Save statistics
            stats = {
                'total_companies': len(self.progress.all_data_rows),
                'completed_batches': len(self.progress.processed_batches),
                'failed_batches': len(self.progress.failed_batches),
                'total_batches': self.progress.total_batches,
                'processing_time': time.time() - self.progress.start_time
            }
            
            stats_path = base_path.with_suffix('.stats.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            print(f"📊 Statistics saved to: {stats_path}")
            
        except Exception as e:
            raise ValueError(f"Error saving results: {e}")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.client:
            try:
                await self.client.__aexit__(None, None, None)
                print("🧹 Cleaned up MCP connections")
            except Exception as e:
                print(f"⚠️ Error during cleanup: {e}")


async def main():
    """Main CLI function with integrated workflow."""
    parser = argparse.ArgumentParser(
        description="Enhanced Company Classification CLI Tool with Integrated Validation and Country Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file")
    parser.add_argument("--output", "-o", required=True, help="Base path for output files")
    parser.add_argument("--servers", "-s", type=str, default="perplexity", 
                       help="MCP servers to use: 'google', 'perplexity', or 'both' (default: perplexity)")
    parser.add_argument("--batch-size", "-b", type=int, default=1, 
                       help="Batch size (default: 1)")
    parser.add_argument("--config", "-c", help="Path to server configuration file")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume from previous run (default behavior)")
    parser.add_argument("--clean-start", action="store_true", 
                       help="Start fresh, ignoring previous progress")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        # Parse server selection
        if args.servers == "google":
            enabled_servers = {"google", "company_tagging"}
        elif args.servers == "perplexity":
            enabled_servers = {"perplexity", "company_tagging"}
        elif args.servers == "both":
            enabled_servers = {"google", "perplexity", "company_tagging"}
        else:
            enabled_servers = {"perplexity", "company_tagging"}
        
        # Initialize CLI tool
        cli = EnhancedCompanyClassificationCLI(
            config_path=args.config,
            batch_size=args.batch_size,
            enabled_servers=enabled_servers,
            output_base=args.output
        )
        
        # Clean start if requested
        if args.clean_start:
            print("🧹 Starting fresh (ignoring previous progress)")
            cli.progress.cleanup_temp_files()
        
        print("🚀 Enhanced Company Classification CLI Tool")
        print("=" * 80)
        print(f"🔧 Enabled servers: {', '.join(enabled_servers)}")
        print(f"📊 Batch size: {args.batch_size}")
        print(f"📋 Output format: 12 columns with validated taxonomy pairs")
        
        # Setup connections
        await cli.setup_connections()
        
        # Read CSV file
        companies = cli.read_csv_file(args.input)
        
        if not companies:
            print("❌ No valid companies found in the CSV file")
            return 1
        
        # Classify companies in batches
        markdown_content, csv_results = await cli.classify_companies_batched(companies)
        
        # Save results
        await cli.save_results(markdown_content, csv_results, args.output)
        
        # Print final summary
        stats = {
            'total_companies': len(companies),
            'processed_companies': len(cli.progress.all_data_rows),
            'completed_batches': len(cli.progress.processed_batches),
            'failed_batches': len(cli.progress.failed_batches),
            'total_batches': cli.progress.total_batches
        }
        
        print(f"\n📊 Final Summary:")
        print(f"   Total companies: {stats['total_companies']}")
        print(f"   Processed: {stats['processed_companies']}")
        print(f"   Batches completed: {stats['completed_batches']}/{stats['total_batches']}")
        print(f"   Output format: 12 columns (CASEACCID + Names + 4 Tech Pairs + Country)")
        print(f"   Taxonomy validation: ✅ All pairs validated against MCP taxonomy")
        
        # Clean up temp files on successful completion
        if stats['completed_batches'] == stats['total_batches']:
            cli.progress.cleanup_temp_files()
            print("✅ Processing completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1
    
    finally:
        if 'cli' in locals():
            await cli.cleanup()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))