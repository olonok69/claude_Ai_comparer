#!/usr/bin/env python3
"""
Enhanced Company Classification CLI Tool
Integrated workflow: Perplexity Search → Azure OpenAI Classification → Taxonomy Validation → Country Inference
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
import pickle
import traceback
import shutil
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


class ProgressTracker:
    """Enhanced progress tracker for integrated workflow."""
    
    def __init__(self, output_base: str):
        self.output_base = output_base
        self.progress_file = f"{output_base}_progress.pkl"
        self.temp_results_file = f"{output_base}_temp_results.json"
        self.error_log_file = f"{output_base}_errors.log"
        
        # Progress tracking
        self.processed_batches = set()
        self.failed_batches = set()
        self.all_data_rows = []  # Store only data rows, no headers
        self.header_row = None   # Store header separately
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
        
        print(f"📁 ProgressTracker initialized for: {output_base}")
        print(f"   📄 Progress file: {self.progress_file}")
        print(f"   📄 Temp results: {self.temp_results_file}")
        print(f"   📄 Error log: {self.error_log_file}")
    
    def save_progress(self):
        """Save current progress to disk with atomic operations."""
        try:
            # Prepare progress data with JSON-compatible types
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
            
            # Atomic save for progress file using temporary file
            temp_progress_file = f"{self.progress_file}.tmp"
            try:
                with open(temp_progress_file, 'wb') as f:
                    pickle.dump(progress_data, f)
                # Atomic move
                shutil.move(temp_progress_file, self.progress_file)
                print(f"   💾 Progress saved to: {self.progress_file}")
            except Exception as e:
                # Clean up temp file if it exists
                if os.path.exists(temp_progress_file):
                    os.remove(temp_progress_file)
                raise e
            
            self.last_save_time = time.time()
            return True
            
        except Exception as e:
            print(f"❌ Error saving progress: {e}")
            return False
    
    def load_progress(self) -> bool:
        """Load existing progress if available."""
        if not os.path.exists(self.progress_file):
            print(f"📂 No existing progress file found: {self.progress_file}")
            return False
            
        try:
            with open(self.progress_file, 'rb') as f:
                data = pickle.load(f)
                
            # Convert lists back to sets for internal processing
            self.processed_batches = set(data.get('processed_batches', []))
            self.failed_batches = set(data.get('failed_batches', []))
            self.all_data_rows = data.get('all_data_rows', [])
            self.header_row = data.get('header_row', None)
            self.total_batches = data.get('total_batches', 0)
            self.start_time = data.get('start_time', time.time())
            self.error_count = data.get('error_count', 0)
            
            print(f"📥 Progress loaded successfully:")
            print(f"   ✅ Processed batches: {len(self.processed_batches)}")
            print(f"   ❌ Failed batches: {len(self.failed_batches)}")
            print(f"   📊 Data rows: {len(self.all_data_rows)}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading progress: {e}")
            return False
    
    def mark_batch_completed(self, batch_num: int, parsed_rows: List[List[str]]):
        """Mark a batch as completed and save results."""
        print(f"   ✅ Marking batch {batch_num} as completed...")
        
        self.processed_batches.add(batch_num)
        self.failed_batches.discard(batch_num)
        
        # Extract header and data rows
        batch_header, data_rows = self.extract_header_and_data(parsed_rows)
        
        # Set header if we don't have one yet
        if self.header_row is None and batch_header is not None:
            self.header_row = batch_header
            print(f"   📋 Header captured: {len(batch_header)} columns")
        
        # Add data rows (no headers)
        if data_rows:
            self.all_data_rows.extend(data_rows)
            print(f"   📊 Added {len(data_rows)} data rows (total: {len(self.all_data_rows)})")
        
        self.consecutive_errors = 0
        
        # Save progress after every batch completion
        self.save_progress()
    
    def mark_batch_failed(self, batch_num: int, error: str, traceback_str: str = ""):
        """Mark a batch as failed and log the error."""
        print(f"   ❌ Marking batch {batch_num} as failed...")
        
        self.failed_batches.add(batch_num)
        self.error_count += 1
        self.consecutive_errors += 1
        
        # Log error immediately
        self.log_error(batch_num, error, traceback_str)
        
        # Save progress immediately after marking failure
        self.save_progress()
    
    def log_error(self, batch_num: int, error: str, traceback_str: str = ""):
        """Log an error to the error log file."""
        try:
            error_entry = f"\n{'='*50}\n"
            error_entry += f"Batch {batch_num} Error - {datetime.now().isoformat()}\n"
            error_entry += f"Error: {error}\n"
            if traceback_str:
                error_entry += f"Traceback:\n{traceback_str}\n"
            error_entry += f"{'='*50}\n"
            
            # Append to existing content
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                f.write(error_entry)
                
        except Exception as e:
            print(f"⚠️  Error writing to error log: {e}")
    
    def extract_header_and_data(self, parsed_rows: List[List[str]]) -> Tuple[Optional[List[str]], List[List[str]]]:
        """Extract header and data rows from parsed table."""
        if not parsed_rows:
            return None, []
        
        header = None
        data_rows = []
        
        for row in parsed_rows:
            if not row or len(row) == 0:
                continue
            
            # Skip separator rows (markdown table separators)
            if all(cell.strip() == '' or all(c in '-|: ' for c in cell.strip()) for cell in row):
                continue
            
            # Check if this is a header row
            first_cell = row[0].lower().strip()
            header_indicators = ['company name', 'account name', 'trading name', 'tech industry', 'industry', 'product']
            
            if any(indicator in first_cell for indicator in header_indicators):
                if header is None:  # Take the first header we find
                    header = row
                continue  # Skip subsequent headers
            
            # This is a data row
            if row and any(cell.strip() for cell in row):  # Has some content
                data_rows.append(row)
        
        return header, data_rows
    
    def get_stats(self) -> Dict:
        """Get current processing statistics."""
        valid_completed = len([b for b in self.processed_batches if b <= self.total_batches])
        valid_failed = len([b for b in self.failed_batches if b <= self.total_batches])
        
        return {
            'total_batches': self.total_batches,
            'completed_batches': valid_completed,
            'failed_batches': valid_failed,
            'remaining_batches': self.total_batches - valid_completed,
            'error_count': self.error_count,
            'consecutive_errors': self.consecutive_errors,
            'success_rate': valid_completed / max(1, self.total_batches) * 100,
            'total_data_rows': len(self.all_data_rows),
            'has_header': self.header_row is not None,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0
        }
    
    def cleanup_temp_files(self):
        """Clean up temporary files after successful completion."""
        files_to_clean = [
            self.progress_file,
            self.temp_results_file,
            f"{self.progress_file}.tmp",
            f"{self.temp_results_file}.tmp",
            f"{self.error_log_file}.tmp",
            f"{self.output_base}_backup.json"
        ]
        
        cleaned_count = 0
        for file_path in files_to_clean:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    cleaned_count += 1
            except Exception as e:
                print(f"⚠️  Error cleaning up {file_path}: {e}")
        
        print(f"🧹 Cleaned up {cleaned_count} temporary files")
    
    def get_final_results(self) -> List[List[str]]:
        """Get final results with header + data rows."""
        if self.header_row is None:
            return self.all_data_rows
        
        return [self.header_row] + self.all_data_rows


class EnhancedCompanyClassificationCLI:
    """Enhanced CLI with integrated Perplexity → Azure OpenAI → Taxonomy Validation → Country Inference workflow."""
    
    def __init__(self, config_path: Optional[str] = None, batch_size: int = 1, 
                 enabled_servers: Set[str] = None, output_base: str = "results"):
        self.config_path = config_path or "cli_servers_config.json"
        self.batch_size = batch_size
        self.enabled_servers = enabled_servers or {"perplexity", "company_tagging"}
        self.output_base = output_base
        
        # Initialize progress tracker
        self.progress = ProgressTracker(output_base)
        
        # MCP components (using existing structure)
        self.client = None
        self.agent = None
        self.tools = []
        self.llm = None
        self.available_tools = {
            "google": [],
            "perplexity": [],
            "company_tagging": []
        }
        
        # Processing state
        self.companies = []
        self.retry_delay = 5
        self.max_retries = 3
        
        print(f"🚀 Enhanced Company Classification CLI initialized")
        print(f"   Batch size: {batch_size}")
        print(f"   Workflow: Perplexity Search → Azure OpenAI → Taxonomy Validation → Country Inference")
    
    def sanitize_json_string(self, text: str) -> str:
        """Sanitize a string to prevent JSON parsing errors."""
        if not isinstance(text, str):
            return str(text)
        
        # Remove or escape problematic characters
        text = text.replace('\x00', '')
        text = text.replace('\x1c', '')
        text = text.replace('\x1d', '')
        text = text.replace('\x1e', '')
        text = text.replace('\x1f', '')
        text = text.replace('\x17', '')
        
        # Remove other control characters except common ones
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        return text
    
    def clean_response_content(self, response_content: str) -> str:
        """Clean response content to prevent parsing errors."""
        if not response_content:
            return ""
        
        # Sanitize the content
        cleaned = self.sanitize_json_string(response_content)
        
        # Try to find and extract only the markdown table
        lines = cleaned.split('\n')
        table_lines = []
        in_table = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a table line
            if '|' in line and ('Company Name' in line or 'Tech Industry' in line):
                in_table = True
                table_lines.append(line)
            elif in_table and '|' in line:
                table_lines.append(line)
            elif in_table and not line and len(table_lines) > 0:
                # Empty line might continue the table
                continue
            elif in_table and line and '|' not in line:
                # Non-table line ends the table
                break
        
        if table_lines:
            return '\n'.join(table_lines)
        
        return cleaned
    
    def validate_server_requirements(self) -> bool:
        """Validate that required environment variables are set."""
        missing_vars = []
        
        # Check AI provider credentials
        if self._has_azure_credentials():
            pass
        elif os.getenv("OPENAI_API_KEY"):
            pass
        else:
            missing_vars.append("AZURE_API_KEY + AZURE_ENDPOINT + AZURE_DEPLOYMENT + AZURE_API_VERSION or OPENAI_API_KEY")
        
        # Check Perplexity credentials if enabled
        if "perplexity" in self.enabled_servers:
            if not os.getenv("PERPLEXITY_API_KEY"):
                missing_vars.append("PERPLEXITY_API_KEY")
        
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
        
        # Company Tagging is always included
        if "Company Tagging" in all_servers:
            filtered_servers["Company Tagging"] = all_servers["Company Tagging"]
        
        # Add Perplexity Search if enabled
        if "perplexity" in self.enabled_servers and "Perplexity Search" in all_servers:
            filtered_servers["Perplexity Search"] = all_servers["Perplexity Search"]
        
        return filtered_servers
    
    async def setup_connections(self):
        """Set up MCP client connections and initialize the agent."""
        print("🔧 Setting up MCP connections...")
        
        if not self.validate_server_requirements():
            raise ValueError("Missing required environment variables")
        
        # Create LLM instance (using existing structure)
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
            
            # Categorize tools
            self.available_tools = self.categorize_tools(self.tools)
            
            # Create agent
            self.agent = create_react_agent(self.llm, self.tools)
            
            print(f"✅ Initialized with {len(self.tools)} available tools")
            
        except Exception as e:
            print(f"❌ MCP Connection Error: {str(e)}")
            raise ValueError(f"Failed to setup MCP connections: {e}")
    
    def categorize_tools(self, tools: List) -> Dict[str, List]:
        """Categorize tools by server type."""
        categorized = {
            "google": [],
            "perplexity": [],
            "company_tagging": []
        }
        
        for tool in tools:
            tool_name = tool.name.lower()
            
            if any(keyword in tool_name for keyword in [
                'search_show_categories', 'company_tagging', 'tag_companies'
            ]):
                categorized["company_tagging"].append(tool)
            elif any(keyword in tool_name for keyword in [
                'perplexity_search_web', 'perplexity_advanced_search', 'perplexity'
            ]):
                categorized["perplexity"].append(tool)
            elif any(keyword in tool_name for keyword in [
                'google-search', 'read-webpage', 'google_search'
            ]):
                categorized["google"].append(tool)
        
        return categorized
    
    def read_csv_file(self, csv_path: str) -> List[Dict]:
        """Read and parse the CSV file."""
        print(f"📖 Reading CSV file: {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        companies = []
        try:
            # Read with UTF-8-sig to handle BOM automatically
            with open(csv_path, 'r', encoding='utf-8-sig', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                print(f"📋 CSV columns detected: {reader.fieldnames}")
                
                for row_num, row in enumerate(reader, 1):
                    # Clean and validate row data
                    cleaned_row = {}
                    for k, v in row.items():
                        if k is not None:
                            # Clean the key (column name) - remove BOM and whitespace
                            clean_key = k.strip().lstrip('\ufeff')
                            # Clean the value
                            clean_value = self.sanitize_json_string(str(v).strip()) if v is not None else ""
                            cleaned_row[clean_key] = clean_value
                    
                    # Debug CASEACCID reading
                    caseaccid_value = cleaned_row.get('CASEACCID', '')
                    if row_num <= 5:  # Only show first 5 for debugging
                        print(f"   Row {row_num}: CASEACCID = '{caseaccid_value}' (Account Name: {cleaned_row.get('Account Name', '')})")
                    
                    # Check for required columns
                    if cleaned_row.get('Account Name'):
                        companies.append(cleaned_row)
                    else:
                        print(f"⚠️  Row {row_num}: Missing Account Name, skipping")
            
            print(f"✅ Successfully read {len(companies)} companies from CSV")
            return companies
            
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
    
    def create_integrated_prompt(self, companies_batch: List[Dict]) -> str:
        """Create integrated research and classification prompt."""
        
        # Format company data
        formatted_companies = []
        for company in companies_batch:
            company_info = []
            company_info.append(f"Account Name = {company.get('Account Name', '')}")
            company_info.append(f"Trading Name = {company.get('Trading Name', '')}")
            company_info.append(f"Domain = {company.get('Domain', '')}")
            company_info.append(f"Industry = {company.get('Industry', '')}")
            company_info.append(f"Product/Service Type = {company.get('Product/Service Type', '')}")
            company_info.append(f"Event = {company.get('Event', '')}")
            formatted_companies.append('\n'.join(company_info))
        
        company_data = '\n\n'.join(formatted_companies)
        
        return f"""You are a professional data analyst tasked with comprehensive company research and classification using an integrated workflow.

COMPANY DATA TO ANALYZE:
{company_data}

INTEGRATED WORKFLOW - EXECUTE ALL STEPS:

**Step 1: Get Complete Taxonomy** (ONCE ONLY)
- Use search_show_categories tool without any filters to get all available industry/product pairs

**Step 2: For EACH Company - Enhanced Research**
- Use perplexity_advanced_search with recency="year" 
- Search query: "[Account Name] [Industry if available] [Product/Service Type if available] company products services technology solutions country headquarters location"
- Focus on: what they sell/offer, their technology solutions, and where they operate

**Step 3: Classification & Country Extraction**
- Based on research results, determine:
  * Up to 4 relevant (Industry | Product) pairs from taxonomy
  * Company's primary operating country/headquarters
- Use EXACT taxonomy pairs from Step 1
- DO NOT create new pairs or modify existing ones

**Step 4: Output Generation**
Generate a markdown table with ONLY DATA ROWS (NO HEADER) in this exact format:
| Company Name | Trading Name | Tech Industry 1 | Tech Product 1 | Tech Industry 2 | Tech Product 2 | Tech Industry 3 | Tech Product 3 | Tech Industry 4 | Tech Product 4 | Country |

CRITICAL REQUIREMENTS:
- MUST use search_show_categories to get complete taxonomy first
- MUST use perplexity_advanced_search with recency="year" for each company
- Use taxonomy pairs EXACTLY as they appear in the official list
- Extract country information from research results
- Company Name = Account Name field value (literal copy)
- Trading Name = Trading Name field value (literal copy)
- Output ONLY data rows (no header row)
- If no valid taxonomy match found, leave Tech Industry/Product fields blank
- If country cannot be determined with confidence, leave Country field blank

Begin the systematic analysis now."""
    
    def parse_markdown_table(self, response: str) -> List[List[str]]:
        """Parse markdown table response into structured data with exact 12-column format."""
        if not response:
            return []
        
        # Clean the response first
        response = self.clean_response_content(response)
        
        lines = response.strip().split('\n')
        table_rows = []
        
        for line in lines:
            line = line.strip()
            if "|" in line and line:
                # Parse markdown table row
                columns = [col.strip() for col in line.split('|')]
                # Remove empty first/last columns from markdown formatting
                if columns and not columns[0]:
                    columns = columns[1:]
                if columns and not columns[-1]:
                    columns = columns[:-1]
                
                # Skip separator lines
                if columns and not all(col == '' or '-' in col for col in columns):
                    # Sanitize each column
                    sanitized_columns = [self.sanitize_json_string(col) for col in columns]
                    
                    # Ensure we have exactly 11 columns (Company Name through Country)
                    if len(sanitized_columns) >= 11:
                        # Take first 11 columns and pad if necessary
                        final_row = sanitized_columns[:11]
                        # Pad with empty strings if needed
                        while len(final_row) < 11:
                            final_row.append("")
                        table_rows.append(final_row)
        
        return table_rows
    
    async def process_batch_with_retry(self, batch_companies: List[Dict], 
                                     batch_num: int, total_batches: int) -> List[List[str]]:
        """Process a batch with retry logic."""
        for attempt in range(self.max_retries):
            try:
                print(f"🔄 Processing batch {batch_num}/{total_batches} (attempt {attempt + 1}/{self.max_retries})")
                
                batch_prompt = self.create_integrated_prompt(batch_companies)
                conversation_memory = [HumanMessage(content=batch_prompt)]
                
                # Use longer timeout for Perplexity + classification workflow
                timeout = 900  # 15 minutes for integrated workflow
                
                try:
                    response = await asyncio.wait_for(
                        self.agent.ainvoke({"messages": conversation_memory}),
                        timeout=timeout
                    )
                except Exception as llm_error:
                    # Check for specific API errors
                    error_str = str(llm_error).lower()
                    if any(keyword in error_str for keyword in ['quota', 'rate limit', 'billing', 'insufficient', 'exceeded']):
                        error_msg = f"Batch {batch_num} API error: {str(llm_error)}"
                        print(f"   🚨 {error_msg}")
                        print(f"   💰 API quota/billing issue detected - stopping to avoid wasting calls")
                        self.progress.mark_batch_failed(batch_num, error_msg, traceback.format_exc())
                        return []
                    else:
                        raise llm_error
                
                # Extract response
                assistant_response = None
                if "messages" in response:
                    for msg in response["messages"]:
                        if isinstance(msg, AIMessage) and hasattr(msg, "content") and msg.content:
                            assistant_response = self.clean_response_content(str(msg.content))
                            break
                
                if assistant_response and "|" in assistant_response:
                    parsed_rows = self.parse_markdown_table(assistant_response)
                    if parsed_rows:
                        print(f"   ✅ Batch {batch_num} completed: {len(parsed_rows)} rows processed")
                        return parsed_rows
                
                # If no valid response, retry
                if attempt < self.max_retries - 1:
                    print(f"   ⚠️  Batch {batch_num} attempt {attempt + 1} failed (no valid response), retrying...")
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    print(f"   ❌ Batch {batch_num} failed after {self.max_retries} attempts")
                    return []
                    
            except asyncio.TimeoutError:
                error_msg = f"Batch {batch_num} timeout after {timeout//60} minutes"
                print(f"   ⏰ {error_msg}")
                if attempt < self.max_retries - 1:
                    print(f"   🔄 Retrying in {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    self.progress.mark_batch_failed(batch_num, error_msg)
                    return []
                    
            except Exception as e:
                error_msg = f"Batch {batch_num} error: {str(e)}"
                traceback_str = traceback.format_exc()
                print(f"   ❌ {error_msg}")
                
                if attempt < self.max_retries - 1:
                    print(f"   🔄 Retrying in {self.retry_delay} seconds...")
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    self.progress.mark_batch_failed(batch_num, error_msg, traceback_str)
                    return []
        
        return []
    
    async def classify_companies_batched(self, companies: List[Dict]) -> Tuple[str, List[List[str]]]:
        """Process companies in batches with integrated workflow."""
        
        # Split companies into batches
        batches = [companies[i:i + self.batch_size] for i in range(0, len(companies), self.batch_size)]
        total_batches = len(batches)
        
        print(f"📊 Created {total_batches} batches of {self.batch_size} companies each")
        
        # Load existing progress if available
        progress_loaded = self.progress.load_progress()
        
        # Initialize progress tracker
        self.progress.total_batches = total_batches
        self.progress.start_time = time.time()
        
        if progress_loaded:
            print(f"📂 Resuming from previous run...")
        
        # Set exact header format for 12 columns
        if self.progress.header_row is None:
            self.progress.header_row = [
                "CASEACCID", "Company Name", "Trading Name", 
                "Tech Industry 1", "Tech Product 1",
                "Tech Industry 2", "Tech Product 2", 
                "Tech Industry 3", "Tech Product 3",
                "Tech Industry 4", "Tech Product 4", 
                "Country"
            ]
            print("📋 Using integrated workflow header structure (12 columns)")
        
        # Get remaining batches to process
        remaining_batches = [i for i in range(1, total_batches + 1) 
                           if i not in self.progress.processed_batches and i <= total_batches]
        
        if not remaining_batches:
            print("✅ All batches already completed!")
        else:
            print(f"📊 Processing {len(remaining_batches)} remaining batches out of {total_batches} total")
        
        # Process remaining batches
        for batch_idx in remaining_batches:
            batch_num = batch_idx
            batch_companies = batches[batch_idx - 1]  # Convert to 0-based for list access
            
            # Process batch with retry logic
            batch_results = await self.process_batch_with_retry(batch_companies, batch_num, total_batches)
            
            if batch_results:
                # Add CASEACCID to results to match header format - LITERAL COPY FROM INPUT
                enhanced_results = []
                for i, row in enumerate(batch_results):
                    if len(row) >= 11:  # Ensure we have at least Company Name through Country
                        # Get CASEACCID from original input (literal copy)
                        company_data = batch_companies[i] if i < len(batch_companies) else {}
                        caseaccid = company_data.get('CASEACCID', '')  # Literal copy from input
                        enhanced_row = [caseaccid] + row[:11]  # CASEACCID + 11 other columns = 12 total
                        enhanced_results.append(enhanced_row)
                
                self.progress.mark_batch_completed(batch_num, enhanced_results)
            else:
                print(f"   ❌ Batch {batch_num} failed completely")
            
            # Print progress update
            completed_batches = len([b for b in self.progress.processed_batches if b <= total_batches])
            success_rate = completed_batches / max(1, total_batches) * 100
            print(f"📊 Progress: {completed_batches}/{total_batches} batches "
                  f"({success_rate:.1f}% success rate, {len(self.progress.all_data_rows)} companies)")
        
        # Final save
        self.progress.save_progress()
        
        # Generate final results
        final_results = self.progress.get_final_results()
        
        # Generate markdown content
        if final_results and len(final_results) > 0:
            header = final_results[0]
            data_rows = final_results[1:] if len(final_results) > 1 else []
            
            # Format as markdown table
            markdown_lines = []
            markdown_lines.append('|' + '|'.join([''] + header + ['']) + '|')
            
            # Add separator line
            separator = ['---'] * len(header)
            markdown_lines.append('|' + '|'.join([''] + separator + ['']) + '|')
            
            # Add data rows
            for row in data_rows:
                if len(row) >= len(header):
                    markdown_lines.append('|' + '|'.join([''] + row[:len(header)] + ['']) + '|')
            
            markdown_content = '\n'.join(markdown_lines)
        else:
            markdown_content = "No results to display"
        
        return markdown_content, final_results
    
    def save_csv_results(self, results: List[List[str]], csv_path: str):
        """Save results to CSV file."""
        if not results:
            print("⚠️  No results to save")
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
            
            # Save final statistics
            stats = self.progress.get_stats()
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
                print(f"⚠️  Error during cleanup: {e}")


async def main():
    """Main CLI function with integrated workflow."""
    parser = argparse.ArgumentParser(
        description="Enhanced Company Classification CLI Tool with Integrated Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file")
    parser.add_argument("--output", "-o", required=True, help="Base path for output files")
    parser.add_argument("--servers", "-s", type=str, default="perplexity", 
                       help="MCP servers to use: 'perplexity' (default)")
    parser.add_argument("--batch-size", "-b", type=int, default=1, 
                       help="Batch size (default: 1 for accuracy)")
    parser.add_argument("--config", "-c", help="Path to server configuration file")
    parser.add_argument("--clean-start", action="store_true", 
                       help="Start fresh, ignoring previous progress")
    parser.add_argument("--perplexity-batch", action="store_true", 
                       help="Use Perplexity for batch processing (default behavior)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        # Server selection - always use perplexity + company_tagging for integrated workflow
        enabled_servers = {"perplexity", "company_tagging"}
        
        # Batch size validation
        batch_size = args.batch_size
        if batch_size > 1:
            print(f"⚠️  Recommended batch size is 1 for integrated workflow (you chose {batch_size})")
        
        # Initialize CLI tool
        cli = EnhancedCompanyClassificationCLI(
            config_path=args.config,
            batch_size=batch_size,
            enabled_servers=enabled_servers,
            output_base=args.output
        )
        
        # Clean start if requested
        if args.clean_start:
            print("🧹 Starting fresh (ignoring previous progress)")
            cli.progress.cleanup_temp_files()
        
        print("🚀 Enhanced Company Classification CLI Tool - Integrated Workflow")
        print("=" * 80)
        print(f"🔧 Enabled servers: {', '.join(enabled_servers)}")
        print(f"📊 Batch size: {batch_size}")
        print("🔄 Workflow: Perplexity Research → Azure OpenAI Classification → Taxonomy Validation → Country Inference")
        print("📋 Output: 12 columns with up to 4 classification pairs + country")
        
        # Setup connections
        await cli.setup_connections()
        
        # Read CSV file
        companies = cli.read_csv_file(args.input)
        
        if not companies:
            print("❌ No valid companies found in the CSV file")
            return 1
        
        print(f"\n🎯 Starting integrated processing workflow...")
        print(f"   ✅ Step 1: Perplexity advanced search for company research")
        print(f"   ✅ Step 2: Azure OpenAI classification (up to 4 pairs + country)")
        print(f"   ✅ Step 3: Taxonomy validation against classes.csv")
        print(f"   ✅ Step 4: Final output with literal field copying")
        
        # Classify companies in batches
        markdown_content, csv_results = await cli.classify_companies_batched(companies)
        
        # Save results
        await cli.save_results(markdown_content, csv_results, args.output)
        
        # Print final summary
        stats = cli.progress.get_stats()
        
        print(f"\n📊 Final Summary:")
        print(f"   Total companies: {len(companies)}")
        print(f"   Completed batches: {stats['completed_batches']}/{stats['total_batches']}")
        print(f"   Success rate: {stats['success_rate']:.1f}%")
        print(f"   Processing time: {stats['elapsed_time']/60:.1f} minutes")
        print(f"   Output format: 12 columns (CASEACCID + Company Name + Trading Name + 4 classification pairs + Country)")
        
        # Clean up temp files on successful completion
        if stats['completed_batches'] == stats['total_batches']:
            cli.progress.cleanup_temp_files()
            print("✅ Processing completed successfully! Temporary files cleaned up.")
        else:
            remaining = stats['total_batches'] - stats['completed_batches']
            print(f"⚠️  {remaining} batches remaining. Use --resume to continue (default behavior)")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            print("\nFull traceback:")
            traceback.print_exc()
        return 1
    
    finally:
        if 'cli' in locals():
            await cli.cleanup()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))