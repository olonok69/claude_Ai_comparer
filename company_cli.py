#!/usr/bin/env python3
"""
Enhanced Company Classification CLI Tool with CSV Sanitization
Classifies companies using multiple MCP servers (Google Search + Perplexity) with Excel-compatible CSV output.
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

# Add the client directory to the path
sys.path.insert(0, str(Path(__file__).parent / "client"))

from dotenv import load_dotenv

# Import MCP components
from services.mcp_service import MCPService
from services.llm_service import LLMService
from utils.config_manager import ConfigManager

# Load environment variables
load_dotenv()


class ProgressTracker:
    """Enhanced progress tracker with atomic saves and comprehensive error handling."""
    
    def __init__(self, output_base: str):
        self.output_base = output_base
        self.progress_file = f"{output_base}_progress.json"
        self.temp_results_file = f"{output_base}_temp_results.json"
        self.error_log_file = f"{output_base}_errors.json"
        
        # State tracking
        self.processed_batches = set()
        self.failed_batches = set()
        self.all_data_rows = []
        self.header_row = None
        self.error_count = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.total_batches = 0
        self.start_time = time.time()
        
        # Load existing progress
        self.load_progress()
    
    def load_progress(self):
        """Load existing progress from disk."""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.processed_batches = set(data.get('processed_batches', []))
                self.failed_batches = set(data.get('failed_batches', []))
                self.error_count = data.get('error_count', 0)
                self.consecutive_errors = data.get('consecutive_errors', 0)
                self.total_batches = data.get('total_batches', 0)
                self.start_time = data.get('start_time', time.time())
                
                print(f"📂 Loaded previous progress: {len(self.processed_batches)} completed batches")
            
            # Load temp results
            if os.path.exists(self.temp_results_file):
                with open(self.temp_results_file, 'r', encoding='utf-8') as f:
                    temp_data = json.load(f)
                
                self.all_data_rows = temp_data.get('data_rows', [])
                self.header_row = temp_data.get('header_row')
                
                print(f"📂 Loaded {len(self.all_data_rows)} previous results")
                
        except Exception as e:
            print(f"⚠️  Error loading progress: {e}")
            print("🔄 Starting fresh")
    
    def save_progress(self) -> bool:
        """Save current progress with atomic operations."""
        try:
            # Save main progress
            progress_data = {
                'processed_batches': list(self.processed_batches),
                'failed_batches': list(self.failed_batches),
                'error_count': self.error_count,
                'consecutive_errors': self.consecutive_errors,
                'total_batches': self.total_batches,
                'start_time': self.start_time,
                'last_update': time.time()
            }
            
            # Atomic save for progress
            temp_progress = f"{self.progress_file}.tmp"
            with open(temp_progress, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2)
            
            os.replace(temp_progress, self.progress_file)
            
            # Save temp results
            temp_data = {
                'header_row': self.header_row,
                'data_rows': self.all_data_rows,
                'last_update': time.time()
            }
            
            temp_results = f"{self.temp_results_file}.tmp"
            with open(temp_results, 'w', encoding='utf-8') as f:
                json.dump(temp_data, f, indent=2, ensure_ascii=False)
            
            os.replace(temp_results, self.temp_results_file)
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving progress: {e}")
            return False
    
    def mark_batch_completed(self, batch_num: int, results: List[List[str]]):
        """Mark a batch as completed and save results."""
        self.processed_batches.add(batch_num)
        self.consecutive_errors = 0  # Reset on success
        
        # Add results to collection
        if results:
            header, data_rows = self.extract_header_and_data(results)
            
            if header and self.header_row is None:
                self.header_row = header
            
            if data_rows:
                self.all_data_rows.extend(data_rows)
        
        # Immediate save after each batch
        self.save_progress()
    
    def mark_batch_failed(self, batch_num: int):
        """Mark a batch as failed."""
        self.failed_batches.add(batch_num)
        self.error_count += 1
        self.consecutive_errors += 1
        
        # Save progress immediately
        self.save_progress()
    
    def extract_header_and_data(self, results: List[List[str]]) -> Tuple[Optional[List[str]], List[List[str]]]:
        """Extract header and data rows from results."""
        if not results:
            return None, []
        
        header = None
        data_rows = []
        
        for row in results:
            if not row or len(row) == 0:
                continue
            
            # Check if this is a header row
            if self.is_header_row(row):
                if header is None:
                    header = row
                continue
            
            # Check if this is a separator row
            if self.is_separator_row(row):
                continue
            
            # This is a data row
            if any(cell.strip() for cell in row):
                data_rows.append(row)
        
        return header, data_rows
    
    def is_header_row(self, row: List[str]) -> bool:
        """Check if a row is a header row."""
        if not row or len(row) == 0:
            return False
        
        first_cell = row[0].lower().strip()
        header_indicators = [
            'company name', 'account name', 'trading name', 
            'tech industry', 'industry', 'product'
        ]
        
        return any(indicator in first_cell for indicator in header_indicators)
    
    def is_separator_row(self, row: List[str]) -> bool:
        """Check if a row is a markdown separator row."""
        if not row:
            return False
        
        return all(cell.strip() == '' or all(c in '-|: ' for c in cell.strip()) for cell in row)
    
    def should_stop_due_to_errors(self) -> bool:
        """Check if we should stop due to too many consecutive errors."""
        return self.consecutive_errors >= self.max_consecutive_errors
    
    def get_remaining_batches(self, total_batches: int) -> List[int]:
        """Get list of batches that still need to be processed."""
        return [i for i in range(1, total_batches + 1) 
                if i not in self.processed_batches]
    
    def get_final_results(self) -> List[List[str]]:
        """Get final results with header + data rows."""
        if self.header_row is None:
            return self.all_data_rows
        
        return [self.header_row] + self.all_data_rows
    
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
    
    def force_save(self):
        """Force an immediate save of current progress."""
        print("🔄 Forcing immediate progress save...")
        success = self.save_progress()
        if success:
            print("✅ Force save completed successfully")
        else:
            print("❌ Force save failed")
        return success


class EnhancedCompanyClassificationCLI:
    """Enhanced CLI with Perplexity integration and improved search strategy."""
    
    def __init__(self, config_path: Optional[str] = None, batch_size: int = 10, 
                 enabled_servers: Set[str] = None, output_base: str = "results"):
        self.config_path = config_path or "cli_servers_config.json"
        self.batch_size = batch_size
        self.enabled_servers = enabled_servers or {"google", "company_tagging"}
        self.output_base = output_base
        
        # Initialize progress tracker
        self.progress = ProgressTracker(output_base)
        
        # MCP components
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
    
    def sanitize_csv_field(self, text: str) -> str:
        """
        Sanitize CSV field content to prevent Excel formatting issues.
        Specifically handles Account Name and Trading Name columns.
        """
        if not isinstance(text, str):
            return str(text)
        
        # Remove or escape characters that cause Excel CSV parsing issues
        text = text.replace(',', ';')  # Replace commas with semicolons
        text = text.replace('\\', '/')  # Replace backslashes with forward slashes
        text = text.replace('"', "'")  # Replace double quotes with single quotes
        text = text.replace('\r', ' ')  # Replace carriage returns with spaces
        text = text.replace('\n', ' ')  # Replace newlines with spaces
        text = text.replace('\t', ' ')  # Replace tabs with spaces
        
        # Remove control characters that can break CSV parsing
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        # Trim whitespace
        text = text.strip()
        
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
        
        # Check Google Search credentials if enabled
        if "google" in self.enabled_servers:
            if not os.getenv("GOOGLE_API_KEY"):
                missing_vars.append("GOOGLE_API_KEY")
            if not os.getenv("GOOGLE_SEARCH_ENGINE_ID"):
                missing_vars.append("GOOGLE_SEARCH_ENGINE_ID")
        
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
        config = {}
        
        if "google" in self.enabled_servers:
            config["Google Search"] = {
                "transport": "sse",
                "url": "http://localhost:8002/sse"
            }
        
        if "perplexity" in self.enabled_servers:
            config["Perplexity"] = {
                "transport": "sse", 
                "url": "http://localhost:8001/sse"
            }
        
        if "company_tagging" in self.enabled_servers:
            config["Company Tagging"] = {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "mcp_servers.company_tagging.server"],
                "env": {
                    "PERPLEXITY_API_KEY": os.getenv("PERPLEXITY_API_KEY", ""),
                    "PERPLEXITY_MODEL": os.getenv("PERPLEXITY_MODEL", "llama-3.1-sonar-large-128k-online")
                }
            }
        
        return config
    
    async def setup_connections(self):
        """Setup MCP connections and LLM service."""
        try:
            # Validate environment first
            if not self.validate_server_requirements():
                raise ValueError("Missing required environment variables")
            
            # Setup LLM service
            self.llm = LLMService()
            print("✅ LLM service initialized")
            
            # Get server configuration
            server_config = self.get_server_config()
            
            # Save configuration file
            with open(self.config_path, 'w') as f:
                json.dump(server_config, f, indent=2)
            
            print(f"💾 Server configuration saved to: {self.config_path}")
            
            # Setup MCP client
            config_manager = ConfigManager(self.config_path)
            self.client = MCPService(config_manager)
            
            # Initialize connections
            await self.client.initialize()
            print("✅ MCP connections established")
            
            # Get available tools
            self.tools = await self.client.get_available_tools()
            
            # Organize tools by server
            for tool in self.tools:
                server_name = getattr(tool, 'server_name', 'unknown')
                if 'google' in server_name.lower():
                    self.available_tools["google"].append(tool)
                elif 'perplexity' in server_name.lower():
                    self.available_tools["perplexity"].append(tool)
                elif 'company' in server_name.lower() or 'tagging' in server_name.lower():
                    self.available_tools["company_tagging"].append(tool)
            
            print(f"🔧 Available tools:")
            for server, tools in self.available_tools.items():
                if tools:
                    print(f"   {server}: {len(tools)} tools")
            
        except Exception as e:
            print(f"❌ Error setting up connections: {e}")
            raise
    
    def read_csv_file(self, csv_path: str) -> List[Dict]:
        """Read and parse the CSV file."""
        print(f"📖 Reading CSV file: {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        companies = []
        try:
            with open(csv_path, 'r', encoding='utf-8', newline='') as csvfile:
                sample = csvfile.read(1024)
                csvfile.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(csvfile, delimiter=delimiter)
                
                for row_num, row in enumerate(reader, 1):
                    # Clean and validate row data
                    cleaned_row = {}
                    for k, v in row.items():
                        if k is not None and v is not None:
                            cleaned_row[k.strip()] = self.sanitize_json_string(str(v).strip())
                    
                    # Check for required columns
                    required_columns = ['Account Name', 'Trading Name', 'Domain', 'Event']
                    if all(col in cleaned_row for col in required_columns):
                        if cleaned_row.get('Account Name'):
                            companies.append(cleaned_row)
            
            print(f"✅ Successfully read {len(companies)} companies from CSV")
            return companies
            
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
    
    def format_companies_for_analysis(self, companies: List[Dict]) -> str:
        """Format companies data for the analysis prompt."""
        formatted_lines = []
        
        for company in companies:
            company_block = []
            company_block.append(f"Account Name = {company.get('Account Name', '')}")
            company_block.append(f"Trading Name = {company.get('Trading Name', '')}")
            company_block.append(f"Domain = {company.get('Domain', '')}")
            company_block.append(f"Industry = {company.get('Industry', '')}")
            company_block.append(f"Product/Service Type = {company.get('Product/Service Type', '')}")
            company_block.append(f"Event = {company.get('Event', '')}")
            
            formatted_lines.append('\n'.join(company_block))
        
        return '\n\n'.join(formatted_lines)
    
    def create_research_prompt(self, companies_batch: List[Dict]) -> str:
        """Create research prompt for the batch."""
        companies_text = self.format_companies_for_analysis(companies_batch)
        
        prompt = f"""
You are a data analyst specializing in technology company classification for trade shows. 

TASK: Research and classify the following companies for technology trade shows (CAI, DOL, CCSE, BDAIW, DCW).

COMPANIES TO RESEARCH:
{companies_text}

INSTRUCTIONS:
1. Research each company thoroughly using available tools
2. Focus on their core technology products/services
3. Classify using the official taxonomy categories
4. Provide up to 4 industry/product pairs per company
5. Format as a clean markdown table

REQUIRED OUTPUT FORMAT:
| Company Name | Trading Name | Tech Industry 1 | Tech Product 1 | Tech Industry 2 | Tech Product 2 | Tech Industry 3 | Tech Product 3 | Tech Industry 4 | Tech Product 4 |

Research thoroughly and provide accurate classifications based on actual company information.
"""
        return prompt
    
    def parse_markdown_table(self, response: str) -> List[List[str]]:
        """Parse markdown table response into structured data."""
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
                    table_rows.append(sanitized_columns)
        
        return table_rows
    
    async def process_batch_with_retry(self, batch_companies: List[Dict], 
                                     batch_num: int, total_batches: int) -> List[List[str]]:
        """Process a batch with retry logic."""
        print(f"\n🔄 Processing batch {batch_num}/{total_batches} ({len(batch_companies)} companies)")
        
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    print(f"   🔄 Retry attempt {attempt + 1}/{self.max_retries}")
                    await asyncio.sleep(self.retry_delay)
                
                # Create research prompt
                prompt = self.create_research_prompt(batch_companies)
                
                # Execute research using MCP client
                response = await self.client.execute_with_tools(prompt)
                
                if response and response.strip():
                    # Parse the response
                    table_rows = self.parse_markdown_table(response)
                    
                    if table_rows:
                        print(f"   ✅ Batch {batch_num} completed: {len(table_rows)} rows")
                        return table_rows
                    else:
                        print(f"   ⚠️  Batch {batch_num} returned no valid table data")
                        if attempt == self.max_retries - 1:
                            return []
                else:
                    print(f"   ⚠️  Batch {batch_num} returned empty response")
                    if attempt == self.max_retries - 1:
                        return []
                
            except Exception as e:
                print(f"   ❌ Error in batch {batch_num}, attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    print(f"   ❌ Batch {batch_num} failed after {self.max_retries} attempts")
                    return []
        
        return []
    
    async def classify_companies_batched(self, companies: List[Dict]) -> Tuple[str, List[List[str]]]:
        """Classify companies in batches with enhanced progress tracking."""
        # Create batches
        batches = [companies[i:i + self.batch_size] 
                  for i in range(0, len(companies), self.batch_size)]
        
        total_batches = len(batches)
        self.progress.total_batches = total_batches
        
        print(f"📊 Processing {len(companies)} companies in {total_batches} batches")
        
        # Get remaining batches to process
        remaining_batches = self.progress.get_remaining_batches(total_batches)
        
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
                    self.progress.mark_batch_completed(batch_num, batch_results)
                else:
                    print(f"   ❌ Batch {batch_num} failed completely")
                    self.progress.mark_batch_failed(batch_num)
                    
                    # Check if we should stop due to too many consecutive errors
                    if self.progress.should_stop_due_to_errors():
                        print(f"❌ Stopping due to {self.progress.consecutive_errors} consecutive errors")
                        break
                
                # Print progress update
                stats = self.progress.get_stats()
                print(f"📊 Progress: {stats['completed_batches']}/{total_batches} batches "
                      f"({stats['success_rate']:.1f}% success rate, {len(self.progress.all_data_rows)} companies)")
        
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
        """Save results to CSV file with enhanced sanitization for Excel compatibility."""
        if not results:
            print("⚠️  No results to save")
            return
            
        try:
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                
                # Get header row to identify column positions
                header_row = results[0] if results else []
                account_name_idx = None
                trading_name_idx = None
                
                # Find the indices of Account Name and Trading Name columns
                for idx, header in enumerate(header_row):
                    if header == 'Account Name' or 'Account Name' in header:
                        account_name_idx = idx
                    elif header == 'Trading Name' or 'Trading Name' in header:
                        trading_name_idx = idx
                
                # Process each row
                for row_idx, row in enumerate(results):
                    if row_idx == 0:  # Header row
                        writer.writerow(row)
                    else:  # Data rows
                        sanitized_row = list(row)  # Create a copy
                        
                        # Sanitize Account Name column if found
                        if account_name_idx is not None and len(sanitized_row) > account_name_idx:
                            sanitized_row[account_name_idx] = self.sanitize_csv_field(
                                str(sanitized_row[account_name_idx])
                            )
                        
                        # Sanitize Trading Name column if found
                        if trading_name_idx is not None and len(sanitized_row) > trading_name_idx:
                            sanitized_row[trading_name_idx] = self.sanitize_csv_field(
                                str(sanitized_row[trading_name_idx])
                            )
                        
                        writer.writerow(sanitized_row)
                        
            print(f"💾 CSV results saved to: {csv_path}")
            if account_name_idx is not None or trading_name_idx is not None:
                print(f"✅ Applied Excel-compatible sanitization to Account Name and Trading Name columns")
            
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
    """Main CLI function with enhanced server support."""
    parser = argparse.ArgumentParser(
        description="Enhanced Company Classification CLI Tool with CSV Sanitization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file")
    parser.add_argument("--output", "-o", required=True, help="Base path for output files")
    parser.add_argument("--servers", "-s", type=str, default="both", 
                       help="MCP servers to use: 'google', 'perplexity', or 'both' (default: both)")
    parser.add_argument("--batch-size", "-b", type=int, default=10, 
                       help="Batch size (default: 10)")
    parser.add_argument("--config", "-c", help="Path to server configuration file")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume from previous run (default behavior)")
    parser.add_argument("--clean-start", action="store_true", 
                       help="Start fresh, ignoring previous progress")
    parser.add_argument("--force-continue", action="store_true",
                       help="Continue with existing progress even if batch size changed")
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
            enabled_servers = {"google", "company_tagging"}  # Default fallback
        
        # Use the batch size exactly as specified by user
        batch_size = args.batch_size
        print(f"📊 Using batch size: {batch_size} (as specified)")
        
        if "perplexity" in enabled_servers and batch_size > 2:
            print(f"⚠️  NOTE: Perplexity works best with batch size 2 or smaller")
            print(f"   You specified {batch_size}. Consider using --batch-size 2 if you encounter issues.")
        
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
        
        print("🚀 Enhanced Company Classification CLI Tool with CSV Sanitization")
        print("=" * 80)
        print(f"🔧 Enabled servers: {', '.join(enabled_servers)}")
        print(f"📊 Batch size: {batch_size}")
        print(f"🔄 System recursion limit: {sys.getrecursionlimit()}")
        
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
        stats = cli.progress.get_stats()
        completed_valid = stats['completed_batches']
        total_batches = stats['total_batches']
        
        print(f"\n📊 Final Processing Summary:")
        print(f"   Total companies in file: {len(companies)}")
        print(f"   Total batches: {total_batches}")
        print(f"   Completed batches: {completed_valid}")
        print(f"   Failed batches: {stats['failed_batches']}")
        print(f"   Success rate: {stats['success_rate']:.1f}%")
        print(f"   Total errors: {stats['error_count']}")
        print(f"   Total data rows: {stats['total_data_rows']}")
        print(f"   Header present: {stats['has_header']}")
        print(f"   Processing time: {stats['elapsed_time']:.1f}s")
        
        # Cleanup if successful
        if stats['success_rate'] > 90:
            print("🧹 Cleaning up temporary files...")
            cli.progress.cleanup_temp_files()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1
    finally:
        # Cleanup connections
        if 'cli' in locals():
            await cli.cleanup()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))