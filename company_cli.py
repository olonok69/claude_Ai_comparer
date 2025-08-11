#!/usr/bin/env python3
"""
Enhanced Company Classification CLI Tool
Integrates Perplexity logic with improved search strategy and maintains Google functionality.
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
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import re

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
    """COMPLETELY FIXED progress tracker with immediate saves and atomic operations."""
    
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
        self.start_time = time.time()  # FIX: Initialize immediately
        self.last_save_time = None
        
        # Error tracking
        self.error_count = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 3
        
        # FIX: Ensure output directory exists
        output_dir = Path(self.output_base).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 ProgressTracker initialized for: {output_base}")
        print(f"   📄 Progress file: {self.progress_file}")
        print(f"   📄 Temp results: {self.temp_results_file}")
        print(f"   📄 Error log: {self.error_log_file}")
    
    def save_progress(self):
        """FIXED: Save current progress to disk with atomic operations and proper error handling."""
        try:
            # Prepare progress data with JSON-compatible types
            progress_data = {
                'processed_batches': list(self.processed_batches),  # FIX: Convert set to list
                'failed_batches': list(self.failed_batches),        # FIX: Convert set to list
                'all_data_rows': self.all_data_rows,
                'header_row': self.header_row,
                'total_batches': self.total_batches,
                'start_time': self.start_time,
                'error_count': self.error_count,
                'last_save_time': time.time()  # FIX: Use time.time() instead of datetime
            }
            
            # FIX: Atomic save for progress file using temporary file
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
            
            # FIX: Atomic save for human-readable results
            if self.header_row and self.all_data_rows:
                all_results = [self.header_row] + self.all_data_rows
                temp_results_file = f"{self.temp_results_file}.tmp"
                try:
                    with open(temp_results_file, 'w', encoding='utf-8') as f:
                        json.dump(all_results, f, indent=2, ensure_ascii=False)
                    # Atomic move
                    shutil.move(temp_results_file, self.temp_results_file)
                    print(f"   💾 Temp results saved: {len(all_results)} rows")
                except Exception as e:
                    # Clean up temp file if it exists
                    if os.path.exists(temp_results_file):
                        os.remove(temp_results_file)
                    raise e
            
            self.last_save_time = time.time()
            return True
            
        except PermissionError as e:
            print(f"❌ Permission error saving progress: {e}")
            print(f"   Check write permissions for: {Path(self.output_base).parent}")
            return False
        except OSError as e:
            print(f"❌ OS error saving progress: {e}")
            print(f"   Check disk space and file system permissions")
            return False
        except Exception as e:
            print(f"❌ Unexpected error saving progress: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            return False
    
    def load_progress(self) -> bool:
        """FIXED: Load existing progress if available with proper error handling."""
        if not os.path.exists(self.progress_file):
            print(f"📂 No existing progress file found: {self.progress_file}")
            return False
            
        try:
            with open(self.progress_file, 'rb') as f:
                data = pickle.load(f)
                
            # FIX: Convert lists back to sets for internal processing
            self.processed_batches = set(data.get('processed_batches', []))
            self.failed_batches = set(data.get('failed_batches', []))
            self.all_data_rows = data.get('all_data_rows', [])
            self.header_row = data.get('header_row', None)
            self.total_batches = data.get('total_batches', 0)
            self.start_time = data.get('start_time', time.time())  # FIX: Fallback to current time
            self.error_count = data.get('error_count', 0)
            
            print(f"📥 Progress loaded successfully:")
            print(f"   ✅ Processed batches: {len(self.processed_batches)}")
            print(f"   ❌ Failed batches: {len(self.failed_batches)}")
            print(f"   📊 Data rows: {len(self.all_data_rows)}")
            print(f"   📋 Header: {'Yes' if self.header_row else 'No'}")
            print(f"   🔢 Total batches: {self.total_batches}")
            return True
            
        except (pickle.PickleError, EOFError) as e:
            print(f"❌ Error loading progress file (corrupted): {e}")
            print(f"   Moving corrupted file to: {self.progress_file}.corrupted")
            try:
                shutil.move(self.progress_file, f"{self.progress_file}.corrupted")
            except:
                pass
            return False
        except Exception as e:
            print(f"❌ Unexpected error loading progress: {e}")
            return False
    
    def mark_batch_completed(self, batch_num: int, parsed_rows: List[List[str]]):
        """FIXED: Mark a batch as completed and save results with immediate persistence."""
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
        
        # FIX: Save progress after EVERY batch completion (not just every 10)
        print(f"   💾 Saving progress for batch {batch_num}...")
        save_success = self.save_progress()
        
        if not save_success:
            print(f"   ⚠️  Warning: Failed to save progress for batch {batch_num}")
            print(f"   📊 Current state: {len(self.processed_batches)} completed, {len(self.all_data_rows)} rows")
            # FIX: Try alternative save method
            try:
                self.force_save_alternative()
            except Exception as e:
                print(f"   ❌ Alternative save also failed: {e}")
        else:
            print(f"   ✅ Progress saved successfully for batch {batch_num}")
    
    def mark_batch_failed(self, batch_num: int, error: str, traceback_str: str = ""):
        """FIXED: Mark a batch as failed and log the error with immediate persistence."""
        print(f"   ❌ Marking batch {batch_num} as failed...")
        
        self.failed_batches.add(batch_num)
        self.error_count += 1
        self.consecutive_errors += 1
        
        # Log error immediately
        self.log_error(batch_num, error, traceback_str)
        
        # FIX: Save progress immediately after marking failure
        print(f"   💾 Saving progress after batch {batch_num} failure...")
        save_success = self.save_progress()
        
        if not save_success:
            print(f"   ⚠️  Warning: Failed to save progress after batch {batch_num} failure")
            # FIX: Try alternative save method
            try:
                self.force_save_alternative()
            except Exception as e:
                print(f"   ❌ Alternative save also failed: {e}")
        else:
            print(f"   ✅ Progress saved after failure for batch {batch_num}")
    
    def force_save_alternative(self):
        """Alternative save method using a different approach."""
        try:
            # Create a minimal save data structure
            minimal_data = {
                'processed': list(self.processed_batches),
                'failed': list(self.failed_batches),
                'rows': len(self.all_data_rows),
                'total': self.total_batches,
                'timestamp': time.time()
            }
            
            backup_file = f"{self.output_base}_backup.json"
            with open(backup_file, 'w') as f:
                json.dump(minimal_data, f)
            
            print(f"   🔄 Alternative save completed: {backup_file}")
            return True
        except Exception as e:
            print(f"   ❌ Alternative save also failed: {e}")
            return False
    
    def log_error(self, batch_num: int, error: str, traceback_str: str = ""):
        """Log an error to the error log file with atomic write."""
        try:
            error_entry = f"\n{'='*50}\n"
            error_entry += f"Batch {batch_num} Error - {datetime.now().isoformat()}\n"
            error_entry += f"Error: {error}\n"
            if traceback_str:
                error_entry += f"Traceback:\n{traceback_str}\n"
            error_entry += f"{'='*50}\n"
            
            # FIX: Atomic write for error log
            temp_error_file = f"{self.error_log_file}.tmp"
            try:
                # Append to existing content
                existing_content = ""
                if os.path.exists(self.error_log_file):
                    with open(self.error_log_file, 'r', encoding='utf-8') as f:
                        existing_content = f.read()
                
                with open(temp_error_file, 'w', encoding='utf-8') as f:
                    f.write(existing_content + error_entry)
                
                # Atomic move
                shutil.move(temp_error_file, self.error_log_file)
                print(f"   📝 Error logged for batch {batch_num}")
                
            except Exception as e:
                # Clean up temp file if it exists
                if os.path.exists(temp_error_file):
                    os.remove(temp_error_file)
                raise e
                
        except Exception as e:
            print(f"⚠️  Error writing to error log: {e}")
    
    def is_header_row(self, row: List[str]) -> bool:
        """Check if a row is a header row."""
        if not row or len(row) == 0:
            return False
        
        # Check for common header indicators
        first_cell = row[0].lower().strip()
        header_indicators = [
            'company name', 'account name', 'trading name', 
            'tech industry', 'industry', 'product'
        ]
        
        return any(indicator in first_cell for indicator in header_indicators)
    
    def is_separator_row(self, row: List[str]) -> bool:
        """Check if a row is a markdown separator row (contains only dashes)."""
        if not row:
            return False
        
        return all(cell.strip() == '' or all(c in '-|: ' for c in cell.strip()) for cell in row)
    
    def extract_header_and_data(self, parsed_rows: List[List[str]]) -> Tuple[Optional[List[str]], List[List[str]]]:
        """Extract header and data rows from parsed table."""
        if not parsed_rows:
            return None, []
        
        header = None
        data_rows = []
        
        for row in parsed_rows:
            if not row or len(row) == 0:
                continue
            
            # Skip separator rows
            if self.is_separator_row(row):
                continue
            
            # Identify header row
            if self.is_header_row(row):
                if header is None:  # Take the first header we find
                    header = row
                # Skip subsequent headers (duplicates)
                continue
            
            # This is a data row
            if row and any(cell.strip() for cell in row):  # Has some content
                data_rows.append(row)
        
        return header, data_rows
    
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
        """Get current processing statistics with corrected calculations."""
        # Calculate valid batches (only those within current total_batches)
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
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'all_processed_batches': len(self.processed_batches),  # For debugging
            'all_failed_batches': len(self.failed_batches)  # For debugging
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
        """Force an immediate save of current progress (useful for debugging)."""
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
        
        # Initialize progress tracker with fixed implementation
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
        
        # Don't set recursion limit here - wait until after LLM initialization
        
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
        
        # Add Google Search if enabled
        if "google" in self.enabled_servers and "Google Search" in all_servers:
            filtered_servers["Google Search"] = all_servers["Google Search"]
        
        # Add Perplexity Search if enabled
        if "perplexity" in self.enabled_servers and "Perplexity Search" in all_servers:
            filtered_servers["Perplexity Search"] = all_servers["Perplexity Search"]
        
        return filtered_servers
    
    async def setup_connections(self):
        """Set up MCP client connections and initialize the agent."""
        print("🔧 Setting up MCP connections...")
        
        if not self.validate_server_requirements():
            raise ValueError("Missing required environment variables")
        
        # Create LLM instance first (before setting recursion limit)
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
        
        # Skip recursion limit adjustment - use system default
        current_limit = sys.getrecursionlimit()
        print(f"🔄 Using system recursion limit: {current_limit}")
        
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
            tool_desc = tool.description.lower() if hasattr(tool, 'description') and tool.description else ""
            
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
        """Create research prompt for the batch with enhanced server-specific strategy."""
        company_data = self.format_companies_for_analysis(companies_batch)
        
        # Determine available search tools
        has_google = len(self.available_tools["google"]) > 0
        has_perplexity = len(self.available_tools["perplexity"]) > 0
        
        # Build research instructions based on available tools
        research_instructions = []
        
        if has_google and has_perplexity:
            # Both available - use both for comprehensive research
            research_instructions.extend([
                "   - Use google-search tool: If domain exists and is not empty, use \"site:[domain] products services\", otherwise use \"[company name] products services technology\"",
                "   - Use perplexity_advanced_search tool with recency=\"year\": \"[company name] [Industry if available] [Product/Service Type if available] products services technology offerings\"",
                "   - Combine insights from both searches for comprehensive understanding"
            ])
            tool_requirements = "- MUST use BOTH google-search AND perplexity_advanced_search for each company"
        elif has_perplexity:
            # Only Perplexity available
            research_instructions.append("   - Use perplexity_advanced_search tool with recency=\"year\": \"[company name] [Industry if available] [Product/Service Type if available] products services technology offerings\"")
            tool_requirements = "- MUST use perplexity_advanced_search for each company with recency=\"year\""
        elif has_google:
            # Only Google available
            research_instructions.append("   - Use google-search tool: If domain exists and is not empty, use \"site:[domain] products services\", otherwise use \"[company name] products services technology\"")
            tool_requirements = "- MUST use google-search tool for each company"
        else:
            # No search tools available
            research_instructions.append("   - No web search tools available - use available knowledge and company tagging tools only")
            tool_requirements = "- No web search tools available - proceed with taxonomy matching based on available information"
        
        return f"""You are a professional data analyst tasked with tagging exhibitor companies with accurate industry and product categories from our established taxonomy.

COMPANY DATA TO ANALYZE:
{company_data}

MANDATORY RESEARCH PROCESS:

1. **Retrieve Complete Taxonomy** (ONCE ONLY):
   - Use search_show_categories tool without any filters to get all available categories

2. **For EACH Company - Enhanced Research Phase:**
   - Choose research name priority: Domain > Trading Name > Account Name
   - If Industry and Product/Service Type are available in the input data, incorporate them into your search queries
{chr(10).join(research_instructions)}
   - Identify what the company actually sells/offers

3. **Analysis Phase:**
   - Map company offerings to relevant shows (CAI, DOL, CCSE, BDAIW, DCW)
   - Match findings to EXACT taxonomy pairs from step 1
   - Select up to 4 (Industry | Product) pairs per company
   - Use pairs EXACTLY as they appear

4. **Output Requirements:**
   - Generate a markdown table with ONLY DATA ROWS (NO HEADER)
   - Use this exact format for each company:
   | Company Name | Trading Name | Tech Industry 1 | Tech Product 1 | Tech Industry 2 | Tech Product 2 | Tech Industry 3 | Tech Product 3 | Tech Industry 4 | Tech Product 4 |
   - Do NOT include the header row in your response
   - Do NOT provide additional text, explanations, or context
   - ONLY the data rows without header

CRITICAL RULES:
- MUST use search_show_categories to get taxonomy before starting
{tool_requirements}
- If Industry and Product/Service Type are available in input data, USE THEM to enhance search queries
- Use taxonomy pairs EXACTLY as written
- Output ONLY data rows (no header row)
- Each row should have company data only

Begin the systematic analysis now."""
    
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
        for attempt in range(self.max_retries):
            try:
                print(f"🔄 Processing batch {batch_num}/{total_batches} (attempt {attempt + 1}/{self.max_retries})")
                
                batch_prompt = self.create_research_prompt(batch_companies)
                conversation_memory = [HumanMessage(content=batch_prompt)]
                
                # Adjust timeout based on server selection
                timeout = 600 if "perplexity" in self.enabled_servers else 300  # 10 min for Perplexity, 5 min for Google
                
                # Run the agent with timeout and comprehensive error handling
                try:
                    response = await asyncio.wait_for(
                        self.agent.ainvoke({"messages": conversation_memory}),
                        timeout=timeout
                    )
                except RecursionError as e:
                    # Handle recursion errors specifically
                    error_msg = f"Batch {batch_num} recursion error - try reducing batch size or complexity"
                    print(f"   🔁 {error_msg}")
                    if attempt < self.max_retries - 1:
                        print(f"   🔄 Retrying with simplified approach...")
                        await asyncio.sleep(self.retry_delay)
                        continue
                    else:
                        self.progress.mark_batch_failed(batch_num, error_msg, str(e))
                        return []
                except Exception as llm_error:
                    # Check for Azure OpenAI specific errors
                    error_str = str(llm_error).lower()
                    if any(keyword in error_str for keyword in ['quota', 'rate limit', 'billing', 'insufficient', 'exceeded']):
                        error_msg = f"Batch {batch_num} Azure OpenAI API error: {str(llm_error)}"
                        print(f"   🚨 {error_msg}")
                        print(f"   💰 Azure OpenAI quota/billing issue detected:")
                        
                        if 'quota' in error_str:
                            print(f"      - Check your Azure OpenAI quota limits")
                            print(f"      - Consider upgrading your Azure subscription")
                        elif 'rate limit' in error_str:
                            print(f"      - Azure API rate limit hit")
                            print(f"      - Wait before retrying or increase rate limits")
                        elif 'billing' in error_str:
                            print(f"      - Azure billing/payment issue")
                            print(f"      - Check your Azure account billing status")
                        
                        print(f"   🛑 Stopping batch processing to avoid wasting API calls")
                        self.progress.mark_batch_failed(batch_num, error_msg, traceback.format_exc())
                        return []
                    else:
                        # Generic LLM error
                        raise llm_error
                
                # Extract and clean the response
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
                
                # If we get here, no valid response was found
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
        """Classify companies using batched processing with proper header management."""
        
        # Split companies into batches
        batches = [companies[i:i + self.batch_size] for i in range(0, len(companies), self.batch_size)]
        total_batches = len(batches)
        
        print(f"📊 Created {total_batches} batches of {self.batch_size} companies each")
        
        # Load existing progress if available
        progress_loaded = self.progress.load_progress()
        
        # Check if batch size has changed
        if progress_loaded and self.progress.total_batches != total_batches:
            print(f"⚠️  Batch configuration changed:")
            print(f"   Previous: {self.progress.total_batches} batches")
            print(f"   Current: {total_batches} batches")
            print(f"   This may cause batch numbering issues. Consider using --clean-start")
            
            # Check for force continue flag via a simple approach
            # (In a real implementation, you'd pass this as a parameter)
            import sys
            if "--force-continue" in sys.argv:
                print("🔄 Forcing continuation with existing progress (--force-continue specified)")
            else:
                # Ask user what to do
                print("Options:")
                print("  1. Continue anyway (may have inconsistencies)")
                print("  2. Start fresh (recommended)")
                print("  3. Exit and use --clean-start")
                
                try:
                    choice = input("Choose [1/2/3]: ").strip()
                    if choice == "2":
                        print("🧹 Starting fresh (clearing previous progress)")
                        self.progress.cleanup_temp_files()
                        self.progress = ProgressTracker(self.output_base)
                        progress_loaded = False
                    elif choice == "3":
                        print("👋 Exiting. Run again with --clean-start to start fresh.")
                        sys.exit(0)
                    else:
                        print("📂 Continuing with existing progress (batch numbers may be inconsistent)")
                except KeyboardInterrupt:
                    print("\n👋 Exiting. Run again with --clean-start to start fresh.")
                    sys.exit(0)
        
        # Initialize progress tracker
        self.progress.total_batches = total_batches
        self.progress.start_time = time.time()
        
        if progress_loaded:
            print(f"📂 Resuming from previous run...")
        
        # Set default header if we don't have one
        if self.progress.header_row is None:
            self.progress.header_row = [
                "Company Name", "Trading Name", 
                "Tech Industry 1", "Tech Product 1",
                "Tech Industry 2", "Tech Product 2", 
                "Tech Industry 3", "Tech Product 3",
                "Tech Industry 4", "Tech Product 4"
            ]
            print("📋 Using default header structure")
        
        # Get remaining batches to process (ensure valid range)
        remaining_batches = [i for i in range(1, total_batches + 1) 
                           if i not in self.progress.processed_batches and i <= total_batches]
        
        if not remaining_batches:
            print("✅ All batches already completed!")
        else:
            print(f"📊 Processing {len(remaining_batches)} remaining batches out of {total_batches} total")
            
            # Show which batches are remaining
            if len(remaining_batches) <= 10:
                print(f"   Remaining batches: {remaining_batches}")
            else:
                print(f"   Remaining batches: {remaining_batches[:5]}...{remaining_batches[-5:]}")
        
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
                
                # Check if we should stop due to too many consecutive errors
                if self.progress.should_stop_due_to_errors():
                    print(f"❌ Stopping due to {self.progress.consecutive_errors} consecutive errors")
                    break
            
            # Print progress update with corrected stats
            completed_batches = len([b for b in self.progress.processed_batches if b <= total_batches])
            success_rate = completed_batches / max(1, total_batches) * 100
            print(f"📊 Progress: {completed_batches}/{total_batches} batches "
                  f"({success_rate:.1f}% success rate, {len(self.progress.all_data_rows)} companies)")
        
        # Final save
        self.progress.save_progress()
        
        # Generate final results with proper header
        final_results = self.progress.get_final_results()
        
        # Generate markdown content with single header
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
    """Main CLI function with enhanced server support."""
    parser = argparse.ArgumentParser(
        description="Enhanced Company Classification CLI Tool with Perplexity Integration",
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
        # Remove recursion limit parameter since it's not being used
        # parser.add_argument("--recursion-limit", type=int, default=30,
        #                    help="Set Python recursion limit (default: 30)")
    
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
        
        # Initialize CLI tool with enhanced features (no recursion limit parameter)
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
        
        print("🚀 Enhanced Company Classification CLI Tool with Perplexity Integration")
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
        
        # Print final summary with corrected calculations
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
        print(f"   Processing time: {stats['elapsed_time']/3600:.1f} hours")
        print(f"   Servers used: {', '.join(enabled_servers)}")
        
        # Add comprehensive error summary if there were failures
        if stats['failed_batches'] > 0:
            print(f"\n🚨 Error Summary:")
            print(f"   Failed batches: {stats['failed_batches']}")
            print(f"   Check error details in: {cli.progress.error_log_file}")
            print(f"\n💡 Common solutions:")
            print(f"   - For quota errors: Add more credits to your API accounts")
            print(f"   - For rate limits: Wait before retrying or upgrade your API plan")
            print(f"   - For auth errors: Check API keys in your .env file")
            print(f"   - For network errors: Check internet connection and try again")
            print(f"   - To retry failed batches: Use --resume flag")
        
        # Clean up temp files on successful completion
        if completed_valid == total_batches:
            cli.progress.cleanup_temp_files()
            print("✅ Processing completed successfully! Temporary files cleaned up.")
            print("🎉 Results saved with single header and no duplicates!")
        else:
            # Find the next batch that needs processing
            all_batches = set(range(1, total_batches + 1))
            processed_batches = set(b for b in cli.progress.processed_batches if b <= total_batches)
            remaining_batches = all_batches - processed_batches
            
            if remaining_batches:
                next_batch = min(remaining_batches)
                print(f"⚠️  Processing incomplete. Use --resume to continue from batch {next_batch}")
            else:
                print("✅ All valid batches completed (some historical batches may be out of range)")
                cli.progress.cleanup_temp_files()
                print("🎉 Results saved with single header and no duplicates!")
            print(f"   Progress saved in: {cli.progress.progress_file}")
        
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