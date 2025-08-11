#!/usr/bin/env python3
"""
Enhanced Company Classification CLI Tool with Integrated Cache Management
Processes companies in batches using Perplexity and/or Google with automatic cache clearing
"""

import os
import sys
import json
import csv
import time
import asyncio
import argparse
import traceback
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Set, Any

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ProgressTracker:
    """Track processing progress with atomic saves and cache management."""
    
    def __init__(self, output_base: str, total_batches: int = 0):
        self.output_base = output_base
        self.total_batches = total_batches
        self.processed_batches: Set[int] = set()
        self.failed_batches: Set[int] = set()
        self.cache_clear_log: List[Dict] = []  # Track cache clearing events
        self.all_data_rows: List[List[str]] = []
        self.header_row: Optional[List[str]] = None
        self.error_count = 0
        self.consecutive_errors = 0
        self.start_time = time.time()
        
        # File paths
        self.progress_file = f"{output_base}_progress.json"
        self.temp_results_file = f"{output_base}_temp_results.json"
        self.error_log_file = f"{output_base}_errors.log"
    
    def save_progress(self):
        """Save progress with atomic operations."""
        progress_data = {
            "total_batches": self.total_batches,
            "processed_batches": list(self.processed_batches),
            "failed_batches": list(self.failed_batches),
            "cache_clear_log": self.cache_clear_log,
            "error_count": self.error_count,
            "consecutive_errors": self.consecutive_errors,
            "header_row": self.header_row,
            "data_rows_count": len(self.all_data_rows),
            "start_time": self.start_time,
            "last_update": time.time()
        }
        
        # Atomic save with temp file
        temp_file = f"{self.progress_file}.tmp"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2)
            
            # Atomic rename
            Path(temp_file).replace(self.progress_file)
            
            # Also save accumulated results
            self._save_temp_results()
            
        except Exception as e:
            print(f"⚠️  Error saving progress: {e}")
            self._emergency_save(progress_data)
    
    def _save_temp_results(self):
        """Save temporary results to file."""
        temp_file = f"{self.temp_results_file}.tmp"
        try:
            results_data = {
                "header": self.header_row,
                "data_rows": self.all_data_rows,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2)
            
            Path(temp_file).replace(self.temp_results_file)
            
        except Exception as e:
            print(f"⚠️  Error saving temp results: {e}")
    
    def _emergency_save(self, data: Dict):
        """Emergency save method if primary fails."""
        backup_file = f"{self.output_base}_backup_{int(time.time())}.json"
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"   📦 Emergency backup saved to {backup_file}")
        except Exception as e:
            print(f"   ❌ Emergency save failed: {e}")
    
    def load_progress(self) -> bool:
        """Load existing progress if available."""
        if not os.path.exists(self.progress_file):
            return False
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.total_batches = data.get("total_batches", self.total_batches)
            self.processed_batches = set(data.get("processed_batches", []))
            self.failed_batches = set(data.get("failed_batches", []))
            self.cache_clear_log = data.get("cache_clear_log", [])
            self.error_count = data.get("error_count", 0)
            self.consecutive_errors = data.get("consecutive_errors", 0)
            self.header_row = data.get("header_row")
            self.start_time = data.get("start_time", time.time())
            
            # Load temp results
            if os.path.exists(self.temp_results_file):
                with open(self.temp_results_file, 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
                    self.all_data_rows = results_data.get("data_rows", [])
            
            return True
            
        except Exception as e:
            print(f"⚠️  Error loading progress: {e}")
            return False
    
    def log_cache_clear(self, server: str, success: bool, details: Optional[str] = None):
        """Log cache clearing event."""
        self.cache_clear_log.append({
            "timestamp": time.time(),
            "server": server,
            "success": success,
            "details": details
        })
        self.save_progress()
    
    def mark_batch_completed(self, batch_num: int, parsed_rows: List[List[str]]):
        """Mark a batch as completed and extract data."""
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
            'cache_clears': len(self.cache_clear_log),
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
                 enabled_servers: Set[str] = None, output_base: str = "results",
                 clear_cache_every: int = 10, no_cache_clear: bool = False):
        self.config_path = config_path or "cli_servers_config.json"
        self.batch_size = batch_size
        self.enabled_servers = enabled_servers or {"perplexity", "company_tagging"}
        self.output_base = output_base
        
        # Cache management settings
        self.clear_cache_every = clear_cache_every
        self.no_cache_clear = no_cache_clear
        self.cache_clear_count = 0
        
        # Initialize progress tracker
        self.progress = ProgressTracker(output_base)
        
        # MCP components (using existing structure)
        self.client = None
        self.clients = {}  # Store multiple client sessions
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
        print(f"   Cache management: {'Disabled' if no_cache_clear else f'Clear every {clear_cache_every} batches'}")
        print(f"   Workflow: Perplexity Search → Azure OpenAI → Taxonomy Validation → Country Inference")
    
    async def clear_perplexity_cache(self) -> bool:
        """Clear Perplexity server cache using the clear_api_cache tool."""
        if "perplexity" not in self.clients:
            print("⚠️  Perplexity client not connected")
            return False
        
        try:
            print("🧹 Clearing Perplexity API cache...")
            
            # Get the perplexity client
            client = self.clients["perplexity"]
            
            # Call the clear_api_cache tool
            result = await client.call_tool("clear_api_cache", {})
            
            if result and result.content:
                # Parse the result
                result_text = result.content[0].text if result.content else ""
                try:
                    result_data = json.loads(result_text)
                    entries_cleared = result_data.get("entries_cleared", 0)
                    print(f"   ✅ Perplexity cache cleared: {entries_cleared} entries removed")
                    self.progress.log_cache_clear("perplexity", True, f"{entries_cleared} entries cleared")
                    return True
                except:
                    print(f"   ✅ Perplexity cache cleared")
                    self.progress.log_cache_clear("perplexity", True, "Cache cleared")
                    return True
            else:
                print(f"   ⚠️  No response from clear_api_cache tool")
                self.progress.log_cache_clear("perplexity", False, "No response")
                return False
                
        except Exception as e:
            print(f"   ❌ Error clearing Perplexity cache: {e}")
            self.progress.log_cache_clear("perplexity", False, str(e))
            return False
    
    async def clear_google_cache(self) -> bool:
        """Clear Google server cache using the clear-cache tool."""
        if "google" not in self.clients:
            print("⚠️  Google client not connected")
            return False
        
        try:
            print("🧹 Clearing Google Search cache...")
            
            # Get the google client
            client = self.clients["google"]
            
            # Call the clear-cache tool with all cache types
            result = await client.call_tool("clear-cache", {"cacheType": "all"})
            
            if result and result.content:
                result_text = result.content[0].text if result.content else ""
                try:
                    result_data = json.loads(result_text)
                    search_cleared = result_data.get("searchCleared", 0)
                    webpage_cleared = result_data.get("webpageCleared", 0)
                    total_cleared = search_cleared + webpage_cleared
                    print(f"   ✅ Google cache cleared: {total_cleared} entries removed")
                    print(f"      - Search cache: {search_cleared} entries")
                    print(f"      - Webpage cache: {webpage_cleared} entries")
                    self.progress.log_cache_clear("google", True, f"{total_cleared} entries cleared")
                    return True
                except:
                    print(f"   ✅ Google cache cleared")
                    self.progress.log_cache_clear("google", True, "Cache cleared")
                    return True
            else:
                print(f"   ⚠️  No response from clear-cache tool")
                self.progress.log_cache_clear("google", False, "No response")
                return False
                
        except Exception as e:
            print(f"   ❌ Error clearing Google cache: {e}")
            self.progress.log_cache_clear("google", False, str(e))
            return False
    
    async def clear_all_caches(self) -> bool:
        """Clear caches for all enabled servers."""
        success = True
        
        if "perplexity" in self.enabled_servers:
            success &= await self.clear_perplexity_cache()
        
        if "google" in self.enabled_servers:
            success &= await self.clear_google_cache()
        
        self.cache_clear_count += 1
        return success
    
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
        # Remove control characters
        response_content = self.sanitize_json_string(response_content)
        
        # Fix common JSON issues
        response_content = response_content.replace('\\"', '"')
        response_content = response_content.replace('\\n', '\n')
        response_content = response_content.replace('\\r', '')
        
        return response_content
    
    async def setup_connections(self):
        """Initialize MCP connections and load tools."""
        print("\n🔌 Setting up MCP connections...")
        
        # Load server configuration
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            servers_config = json.load(f)
        
        # Store client sessions for each server
        self.clients = {}
        
        for server_name, config in servers_config.items():
            server_key = server_name.lower().replace(" ", "_")
            if server_key not in self.enabled_servers:
                continue
                
            print(f"   📡 Connecting to {server_name}...")
            
            try:
                if config.get("transport") == "stdio":
                    # stdio transport
                    server_params = StdioServerParameters(
                        command=config["command"],
                        args=config.get("args", []),
                        env={k: os.path.expandvars(v) for k, v in config.get("env", {}).items()}
                    )
                    
                    read_stream, write_stream = await stdio_client(server_params)
                    client = ClientSession(read_stream, write_stream)
                    
                    await client.__aenter__()
                    self.clients[server_key] = client
                    
                    # Get available tools
                    tools = await client.list_tools()
                    self.available_tools[server_key] = [tool.name for tool in tools]
                    
                    print(f"   ✅ Connected to {server_name} ({len(tools)} tools available)")
                    
                else:
                    # SSE transport would need different handling
                    print(f"   ⚠️  SSE transport for {server_name} not implemented in this example")
                    continue
                    
            except Exception as e:
                print(f"   ❌ Failed to connect to {server_name}: {e}")
                continue
        
        # Set the main client to the first available client for compatibility
        if self.clients:
            self.client = list(self.clients.values())[0]
        
        # Clear caches at startup unless disabled
        if not self.no_cache_clear:
            print("\n🚀 Clearing caches before starting...")
            await self.clear_all_caches()
    
    def read_csv_file(self, file_path: str) -> List[Dict[str, str]]:
        """Read and parse CSV file."""
        companies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Clean and validate row
                    if row.get('Account Name') or row.get('CASEACCID'):
                        companies.append(row)
            
            print(f"📂 Loaded {len(companies)} companies from {file_path}")
            return companies
            
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
    
    async def process_batch(self, batch_num: int, batch_companies: List[Dict[str, str]]) -> Tuple[bool, List[List[str]]]:
        """Process a batch of companies with optional cache clearing."""
        
        # Clear cache periodically if enabled
        if not self.no_cache_clear and batch_num > 1 and batch_num % self.clear_cache_every == 0:
            print(f"\n🔄 Periodic cache clear at batch {batch_num}")
            await self.clear_all_caches()
        
        # Continue with existing processing logic
        # This is where your existing batch processing code goes
        # For this example, I'm returning placeholder results
        
        print(f"   🔍 Processing batch {batch_num} with {len(batch_companies)} companies...")
        
        # Your existing processing logic here...
        # This would include calling Perplexity/Google tools, classification, etc.
        
        # Placeholder result
        results = []
        for company in batch_companies:
            results.append([
                company.get('CASEACCID', ''),
                company.get('Account Name', ''),
                company.get('Trading Name', ''),
                # ... other fields
            ])
        
        return True, results
    
    async def classify_companies_batched(self, companies: List[Dict[str, str]]) -> Tuple[str, List[List[str]]]:
        """Process companies in batches with progress tracking and cache management."""
        total_companies = len(companies)
        total_batches = (total_companies + self.batch_size - 1) // self.batch_size
        
        self.progress.total_batches = total_batches
        
        # Check for existing progress
        if self.progress.load_progress():
            completed = len(self.progress.processed_batches)
            print(f"\n📂 Resuming from previous progress:")
            print(f"   ✅ Completed batches: {completed}/{total_batches}")
            print(f"   ❌ Failed batches: {len(self.progress.failed_batches)}")
            print(f"   🧹 Cache clears performed: {len(self.progress.cache_clear_log)}")
        
        # Process batches
        for batch_idx in range(0, total_companies, self.batch_size):
            batch_num = (batch_idx // self.batch_size) + 1
            
            # Skip if already processed
            if batch_num in self.progress.processed_batches:
                continue
            
            batch_companies = companies[batch_idx:batch_idx + self.batch_size]
            
            print(f"\n{'='*60}")
            print(f"📦 Processing Batch {batch_num}/{total_batches}")
            print(f"{'='*60}")
            
            try:
                success, results = await self.process_batch(batch_num, batch_companies)
                
                if success:
                    self.progress.mark_batch_completed(batch_num, results)
                else:
                    self.progress.mark_batch_failed(batch_num, "Processing failed")
                    
            except Exception as e:
                error_msg = str(e)
                traceback_str = traceback.format_exc()
                self.progress.mark_batch_failed(batch_num, error_msg, traceback_str)
                
                if self.progress.consecutive_errors >= 3:
                    print(f"\n❌ Too many consecutive errors ({self.progress.consecutive_errors}), stopping...")
                    break
        
        # Generate final output
        return self.format_results()
    
    def format_results(self) -> Tuple[str, List[List[str]]]:
        """Format results as markdown and CSV."""
        final_results = self.progress.get_final_results()
        
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
        if self.clients:
            for server_name, client in self.clients.items():
                try:
                    await client.__aexit__(None, None, None)
                    print(f"🧹 Cleaned up {server_name} connection")
                except Exception as e:
                    print(f"⚠️  Error cleaning up {server_name}: {e}")
        elif self.client:
            try:
                await self.client.__aexit__(None, None, None)
                print("🧹 Cleaned up MCP connections")
            except Exception as e:
                print(f"⚠️  Error during cleanup: {e}")


async def main():
    """Main CLI function with integrated cache management."""
    parser = argparse.ArgumentParser(
        description="Enhanced Company Classification CLI Tool with Cache Management",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file")
    parser.add_argument("--output", "-o", required=True, help="Base path for output files")
    parser.add_argument("--servers", "-s", type=str, default="perplexity", 
                       help="MCP servers to use: 'perplexity' (default), 'google', or 'both'")
    parser.add_argument("--batch-size", "-b", type=int, default=1, 
                       help="Batch size (default: 1 for accuracy)")
    parser.add_argument("--config", "-c", help="Path to server configuration file")
    parser.add_argument("--clean-start", action="store_true", 
                       help="Start fresh, ignoring previous progress")
    parser.add_argument("--perplexity-batch", action="store_true", 
                       help="Use Perplexity for batch processing (default behavior)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous progress (default behavior)")
    parser.add_argument("--clear-cache-every", type=int, default=10,
                       help="Clear cache every N batches (default: 10)")
    parser.add_argument("--no-cache-clear", action="store_true",
                       help="Disable automatic cache clearing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        # Parse server selection
        if args.servers == "google":
            enabled_servers = {"google", "company_tagging"}
        elif args.servers == "both":
            enabled_servers = {"google", "perplexity", "company_tagging"}
        else:
            # Default to perplexity
            enabled_servers = {"perplexity", "company_tagging"}
        
        # Batch size validation
        batch_size = args.batch_size
        if batch_size > 1:
            print(f"⚠️  Recommended batch size is 1 for integrated workflow (you chose {batch_size})")
        
        # Initialize CLI tool with cache management
        cli = EnhancedCompanyClassificationCLI(
            config_path=args.config,
            batch_size=batch_size,
            enabled_servers=enabled_servers,
            output_base=args.output,
            clear_cache_every=args.clear_cache_every,
            no_cache_clear=args.no_cache_clear
        )
        
        # Clean start if requested
        if args.clean_start:
            print("🧹 Starting fresh (ignoring previous progress)")
            cli.progress.cleanup_temp_files()
        
        print("🚀 Enhanced Company Classification CLI Tool with Cache Management")
        print("=" * 80)
        print(f"🔧 Enabled servers: {', '.join(enabled_servers)}")
        print(f"📊 Batch size: {batch_size}")
        print(f"🧹 Cache management: {'Disabled' if args.no_cache_clear else f'Clear every {args.clear_cache_every} batches'}")
        print(f"📋 Output: 12 columns with up to 4 classification pairs + country")
        
        # Setup connections
        await cli.setup_connections()
        
        # Read CSV file
        companies = cli.read_csv_file(args.input)
        
        if not companies:
            print("❌ No valid companies found in the CSV file")
            return 1
        
        print(f"\n🎯 Starting integrated processing workflow...")
        print(f"   Total companies: {len(companies)}")
        
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
        print(f"   Cache clears performed: {stats['cache_clears']}")
        print(f"   Processing time: {stats['elapsed_time']/60:.1f} minutes")
        
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