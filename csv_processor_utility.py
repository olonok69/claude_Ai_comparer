"""
Enhanced CSV Processor Utility Functions - With Server Selection Support
Additional utilities for the Company Classification CLI Tool
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import re
import subprocess

class EnhancedCSVProcessor:
    """Enhanced utility class for processing CSV files with server selection support."""
    
    @staticmethod
    def validate_csv_structure(csv_path: str) -> Tuple[bool, List[str]]:
        """Validate CSV structure and return status and missing columns."""
        required_columns = ['Account Name', 'Trading Name', 'Domain', 'Event']
        
        if not os.path.exists(csv_path):
            return False, [f"File not found: {csv_path}"]
        
        try:
            with open(csv_path, 'r', encoding='utf-8', newline='') as csvfile:
                sample = csvfile.read(1024)
                csvfile.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(csvfile, delimiter=delimiter)
                headers = reader.fieldnames
                
                if not headers:
                    return False, ["No headers found in CSV"]
                
                # Check for required columns
                missing_columns = []
                for col in required_columns:
                    if col not in headers:
                        missing_columns.append(col)
                
                if missing_columns:
                    return False, missing_columns
                
                return True, []
                
        except Exception as e:
            return False, [f"Error reading CSV: {e}"]
    
    @staticmethod
    def count_valid_companies(csv_path: str) -> int:
        """Count valid companies in CSV file."""
        try:
            with open(csv_path, 'r', encoding='utf-8', newline='') as csvfile:
                sample = csvfile.read(1024)
                csvfile.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(csvfile, delimiter=delimiter)
                
                count = 0
                for row in reader:
                    cleaned_row = {k.strip(): v.strip() if v else "" for k, v in row.items()}
                    if cleaned_row.get('Account Name'):
                        count += 1
                
                return count
                
        except Exception as e:
            return 0
    
    @staticmethod
    def validate_server_selection(servers: str) -> Tuple[bool, Set[str], List[str]]:
        """Validate server selection string and return parsed servers."""
        if not servers:
            return False, set(), ["Empty servers string"]
        
        valid_servers = {"google", "perplexity", "both"}
        server_set = set(server.strip().lower() for server in servers.split(','))
        
        # Handle 'both' shorthand
        if "both" in server_set:
            server_set.remove("both")
            server_set.update({"google", "perplexity"})
        
        # Validate servers
        invalid_servers = server_set - {"google", "perplexity"}
        if invalid_servers:
            return False, set(), [f"Invalid servers: {invalid_servers}"]
        
        if not server_set:
            return False, set(), ["No valid servers specified"]
        
        return True, server_set, []
    
    @staticmethod
    def get_server_requirements(servers: Set[str]) -> Dict[str, List[str]]:
        """Get required environment variables for server selection."""
        requirements = {
            "always_required": ["OPENAI_API_KEY or Azure OpenAI credentials"],
            "conditional": []
        }
        
        if "google" in servers:
            requirements["conditional"].extend([
                "GOOGLE_API_KEY",
                "GOOGLE_SEARCH_ENGINE_ID"
            ])
        
        if "perplexity" in servers:
            requirements["conditional"].append("PERPLEXITY_API_KEY")
        
        return requirements
    
    @staticmethod
    def check_environment_variables(servers: Set[str]) -> Tuple[bool, List[str], List[str]]:
        """Check if required environment variables are set."""
        missing_vars = []
        available_vars = []
        
        # Check AI provider credentials
        if os.getenv("OPENAI_API_KEY"):
            available_vars.append("OPENAI_API_KEY")
        else:
            azure_vars = ["AZURE_API_KEY", "AZURE_ENDPOINT", "AZURE_DEPLOYMENT", "AZURE_API_VERSION"]
            azure_available = all(os.getenv(var) for var in azure_vars)
            if azure_available:
                available_vars.extend(azure_vars)
            else:
                missing_vars.append("OPENAI_API_KEY or Azure OpenAI credentials")
        
        # Check Google Search credentials
        if "google" in servers:
            if os.getenv("GOOGLE_API_KEY"):
                available_vars.append("GOOGLE_API_KEY")
            else:
                missing_vars.append("GOOGLE_API_KEY")
                
            if os.getenv("GOOGLE_SEARCH_ENGINE_ID"):
                available_vars.append("GOOGLE_SEARCH_ENGINE_ID")
            else:
                missing_vars.append("GOOGLE_SEARCH_ENGINE_ID")
        
        # Check Perplexity credentials
        if "perplexity" in servers:
            if os.getenv("PERPLEXITY_API_KEY"):
                available_vars.append("PERPLEXITY_API_KEY")
            else:
                missing_vars.append("PERPLEXITY_API_KEY")
        
        return len(missing_vars) == 0, available_vars, missing_vars
    
    @staticmethod
    def run_classification_with_servers(
        csv_path: str, 
        output_base: str, 
        servers: Set[str], 
        batch_size: int = 10,
        verbose: bool = False
    ) -> Tuple[bool, str]:
        """Run classification with specified servers."""
        try:
            # Build command
            servers_str = ",".join(servers) if len(servers) > 1 else list(servers)[0]
            
            cmd = [
                "python3", "company_cli.py",
                "--input", csv_path,
                "--output", output_base,
                "--servers", servers_str,
                "--batch-size", str(batch_size)
            ]
            
            if verbose:
                cmd.append("--verbose")
            
            # Run command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            return False, "Command timed out after 1 hour"
        except Exception as e:
            return False, f"Error running classification: {e}"
    
    @staticmethod
    def compare_server_results(results_dir: str) -> Dict[str, any]:
        """Compare results from different server configurations."""
        comparison = {
            "google": {},
            "perplexity": {},
            "both": {},
            "summary": {}
        }
        
        # Look for result files
        for server_type in ["google", "perplexity", "both"]:
            pattern = f"*_{server_type}.csv"
            csv_files = list(Path(results_dir).glob(pattern))
            
            if csv_files:
                # Get the most recent file
                latest_file = max(csv_files, key=os.path.getmtime)
                
                try:
                    with open(latest_file, 'r', encoding='utf-8', newline='') as f:
                        reader = csv.reader(f)
                        rows = list(reader)
                        
                        if rows:
                            comparison[server_type] = {
                                "file": str(latest_file),
                                "total_rows": len(rows) - 1,  # Exclude header
                                "header": rows[0] if rows else [],
                                "sample_rows": rows[1:6] if len(rows) > 1 else []  # First 5 data rows
                            }
                            
                except Exception as e:
                    comparison[server_type] = {
                        "error": f"Error reading {latest_file}: {e}"
                    }
        
        # Create summary
        total_companies = {}
        for server_type, data in comparison.items():
            if server_type != "summary" and isinstance(data, dict) and "total_rows" in data:
                total_companies[server_type] = data["total_rows"]
        
        comparison["summary"] = {
            "servers_tested": list(total_companies.keys()),
            "company_counts": total_companies,
            "recommendations": EnhancedCSVProcessor._generate_recommendations(total_companies)
        }
        
        return comparison
    
    @staticmethod
    def _generate_recommendations(company_counts: Dict[str, int]) -> List[str]:
        """Generate recommendations based on server comparison."""
        recommendations = []
        
        if not company_counts:
            recommendations.append("No valid results found for comparison")
            return recommendations
        
        # Compare counts
        if "both" in company_counts:
            both_count = company_counts["both"]
            
            google_count = company_counts.get("google", 0)
            perplexity_count = company_counts.get("perplexity", 0)
            
            if both_count >= max(google_count, perplexity_count):
                recommendations.append("Using both servers provides the best results")
            
            if google_count > 0 and perplexity_count > 0:
                if abs(google_count - perplexity_count) / max(google_count, perplexity_count) < 0.1:
                    recommendations.append("Google and Perplexity show similar performance")
                elif google_count > perplexity_count:
                    recommendations.append("Google Search shows better coverage for this dataset")
                else:
                    recommendations.append("Perplexity shows better coverage for this dataset")
        
        # Add general recommendations
        if len(company_counts) == 1:
            recommendations.append("Consider testing multiple server configurations for comparison")
        
        return recommendations
    
    @staticmethod
    def generate_server_performance_report(results_dir: str, output_path: str):
        """Generate a comprehensive server performance report."""
        comparison = EnhancedCSVProcessor.compare_server_results(results_dir)
        
        report_lines = [
            "# Server Performance Report",
            f"Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- Servers tested: {len(comparison['summary']['servers_tested'])}",
            f"- Results directory: {results_dir}",
            "",
            "## Results by Server",
            ""
        ]
        
        for server_type in ["google", "perplexity", "both"]:
            data = comparison[server_type]
            if isinstance(data, dict) and "total_rows" in data:
                report_lines.extend([
                    f"### {server_type.title()} Server",
                    f"- Companies processed: {data['total_rows']}",
                    f"- Results file: {data['file']}",
                    ""
                ])
            elif isinstance(data, dict) and "error" in data:
                report_lines.extend([
                    f"### {server_type.title()} Server",
                    f"- Status: Error",
                    f"- Details: {data['error']}",
                    ""
                ])
        
        # Add recommendations
        recommendations = comparison["summary"]["recommendations"]
        if recommendations:
            report_lines.extend([
                "## Recommendations",
                ""
            ])
            for rec in recommendations:
                report_lines.append(f"- {rec}")
            report_lines.append("")
        
        # Add company counts comparison
        company_counts = comparison["summary"]["company_counts"]
        if company_counts:
            report_lines.extend([
                "## Company Processing Comparison",
                ""
            ])
            for server, count in company_counts.items():
                report_lines.append(f"- {server.title()}: {count} companies")
            report_lines.append("")
        
        # Save report
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            print(f"✅ Performance report saved to: {output_path}")
        except Exception as e:
            raise ValueError(f"Error saving report: {e}")
    
    @staticmethod
    def create_server_comparison_script(output_path: str = "compare_servers.sh"):
        """Create a script to compare different server configurations."""
        script_content = '''#!/bin/bash
# Server Comparison Script
# Tests all three server configurations and compares results

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_csv> <output_directory> [batch_size]"
    echo "Example: $0 companies.csv comparison_results 10"
    exit 1
fi

INPUT_CSV="$1"
OUTPUT_DIR="$2"
BATCH_SIZE="${3:-10}"

echo "🔍 Starting server comparison for: $INPUT_CSV"
echo "📁 Output directory: $OUTPUT_DIR"
echo "📊 Batch size: $BATCH_SIZE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Test each server configuration
SERVERS=("google" "perplexity" "both")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🧪 Testing server configurations..."
echo ""

for SERVER in "${SERVERS[@]}"; do
    echo "▶️  Testing: $SERVER"
    OUTPUT_BASE="$OUTPUT_DIR/comparison_${TIMESTAMP}_${SERVER}"
    
    if python3 company_cli.py --input "$INPUT_CSV" --output "$OUTPUT_BASE" --servers "$SERVER" --batch-size "$BATCH_SIZE"; then
        echo "   ✅ $SERVER: Success"
    else
        echo "   ❌ $SERVER: Failed"
    fi
    echo ""
done

echo "📊 Generating comparison report..."
python3 -c "
import sys
sys.path.append('.')
from csv_processor_utility import EnhancedCSVProcessor
EnhancedCSVProcessor.generate_server_performance_report('$OUTPUT_DIR', '$OUTPUT_DIR/performance_report.md')
"

echo "✅ Server comparison completed!"
echo "📄 Results available in: $OUTPUT_DIR"
echo "📋 Performance report: $OUTPUT_DIR/performance_report.md"
'''
        
        try:
            with open(output_path, 'w') as f:
                f.write(script_content)
            os.chmod(output_path, 0o755)
            print(f"✅ Server comparison script created: {output_path}")
        except Exception as e:
            raise ValueError(f"Error creating script: {e}")

def main():
    """Main function for enhanced CSV processor utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced CSV Processing Utilities with Server Selection")
    parser.add_argument("--validate", help="Validate CSV structure")
    parser.add_argument("--count", help="Count valid companies in CSV")
    parser.add_argument("--validate-servers", help="Validate server selection string")
    parser.add_argument("--check-env", help="Check environment variables for servers")
    parser.add_argument("--run-classification", help="Run classification with specific servers")
    parser.add_argument("--output", help="Output file/directory path")
    parser.add_argument("--servers", default="both", help="Server selection (google, perplexity, both)")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--compare-results", help="Compare results from different servers")
    parser.add_argument("--generate-report", help="Generate performance report for results directory")
    parser.add_argument("--create-comparison-script", action="store_true", help="Create server comparison script")
    
    args = parser.parse_args()
    
    try:
        if args.validate:
            valid, issues = EnhancedCSVProcessor.validate_csv_structure(args.validate)
            if valid:
                print("✅ CSV structure is valid")
            else:
                print(f"❌ CSV validation failed: {issues}")
        
        elif args.count:
            count = EnhancedCSVProcessor.count_valid_companies(args.count)
            print(f"📊 Valid companies in file: {count}")
        
        elif args.validate_servers:
            valid, server_set, errors = EnhancedCSVProcessor.validate_server_selection(args.validate_servers)
            if valid:
                print(f"✅ Server selection is valid: {server_set}")
                
                # Show requirements
                requirements = EnhancedCSVProcessor.get_server_requirements(server_set)
                print(f"📋 Required environment variables:")
                for var in requirements["always_required"]:
                    print(f"   - {var}")
                for var in requirements["conditional"]:
                    print(f"   - {var}")
            else:
                print(f"❌ Server validation failed: {errors}")
        
        elif args.check_env:
            valid, server_set, errors = EnhancedCSVProcessor.validate_server_selection(args.check_env)
            if valid:
                env_ok, available, missing = EnhancedCSVProcessor.check_environment_variables(server_set)
                
                print(f"🔍 Environment check for servers: {server_set}")
                print(f"✅ Available variables: {available}")
                if missing:
                    print(f"❌ Missing variables: {missing}")
                print(f"Status: {'✅ Ready' if env_ok else '❌ Not ready'}")
            else:
                print(f"❌ Invalid server selection: {errors}")
        
        elif args.run_classification:
            if not args.output:
                print("❌ --output is required for running classification")
                sys.exit(1)
            
            valid, server_set, errors = EnhancedCSVProcessor.validate_server_selection(args.servers)
            if not valid:
                print(f"❌ Invalid server selection: {errors}")
                sys.exit(1)
            
            success, output = EnhancedCSVProcessor.run_classification_with_servers(
                args.run_classification, args.output, server_set, args.batch_size, args.verbose
            )
            
            if success:
                print("✅ Classification completed successfully")
                if args.verbose:
                    print(output)
            else:
                print(f"❌ Classification failed: {output}")
        
        elif args.compare_results:
            comparison = EnhancedCSVProcessor.compare_server_results(args.compare_results)
            print(f"📊 Server Comparison Results:")
            print(json.dumps(comparison, indent=2))
        
        elif args.generate_report:
            if not args.output:
                print("❌ --output is required for generating report")
                sys.exit(1)
            
            EnhancedCSVProcessor.generate_server_performance_report(args.generate_report, args.output)
        
        elif args.create_comparison_script:
            EnhancedCSVProcessor.create_server_comparison_script()
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()