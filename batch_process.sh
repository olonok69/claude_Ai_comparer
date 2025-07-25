#!/bin/bash
# Enhanced Batch Processing Script with Perplexity Integration
# Supports Google, Perplexity, or both MCP servers

set -e

# Default values
DEFAULT_BATCH_SIZE=10
DEFAULT_SERVERS="both"

# Function to display usage
show_usage() {
    echo "Usage: $0 <input_csv> <output_directory> [batch_size] [servers]"
    echo ""
    echo "Arguments:"
    echo "  input_csv        Path to the input CSV file"
    echo "  output_directory Directory for output files"
    echo "  batch_size       Number of companies per batch (default: $DEFAULT_BATCH_SIZE)"
    echo "  servers          MCP servers to use: 'google', 'perplexity', or 'both' (default: $DEFAULT_SERVERS)"
    echo ""
    echo "Examples:"
    echo "  # Use both servers (default) - RECOMMENDED"
    echo "  $0 companies.csv results"
    echo ""
    echo "  # Use only Google Search"
    echo "  $0 companies.csv results 3 google"
    echo ""
    echo "  # Use only Perplexity (with enhanced search)"
    echo "  $0 companies.csv results 3 perplexity"
    echo ""
    echo "  # Use both servers with custom batch size"
    echo "  $0 companies.csv results 5 both"
    echo ""
    echo "Server Options:"
    echo "  google      - Use only Google Search MCP server"
    echo "  perplexity  - Use only Perplexity MCP server (uses advanced search with 1-year recency)"
    echo "  both        - Use both Google and Perplexity servers (RECOMMENDED for best results)"
    echo ""
    echo "Performance Notes:"
    echo "  - Perplexity works best with smaller batch sizes (3 recommended)"
    echo "  - Google can handle larger batch sizes (up to 10)"
    echo "  - 'both' option provides the most comprehensive research"
    echo ""
    exit 1
}

# Function to validate servers argument
validate_servers() {
    local servers="$1"
    case "$servers" in
        "google"|"perplexity"|"both")
            return 0
            ;;
        *)
            echo "❌ Error: Invalid servers option '$servers'"
            echo "   Valid options: google, perplexity, both"
            return 1
            ;;
    esac
}

# Function to get server description
get_server_description() {
    local servers="$1"
    case "$servers" in
        "google")
            echo "Google Search only"
            ;;
        "perplexity")
            echo "Perplexity AI only (with advanced search)"
            ;;
        "both")
            echo "Google Search + Perplexity AI (RECOMMENDED)"
            ;;
        *)
            echo "Unknown"
            ;;
    esac
}

# Get optimal batch size
get_optimal_batch_size() {
    local servers="$1"
    local requested_size="$2"
    local default_size="$3"
    
    # Only auto-adjust if user is using the default batch size
    if [ "$requested_size" -eq "$default_size" ]; then
        case "$servers" in
            "perplexity"|"both")
                echo "2"  # Auto-adjust default for Perplexity
                ;;
            *)
                echo "$requested_size"
                ;;
        esac
    else
        # User specified a custom batch size - respect their choice
        echo "$requested_size"
    fi
}

# Check arguments
if [ $# -lt 2 ]; then
    show_usage
fi

INPUT_CSV="$1"
OUTPUT_DIR="$2"
BATCH_SIZE="${3:-$DEFAULT_BATCH_SIZE}"
SERVERS="${4:-$DEFAULT_SERVERS}"

# Validate input file
if [ ! -f "$INPUT_CSV" ]; then
    echo "❌ Error: Input CSV file not found: $INPUT_CSV"
    exit 1
fi

# Validate servers option
if ! validate_servers "$SERVERS"; then
    exit 1
fi

# Validate batch size
if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -lt 1 ]; then
    echo "❌ Error: Batch size must be a positive integer, got: $BATCH_SIZE"
    exit 1
fi

# Get optimal batch size for the selected servers
OPTIMAL_BATCH_SIZE=$(get_optimal_batch_size "$SERVERS" "$BATCH_SIZE" "$DEFAULT_BATCH_SIZE")

if [ "$OPTIMAL_BATCH_SIZE" != "$BATCH_SIZE" ]; then
    echo "⚠️  Auto-adjusting default batch size from $BATCH_SIZE to $OPTIMAL_BATCH_SIZE for optimal performance with '$SERVERS' servers"
    echo "   (Use a custom batch size like --batch-size $BATCH_SIZE to override this behavior)"
    BATCH_SIZE="$OPTIMAL_BATCH_SIZE"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "🚀 Starting Enhanced Batch Processing with Perplexity Integration"
echo "=================================================================="
echo "📁 Input CSV: $INPUT_CSV"
echo "📁 Output Directory: $OUTPUT_DIR"
echo "📊 Batch Size: $BATCH_SIZE (optimized)"
echo "🔧 Servers: $SERVERS ($(get_server_description "$SERVERS"))"
echo ""

# Check if the enhanced CLI exists
if [ ! -f "company_cli.py" ]; then
    echo "❌ Error: company_cli.py not found in current directory"
    echo "   Please run this script from the project root directory"
    exit 1
fi

# Get company count information
echo "📋 Analyzing input CSV..."
if command -v python3 &> /dev/null; then
    COMPANY_COUNT=$(python3 -c "
import sys
sys.path.append('.')
from csv_processor_utility import CSVProcessor
try:
    count = CSVProcessor.count_valid_companies('$INPUT_CSV')
    print(count)
except:
    print(0)
" 2>/dev/null || echo "0")
    
    if [ "$COMPANY_COUNT" -gt 0 ]; then
        TOTAL_BATCHES=$(( (COMPANY_COUNT + BATCH_SIZE - 1) / BATCH_SIZE ))
        echo "   ✅ Found $COMPANY_COUNT valid companies"
        echo "   📊 Will process in $TOTAL_BATCHES batches"
    else
        echo "   ⚠️  Unable to validate companies, proceeding anyway"
    fi
else
    echo "   ⚠️  Python3 not found, skipping pre-validation"
fi

echo ""

# Run the enhanced classification
echo "🔄 Starting company classification with enhanced features..."
OUTPUT_BASE="$OUTPUT_DIR/results_$(date +%Y%m%d_%H%M%S)_${SERVERS}"

# Build the command with enhanced features
CMD="python3 company_cli.py --input \"$INPUT_CSV\" --output \"$OUTPUT_BASE\" --batch-size $BATCH_SIZE --servers $SERVERS"

# Add server-specific optimizations
case "$SERVERS" in
    "perplexity")
        # Perplexity-specific optimizations
        CMD="$CMD --recursion-limit 100"
        echo "🔧 Applying Perplexity optimizations:"
        echo "   - Higher recursion limit (100)"
        echo "   - Using perplexity_advanced_search with recency=year"
        echo "   - Enhanced search queries with Industry/Product data"
        ;;
    "both")
        # Both servers optimizations
        CMD="$CMD --recursion-limit 75"
        echo "🔧 Applying dual-server optimizations:"
        echo "   - Balanced recursion limit (75)"
        echo "   - Google site-specific searches + Perplexity advanced search"
        echo "   - Comprehensive research combining both engines"
        ;;
    "google")
        # Google-specific optimizations
        echo "🔧 Applying Google Search optimizations:"
        echo "   - Standard recursion limit"
        echo "   - Site-specific domain searches when available"
        ;;
esac

# Add verbose flag if DEBUG environment variable is set
if [ -n "$DEBUG" ]; then
    CMD="$CMD --verbose"
    echo "🔍 Debug mode enabled"
fi

echo ""
echo "💻 Running command: $CMD"
echo ""

# Execute the command
if eval $CMD; then
    echo ""
    echo "✅ Batch processing completed successfully!"
    echo ""
    echo "📄 Output files created:"
    
    # Check and report on output files
    MD_FILE="${OUTPUT_BASE}.md"
    CSV_FILE="${OUTPUT_BASE}.csv"
    STATS_FILE="${OUTPUT_BASE}.stats.json"
    
    if [ -f "$MD_FILE" ]; then
        MD_LINES=$(wc -l < "$MD_FILE" 2>/dev/null || echo "0")
        echo "   📝 Markdown: $MD_FILE ($MD_LINES lines)"
    fi
    
    if [ -f "$CSV_FILE" ]; then
        CSV_ROWS=$(( $(wc -l < "$CSV_FILE" 2>/dev/null || echo "1") - 1 ))
        echo "   📊 CSV: $CSV_FILE ($CSV_ROWS data rows)"
    fi
    
    if [ -f "$STATS_FILE" ]; then
        echo "   📈 Statistics: $STATS_FILE"
    fi
    
    echo ""
    echo "🔧 Server Configuration Used:"
    echo "   $(get_server_description "$SERVERS")"
    
    if [ "$SERVERS" = "perplexity" ]; then
        echo "   Enhanced Features:"
        echo "   - Advanced search with 1-year recency filter"
        echo "   - Industry and Product/Service Type integration"
        echo "   - Optimized for Perplexity AI capabilities"
    elif [ "$SERVERS" = "both" ]; then
        echo "   Enhanced Features:"
        echo "   - Dual-engine comprehensive research"
        echo "   - Google domain-specific + Perplexity advanced search"
        echo "   - Combined insights from both platforms"
    fi
    
    echo ""
    echo "📊 Processing Statistics:"
    if [ -n "$COMPANY_COUNT" ] && [ "$COMPANY_COUNT" -gt 0 ]; then
        echo "   Total companies processed: $COMPANY_COUNT"
        echo "   Batch size used: $BATCH_SIZE"
        echo "   Total batches: $TOTAL_BATCHES"
    fi
    
    # Show file sizes
    if [ -f "$MD_FILE" ]; then
        MD_SIZE=$(du -h "$MD_FILE" 2>/dev/null | cut -f1 || echo "unknown")
        echo "   Markdown file size: $MD_SIZE"
    fi
    
    if [ -f "$CSV_FILE" ]; then
        CSV_SIZE=$(du -h "$CSV_FILE" 2>/dev/null | cut -f1 || echo "unknown")
        echo "   CSV file size: $CSV_SIZE"
    fi
    
    echo ""
    echo "🎉 Ready for analysis! Files are in: $OUTPUT_DIR"
    
    # Show next steps based on server used
    echo ""
    echo "📋 Recommended Next Steps:"
    case "$SERVERS" in
        "perplexity")
            echo "   1. Review results for AI-powered insights and analysis"
            echo "   2. Consider running with 'both' servers for comparison"
            echo "   3. Check taxonomy accuracy against Perplexity research"
            ;;
        "google")
            echo "   1. Review results for comprehensive web research"
            echo "   2. Consider running with 'both' servers for AI insights"
            echo "   3. Validate domain-specific search accuracy"
            ;;
        "both")
            echo "   1. Review comprehensive results from dual engines"
            echo "   2. Compare Google web data with Perplexity AI insights"
            echo "   3. This is the most thorough analysis available"
            ;;
    esac
    
else
    echo ""
    echo "❌ Batch processing failed!"
    echo ""
    echo "🔍 Troubleshooting tips:"
    echo "   1. Check that your .env file contains the required API keys"
    echo "   2. Ensure MCP servers are running (if using Docker)"
    echo "   3. Verify the CSV file format matches the expected structure"
    echo "   4. Try running with DEBUG=1 for verbose output:"
    echo "      DEBUG=1 $0 $INPUT_CSV $OUTPUT_DIR $BATCH_SIZE $SERVERS"
    echo ""
    echo "📋 Required environment variables by server:"
    case "$SERVERS" in
        "google")
            echo "   - GOOGLE_API_KEY"
            echo "   - GOOGLE_SEARCH_ENGINE_ID"
            echo "   - OPENAI_API_KEY or Azure OpenAI credentials"
            ;;
        "perplexity")
            echo "   - PERPLEXITY_API_KEY"
            echo "   - OPENAI_API_KEY or Azure OpenAI credentials"
            ;;
        "both")
            echo "   - GOOGLE_API_KEY"
            echo "   - GOOGLE_SEARCH_ENGINE_ID"
            echo "   - PERPLEXITY_API_KEY"
            echo "   - OPENAI_API_KEY or Azure OpenAI credentials"
            ;;
    esac
    
    # Enhanced troubleshooting for different server types
    echo ""
    echo "🔍 Server-specific troubleshooting:"
    case "$SERVERS" in
        "perplexity")
            echo "   Perplexity-specific issues:"
            echo "   1. If you see recursion limit errors, try: --recursion-limit 150"
            echo "   2. Reduce batch size to 1 or 2 for complex companies"
            echo "   3. Check Perplexity API key and rate limits"
            echo "   4. Perplexity works best with smaller, focused batches"
            ;;
        "google")
            echo "   Google Search-specific issues:"
            echo "   1. Verify Google Custom Search Engine ID is correct"
            echo "   2. Check Google API quotas and billing"
            echo "   3. Ensure CSE is set to search the entire web"
            echo "   4. Google can handle larger batch sizes (up to 10)"
            ;;
        "both")
            echo "   Dual-server issues:"
            echo "   1. Check both Google and Perplexity API credentials"
            echo "   2. If one server fails, try running with single server first"
            echo "   3. Monitor API rate limits for both services"
            echo "   4. Consider reducing batch size if experiencing timeouts"
            ;;
    esac
    
    echo ""
    echo "🔄 Recovery options:"
    echo "   1. The script supports automatic resume - just run the same command again"
    echo "   2. Check for partial results in: $OUTPUT_DIR"
    echo "   3. Use --clean-start to ignore previous progress and start fresh"
    echo "   4. Try a different server configuration if one fails consistently"
    
    exit 1
fi