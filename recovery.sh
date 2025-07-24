#!/bin/bash
# Recovery and Progress Management Script for Company Classification

set -e

show_usage() {
    echo "Recovery and Progress Management Script"
    echo "====================================="
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  status <output_base>     - Show current progress status"
    echo "  resume <input_csv> <output_base> [options] - Resume processing"
    echo "  clean <output_base>      - Clean up temporary files"
    echo "  analyze <output_base>    - Analyze errors and progress"
    echo ""
    echo "Resume Options:"
    echo "  --batch-size N          - Batch size (default: 5)"
    echo "  --servers TYPE          - Server type (default: google)"
    echo "  --clean-start           - Start fresh, ignore progress"
    echo "  --verbose               - Enable verbose output"
    echo ""
    echo "Examples:"
    echo "  $0 status All_accounts"
    echo "  $0 resume All_Tech_Account_Data.csv All_accounts"
    echo "  $0 resume All_Tech_Account_Data.csv All_accounts --batch-size 3"
    echo ""
    exit 1
}

# Function to show progress status
show_status() {
    local output_base="$1"
    
    echo "📊 Progress Status for: $output_base"
    echo "================================="
    echo ""
    
    # Check for progress file
    if [ -f "${output_base}_progress.pkl" ]; then
        echo "✅ Progress file found: ${output_base}_progress.pkl"
        
        # Try to get stats using Python
        python3 -c "
import pickle
import sys
try:
    with open('${output_base}_progress.pkl', 'rb') as f:
        data = pickle.load(f)
    
    completed = len(data.get('processed_batches', set()))
    failed = len(data.get('failed_batches', set()))
    total = data.get('total_batches', 0)
    errors = data.get('error_count', 0)
    
    print(f'📈 Progress Statistics:')
    print(f'   Total batches: {total}')
    print(f'   Completed batches: {completed}')
    print(f'   Failed batches: {failed}')
    print(f'   Remaining batches: {total - completed}')
    print(f'   Success rate: {completed/max(1,total)*100:.1f}%')
    print(f'   Total errors: {errors}')
    
    if data.get('start_time'):
        import time
        elapsed = time.time() - data['start_time']
        print(f'   Elapsed time: {elapsed/3600:.1f} hours')
        
        if completed > 0:
            rate = completed / (elapsed / 3600)
            remaining_time = (total - completed) / max(rate, 0.1)
            print(f'   Processing rate: {rate:.1f} batches/hour')
            print(f'   Estimated remaining time: {remaining_time:.1f} hours')
    
except Exception as e:
    print(f'❌ Error reading progress file: {e}')
    sys.exit(1)
"
    else
        echo "❌ No progress file found: ${output_base}_progress.pkl"
        echo "   This might be a fresh start or the process hasn't been run yet"
    fi
    
    # Check for error log
    if [ -f "${output_base}_errors.log" ]; then
        echo "⚠️  Error log found: ${output_base}_errors.log"
        local error_lines=$(wc -l < "${output_base}_errors.log" 2>/dev/null || echo "0")
        echo "   Log size: $error_lines lines"
        
        if [ "$error_lines" -gt 0 ]; then
            echo "   Recent errors:"
            tail -n 5 "${output_base}_errors.log" | grep -E "^(Batch|Error)" | head -n 3 | sed 's/^/      /'
        fi
    else
        echo "✅ No error log found (good sign!)"
    fi
}

# Function to clean up temporary files
clean_temp_files() {
    local output_base="$1"
    
    echo "🧹 Cleaning temporary files for: $output_base"
    echo "============================================="
    
    local files_to_clean=(
        "${output_base}_progress.pkl"
        "${output_base}_temp_results.json"
        "${output_base}_errors.log"
    )
    
    local cleaned_count=0
    
    for file in "${files_to_clean[@]}"; do
        if [ -f "$file" ]; then
            rm -f "$file"
            echo "   ✅ Removed: $file"
            ((cleaned_count++))
        else
            echo "   ❌ Not found: $file"
        fi
    done
    
    echo ""
    echo "📊 Cleanup Summary:"
    echo "   Files removed: $cleaned_count"
}

# Function to analyze errors and progress
analyze_progress() {
    local output_base="$1"
    
    echo "🔍 Analyzing progress and errors for: $output_base"
    echo "=============================================="
    echo ""
    
    # Analyze error log
    if [ -f "${output_base}_errors.log" ]; then
        echo "📋 Error Analysis:"
        
        # Count different types of errors
        echo "   Error types found:"
        grep -E "^Error:" "${output_base}_errors.log" | sort | uniq -c | sort -nr | head -5 | while read count error; do
            echo "      $count times: $error"
        done
        
        echo ""
        echo "   Recent errors:"
        tail -n 10 "${output_base}_errors.log" | grep -E "^(Batch|Error)" | tail -3 | sed 's/^/      /'
        
        # Check for patterns
        echo ""
        echo "   Common error patterns:"
        if grep -q "timeout" "${output_base}_errors.log"; then
            echo "      ⏰ Timeout errors detected - consider reducing batch size"
        fi
        if grep -q "JSON" "${output_base}_errors.log"; then
            echo "      🔧 JSON parsing errors detected - enhanced version should handle these"
        fi
        if grep -q "connection" "${output_base}_errors.log"; then
            echo "      🌐 Connection errors detected - check network/server status"
        fi
    else
        echo "✅ No error log found - good sign!"
    fi
}

# Function to resume processing
resume_processing() {
    local input_csv="$1"
    local output_base="$2"
    shift 2
    
    echo "🔄 Resuming processing..."
    echo "Input CSV: $input_csv"
    echo "Output base: $output_base"
    echo ""
    
    # Check if input file exists
    if [ ! -f "$input_csv" ]; then
        echo "❌ Error: Input CSV file not found: $input_csv"
        exit 1
    fi
    
    # Run the enhanced CLI tool
    echo "🚀 Starting enhanced CLI tool..."
    python3 company_cli.py --input "$input_csv" --output "$output_base" "$@"
}

# Main script logic
if [ $# -lt 2 ]; then
    show_usage
fi

command="$1"
shift

case "$command" in
    "status")
        show_status "$1"
        ;;
    "clean")
        clean_temp_files "$1"
        ;;
    "analyze")
        analyze_progress "$1"
        ;;
    "resume")
        resume_processing "$@"
        ;;
    *)
        echo "❌ Error: Unknown command: $command"
        show_usage
        ;;
esac
