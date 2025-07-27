# Enhanced Company Classification CLI Tool with Validation System

A **comprehensive** command-line interface for classifying companies using dual-engine search (Google + Perplexity) with the existing MCP (Model Context Protocol) server infrastructure. This tool leverages both Google Search and Perplexity AI for comprehensive company research and taxonomy matching, **now with validation and re-evaluation system to ensure 100% taxonomy compliance**.

## 🚀 Features

### **🔍 Taxonomy Validation and Re-evaluation System** ⭐ **NEW MAJOR FEATURE**
- **Automatic Validation**: Validates all classifications against the official 78-pair taxonomy
- **Smart Re-evaluation**: Re-processes only invalid or blank entries with enhanced prompts
- **Strict Compliance**: Ensures only exact taxonomy matches are used (no made-up pairs)
- **Batch Re-processing**: Re-evaluates flagged entries with configurable batch sizes
- **Comprehensive Reporting**: Detailed validation reports with statistics and examples
- **Safe Updates**: Creates backups before modifying any files

### **🔄 Dual-Engine Search Integration** 
- **Google Search**: Domain-specific searches with comprehensive web research
- **Perplexity AI**: AI-powered search with intelligent analysis and synthesis
- **Combined Research**: Uses both engines for maximum accuracy and coverage
- **Adaptive Queries**: Incorporates Industry and Product/Service Type from CSV data
- **Server Selection**: Choose Google only, Perplexity only, or both engines

### **🔧 Enhanced Data Persistence**
- **Atomic File Operations**: All file writes use temporary files + atomic moves to prevent corruption
- **Immediate Progress Saves**: Progress saved after **every single batch completion**
- **Comprehensive Error Handling**: Enhanced error recovery with specific handling for permissions, disk space, and corruption
- **Alternative Save Methods**: Emergency backup saves when primary save methods fail
- **100% Reliable Resume**: Resume functionality works perfectly with atomic progress tracking

### **🚨 Advanced API Error Handling**
- **Azure OpenAI Error Detection**: Specific handling for quota, rate limit, and billing errors
- **Perplexity API Error Management**: Budget tracking and quota management
- **Smart Error Recovery**: Different retry strategies for different error types
- **Cost Tracking**: Real-time API usage estimates and budget monitoring
- **Graceful Degradation**: Stops processing when API limits are reached to prevent waste

### **⚙️ Intelligent Server Configuration**
- **Flexible Server Selection**: Use Google, Perplexity, or both engines
- **Automatic Optimization**: Smart batch size recommendations per server type
- **Parameter Respect**: User-specified parameters are honored (no forced overrides)
- **Performance Tuning**: Server-specific timeout and retry configurations
- **Cost Optimization**: Budget-aware processing with usage estimates

### **🏗️ Robust Infrastructure**
- **Reuses Existing Infrastructure**: Leverages the same MCP servers and AI models used in the Streamlit application
- **CSV Processing**: Processes company data from CSV files with enhanced column structure support
- **Batch Processing**: Handles large CSV files by splitting them into manageable batches
- **Enhanced Research**: Uses Industry and Product/Service Type data from CSV for better search queries
- **Taxonomy Matching**: Matches companies to exact industry/product pairs from the established taxonomy
- **Automated Research**: Systematically researches each company using multiple data points
- **Markdown Output**: Generates results in markdown table format for easy viewing and processing

## 📋 Prerequisites

- Python 3.11+
- All dependencies from the main Streamlit application
- Environment variables configured (same as Streamlit app)
- Access to Google Search, Perplexity, and Company Tagging MCP servers

## 🛠️ Installation

1. **Run the setup script**:
   ```bash
   chmod +x setup_cli.sh
   ./setup_cli.sh
   ```

2. **Configure environment variables**:
   Edit the `.env` file with your API keys:
   ```env
   # AI Provider (choose one)
   OPENAI_API_KEY=your_openai_api_key_here
   # OR
   AZURE_API_KEY=your_azure_api_key
   AZURE_ENDPOINT=https://your-endpoint.openai.azure.com/
   AZURE_DEPLOYMENT=your_deployment_name
   AZURE_API_VERSION=2023-12-01-preview
   
   # Google Search (required for google server)
   GOOGLE_API_KEY=your_google_api_key
   GOOGLE_SEARCH_ENGINE_ID=your_custom_search_engine_id
   
   # Perplexity (required for perplexity server)
   PERPLEXITY_API_KEY=your_perplexity_api_key
   PERPLEXITY_MODEL=sonar
   ```

3. **Start the MCP servers**:
   ```bash
   # In the main project directory
   docker-compose up mcpserver1 mcpserver2 -d
   ```

## 📊 Enhanced CSV Input Format

Your CSV file must contain these columns:

| Column | Description | Required | Enhancement |
|--------|-------------|----------|-------------|
| `CASEACCID` | Case/Account ID | Optional | Used for tracking |
| `Account Name` | Company name | **Required** | Primary research identifier |
| `Trading Name` | Trading name | Optional | **Used in search queries** ⭐ |
| `Domain` | Company domain | Optional | **Used for site-specific searches** ⭐ |
| `Industry` | Industry classification | Optional | **Integrated into search queries** ⭐ |
| `Product/Service Type` | Product/service type | Optional | **Enhanced search targeting** ⭐ |
| `Event` | Trade show events | Optional | Context for taxonomy matching |

### Example Enhanced CSV Structure:
```csv
CASEACCID,Account Name,Trading Name,Domain,Industry,Product/Service Type,Event
CASE001,Microsoft Corporation,Microsoft,microsoft.com,Cloud Infrastructure,Cloud Computing Services,"Cloud and AI Infrastructure"
CASE002,Amazon Web Services,AWS,aws.amazon.com,Cloud Infrastructure,Web Services,"Cloud and AI Infrastructure"
CASE003,Google LLC,Google,google.com,Technology,Search and Advertising,"Big Data and AI World"
```

## 🎯 Enhanced Usage

### **Classification with Server Selection**

```bash
# Use Google Search only
python3 company_cli.py --input companies.csv --output results --servers google

# Use Perplexity AI only (with enhanced search)
python3 company_cli.py --input companies.csv --output results --servers perplexity

# Use both engines for maximum accuracy (RECOMMENDED)
python3 company_cli.py --input companies.csv --output results --servers both

# Specify batch size (respects your choice)
python3 company_cli.py --input companies.csv --output results --batch-size 5 --servers perplexity
```

### **🔍 Validation and Re-evaluation Workflow** ⭐ **NEW**

```bash
# Step 1: Run initial classification
python3 company_cli.py --input All_Tech_0_8000.csv --output output_0_8000_p --batch-size 1 --servers perplexity

# Step 2: Validate and re-evaluate invalid entries (using Perplexity by default)
python3 company_validator_cli.py --input All_Tech_0_8000.csv --output output_0_8000_p.csv

# Or explicitly specify the server for re-evaluation
python3 company_validator_cli.py --input All_Tech_0_8000.csv --output output_0_8000_p.csv --servers perplexity

# Optional: Generate validation report only (no re-evaluation)
python3 company_validator_cli.py --input All_Tech_0_8000.csv --output output_0_8000_p.csv --report-only

# Optional: Export valid taxonomy pairs for reference
python3 company_validator_cli.py --input All_Tech_0_8000.csv --output output_0_8000_p.csv --export-taxonomy valid_pairs.md
```

### **Validation Options** ⭐ **NEW**

```bash
# Choose different servers for re-evaluation
python3 company_validator_cli.py --input input.csv --output output.csv --servers google
python3 company_validator_cli.py --input input.csv --output output.csv --servers perplexity
python3 company_validator_cli.py --input input.csv --output output.csv --servers both

# Custom batch size for re-evaluation (default: 1)
python3 company_validator_cli.py --input input.csv --output output.csv --batch-size 2

# Specify custom taxonomy file
python3 company_validator_cli.py --input input.csv --output output.csv --taxonomy path/to/taxonomy.csv
```

### **Budget-Aware Processing**

```bash
# Start large dataset processing with cost tracking
python3 company_cli.py --input large_23500.csv --output output_23500 --batch-size 10 --servers perplexity

# System shows cost estimates:
# 💰 API Usage Estimates:
#    Perplexity calls completed: ~15000
#    Perplexity calls remaining: ~8500  
#    Estimated cost completed: ~$75.00
#    Estimated cost remaining: ~$42.50

# Resume when budget allows
python3 company_cli.py --input large_23500.csv --output output_23500 --servers perplexity --resume
```

### **Enhanced Error Recovery**

```bash
# Resume after API quota errors
python3 company_cli.py --input companies.csv --output results --servers perplexity --resume

# Force continue with existing progress (skip batch size warnings)
python3 company_cli.py --input companies.csv --output results --force-continue

# Clean start (ignore previous progress)
python3 company_cli.py --input companies.csv --output results --clean-start

# With verbose API error reporting
python3 company_cli.py --input companies.csv --output results --verbose
```

## 🔍 Validation System Details ⭐ **NEW**

### **How Validation Works**

1. **Taxonomy Loading**: Loads the official 78 industry/product pairs from `client/mcp_servers/company_tagging/categories/classes.csv`

2. **Row Validation**: Each row is checked for:
   - **Invalid Pairs**: Industry/Product combinations not in the official taxonomy
   - **Blank Rows**: All tech fields are empty
   - **Incomplete Pairs**: Only industry or only product specified

3. **Flagging**: Rows are flagged for re-evaluation if they have:
   - Invalid taxonomy pairs (e.g., "APAC PRO | eCommerce Solutions")
   - All blank tech fields
   - Incomplete pairs

4. **Re-evaluation Process**:
   - Creates backup of original file
   - Re-processes only flagged rows
   - Uses enhanced prompts with strict taxonomy enforcement
   - Updates only rows with valid matches
   - Leaves fields blank if no valid match found

5. **Final Update**:
   - Merges re-evaluated results back into original file
   - Preserves all valid rows unchanged
   - Generates comprehensive report

### **Example Validation Report**

```markdown
# Taxonomy Validation Report
Generated on: 2025-07-25 13:02:15

## Summary
- Total rows: 7950
- Valid rows: 6721
- Invalid rows: 126
- Blank rows: 1103
- Rows flagged for re-evaluation: 1229

## Invalid Details

### Row 188: SICK (Thailand) Co., Ltd.
- Reason: Pair 2: Not in taxonomy (Industry='IT Infrastructure & Hardware', Product='Sensor Technologies')

### Row 281: CJ Logistics Asia Pte Ltd
- Reason: Pair 4: Not in taxonomy (Industry='IT Infrastructure & Hardware', Product='Handling Equipment')
```

## 🔄 Enhanced Recovery and Progress Management

### **Recovery Script**
```bash
# Show current progress status with cost estimates
./recovery.sh status output_base

# Resume processing with smart server selection
./recovery.sh resume input.csv output_base --servers perplexity

# Clean up temporary files
./recovery.sh clean output_base

# Analyze errors with API-specific recommendations
./recovery.sh analyze output_base
```

### **Validation Utilities** ⭐ **NEW**
```bash
# Simple validation check
python3 validate_simple.py output.csv client/mcp_servers/company_tagging/categories/classes.csv

# Find all CSV files that might be taxonomy files
python3 find_taxonomy_files.py

# Debug CSV structure
python3 debug_csv_test.py output.csv

# Check taxonomy contents
python3 check_taxonomy.py
```

## 📤 Enhanced Output Format

The tool generates comprehensive markdown tables with validated categorization:

```markdown
| Company Name | Trading Name | Tech Industry 1 | Tech Product 1 | Tech Industry 2 | Tech Product 2 | Tech Industry 3 | Tech Product 3 | Tech Industry 4 | Tech Product 4 |
|--------------|--------------|-----------------|----------------|-----------------|----------------|-----------------|----------------|-----------------|----------------|
| Microsoft Corporation | Microsoft | Cloud and AI Infrastructure Services | Hyperscale Cloud Solutions | Platforms & Software | AI Applications | | | | |
| Amazon Web Services | AWS | Cloud and AI Infrastructure Services | Cloud Service Provider | Cloud and AI Infrastructure Services | Compute-as-a-service | | | | |
```

**Note**: All entries are validated against the official taxonomy - no made-up pairs!

## 🔧 Enhanced Processing Flow

The CLI tool follows an enhanced systematic process with validation:

1. **Initial Classification**:
   - Uses dual-engine search (Google/Perplexity/Both)
   - Applies taxonomy matching with LLM
   - Generates initial results

2. **Validation Phase** ⭐ **NEW**:
   - Loads official taxonomy (78 pairs)
   - Validates each row against exact matches
   - Identifies invalid pairs and blank entries
   - Generates validation report

3. **Re-evaluation Phase** ⭐ **NEW**:
   - Creates backup of original file
   - Re-processes only flagged rows
   - Uses enhanced prompts with strict enforcement
   - Validates results before updating

4. **Final Output**:
   - Merges validated results
   - All entries comply with official taxonomy
   - Comprehensive report generated

## 📁 Enhanced File Structure

```
project_root/
├── company_cli.py              # Main CLI with dual-engine support
├── company_validator_cli.py    # ⭐ NEW: Validation and re-evaluation tool
├── taxonomy_validator.py       # ⭐ NEW: Taxonomy validation module
├── validate_simple.py          # ⭐ NEW: Simple validation script
├── find_taxonomy_files.py      # ⭐ NEW: Taxonomy file finder
├── check_taxonomy.py           # ⭐ NEW: Taxonomy checker
├── debug_csv_test.py           # ⭐ NEW: CSV debug tool
├── csv_processor_utility.py    # Enhanced CSV processing utilities
├── setup_cli.sh               # Setup script
├── batch_process.sh           # Enhanced batch script with server selection
├── recovery.sh                # Recovery and progress management
├── sample_companies.csv       # Sample data
├── cli_servers_config.json    # Enhanced MCP server configuration
├── client/                    # Existing Streamlit app
│   ├── services/             # Enhanced service modules
│   ├── utils/                # Enhanced utility modules
│   ├── mcp_servers/          # Enhanced MCP server definitions
│   │   └── company_tagging/  # Embedded company tagging server
│   │       └── categories/   
│   │           └── classes.csv  # Official taxonomy (78 pairs)
│   └── config.py             # Enhanced configuration
└── .env                      # Environment variables
```

## 🔍 Enhanced MCP Tools

### **Google Search Tools** (When `--servers google` or `--servers both`)
- `google-search`: Web search with domain-specific optimization
- `read-webpage`: Clean webpage content extraction

### **Perplexity AI Tools** (When `--servers perplexity` or `--servers both`)
- `perplexity_advanced_search`: Enhanced search with recency filtering
- AI-powered analysis with intelligent response synthesis

### **Company Tagging Tools** (Always Available)
- `search_show_categories`: Access to official taxonomy (78 industry/product pairs)
- Complete taxonomy validation and matching
- Embedded stdio MCP server for specialized workflows

## 💰 Budget Management Features

### **Cost Tracking**
```bash
# Real-time cost estimates during processing
💰 API Usage Estimates:
   Perplexity calls completed: ~12000
   Perplexity calls remaining: ~11500
   Estimated cost completed: ~$60.00
   Estimated cost remaining: ~$57.50
```

### **Budget Planning**
- **Perplexity API**: ~$0.005 per search call
- **Google API**: Based on your custom search engine quota
- **Re-evaluation costs**: Same as initial processing per company

## 🛡️ Enhanced Data Reliability Features

### **Validation System** ⭐ **NEW**
- **Exact Match Enforcement**: Only taxonomy pairs that exist exactly are allowed
- **No Creative Interpretations**: Prevents "APAC PRO", "WIP", etc.
- **Safe Fallback**: Leaves fields blank rather than using invalid values
- **Backup Before Changes**: Always creates timestamped backup

### **Atomic File Operations**
- Write to temporary file first
- Atomic move prevents corruption
- Cleanup on failure
- Alternative save methods

## 🐛 Enhanced Troubleshooting

### **Validation-Specific Issues** ⭐ **NEW**

1. **"Taxonomy file not found"**:
   ```bash
   # Ensure the taxonomy file exists
   ls -la client/mcp_servers/company_tagging/categories/classes.csv
   
   # Use explicit path
   python3 company_validator_cli.py --taxonomy client/mcp_servers/company_tagging/categories/classes.csv
   ```

2. **"No original data found for flagged rows"**:
   ```bash
   # Ensure input CSV matches the one used for initial classification
   # Company names must match between files
   ```

3. **"Invalid pairs still appearing after re-evaluation"**:
   ```bash
   # Check if the pair actually exists in taxonomy
   python3 check_taxonomy.py
   
   # Export valid pairs for reference
   python3 company_validator_cli.py --export-taxonomy valid_pairs.md
   ```

### **API-Specific Issues**

1. **Quota/Budget Issues**:
   ```bash
   # Check current usage
   ./recovery.sh status output_base
   
   # Resume with different server if one is exhausted
   python3 company_validator_cli.py --servers google  # If Perplexity exhausted
   ```

2. **Timeout Errors**:
   ```bash
   # Reduce batch size for re-evaluation
   python3 company_validator_cli.py --batch-size 1  # Default, most reliable
   ```

## 📊 Enhanced Performance Considerations

### **Validation Performance** ⭐ **NEW**
- **Validation Speed**: ~10,000 rows per second
- **Re-evaluation Speed**: 30-60 seconds per company
- **Memory Usage**: Minimal (loads taxonomy once)
- **Disk Usage**: Creates one backup file

### **Server-Specific Performance**
- **Google Search**: Fast, handles larger batch sizes (up to 10)
- **Perplexity AI**: Slower but more intelligent, works best with smaller batches (1-2)
- **Both Engines**: Most comprehensive but higher cost

### **Time Estimates**
- **Initial Classification**: 50-100 companies/hour
- **Validation**: Instant (< 1 second for 8000 rows)
- **Re-evaluation**: 30-60 companies/hour

## 🔒 Enhanced Security

- Environment variables loaded securely with validation
- API keys not logged or exposed
- Atomic file operations prevent corruption
- Backup before any modifications
- Validation ensures data integrity

## 📈 Enhanced Monitoring

### **Validation Metrics** ⭐ **NEW**
```bash
📊 Validation Summary:
   Total rows: 7950
   Valid rows: 6721 (84.5%)
   Invalid rows: 126
   Blank rows: 1103
   Rows to re-evaluate: 1229
```

### **Re-evaluation Progress** ⭐ **NEW**
```bash
🔄 Step 3: Re-evaluating 1229 flagged rows...
   Using batch size: 1
🔍 Re-evaluating batch 1/1229 (1 companies)
   - Row 25: Epost Express
   ✅ Row 25 successfully re-evaluated with valid taxonomy pairs
```

## 🤝 Contributing

To extend the Enhanced CLI tool:

1. **Add new validation rules** in `taxonomy_validator.py`
2. **Enhance re-evaluation prompts** in `company_validator_cli.py`
3. **Add new server integrations** in `client/services/mcp_service.py`
4. **Extend error handling** in the main CLI error handling blocks
5. **Update taxonomy** in `client/mcp_servers/company_tagging/categories/classes.csv`

## 📄 License

This CLI tool inherits the same license as the main Streamlit application.

---

**Version**: 4.0.0 ⭐ **VALIDATION SYSTEM UPDATE**  
**Compatibility**: Python 3.11+, Enhanced MCP servers (Google + Perplexity)  
**Dependencies**: Enhanced dependencies with validation support  
**Architecture**: Dual-engine search with validation and re-evaluation system  
**MAJOR ENHANCEMENT**: **Complete validation and re-evaluation system for 100% taxonomy compliance**  
**NEW FEATURES**: **Automatic validation, smart re-evaluation, strict taxonomy enforcement, comprehensive reporting**  
**Data Quality**: **Ensures all classifications use only official taxonomy pairs - no made-up values!** ⭐