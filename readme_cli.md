# Company Classification CLI Tool

## 🚀 **Enhanced Integrated Workflow**

A powerful command-line interface for automated company classification with an integrated **Perplexity → Azure OpenAI → Taxonomy Validation → Country Inference** workflow. All processing happens in a single streamlined process with comprehensive validation and real-time progress tracking.

## 🌟 **Key Features**

### **🔄 Integrated Single-Step Workflow**
- **Perplexity Advanced Search**: Comprehensive company research with recency="year"
- **Azure OpenAI Classification**: Intelligent classification into up to 4 industry/product pairs
- **Real-time Taxonomy Validation**: Validates against official classes.csv taxonomy
- **Automatic Country Inference**: Extracts country information during research
- **Literal Field Preservation**: Exact copying of critical fields (CASEACCID, Company Name, Trading Name)

### **📊 Advanced Output Format**
- **12-Column Structure**: Comprehensive classification with up to 4 pairs + country
- **Exact Column Order**: `CASEACCID, Company Name, Trading Name, Tech Industry 1, Tech Product 1, Tech Industry 2, Tech Product 2, Tech Industry 3, Tech Product 3, Tech Industry 4, Tech Product 4, Country`
- **Conservative Classification**: Leaves fields blank rather than using invalid taxonomy pairs
- **100% Taxonomy Compliance**: Only uses verified industry/product combinations

### **🛡️ Robust Processing**
- **Atomic Progress Tracking**: Never lose progress with immediate saves after each batch
- **Automatic Resume**: Continues from last successful batch on restart
- **Error Recovery**: Comprehensive error handling with retry logic
- **BOM Handling**: Automatically handles UTF-8 BOM in CSV files
- **Progress Persistence**: Multiple backup mechanisms ensure data integrity

## 📋 **Prerequisites**

- Python 3.11+
- Azure OpenAI or OpenAI API access
- Perplexity API access
- Docker (for MCP servers)
- All dependencies from the main Streamlit application

## 🛠️ **Installation**

1. **Run the setup script**:
   ```bash
   chmod +x setup_cli.sh
   ./setup_cli.sh
   ```

2. **Configure environment variables**:
   Edit the `.env` file with your API keys:
   ```env
   # Azure OpenAI (recommended)
   AZURE_API_KEY=your_azure_api_key
   AZURE_ENDPOINT=https://your-endpoint.openai.azure.com/
   AZURE_DEPLOYMENT=your_deployment_name
   AZURE_API_VERSION=2023-12-01-preview
   
   # OR OpenAI (alternative)
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Perplexity (required)
   PERPLEXITY_API_KEY=your_perplexity_api_key
   PERPLEXITY_MODEL=sonar
   ```

3. **Start the MCP servers**:
   ```bash
   # In the main project directory
   docker-compose up mcpserver1 -d  # Perplexity server
   ```

## 📊 **CSV Input Format**

Your CSV file must contain these columns:

| Column | Description | Required | Usage |
|--------|-------------|----------|-------|
| `CASEACCID` | Salesforce Case/Account ID | **Required** | Literal copy to output |
| `Account Name` | Company name | **Required** | Copied as "Company Name" in output |
| `Trading Name` | Trading/brand name | **Required** | Literal copy to output |
| `Domain` | Company domain | Optional | Used in Perplexity searches |
| `Industry` | Industry classification | Optional | Enhanced search targeting |
| `Product/Service Type` | Product/service type | Optional | Enhanced search targeting |
| `Event` | Trade show events | Optional | Context for classification |

### Example Input CSV:
```csv
CASEACCID,Account Name,Trading Name,Domain,Industry,Product/Service Type,Event
00158000005d8uJAAQ,Microsoft Corporation,Microsoft,microsoft.com,Cloud Infrastructure,Cloud Computing Services,Cloud and AI Infrastructure
00158000005d9GxAAI,Amazon Web Services,AWS,aws.amazon.com,Cloud Infrastructure,Web Services,Cloud and AI Infrastructure
00158000005d9eQAAQ,Google LLC,Google,google.com,Technology,Search and Advertising,Big Data and AI World
```

## 🎯 **Usage**

### **Basic Usage (Recommended)**
```bash
# Standard integrated workflow processing
python company_cli.py --input companies.csv --output results_perplexity --servers perplexity
```

### **Advanced Usage Options**
```bash
# Custom batch size (default: 1 for maximum accuracy)
python company_cli.py --input companies.csv --output results --batch-size 1

# Start fresh (ignore previous progress)
python company_cli.py --input companies.csv --output results --clean-start

# Verbose output for debugging
python company_cli.py --input companies.csv --output results --verbose

# Custom MCP server configuration
python company_cli.py --input companies.csv --output results --config custom_config.json
```

## 📈 **Integrated Workflow Details**

### **Step 1: Perplexity Advanced Search**
- Uses `perplexity_advanced_search` with `recency="year"`
- Search query: `"[Company Name] [Industry] [Product/Service Type] company products services technology solutions country headquarters location"`
- Focuses on: what they sell/offer, technology solutions, operating country

### **Step 2: Azure OpenAI Classification**
- Sends Perplexity results + company data to Azure OpenAI
- Requests up to 4 relevant (Industry | Product) pairs
- Extracts country information from research results
- Uses structured prompts for consistent output

### **Step 3: Taxonomy Validation**
- Validates each classification pair against official classes.csv
- Uses company_tagging MCP server for taxonomy access
- Discards invalid pairs (conservative approach)
- Only retains exact taxonomy matches

### **Step 4: Final Output Generation**
- Creates 12-column output with literal field copying
- Preserves CASEACCID, Company Name (from Account Name), Trading Name exactly
- Fills up to 4 validated classification pairs
- Includes inferred country information

## 📊 **Output Format**

### **CSV Output (12 Columns)**
```csv
CASEACCID,Company Name,Trading Name,Tech Industry 1,Tech Product 1,Tech Industry 2,Tech Product 2,Tech Industry 3,Tech Product 3,Tech Industry 4,Tech Product 4,Country
00158000005d8uJAAQ,Microsoft Corporation,Microsoft,Cloud Infrastructure,Cloud Computing Services,Software,Enterprise Software,,,,United States
00158000005d9GxAAI,Amazon Web Services,AWS,Cloud Infrastructure,Web Services,Data Management,Big Data Analytics,,,,United States
```

### **Additional Output Files**
- **Markdown Report** (`results.md`): Human-readable classification results
- **Statistics** (`results.stats.json`): Processing statistics and performance metrics
- **Progress Files**: Automatic progress tracking for resume functionality

## 🔧 **Performance & Optimization**

### **Batch Processing**
- **Default Batch Size**: 1 (recommended for accuracy)
- **Perplexity Optimization**: Batch size 1 works best with Perplexity API
- **Progress Tracking**: Saves after every batch completion
- **Memory Efficient**: Processes large files without memory issues

### **Error Handling**
- **Automatic Retry**: Up to 3 attempts per batch with exponential backoff
- **API Error Detection**: Identifies quota, rate limit, and billing issues
- **Graceful Degradation**: Continues processing valid batches despite individual failures
- **Comprehensive Logging**: Detailed error logs for troubleshooting

### **Resume Functionality**
- **Automatic Resume**: Continues from last successful batch (default behavior)
- **Clean Start Option**: `--clean-start` to ignore previous progress
- **Progress Preservation**: Multiple backup mechanisms prevent data loss

## 📊 **Expected Performance**

### **Processing Speed**
- **Single Company**: 30-60 seconds (comprehensive research + classification)
- **Batch of 30**: 15-30 minutes (depending on API response times)
- **Large Datasets**: Scales linearly with automatic progress tracking

### **Accuracy Metrics**
- **Taxonomy Compliance**: 100% (only uses verified pairs from classes.csv)
- **Country Inference**: ~85-90% success rate
- **Classification Success**: ~80-90% of companies receive valid classifications
- **Data Integrity**: 100% preservation of critical fields

## 🐛 **Troubleshooting**

### **Common Issues**

#### **CSV Reading Issues**
```bash
# BOM (Byte Order Mark) handling
✅ Automatically handled with utf-8-sig encoding

# Column name mismatches
✅ Automatic BOM stripping and whitespace cleanup

# Missing CASEACCID values
📋 Check that input CSV has CASEACCID column with actual values
```

#### **API Issues**
```bash
# Azure OpenAI quota errors
💰 Check your Azure subscription and increase quotas
🔧 Monitor usage in Azure portal

# Perplexity rate limits
⏰ Default batch size 1 respects rate limits
🔄 Automatic retry with exponential backoff

# Network timeouts
🌐 15-minute timeout per batch for comprehensive processing
🔄 Automatic resume on restart
```

#### **Processing Issues**
```bash
# Resume from interruption
python company_cli.py --input companies.csv --output results  # Automatically resumes

# Start completely fresh
python company_cli.py --input companies.csv --output results --clean-start

# Debug processing issues
python company_cli.py --input companies.csv --output results --verbose
```

### **Performance Optimization**

#### **For Large Datasets**
- Use batch size 1 for maximum accuracy
- Monitor API quotas and rate limits
- Consider processing during off-peak hours
- Use resume functionality for long-running jobs

#### **For Better Accuracy**
- Ensure input CSV has Domain, Industry, and Product/Service Type filled
- Use descriptive company and trading names
- Verify taxonomy file is up to date
- Review failed batches in error logs

## 📈 **Monitoring & Analytics**

### **Real-time Progress**
```bash
📊 Progress: 25/30 batches (83.3% success rate, 25 companies)
   ✅ Step 1: Perplexity advanced search for company research
   ✅ Step 2: Azure OpenAI classification (up to 4 pairs + country)
   ✅ Step 3: Taxonomy validation against classes.csv
   ✅ Step 4: Final output with literal field copying
```

### **Final Statistics**
```bash
📊 Final Summary:
   Total companies: 30
   Completed batches: 30/30
   Success rate: 100.0%
   Processing time: 25.3 minutes
   Output format: 12 columns (CASEACCID + Company Name + Trading Name + 4 classification pairs + Country)
```

### **Performance Files**
- **Progress Tracking**: `results_progress.pkl` (resume state)
- **Error Logs**: `results_errors.log` (detailed error information)
- **Statistics**: `results.stats.json` (processing metrics)

## 🔒 **Security & Best Practices**

### **API Key Management**
- Store API keys in `.env` file (never commit to git)
- Use Azure Key Vault or similar for production environments
- Rotate API keys regularly
- Monitor API usage and costs

### **Data Handling**
- Input CSV files processed locally
- API calls only send necessary company information
- No sensitive data stored in logs
- Results saved locally with proper file permissions

### **Error Recovery**
- Progress files enable safe resume after interruption
- Atomic file operations prevent data corruption
- Multiple backup mechanisms ensure data integrity
- Comprehensive error logging for audit trails

## 🤝 **Contributing**

To extend the Enhanced CLI tool:

1. **Add new classification logic** in the `create_integrated_prompt()` method
2. **Enhance taxonomy validation** in the taxonomy validation step
3. **Add new API integrations** by extending the MCP server configuration
4. **Improve error handling** in the batch processing retry logic
5. **Update output formats** by modifying the result generation methods

## 📄 **License**

This CLI tool inherits the same license as the main Streamlit application.

---

**Version**: 5.0.0 ⭐ **INTEGRATED WORKFLOW RELEASE**  
**Compatibility**: Python 3.11+, Azure OpenAI, Perplexity API  
**Dependencies**: Enhanced MCP integration with Perplexity advanced search  
**Architecture**: Single integrated workflow: Perplexity → Azure OpenAI → Taxonomy → Country  
**MAJOR ENHANCEMENT**: **Complete integrated processing workflow with real-time validation**  
**NEW FEATURES**: **Automated research, intelligent classification, taxonomy enforcement, country inference**  
**Data Quality**: **100% taxonomy compliance with exact field preservation and comprehensive progress tracking** ⭐