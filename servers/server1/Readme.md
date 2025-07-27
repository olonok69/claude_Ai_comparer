# Perplexity MCP Server with Company Tagging & CSV Categories

A comprehensive Model Context Protocol (MCP) server that provides AI-powered web search capabilities using the Perplexity API, plus specialized company tagging functionality for trade show exhibitor categorization. This version uses Server-Sent Events (SSE) for real-time communication, making it compatible with web-based MCP clients and browsers.

## 🚀 Features

### **AI-Powered Web Search**
- **Intelligent Search**: Leverage Perplexity's AI-powered search across the web
- **Recency Filtering**: Filter results by time period (day, week, month, year)
- **Multiple Models**: Support for all Perplexity AI models
- **Citation Support**: Automatic source citations for all results
- **Advanced Parameters**: Fine-tune search with custom parameters

### **Company Tagging & Categorization** ⭐ NEW
- **Automated Company Research**: Research companies using web sources and LinkedIn
- **Taxonomy Matching**: Match companies to precise industry/product categories
- **Trade Show Context**: Focus on relevant shows (CAI, DOL, CCSE, BDAIW, DCW)
- **Structured Output**: Generate tables with up to 4 industry/product pairs per company
- **Data Analyst Workflow**: Professional data analysis approach with accuracy focus

### **CSV Categories Data Access**
- **Show Categories**: Access categorized data organized by shows
- **Industry Organization**: Browse categories by industry and product classifications
- **Search Functionality**: Search across all category data with flexible filtering
- **Dynamic Resources**: Real-time access to CSV data through MCP resources
- **Tagging Context**: Formatted categories specifically for company tagging

### **Technical Features**
- **SSE Protocol**: Real-time communication using Server-Sent Events
- **Health Monitoring**: Built-in health check endpoints with comprehensive status
- **Async Operations**: High-performance async/await architecture
- **Error Handling**: Comprehensive error management and logging
- **Environment Configuration**: Flexible configuration via environment variables

## 📋 Prerequisites

- Python 3.11+
- Perplexity API key (get one at [perplexity.ai](https://perplexity.ai))
- Docker (optional, for containerized deployment)
- CSV file with categories data (included: `src/perplexity_mcp/categories/classes.csv`)

## 🛠️ Installation & Setup

### 1. Environment Configuration

Create a `.env` file in the server directory:

```env
# Perplexity API Configuration
PERPLEXITY_API_KEY=your_perplexity_api_key_here
PERPLEXITY_MODEL=sonar
```

### 2. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python perplexity_sse_server.py
```

### 3. Docker Deployment

```bash
# Build the Docker image
docker build -t perplexity-mcp-sse .

# Run the container
docker run -p 8001:8001 \
  -e PERPLEXITY_API_KEY=your_api_key_here \
  -e PERPLEXITY_MODEL=sonar \
  perplexity-mcp-sse
```

## 🔧 Available Tools (4 Tools)

### **Web Search Tools**

#### **perplexity_search_web**
Standard web search with recency filtering.

**Parameters:**
- `query` (string, required): The search query
- `recency` (string, optional): Time filter - "day", "week", "month", "year" (default: "month")

#### **perplexity_advanced_search**
Advanced search with custom parameters for fine-tuned control.

**Parameters:**
- `query` (string, required): The search query
- `recency` (string, optional): Time filter (default: "month")
- `model` (string, optional): Override the default model
- `max_tokens` (int, optional): Maximum response tokens (default: 512, max: 2048)
- `temperature` (float, optional): Response randomness 0.0-1.0 (default: 0.2)

### **Category Management Tools**

#### **search_show_categories**
Search and filter show categories from the CSV data.

**Parameters:**
- `show_name` (string, optional): Filter by specific show (CAI, DOL, CCSE, BDAIW, DCW)
- `industry_filter` (string, optional): Filter by industry name (partial match)
- `product_filter` (string, optional): Filter by product name (partial match)

### **Company Tagging Tool** ⭐ NEW

#### **tag_company**
Advanced company research and taxonomy tagging for trade show exhibitors.

**Parameters:**
- `company_name` (string, required): The main company name
- `trading_name` (string, optional): Alternative trading name
- `target_shows` (string, optional): Comma-separated show codes (e.g., "CAI,DOL,BDAIW")
- `company_description` (string, optional): Brief description of the company

**Example:**
```python
await tag_company(
    company_name="Microsoft Corporation",
    trading_name="Microsoft",
    target_shows="CAI,BDAIW",
    company_description="Technology company specializing in cloud and AI solutions"
)
```

**Output Format:**
The tool generates a comprehensive analysis including:
- Research findings from web sources and LinkedIn
- Taxonomy analysis and matching
- Structured table with up to 4 industry/product pairs
- Complete audit trail of the tagging process

## 📋 Available Prompts (1 Prompt)

### **company_tagging_analyst** ⭐ NEW
Professional data analyst prompt for company categorization.

**Parameters:**
- `company_name` (string): The main company name
- `trading_name` (string): Alternative trading name (optional)
- `target_shows` (string): Shows the company is interested in
- `company_description` (string): Brief description (optional)

**Purpose:**
This prompt provides the AI with a complete data analyst persona and workflow for accurately tagging companies with industry and product categories from the taxonomy.

**Key Features:**
- Professional data analyst role definition
- Complete taxonomy context
- Step-by-step analysis process
- Accuracy and consistency focus
- Structured output format requirements

## 🗂️ Available Resources (7 Resources)

### **Basic Category Resources**

#### **categories://all**
Complete CSV data with all show categories in JSON format.

#### **categories://shows**
Categories organized by show with statistics and industry breakdowns.

#### **categories://shows/{show_name}**
Categories for a specific show (CAI, DOL, CCSE, BDAIW, DCW).

#### **categories://industries**
Categories organized by industry with product associations.

#### **categories://industries/{industry_name}**
Categories for a specific industry (case-insensitive, partial match).

#### **categories://search/{query}**
Search across all category data with flexible query matching.

### **Company Tagging Resource** ⭐ NEW

#### **categories://for-tagging**
Categories formatted specifically for company tagging analysis.

**Special Features:**
- Formatted for prompt consumption
- Organized by show with full names
- Exact spelling and formatting preservation
- Usage instructions included

**Output Format:**
```
TAXONOMY CATEGORIES - Industry and Product Pairs by Show:

**CAI (Cloud and AI Infrastructure):**
- IT Infrastructure & Hardware | Semiconductor Technologies
- Cloud and AI Infrastructure Services | Hyperscale Cloud Solutions
...

**DOL (DevOps Live):**
- Application Delivery & Runtime | Application Delivery
- CI/CD & Automation | CI/CD Pipelines
...

NOTE: Use industry and product pairs EXACTLY as shown above.
```

## 🎯 Company Tagging Workflow

### **Complete Analysis Process**

The `tag_company` tool follows a comprehensive research and analysis workflow:

1. **Input Validation**
   - Validates company name (required)
   - Normalizes trading name and show targets
   - Determines research name priority (Trading Name > Company Name)

2. **Web Research Phase**
   - Initial company research using Perplexity AI
   - Additional context from LinkedIn and company websites
   - Focus on products/services relevant to target shows

3. **Taxonomy Matching Phase**
   - Analysis of research findings
   - Matching to exact taxonomy categories
   - Selection of up to 4 most relevant industry/product pairs

4. **Structured Output Generation**
   - Professional analysis summary
   - Research audit trail
   - Formatted table with categorization results

### **Usage Examples**

#### **Basic Company Tagging**
```python
# Tag a technology company
result = await tag_company(
    company_name="NVIDIA Corporation",
    target_shows="CAI,BDAIW"
)
```

#### **Company with Trading Name**
```python
# Use trading name for research
result = await tag_company(
    company_name="International Business Machines Corporation",
    trading_name="IBM",
    target_shows="CAI,DOL,BDAIW",
    company_description="Enterprise technology and consulting company"
)
```

#### **Multi-Show Analysis**
```python
# Analyze for multiple relevant shows
result = await tag_company(
    company_name="Amazon Web Services",
    trading_name="AWS",
    target_shows="CAI,DOL,CCSE,DCW"
)
```

### **Expected Output Structure**

The tool generates a comprehensive report including:

```
COMPANY TAGGING ANALYSIS FOR: [Company Name]
============================================================

RESEARCH NAME USED: [Name used for web research]
TARGET SHOWS: [Show codes provided]

INITIAL RESEARCH:
[Perplexity search results about company products/services]

ADDITIONAL RESEARCH:
[LinkedIn and website specific research findings]

TAXONOMY ANALYSIS AND TAGGING:
[Analysis explaining categorization decisions]

TAXONOMY MATCHES:
| Tech Industry 1 | Tech Product 1 | Tech Industry 2 | Tech Product 2 | ... |
|-----------------|----------------|-----------------|----------------|-----|
| [Exact Industry] | [Exact Product] | [Exact Industry] | [Exact Product] | ... |

TABLE FORMAT:
| Company Name | Trading Name | Tech Industry 1 | Tech Product 1 | ... |
|--------------|--------------|-----------------|----------------|-----|
| [Company]    | [Trading]    | [Industry]      | [Product]      | ... |

CATEGORIES REFERENCE:
[Complete taxonomy for verification]
```

## 🎯 Trade Show Context

### **Supported Shows**
- **CAI**: Cloud and AI Infrastructure
- **DOL**: DevOps Live
- **CCSE**: Cloud and Cyber Security Expo
- **BDAIW**: Big Data and AI World
- **DCW**: Data Centre World

### **Show-Specific Categories**

Each show has specific industry and product categories relevant to its theme:

**CAI (Cloud and AI Infrastructure):**
- Focus: Cloud infrastructure, AI platforms, semiconductor technologies
- Key Industries: IT Infrastructure & Hardware, Cloud and AI Infrastructure Services
- Example Products: Hyperscale Cloud Solutions, AI Applications

**DOL (DevOps Live):**
- Focus: Development operations, CI/CD, automation
- Key Industries: Application Delivery & Runtime, CI/CD & Automation
- Example Products: DevOps Tools, Configuration Management

**CCSE (Cloud and Cyber Security Expo):**
- Focus: Security solutions, compliance, threat detection
- Key Industries: Application Security, Governance Risk and Compliance
- Example Products: Cloud Security Solutions, Threat Intelligence & Analytics

**BDAIW (Big Data & AI World):**
- Focus: Data analytics, machine learning, AI platforms
- Key Industries: AI & ML Platforms, Data Management
- Example Products: Cloud AI Platform, Analytics Platforms

**DCW (Data Centre World):**
- Focus: Data center infrastructure, power, cooling
- Key Industries: Power & Energy, Cooling & Environment
- Example Products: Energy Storage, Cooling systems

## 🔌 API Endpoints

### Health Check
```
GET /health
```

**Enhanced Response with Company Tagging:**
```json
{
  "status": "healthy",
  "version": "0.1.7",
  "model": "sonar",
  "api_key_configured": true,
  "test_query_successful": true,
  "csv_data": {
    "available": true,
    "total_records": 45,
    "shows": ["CAI", "DOL", "CCSE", "BDAIW", "DCW"]
  },
  "available_models": [...],
  "available_resources": [
    "categories://all",
    "categories://shows",
    "categories://shows/{show_name}",
    "categories://industries", 
    "categories://industries/{industry_name}",
    "categories://search/{query}",
    "categories://for-tagging"
  ],
  "available_prompts": [
    "company_tagging_analyst"
  ],
  "available_tools": [
    "perplexity_search_web",
    "perplexity_advanced_search",
    "search_show_categories", 
    "tag_company"
  ]
}
```

## 🎯 Advanced Usage Examples

### **Company Research Workflow**
```python
# Step 1: Research company categories context
categories = await search_show_categories(show_name="CAI")

# Step 2: Tag the company with full analysis
result = await tag_company(
    company_name="Salesforce",
    target_shows="CAI,BDAIW",
    company_description="CRM and cloud computing company"
)

# Step 3: Verify categories used
verification = client.read_resource("categories://for-tagging")
```

### **Batch Company Processing**
```python
companies = [
    {"name": "Microsoft", "trading": "Microsoft", "shows": "CAI,BDAIW"},
    {"name": "Google LLC", "trading": "Google", "shows": "CAI,BDAIW,DOL"},
    {"name": "Oracle Corporation", "trading": "Oracle", "shows": "CAI,DCW"}
]

for company in companies:
    result = await tag_company(
        company_name=company["name"],
        trading_name=company["trading"],
        target_shows=company["shows"]
    )
    # Process results...
```

### **Show-Specific Analysis**
```python
# Focus on specific show categories
cai_categories = client.read_resource("categories://shows/CAI")

# Tag company for CAI show specifically
result = await tag_company(
    company_name="Intel Corporation",
    target_shows="CAI"
)

# Compare with available CAI categories
```

## 🔒 Security & Best Practices

### **API Security**
- Use secure API keys with proper scoping
- Implement rate limiting for search requests
- Enable SSL/TLS for all communications
- Regularly rotate API keys and credentials

### **Data Privacy**
- Company research respects public information sources
- No storage of sensitive company data
- Audit trail for all tagging decisions
- Transparent research methodology

### **Taxonomy Integrity**
- Exact matching to prevent category drift
- No modification of industry/product terms
- Consistent spelling and formatting
- Validation against source taxonomy

## 🐛 Troubleshooting

### **Company Tagging Issues**

**Research Failures:**
```
Error: Unable to find information about company
```
- **Solution**: Verify company name spelling and try trading name
- **Check**: Company has web presence and is publicly searchable
- **Alternative**: Add company description for additional context

**Taxonomy Matching Errors:**
```
Warning: No matching categories found for company
```
- **Solution**: Review target shows - company may not fit these show themes
- **Check**: Expand target shows or verify company's actual business focus
- **Verify**: CSV data contains relevant categories for the company type

**API Rate Limits:**
```
Error: Perplexity API rate limit exceeded
```
- **Solution**: Implement delays between requests
- **Monitor**: API usage against quotas
- **Consider**: Upgrading Perplexity subscription for higher limits

### **CSV Data Issues**

**Missing Categories:**
```
Warning: CSV file not found or empty
```
- **Solution**: Verify CSV file at `src/perplexity_mcp/categories/classes.csv`
- **Check**: File format with headers `Show,Industry,Product`
- **Restart**: Server after CSV file changes

## 🚀 Production Deployment

### **Docker Compose with Company Tagging**
```yaml
services:
  perplexity-mcp:
    build: .
    ports:
      - "8001:8001"
    environment:
      - PERPLEXITY_API_KEY=${PERPLEXITY_API_KEY}
      - PERPLEXITY_MODEL=sonar-pro
    volumes:
      - ./src/perplexity_mcp/categories:/app/src/perplexity_mcp/categories:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
```

### **Company Tagging Performance**
- **Batch Processing**: Process multiple companies with delays
- **Caching**: Cache research results for repeated company analysis
- **Rate Limiting**: Respect Perplexity API limits
- **Quality Control**: Implement validation of tagging results

## 🔄 Integration Examples

### **Web-based Clients**
```javascript
// Tag a company with full analysis
const tagResult = await client.callTool('tag_company', {
    company_name: 'Adobe Inc.',
    trading_name: 'Adobe',
    target_shows: 'CAI,BDAIW',
    company_description: 'Digital media and marketing software company'
});

// Access tagging-specific categories
const taggingCategories = await client.readResource('categories://for-tagging');

// Use the tagging prompt
const promptResult = await client.getPrompt('company_tagging_analyst', {
    company_name: 'Adobe Inc.',
    trading_name: 'Adobe',
    target_shows: 'CAI,BDAIW'
});
```

### **Automated Workflows**
```python
# Research and tag workflow
async def research_and_tag_company(company_info):
    # Step 1: Get relevant categories
    categories = await search_show_categories(
        show_name=company_info.get('primary_show')
    )
    
    # Step 2: Tag the company
    result = await tag_company(
        company_name=company_info['name'],
        trading_name=company_info.get('trading_name'),
        target_shows=company_info.get('target_shows'),
        company_description=company_info.get('description')
    )
    
    return {
        'company': company_info,
        'categories': categories,
        'tagging_result': result
    }
```

## 📈 Performance Monitoring

### **Company Tagging Metrics**
- **Research Success Rate**: Percentage of successful company research
- **Taxonomy Match Rate**: Percentage of companies successfully categorized
- **API Usage**: Perplexity API calls per company analysis
- **Processing Time**: Average time per company tagging
- **Accuracy Validation**: Manual verification of tagging results

### **Quality Assurance**
- **Manual Spot Checks**: Random verification of tagging accuracy
- **Category Distribution**: Monitor usage of different taxonomy categories
- **Show Relevance**: Validate categories match intended show themes
- **Research Quality**: Assess depth and relevance of company research

## 🤝 Contributing

### **Extending Company Tagging**

1. **Add New Shows**: Extend CSV with new show categories
2. **Enhance Research**: Add new data sources for company information
3. **Improve Accuracy**: Refine taxonomy matching algorithms
4. **Add Validation**: Implement quality checking mechanisms

### **CSV Data Management**
- **Category Updates**: Process for updating taxonomy categories
- **Show Management**: Adding or modifying show definitions
- **Quality Control**: Validation of CSV data integrity
- **Version Control**: Track changes to taxonomy over time

## 📄 License

This project is licensed under the MIT License.

---

**Version**: 0.1.7  
**Last Updated**: July 2025  
**Compatibility**: Perplexity API v1, MCP 1.0+, Python 3.11+  
**Tools**: 4 (Web search + Categories + Company tagging)  
**Resources**: 7 (Complete CSV access + Tagging context)  
**Prompts**: 1 (Professional company tagging analyst)  
**Shows Supported**: 5 (CAI, DOL, CCSE, BDAIW, DCW)