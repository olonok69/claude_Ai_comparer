# Google Search/Perplexity MCP Host - AI Chat Interface with Company Tagging & CLI Tool

A secure Streamlit-based chat application that connects to Google Search and Perplexity MCP servers to provide AI-powered web search, content extraction, and **specialized company tagging capabilities**. Features comprehensive user authentication, session management, SSL/HTTPS support, multi-provider AI integration, **embedded stdio MCP server**, and a **command-line tool for batch processing**.

## 🚀 Features

### **Security & Authentication**
- **User Authentication System**: Secure login with bcrypt password hashing
- **Session Management**: Persistent user sessions with configurable expiry
- **SSL/HTTPS Support**: Secure connections with self-signed certificates on port 8503
- **Role-Based Access**: Pre-authorized email domains and user management
- **Secure Cookies**: Configurable authentication cookies with custom keys

### **AI & Search Integration**
- **Multi-Provider AI Support**: Compatible with OpenAI and Azure OpenAI
- **Dual Search Engine Integration**: Google Search + Perplexity AI MCP servers
- **Real-time Chat Interface**: Interactive chat with conversation history
- **Web Search Tools**: Google Custom Search API integration with caching
- **AI-Powered Analysis**: Perplexity AI with intelligent responses and caching
- **Content Extraction**: Clean webpage content extraction and analysis
- **Tool Execution Tracking**: Monitor and debug tool usage

### **Company Tagging Workflow** ⭐ **NEW**
- **Specialized stdio MCP Server**: Embedded company categorization system
- **Trade Show Taxonomy**: Access to industry/product categories for 5 major trade shows
- **Automated Research**: Systematic company research using Google Search + Perplexity AI
- **Exact Taxonomy Matching**: Strict enforcement of existing category pairs
- **Interactive Workflow**: Simply type "tag companies" to activate the specialized workflow
- **Batch Support**: Process multiple companies in a single request

### **CLI Tool for Batch Processing** ⭐ **NEW**
- **Command-Line Interface**: Process large CSV files with company data
- **Batch Operations**: Handle hundreds of companies efficiently
- **CSV Input/Output**: Standard CSV format for easy integration
- **Automated Classification**: Reuses the same AI workflow as the web interface
- **Progress Tracking**: Real-time progress updates and error handling
- **Utility Scripts**: CSV validation, merging, and conversion tools

### **User Experience**
- **Modern Tabbed Interface**: Configuration, Connections, Tools, and Chat tabs
- **Responsive Design**: Modern UI with customizable themes and animations
- **User Dashboard**: Personal information and session management
- **Conversation Management**: Create, switch, and delete chat sessions
- **Research Workflows**: Multi-step search and analysis capabilities

## 📋 Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- Active Google Search MCP server
- Active Perplexity MCP server
- Google Custom Search API key and Search Engine ID
- Perplexity API key
- OpenAI API key or Azure OpenAI configuration

## 🛠️ Installation

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd client
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create a `.env` file in the client directory:
   ```env
   # OpenAI Configuration (choose one)
   OPENAI_API_KEY=your_openai_api_key_here
   
   # OR Azure OpenAI Configuration
   AZURE_API_KEY=your_azure_api_key
   AZURE_ENDPOINT=https://your-endpoint.openai.azure.com/
   AZURE_DEPLOYMENT=your_deployment_name
   AZURE_API_VERSION=2023-12-01-preview
   
   # Google Search Configuration
   GOOGLE_API_KEY=your_google_api_key
   GOOGLE_SEARCH_ENGINE_ID=your_custom_search_engine_id
   
   # Perplexity Configuration
   PERPLEXITY_API_KEY=your_perplexity_api_key
   PERPLEXITY_MODEL=sonar
   
   # SSL Configuration (Optional)
   SSL_ENABLED=true
   ```

4. **Set up user authentication**
   
   Generate user credentials:
   ```bash
   python simple_generate_password.py
   ```
   
   This creates `keys/config.yaml` with default users:
   - **admin**: very_Secure_p@ssword_123!
   - **juan**: Larisa1000@
   - **giovanni_romero**: MrRomero2024!
   - **demo_user**: strong_password_123!

5. **Update MCP server configuration**
   
   Edit `servers_config.json` to match your server endpoints:
   ```json
   {
     "mcpServers": {
       "Google Search": {
         "transport": "sse",
         "url": "http://your-google-search-mcp-server:8002/sse",
         "timeout": 600,
         "headers": null,
         "sse_read_timeout": 900
       },
       "Perplexity Search": {
         "transport": "sse",
         "url": "http://your-perplexity-mcp-server:8001/sse",
         "timeout": 600,
         "headers": null,
         "sse_read_timeout": 900
       },
       "Company Tagging": {
         "transport": "stdio",
         "command": "python",
         "args": ["-m", "mcp_servers.company_tagging.server"],
         "env": {
           "PERPLEXITY_API_KEY": "${PERPLEXITY_API_KEY}",
           "PERPLEXITY_MODEL": "${PERPLEXITY_MODEL}"
         }
       }
     }
   }
   ```

6. **Run the application**
   ```bash
   # HTTP mode
   streamlit run app.py
   
   # HTTPS mode (if SSL certificates are set up)
   streamlit run app.py --server.port=8503 --server.sslCertFile=ssl/cert.pem --server.sslKeyFile=ssl/private.key
   ```

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t google-search-mcp-client .
   ```

2. **Run with environment variables**
   ```bash
   docker run -p 8501:8501 -p 8503:8503 \
     -e OPENAI_API_KEY=your_key \
     -e GOOGLE_API_KEY=your_google_key \
     -e GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id \
     -e PERPLEXITY_API_KEY=your_perplexity_key \
     -e SSL_ENABLED=true \
     -v $(pwd)/.env:/app/.env \
     -v $(pwd)/keys:/app/keys \
     google-search-mcp-client
   ```

### CLI Tool Setup ⭐ **NEW**

1. **Set up the CLI tool**
   ```bash
   # From project root
   ./setup_cli.sh
   ```

2. **Test with sample data**
   ```bash
   # Process sample CSV
   python3 company_cli.py --input sample_companies.csv --output test_results
   ```

3. **Process your own CSV file**
   ```bash
   # Basic processing
   python3 company_cli.py --input your_companies.csv --output results
   
   # With custom batch size
   python3 company_cli.py --input your_companies.csv --output results --batch-size 5
   
   # Use the batch script for large files
   ./batch_process.sh large_companies.csv output_directory
   ```

## 🎯 Usage

### Getting Started

1. **Launch the application** and navigate to:
   - **HTTP**: `http://localhost:8501`
   - **HTTPS**: `https://localhost:8503` (accept browser security warning)

2. **Authenticate**:
   - Use the sidebar authentication panel
   - Login with generated credentials (default: admin/very_Secure_p@ssword_123!)
   - View welcome message and user information

3. **Configure your AI provider** (Configuration tab):
   - Select between OpenAI or Azure OpenAI
   - Verify your credentials are loaded (green checkmark)
   - Adjust model parameters (temperature, max tokens)

4. **Connect to MCP servers** (Connections tab):
   - Click "Connect to MCP Servers"
   - Verify successful connection (you'll see available tools)
   - Check server health status

5. **Explore available tools** (Tools tab):
   - Browse Google Search tools (google-search, read-webpage)
   - Browse Perplexity AI tools (perplexity_search_web, perplexity_advanced_search)
   - Browse Company Tagging tools (search_show_categories, tag_company) ⭐ **NEW**
   - View tool documentation and parameters

6. **Start chatting** (Chat tab):
   - Ask questions to search the web and extract content
   - Use company tagging by typing "tag companies" ⭐ **NEW**
   - The AI will automatically use appropriate tools to answer
   - View tool execution history

### Example Queries

#### **Web Search Operations**
```
"Search for the latest developments in artificial intelligence"
"Find recent news about climate change"
"What are the current trends in web development?"
"Search for Python programming tutorials"
```

#### **Content Extraction Operations**
```
"Search for climate reports and read the full content from the first result"
"Find the latest tech news and extract content from TechCrunch"
"Search for React documentation and read the official guide"
```

#### **Company Tagging Operations** ⭐ **NEW**
```
"Tag the following companies:
Account Name = Microsoft Trading Name = Microsoft Domain = microsoft.com Event = Cloud and AI Infrastructure"

"Categorize companies for CAI and BDAIW shows"

"Research and tag these exhibitors for trade show categorization"

"tag companies: GMV, IQVIA, Keepler"
```

#### **Research Workflows**
```
"Research the current state of renewable energy technology"
"Find and analyze multiple sources about cryptocurrency trends"
"Search for best practices in software engineering and summarize them"
```

### CLI Tool Usage ⭐ **NEW**

#### **Basic Commands**
```bash
# Process a CSV file
python3 company_cli.py --input companies.csv --output results

# Process with verbose output
python3 company_cli.py --input companies.csv --output results --verbose

# Process with custom batch size
python3 company_cli.py --input companies.csv --output results --batch-size 5
```

#### **CSV Utilities**
```bash
# Validate CSV structure
python3 csv_processor_utility.py --validate companies.csv

# Generate sample CSV
python3 csv_processor_utility.py --generate-sample test_companies.csv

# Get batch processing info
python3 csv_processor_utility.py --batch-info companies.csv --batch-size 10

# Convert markdown results to CSV
python3 csv_processor_utility.py --to-csv results.md --output results.csv

# Merge multiple result files
python3 csv_processor_utility.py --merge-md result1.md result2.md --output final.md
```

#### **Expected CSV Input Format**
```csv
CASEACCID,Account Name,Trading Name,Domain,Industry,Product/Service Type,Event
CASE001,GMV,GMV,gmv.com,,"100 Optical; ADAS and Autonomous Vehicle Technology Expo Europe"
CASE002,IQVIA,IQVIA,iqvia.com,,"100 Optical; Best Practice; Big Data Expo"
CASE003,Keepler,Keepler,keepler.io,,"100 Optical; Best Practice; Big Data Expo"
```

#### **CLI Output Example**
```markdown
| Company Name | Trading Name | Tech Industry 1 | Tech Product 1 | Tech Industry 2 | Tech Product 2 |
|--------------|--------------|-----------------|----------------|-----------------|----------------|
| GMV | GMV | Cloud and AI Infrastructure Services | Cloud Security Solutions | IT Infrastructure & Hardware | Semiconductor Technologies |
| IQVIA | IQVIA | AI & ML Platforms | Applications & AI Tools | Data Management | Data Analytics & Integration |
```

### Advanced Configuration

**Model Parameters:**
- **Temperature**: Control response creativity (0.0-1.0)
- **Max Tokens**: Set response length limit (1024-10240)

**Chat Management:**
- Create new conversations with "New Chat"
- Access conversation history in the sidebar
- Delete conversations as needed
- Switch between conversations seamlessly

**User Management:**
- View current user information in the sidebar
- Monitor session time and activity
- Secure logout functionality

## 🏗️ Architecture

```
┌───────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI    │    │   LangChain      │    │ Google Search   │
│                   │◄──►│   Agent          │◄──►│ MCP Server      │
│  - Chat Interface │    │  - Tool Routing  │    │                 │
│  - Authentication │    │  - LLM Provider  │    │  - Web Search   │
│  - Config Panel   │    │  - Memory Mgmt   │    │  - Content Ext  │
│  - Tool Display   │    │                  │    │                 │
│  - Company Tags   │    └──────────────────┘    └─────────────────┘
└───────────────────┘               │                       
           ▲                        │              ┌─────────────────┐
           │                        │              │ Perplexity AI   │
           │                        └─────────────►│ MCP Server      │
           │                                       │                 │
           │                                       │  - AI Search    │
           │                                       │  - Analysis     │
           │                                       │  - Taxonomy     │
           │                                       └─────────────────┘
           │                        
    ┌──────┴──────┐                 ┌─────────────────┐
    │ stdio MCP   │                 │ CLI Tool        │
    │ Server      │                 │                 │
    │             │                 │ - Batch Process │
    │ - Company   │                 │ - CSV I/O       │
    │   Tagging   │                 │ - Automation    │
    │ - Taxonomy  │                 │ - Utilities     │
    └─────────────┘                 └─────────────────┘
```

### Key Components

- **`app.py`**: Main Streamlit application with authentication
- **`services/`**: Core business logic (AI, MCP, Chat management)
  - **`ai_service.py`**: LLM provider management (OpenAI, Azure OpenAI)
  - **`mcp_service.py`**: MCP client management (SSE + stdio)
  - **`chat_service.py`**: Conversation and session management
- **`ui_components/`**: Reusable UI components and widgets
  - **`tab_components.py`**: Configuration, Connections, Tools, Chat tabs
  - **`sidebar_components.py`**: Sidebar elements and chat history
- **`mcp_servers/company_tagging/`**: Embedded stdio MCP server ⭐ **NEW**
  - **`server.py`**: Main stdio MCP server implementation
  - **`categories/classes.csv`**: Trade show taxonomy data
- **`utils/`**: Helper functions and utilities
- **`config.py`**: Configuration management
- **CLI Tools** ⭐ **NEW**:
  - **`company_cli.py`**: Main command-line interface
  - **`csv_processor_utility.py`**: CSV processing utilities
  - **`setup_cli.sh`**: CLI setup script
  - **`batch_process.sh`**: Batch processing wrapper

### Authentication System

- **Password Hashing**: bcrypt with salt for secure storage
- **Session Management**: Streamlit-authenticator integration
- **Cookie Configuration**: Secure, configurable authentication cookies
- **User Validation**: Pre-authorized email domains

### MCP Server Integration ⭐ **ENHANCED**

- **SSE Transport**: Google Search and Perplexity servers via HTTP/SSE
- **stdio Transport**: Company Tagging server embedded in client
- **Environment Variables**: Automatic expansion and validation
- **Health Monitoring**: Built-in health checks and performance monitoring
- **Cache Management**: Intelligent caching with TTL and cleanup

## 🔧 Configuration

### Model Providers

The application supports multiple AI providers configured in `config.py`:

```python
MODEL_OPTIONS = {
    'OpenAI': 'gpt-4o',
    'Azure OpenAI': 'gpt-4.1',
}
```

### User Management

User credentials are managed in `keys/config.yaml`. To add/modify users:

1. Edit `simple_generate_password.py`
2. Modify the `users` dictionary with desired credentials
3. Run the script to generate new `config.yaml`

### MCP Server Configuration ⭐ **ENHANCED**

Server endpoints are defined in `servers_config.json`. The configuration now supports both SSE and stdio transports:

```json
{
  "mcpServers": {
    "Google Search": {
      "transport": "sse",
      "url": "http://mcpserver2:8002/sse",
      "timeout": 600,
      "headers": null,
      "sse_read_timeout": 900
    },
    "Perplexity Search": {
      "transport": "sse",
      "url": "http://mcpserver1:8001/sse",
      "timeout": 600,
      "headers": null,
      "sse_read_timeout": 900
    },
    "Company Tagging": {
      "transport": "stdio",
      "command": "python",
      "args": ["-m", "mcp_servers.company_tagging.server"],
      "env": {
        "PERPLEXITY_API_KEY": "${PERPLEXITY_API_KEY}",
        "PERPLEXITY_MODEL": "${PERPLEXITY_MODEL}"
      }
    }
  }
}
```

### Company Tagging Configuration ⭐ **NEW**

The embedded stdio MCP server provides:
- **Trade Show Taxonomy**: Access to 200+ industry/product category pairs
- **Specialized Prompts**: Systematic company research and categorization
- **CSV Data Management**: Real-time access to show categories
- **Exact Taxonomy Enforcement**: Strict matching to existing pairs only

### SSL Configuration

For HTTPS support on port 8503:
1. Set `SSL_ENABLED=true` in environment variables
2. Certificates will be automatically generated on startup
3. Access via https://localhost:8503 (accept browser warning for self-signed cert)

### Styling

Custom CSS is located in `.streamlit/style.css` for UI customization:
- Tab styling and animations
- Modern button designs
- Responsive layout adjustments
- Color scheme and themes

## 🔒 Security Features

### Authentication Security
- **Bcrypt Hashing**: Industry-standard password protection
- **Session Timeout**: Configurable session expiry (30 days default)
- **Secure Cookies**: HTTPOnly and secure cookie attributes
- **Access Control**: Pre-authorized email validation

### SSL/HTTPS Security
- **Self-signed Certificates**: Automatic generation for development
- **Port 8503**: Dedicated HTTPS port separate from HTTP (8501)
- **Secure Headers**: Proper SSL configuration for Streamlit
- **Certificate Management**: Automatic renewal and permission handling

### API Security
- **Environment Variables**: Secure credential storage
- **Token Validation**: Real-time API key verification
- **Input Sanitization**: XSS and injection protection
- **Error Handling**: Secure error messages without data exposure

### Session Management
- **User Isolation**: Separate conversation histories per user
- **Session Tracking**: Login time and activity monitoring
- **Automatic Cleanup**: Session data management
- **Cross-Session Security**: Protected against session hijacking

### CLI Tool Security ⭐ **NEW**
- **Environment Variables**: Secure API key loading
- **No Credential Logging**: API keys never exposed in logs
- **Same Security Model**: Inherits all web application security features
- **Validation**: Input validation for CSV files and parameters

## 🐛 Troubleshooting

### Common Issues

**Authentication Problems:**
- Verify `keys/config.yaml` exists and is properly formatted
- Check user credentials match the generated passwords
- Ensure email domains are in preauthorized list
- Clear browser cookies if experiencing login issues

**Connection Problems:**
- Verify Google Search and Perplexity MCP servers are running and accessible
- Check network connectivity to server endpoints (ports 8001, 8002)
- Ensure proper server configuration in `servers_config.json`
- Review Google API credentials and quotas

**Company Tagging Issues:** ⭐ **NEW**
- Verify `PERPLEXITY_API_KEY` is set in environment variables
- Check CSV file exists at `mcp_servers/company_tagging/categories/classes.csv`
- Test stdio server using the "Test Company Tagging Server" button
- Ensure proper import paths for embedded MCP server

**CLI Tool Issues:** ⭐ **NEW**
- Run `./setup_cli.sh` to verify setup
- Check Python path includes client directory
- Validate CSV structure with `python3 csv_processor_utility.py --validate file.csv`
- Ensure MCP servers are running: `docker-compose ps`

**SSL/HTTPS Issues:**
- Certificates are automatically generated on startup
- Accept browser security warning for self-signed certificates
- Check container logs for certificate generation errors
- Verify SSL_ENABLED environment variable is set to "true"

**API Key Issues:**
- Confirm Google API key and Search Engine ID are properly set
- Check API key permissions and quotas in Google Cloud Console
- Verify Custom Search API is enabled
- Test API connectivity outside the application

**Tool Execution Errors:**
- Review tool execution history in the expandable section
- Check Google Search and Perplexity MCP server logs for detailed error information
- Ensure MCP servers are properly configured and running
- Verify API quotas haven't been exceeded

### Debug Mode

Enable debug information by:
1. Using the "Tool Execution History" expander
2. Checking browser console for JavaScript errors
3. Monitoring Streamlit logs in terminal
4. Reviewing authentication logs
5. Using CLI tool verbose mode: `--verbose`

### Performance Optimization

- Adjust `max_tokens` for faster responses
- Use appropriate search result counts (1-10) based on needs
- Monitor memory usage with multiple concurrent sessions
- Optimize conversation history management
- Use batch processing for large-scale operations

## 🔄 User Management

### Adding New Users

1. **Edit the password generation script**:
   ```bash
   # Edit simple_generate_password.py
   # Add new users to the users dictionary
   ```

2. **Generate new configuration**:
   ```bash
   python simple_generate_password.py
   ```

3. **Restart the application** to load new users

### Managing Existing Users

- **Password Changes**: Regenerate `config.yaml` with new passwords
- **Email Updates**: Modify user information in the generation script
- **Access Revocation**: Remove users from the configuration and regenerate

### Session Management

- **View Active Sessions**: Check sidebar for current user information
- **Session Timeout**: Configure in `config.yaml` (expiry_days)
- **Forced Logout**: Clear cookies or restart application

## 📊 Available Tools & Features

### **Total Tools Available: 11 Tools**

#### **Google Search MCP Server (4 Tools)**
1. **google-search**: Google Custom Search with caching (30-min TTL)
2. **read-webpage**: Clean content extraction with caching (2-hour TTL)
3. **clear-cache**: Cache management for search and webpage data
4. **cache-stats**: Performance monitoring and cache statistics

#### **Perplexity AI MCP Server (5 Tools)**
1. **perplexity_search_web**: AI-powered web search with caching (30-min TTL)
2. **perplexity_advanced_search**: Advanced search with custom parameters
3. **search_show_categories**: Access to CSV-based category taxonomy
4. **clear_api_cache**: Clear Perplexity API response cache
5. **get_cache_stats**: Detailed cache performance statistics

#### **Company Tagging MCP Server (2 Tools)** ⭐ **NEW**
1. **search_show_categories**: Access trade show taxonomy with filtering
2. **tag_company**: Systematic company categorization workflow

### **CLI Tool Features** ⭐ **NEW**
- **Batch Processing**: Process CSV files with multiple companies
- **Automated Research**: Systematic research using all available tools
- **Progress Tracking**: Real-time updates and error handling
- **Multiple Formats**: Markdown and CSV output files
- **Validation Tools**: CSV structure validation and preview
- **Utility Scripts**: Merging, conversion, and batch processing helpers

### **Available Resources**
- **categories://all**: Complete CSV data with all show categories
- **categories://shows**: Categories organized by show with statistics
- **categories://shows/{show_name}**: Categories for specific shows
- **categories://industries**: Categories organized by industry
- **categories://for-analysis**: Categories formatted for analysis with strict enforcement

## 🔄 Version History

- **v3.0.0**: Major update with Company Tagging stdio MCP server and CLI tool ⭐ **NEW**
- **v2.0.0**: Google Search and Perplexity MCP integration with SSL support
- **v1.0.0**: Initial release with authentication system
- Basic chat interface with multi-provider AI support
- Tool execution tracking and conversation management

## 🤝 Contributing

### Development Guidelines

1. **Follow authentication patterns** when adding new features
2. **Test with multiple user accounts** to ensure proper isolation
3. **Maintain security best practices** for credential handling
4. **Update documentation** for new authentication features
5. **Test both web interface and CLI tool** for consistency

### Security Considerations

- Never log or expose user passwords or session tokens
- Validate all user inputs for security vulnerabilities
- Follow secure coding practices for authentication flows
- Test authentication edge cases and error conditions
- Ensure CLI tool follows same security patterns as web interface

### Testing New Features

#### **Company Tagging Testing** ⭐ **NEW**
- Test stdio MCP server startup and connection
- Verify taxonomy data loading and filtering
- Test company research workflow with various company types
- Validate exact taxonomy matching enforcement
- Test error handling for invalid company data

#### **CLI Tool Testing** ⭐ **NEW**
- Test with various CSV formats and sizes
- Verify batch processing works correctly
- Test error handling and recovery mechanisms
- Validate output formats match web interface results
- Test utility scripts for CSV processing

---

**Version**: 3.0.0 ⭐ **MAJOR UPDATE**  
**Last Updated**: January 2025  
**New Features**: Company Tagging stdio MCP Server, CLI Tool for Batch Processing  
**Security**: Streamlit Authenticator 0.3.2, bcrypt password hashing, SSL/HTTPS support  
**Compatibility**: Python 3.11+, Streamlit 1.44+, Google Custom Search API v1, Perplexity API v1  
**Total Tools**: 11 tools (4 Google Search, 5 Perplexity AI, 2 Company Tagging)  
**MCP Servers**: 3 total (2 SSE + 1 embedded stdio)  
**CLI Tool**: Full-featured batch processing with CSV input/output  
**Architecture**: Multi-transport MCP integration (SSE + stdio) with comprehensive authentication and caching