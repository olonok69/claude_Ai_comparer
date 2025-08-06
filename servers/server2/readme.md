# Google Search MCP Server

A Model Context Protocol (MCP) server implementation for Google Search integration, providing web search and webpage reading capabilities through Server-Sent Events (SSE) transport for real-time communication with AI assistants and applications.

## 🎯 Overview

This MCP server enables seamless integration with Google's Custom Search API, allowing you to perform web searches and extract content from web pages through a standardized protocol. The implementation provides search capabilities and webpage content extraction.

## ✨ Features

### **Google Search Integration (2 Tools)**
- **Web Search**: Perform Google searches with customizable result counts
- **Webpage Reading**: Extract and clean content from web pages

### **Technical Capabilities**
- **Server-Sent Events (SSE)**: Real-time bidirectional communication
- **Docker Support**: Containerized deployment with Docker Compose
- **Schema Validation**: Zod-based input validation for all tools
- **Error Handling**: Comprehensive error messages and debugging
- **Auto-Registration**: Dynamic tool discovery and registration

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ or Docker
- Google Custom Search API key
- Google Custom Search Engine ID
- MCP-compatible client (Claude Desktop, custom applications, etc.)

### 1. Environment Setup

Create a `.env` file in the server directory:
```env
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_custom_search_engine_id
PORT=8002
HOST=0.0.0.0
```

### 2. Google Custom Search Setup

1. **Get API Key**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Enable the Custom Search API
   - Create credentials (API Key)

2. **Create Custom Search Engine**:
   - Go to [Google Custom Search](https://cse.google.com/cse/)
   - Create a new search engine
   - Configure to search the entire web or specific sites
   - Get the Search Engine ID (cx parameter)

### 3. Installation & Running

#### Option A: Docker (Recommended)
```bash
# Build and run with Docker Compose
docker-compose build --no-cache mcpserver2
docker-compose up mcpserver2

# Or build individually
docker build -t google-search-mcp-server .
docker run -p 8002:8002 --env-file .env google-search-mcp-server
```

#### Option B: Node.js
```bash
# Install dependencies
npm install

# Start the server
npm start

# Or run in development mode with auto-reload
npm run dev
```

### 4. Verify Installation
- **Health check**: http://localhost:8002/health
- **MCP endpoint**: http://localhost:8002/sse
- **Expected response**: Server status and version information

## 🔧 MCP Client Configuration

Add to your MCP client configuration:
```json
{
  "mcpServers": {
    "google_search": {
      "transport": "sse",
      "url": "http://localhost:8002/sse",
      "timeout": 600,
      "sse_read_timeout": 900
    }
  }
}
```

## 📚 Available Tools (2 Tools)

### **Web Search (1 tool)**
- `google-search` - Perform web searches with Google Custom Search API

### **Content Extraction (1 tool)**
- `read-webpage` - Extract and clean content from web pages

## 💡 Usage Examples

### Basic Web Search
```javascript
// Search for information
google-search {
  "query": "latest developments in artificial intelligence",
  "num": 5
}

// Search for specific topics
google-search {
  "query": "Model Context Protocol MCP documentation",
  "num": 3
}
```

### Content Extraction Workflow
```javascript
// 1. First, search for information
google-search {
  "query": "climate change 2024 report",
  "num": 5
}

// 2. Then read full content from interesting results
read-webpage {
  "url": "https://example.com/climate-report-2024"
}
```

### Research Workflow
```javascript
// 1. Search for recent news
google-search {
  "query": "technology trends 2024",
  "num": 10
}

// 2. Extract content from multiple sources
read-webpage {
  "url": "https://techcrunch.com/article-url"
}

read-webpage {
  "url": "https://wired.com/another-article"
}
```

## 🏗️ Architecture

### Project Structure
```
servers/server2/
├── main.js                 # Express server with SSE transport
├── package.json           # Dependencies and scripts
├── Dockerfile            # Container configuration
├── Readme.md             # This documentation
├── tools/                # Tool implementations
│   ├── index.js          # Tool registry and handler
│   ├── baseTool.js       # Base tool class with validation
│   ├── toolsRegistry.js  # Auto-registration of all tools
│   ├── searchTool.js     # Google search implementation
│   └── readWebpageTool.js # Webpage content extraction
└── prompts/              # MCP prompts (extensible)
    ├── index.js
    └── promptsRegistry.js
```

### Key Components

#### **BaseTool Class**
All tools extend the `BaseTool` class which provides:
- **Zod Schema Validation**: Input parameter validation
- **Standardized Error Handling**: Consistent error responses
- **Consistent Response Formatting**: Uniform tool output structure

#### **Google Search Client**
Centralized HTTP client with:
- **Google Custom Search API Integration**: Direct API communication
- **Request/Response Handling**: Standardized API communication
- **Error Management**: Comprehensive error handling for API limits and failures

#### **Content Extraction**
Web page processing with:
- **HTML Parsing**: Cheerio-based content extraction
- **Content Cleaning**: Removal of scripts, styles, and navigation elements
- **Text Normalization**: Clean, readable text output

## 🔍 Advanced Features

### **Search Capabilities**
The `google-search` tool supports:

```javascript
// Basic search with default 5 results
{
  "query": "machine learning tutorials"
}

// Search with specific result count
{
  "query": "React best practices 2024",
  "num": 10
}
```

### **Content Extraction Features**
The `read-webpage` tool provides:

```javascript
// Extract content from any accessible webpage
{
  "url": "https://example.com/article"
}

// Returns:
// - Page title
// - Clean text content (scripts/styles removed)
// - Content length information
// - Truncation handling for large pages
```

## 🔍 Debugging & Monitoring

### Health Check
```bash
curl http://localhost:8002/health
```

**Response includes:**
- Server status and version
- Active connection count
- Timestamp information

### Common Issues

#### "API key not found" Error
- **Solution**: Verify `GOOGLE_API_KEY` in `.env`
- **Check**: Google Cloud Console API key configuration
- **Ensure**: Custom Search API is enabled in your project

#### "Search Engine ID not found"
- **Solution**: Verify `GOOGLE_SEARCH_ENGINE_ID` in `.env`
- **Check**: Google Custom Search Engine configuration
- **Verify**: Search engine is active and configured correctly

#### "403 Forbidden" API Error
- **Solution**: Check API key permissions and quotas
- **Monitor**: Daily usage against your quota limits
- **Verify**: API key has Custom Search API access

#### Connection Refused
- **Solution**: Verify server is running on correct port (8002)
- **Check**: Firewall settings and network connectivity
- **Confirm**: MCP client configuration matches server endpoint

## 📈 Performance & Optimization

### Optimization Features
- **Request Timeout Handling**: 10-second timeout for webpage requests
- **Content Truncation**: Large pages truncated to prevent token overflow
- **Error Recovery**: Graceful handling of failed requests
- **Connection Pooling**: Efficient HTTP connection management

### Google API Limits
- **Daily Limits**: Respect Google's daily API call limits
- **Rate Limiting**: Built-in handling of rate limit responses
- **Quota Monitoring**: Track usage against your account limits

## 🧪 Testing

### Basic Connectivity
```bash
# Test health endpoint
curl http://localhost:8002/health

# Test MCP connection (requires MCP client)
# Connect your MCP client to http://localhost:8002/sse
```

### Tool Validation Workflow
1. **Search Testing**: Use `google-search` with simple queries
2. **Content Extraction**: Try `read-webpage` with public URLs
3. **Error Handling**: Test with invalid URLs and API scenarios

### Common Test Scenarios
```javascript
// Test basic search
google-search {"query": "test search", "num": 3}

// Test webpage reading
read-webpage {"url": "https://example.com"}

// Test error handling
read-webpage {"url": "https://invalid-url-example"}
```

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd servers/server2

# Install dependencies
npm install

# Start in development mode
npm run dev
```

### Adding New Tools

1. **Create Tool Class**: Extend `BaseTool` with required functionality
2. **Implement Process Method**: Handle the tool logic and API calls
3. **Define Schema**: Use Zod for input validation
4. **Add to Registry**: Import and register in `toolsRegistry.js`
5. **Update Documentation**: Add tool details to this guide

## 🐛 Troubleshooting

### Environment Variables
```bash
# Verify all required environment variables are set
echo $GOOGLE_API_KEY
echo $GOOGLE_SEARCH_ENGINE_ID
```

### API Connectivity
```bash
# Test Google Custom Search API directly
curl "https://www.googleapis.com/customsearch/v1?key=${GOOGLE_API_KEY}&cx=${GOOGLE_SEARCH_ENGINE_ID}&q=test"
```

### Server Logs
```bash
# Check server logs for detailed error information
docker-compose logs mcpserver2

# Or for Node.js direct run
npm start
```

## 📄 License

This project is licensed under the MIT License.

## 🔗 Resources

- [Google Custom Search JSON API](https://developers.google.com/custom-search/v1/overview)
- [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
- [Express.js Documentation](https://expressjs.com/)
- [Zod Schema Validation](https://zod.dev/)
- [Cheerio HTML Parsing](https://cheerio.js.org/)

---

**Version**: 1.0.0  
**Last Updated**: June 2025  
**API Compatibility**: Google Custom Search API v1  
**Node.js**: 18+  
**Tools**: 2 complete implementations