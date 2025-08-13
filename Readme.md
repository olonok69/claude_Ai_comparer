# MSSQL MCP Server

A Model Context Protocol (MCP) server that provides AI-powered interactions with Microsoft SQL Server databases. This server enables natural language queries, table exploration, and data management operations through a secure, containerized interface as part of the CSM MCP Servers platform.

## 🚀 Features

### **Database Operations**
- **Table Management**: List all tables, describe structure, get sample data
- **SQL Execution**: Execute SELECT, INSERT, UPDATE, DELETE queries
- **Schema Discovery**: Automatic table and column information retrieval
- **Data Exploration**: Sample data retrieval with configurable limits
- **CRUD Operations**: Full Create, Read, Update, Delete support

### **Technical Capabilities**
- **ODBC Integration**: Uses Microsoft ODBC Driver 18 for SQL Server
- **JSON Serialization**: Handles complex data types (Decimal, DateTime, etc.)
- **Error Handling**: Comprehensive error management and logging
- **Health Monitoring**: Built-in health check endpoints
- **SSE Transport**: Server-Sent Events for real-time communication

### **Security Features**
- **Secure Authentication**: Username/password authentication
- **Connection Pooling**: Efficient database connection management
- **Input Validation**: SQL injection protection
- **Certificate Trust**: Configurable SSL certificate handling

### **Integration with CSM Platform**
- **MCP Protocol**: Seamless integration with Streamlit client
- **Multi-Database Environment**: Works alongside Neo4j and HubSpot servers
- **Cross-Platform Queries**: Compare and analyze data across database systems
- **Unified AI Interface**: Natural language queries through the main application

## 📋 Prerequisites

- Docker and Docker Compose
- Microsoft SQL Server instance
- ODBC Driver 18 for SQL Server
- Network access to SQL Server instance
- Part of the CSM MCP Servers platform

## 🛠️ Installation & Setup

### 1. Environment Configuration

As part of the CSM MCP Servers platform, configure your `.env` file in the project root:

```env
# MSSQL Database Configuration
MSSQL_HOST=your-sql-server-host
MSSQL_USER=your-username
MSSQL_PASSWORD=your-password
MSSQL_DATABASE=your-database-name
MSSQL_DRIVER=ODBC Driver 18 for SQL Server
TrustServerCertificate=yes
Trusted_Connection=no

# Other platform configurations
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=bolt://host.docker.internal:7687
PRIVATE_APP_ACCESS_TOKEN=your_hubspot_token
```

### 2. Platform Deployment

```bash
# Build and run all CSM MCP servers
docker-compose up --build

# Or run just the MSSQL server
docker-compose up mcpserver3

# Access the main application
# HTTP: http://localhost:8501
# HTTPS: https://localhost:8502 (if SSL enabled)
```

### 3. Local Development

```bash
# Install dependencies
cd servers/server3
pip install -r requirements.txt

# Run the server
python serversse.py
```

## 🔧 Available Tools

### **list_tables**
List all tables in the database with schema information.

```json
{
  "name": "list_tables",
  "description": "List all tables in the database.",
  "parameters": {}
}
```

**Example Response:**
```json
{
  "total_tables": 5,
  "tables": [
    {
      "table_name": "Customers",
      "schema": "dbo",
      "full_name": "Customers"
    },
    {
      "table_name": "Orders",
      "schema": "dbo", 
      "full_name": "Orders"
    }
  ]
}
```

### **describe_table**
Get detailed information about a specific table including columns, data types, and constraints.

```json
{
  "name": "describe_table",
  "description": "Get detailed information about a table including columns, data types, and constraints.",
  "parameters": {
    "table_name": {
      "type": "string",
      "description": "Name of the table to describe"
    }
  }
}
```

**Example Response:**
```json
{
  "table_name": "Customers",
  "row_count": 1250,
  "total_columns": 6,
  "columns": [
    {
      "column_name": "CustomerID",
      "data_type": "int",
      "nullable": false,
      "default_value": null,
      "max_length": null,
      "precision": 10,
      "scale": 0
    },
    {
      "column_name": "CustomerName",
      "data_type": "varchar",
      "nullable": false,
      "default_value": null,
      "max_length": 100,
      "precision": null,
      "scale": null
    }
  ]
}
```

### **execute_sql**
Execute SQL queries with support for all standard SQL operations.

```json
{
  "name": "execute_sql",
  "description": "Execute an SQL query on the MSSQL database. Supports SELECT, INSERT, UPDATE, DELETE, and other SQL commands.",
  "parameters": {
    "query": {
      "type": "string",
      "description": "The SQL query to execute"
    }
  }
}
```

**Example SELECT Response:**
```json
{
  "query": "SELECT TOP 5 * FROM Customers",
  "columns": ["CustomerID", "CustomerName", "City"],
  "rows": [
    {
      "CustomerID": 1,
      "CustomerName": "Acme Corp",
      "City": "New York"
    }
  ],
  "row_count": 5
}
```

**Example INSERT/UPDATE Response:**
```json
{
  "query": "INSERT INTO Customers (CustomerName, City) VALUES ('New Company', 'Boston')",
  "rows_affected": 1,
  "status": "success"
}
```

### **get_table_sample**
Retrieve sample records from a specific table.

```json
{
  "name": "get_table_sample",
  "description": "Get a sample of records from a specific table.",
  "parameters": {
    "table_name": {
      "type": "string",
      "description": "Name of the table to sample"
    },
    "limit": {
      "type": "integer",
      "description": "Number of records to retrieve (default: 5, max: 1000)",
      "default": 5
    }
  }
}
```

## 🔌 API Endpoints

### Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "database": "YourDatabase",
  "server": "your-server-host",
  "driver": "ODBC Driver 18 for SQL Server"
}
```

### Server-Sent Events
```
GET /sse
```
Main endpoint for MCP communication via Server-Sent Events.

### Message Handling
```
POST /messages/
```
Handles MCP protocol messages.

## 📊 SQL Server Syntax Support

The server uses proper SQL Server syntax and functions:

### **Query Patterns**
```sql
-- Limiting results (use TOP, not LIMIT)
SELECT TOP 10 * FROM Customers

-- Date functions
SELECT GETDATE() as CurrentDateTime

-- String functions
SELECT LEN(CustomerName) as NameLength FROM Customers
SELECT CHARINDEX('Corp', CustomerName) as Position FROM Customers

-- Case-insensitive comparisons
SELECT * FROM Customers WHERE UPPER(City) = 'NEW YORK'

-- Text search with wildcards
SELECT * FROM Customers WHERE CustomerName LIKE '%Corp%'
```

### **Data Type Handling**
- **DECIMAL/NUMERIC**: Automatically converted to float
- **DATETIME/DATE**: Converted to ISO format strings
- **VARCHAR/NVARCHAR**: Handled as strings
- **INT/BIGINT**: Handled as integers
- **BIT**: Converted to boolean

## 🔒 Security Considerations

### **Connection Security**
- Uses encrypted connections when TrustServerCertificate=yes
- Supports both SQL Server and Windows authentication
- Connection string parameters are validated

### **SQL Injection Prevention**
- Input validation on all queries
- Parameterized query support
- Error handling that doesn't expose system details

### **Access Control**
- Database-level permissions enforced
- User-based access control through SQL Server roles
- Connection pooling with secure credential management

## 💬 Usage Through CSM Platform

### **Natural Language Queries**
Access the MSSQL server through the main CSM application using natural language:

```
"Show me all tables in the database"
"Describe the structure of the customers table"
"Get 5 sample records from the orders table"
"Find all customers with orders over $1000"
"Count the total number of products in inventory"
"Show me the top 10 best-selling products"
```

### **Cross-Database Integration**
Combine MSSQL queries with other data sources:

```
"Compare customer data between Neo4j graph database and MSSQL tables"
"Find customers in HubSpot that don't exist in our SQL database"
"Show me sales data from MSSQL and corresponding deal information from HubSpot"
```

### **Authentication**
- Log into the main CSM application using provided credentials
- All database operations are performed through the authenticated session
- Full audit trail of all SQL operations

## 🐛 Troubleshooting

### **Common Connection Issues**

**ODBC Driver Not Found**
```bash
# Check installed drivers
odbcinst -j

# Install ODBC Driver 18 for SQL Server
# On Ubuntu/Debian:
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
apt-get update
ACCEPT_EULA=Y apt-get install -y msodbcsql18
```

**Connection Timeout**
```env
# Increase connection timeout
MSSQL_CONNECTION_TIMEOUT=30
```

**Trust Certificate Issues**
```env
# For self-signed certificates
TrustServerCertificate=yes

# For production with valid certificates
TrustServerCertificate=no
```

### **Query Issues**

**Syntax Errors**
- Use `SELECT TOP n` instead of `LIMIT n`
- Use `LEN()` instead of `LENGTH()`
- Use `CHARINDEX()` instead of `LOCATE()`
- Use `GETDATE()` for current timestamp

**Permission Errors**
- Verify user has SELECT/INSERT/UPDATE/DELETE permissions
- Check database access permissions
- Ensure user has CONNECT permission

### **Performance Issues**

**Slow Queries**
- Add appropriate indexes to frequently queried columns
- Use `SELECT TOP` to limit result sets
- Consider query optimization techniques

**Connection Pool Exhaustion**
- Monitor connection usage
- Implement connection pooling best practices
- Set appropriate connection timeouts

## 📈 Monitoring & Logging

### **Health Monitoring**
The server provides comprehensive health checks:

```bash
# Check server health
curl http://localhost:8008/health

# Expected response
{
  "status": "healthy",
  "database": "YourDatabase",
  "server": "your-server-host",
  "driver": "ODBC Driver 18 for SQL Server"
}
```

### **Logging**
The server logs all important events:

- Connection attempts and failures
- Query execution and performance
- Error conditions and recovery
- Tool usage and parameters

### **Performance Metrics**
Monitor these key metrics:

- Query execution time
- Connection pool utilization
- Error rates by query type
- Memory usage and garbage collection

## 🚀 Production Deployment

### **Docker Compose Configuration**
```yaml
services:
  mcpserver3:
    build: ./servers/server3
    ports:
      - "8008:8008"
    environment:
      - MSSQL_HOST=production-sql-server
      - MSSQL_USER=${MSSQL_USER}
      - MSSQL_PASSWORD=${MSSQL_PASSWORD}
      - MSSQL_DATABASE=${MSSQL_DATABASE}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8008/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### **Production Considerations**
- Use environment-specific configurations
- Implement proper secret management
- Set up monitoring and alerting
- Use connection pooling for high-traffic scenarios
- Regular backup and recovery testing

## 🔄 Development & Extensions

### **Adding Custom Tools**
To add new tools, extend the server:

```python
@mcp.tool()
async def your_custom_tool(parameter: str) -> list[TextContent]:
    """Your custom tool description."""
    # Implementation here
    return [TextContent(type="text", text="Result")]
```

### **Custom Data Serialization**
Extend the `serialize_row_data` function for custom data types:

```python
def serialize_row_data(data):
    if isinstance(data, your_custom_type):
        return your_custom_serialization(data)
    # ... existing serialization logic
```

### **Advanced SQL Operations**
Add support for:
- Stored procedure execution
- Bulk data operations
- Transaction management
- Advanced query optimization

## 📚 Examples

### **Basic Usage Through CSM Platform**
Access through the main application at http://localhost:8501 or https://localhost:8502:

```
# Natural language queries
"List all available tables"
"Get table structure for Customers"
"Show me sample data from Orders with limit 10"
"Execute a query to find top customers by revenue"
```

### **Advanced Queries**
```sql
-- Complex aggregation
SELECT 
    City,
    COUNT(*) as CustomerCount,
    AVG(OrderAmount) as AvgOrderAmount
FROM Customers c
JOIN Orders o ON c.CustomerID = o.CustomerID
WHERE o.OrderDate >= '2024-01-01'
GROUP BY City
ORDER BY CustomerCount DESC

-- Data modification
UPDATE Customers 
SET LastContactDate = GETDATE() 
WHERE CustomerID = 123
```

## 🤝 Integration with CSM Platform

This MSSQL MCP Server is designed to work seamlessly with:

- **Streamlit Client**: Main AI chat interface with authentication
- **Neo4j MCP Server**: Graph database operations and schema discovery
- **HubSpot MCP Server**: Complete CRM integration with 25+ tools

All servers communicate through the Model Context Protocol (MCP) and are accessible through a unified AI-powered interface.

---

**Version**: 1.0.0  
**Last Updated**: June 2025  
**Compatibility**: MSSQL Server 2019+, ODBC Driver 18+, Python 3.11+  
**Platform**: CSM MCP Servers v3.0.0  
**License**: MIT