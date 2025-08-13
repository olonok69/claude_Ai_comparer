import os
import ssl
import json
import logging
from typing import Optional
from dotenv import load_dotenv
from pyodbc import connect, Error
from decimal import Decimal
from datetime import datetime, date

from mcp.types import TextContent
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse
import uvicorn

# Load environment variables
load_dotenv("mcp-chatbot/servers/server_mssql/mssql_mcp_server/.env4test")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mssql_mcp_server")

# Check if SSL is enabled via environment variable
USE_HTTPS = os.getenv("USE_HTTPS", "false").lower() == "true"
PORT = int(os.getenv("SERVER_PORT", "8443" if USE_HTTPS else "8008"))

# Create MCP server and transport
mcp = FastMCP("MSSQL MCP Server", host="0.0.0.0", port=PORT)
transport = SseServerTransport("/messages/")

def get_db_config():
    config = {
        "driver": os.getenv("MSSQL_DRIVER", "ODBC Driver 18 for SQL Server"),
        "server": os.getenv("MSSQL_HOST", "localhost"),
        "user": os.getenv("MSSQL_USER"),
        "password": os.getenv("MSSQL_PASSWORD"),
        "database": os.getenv("MSSQL_DATABASE"),
        "trusted_server_certificate": os.getenv("TrustServerCertificate", "yes"),
        "trusted_connection": os.getenv("Trusted_Connection", "no")
    }
    if not all([config["user"], config["password"], config["database"]]):
        logger.error("Missing required database configuration.")
        raise ValueError("Missing required database configuration")

    connection_string = (
        f"Driver={{{config['driver']}}};Server={config['server']};"
        f"UID={config['user']};PWD={config['password']};"
        f"Database={config['database']};"
        f"TrustServerCertificate={config['trusted_server_certificate']};"
        f"Trusted_Connection={config['trusted_connection']};"
    )

    return config, connection_string

def serialize_row_data(data):
    """Convert pyodbc Row objects and other non-JSON serializable types to JSON-compatible format."""
    if data is None:
        return None
    elif isinstance(data, (str, int, float, bool)):
        return data
    elif isinstance(data, Decimal):
        return float(data)
    elif isinstance(data, (datetime, date)):
        return data.isoformat()
    elif hasattr(data, '_asdict'):  # pyodbc Row object
        return dict(data._asdict())
    elif isinstance(data, (list, tuple)):
        return [serialize_row_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: serialize_row_data(value) for key, value in data.items()}
    else:
        return str(data)

@mcp.tool()
async def list_tables() -> list[TextContent]:
    """List all tables in the database."""
    try:
        _, connection_string = get_db_config()
        with connect(connection_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
                    FROM INFORMATION_SCHEMA.TABLES
                    ORDER BY TABLE_SCHEMA, TABLE_NAME
                """)
                tables = cursor.fetchall()
                
                if not tables:
                    return [TextContent(text="No tables found in the database.")]
                
                result = "Tables in database:\n\n"
                for table in tables:
                    result += f"• {table.TABLE_SCHEMA}.{table.TABLE_NAME} ({table.TABLE_TYPE})\n"
                
                return [TextContent(text=result)]
    except Exception as e:
        logger.error(f"Failed to list tables: {e}")
        return [TextContent(text=f"Error listing tables: {str(e)}")]

@mcp.tool()
async def describe_table(table_name: str) -> list[TextContent]:
    """Get the structure of a specific table."""
    try:
        _, connection_string = get_db_config()
        with connect(connection_string) as conn:
            with conn.cursor() as cursor:
                # Parse schema and table name
                parts = table_name.split('.')
                if len(parts) == 2:
                    schema, table = parts
                else:
                    schema = 'dbo'
                    table = table_name
                
                cursor.execute("""
                    SELECT 
                        COLUMN_NAME,
                        DATA_TYPE,
                        CHARACTER_MAXIMUM_LENGTH,
                        IS_NULLABLE,
                        COLUMN_DEFAULT
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                    ORDER BY ORDINAL_POSITION
                """, (schema, table))
                
                columns = cursor.fetchall()
                
                if not columns:
                    return [TextContent(text=f"Table '{table_name}' not found.")]
                
                result = f"Structure of {schema}.{table}:\n\n"
                for col in columns:
                    length = f"({col.CHARACTER_MAXIMUM_LENGTH})" if col.CHARACTER_MAXIMUM_LENGTH else ""
                    nullable = "NULL" if col.IS_NULLABLE == 'YES' else "NOT NULL"
                    default = f" DEFAULT {col.COLUMN_DEFAULT}" if col.COLUMN_DEFAULT else ""
                    result += f"• {col.COLUMN_NAME}: {col.DATA_TYPE}{length} {nullable}{default}\n"
                
                return [TextContent(text=result)]
    except Exception as e:
        logger.error(f"Failed to describe table {table_name}: {e}")
        return [TextContent(text=f"Error describing table: {str(e)}")]

@mcp.tool()
async def query_database(query: str, max_rows: Optional[int] = 100) -> list[TextContent]:
    """Execute a SQL query and return the results."""
    try:
        _, connection_string = get_db_config()
        with connect(connection_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                
                # Check if this is a SELECT query
                if cursor.description:
                    columns = [column[0] for column in cursor.description]
                    rows = cursor.fetchmany(max_rows) if max_rows else cursor.fetchall()
                    
                    if not rows:
                        return [TextContent(text="Query executed successfully. No rows returned.")]
                    
                    # Convert rows to serializable format
                    serialized_rows = [serialize_row_data(row) for row in rows]
                    
                    result = {
                        "columns": columns,
                        "rows": serialized_rows,
                        "row_count": len(rows),
                        "truncated": len(rows) == max_rows if max_rows else False
                    }
                    
                    return [TextContent(text=json.dumps(result, indent=2, default=str))]
                else:
                    # For non-SELECT queries
                    conn.commit()
                    return [TextContent(text=f"Query executed successfully. Rows affected: {cursor.rowcount}")]
                    
    except Exception as e:
        logger.error(f"Failed to execute query: {e}")
        return [TextContent(text=f"Error executing query: {str(e)}")]

# Health check endpoint
async def health_check(request):
    try:
        config, connection_string = get_db_config()
        # Test database connection
        with connect(connection_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        
        return JSONResponse({
            "status": "healthy",
            "database": config["database"],
            "server": config["server"],
            "driver": config["driver"],
            "protocol": "HTTPS" if USE_HTTPS else "HTTP"
        })
    except Exception as e:
        return JSONResponse({
            "status": "unhealthy",
            "error": str(e)
        })

async def handle_sse(request):
    async with transport.connect_sse(request.scope, request.receive, request._send) as (
        in_stream, out_stream,
    ):
        await mcp._mcp_server.run(
            in_stream, out_stream, mcp._mcp_server.create_initialization_options()
        )

# Build the complete Starlette app with all routes
app = Starlette(
    routes=[
        Route("/health", health_check, methods=["GET"]),
        Route("/sse", handle_sse, methods=["GET"]),
        Mount("/messages/", app=transport.handle_post_message),
    ]
)

if __name__ == "__main__":
    protocol = "https" if USE_HTTPS else "http"
    
    if USE_HTTPS:
        # SSL certificate configuration
        cert_path = os.getenv("SSL_CERT_PATH", "certs/server.crt")
        key_path = os.getenv("SSL_KEY_PATH", "certs/server.key")
        
        # Check if certificate files exist
        if not os.path.exists(cert_path) or not os.path.exists(key_path):
            logger.error(f"Certificate files not found at {cert_path} and {key_path}")
            logger.error("Please generate them first using: ./generate_certs.sh")
            logger.error("Or run without HTTPS by setting USE_HTTPS=false")
            exit(1)
        
        logger.info("Starting MSSQL MCP Server with SSE transport over HTTPS...")
        logger.info(f"MSSQL HOST: {os.getenv('MSSQL_HOST')}")
        logger.info(f"MSSQL DB: {os.getenv('MSSQL_DATABASE')}")
        logger.info(f"MSSQL username: {os.getenv('MSSQL_USER')}")
        logger.info(f"MSSQL MCP Server running on https://0.0.0.0:{PORT}")
        logger.info(f"SSE endpoint: https://0.0.0.0:{PORT}/sse")
        logger.info(f"Health check: https://0.0.0.0:{PORT}/health")
        logger.info("⚠️  Using self-signed certificate - browsers will show security warning")
        
        # Run with SSL
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=PORT, 
            log_level="info",
            ssl_keyfile=key_path,
            ssl_certfile=cert_path
        )
    else:
        logger.info("Starting MSSQL MCP Server with SSE transport over HTTP...")
        logger.info(f"MSSQL HOST: {os.getenv('MSSQL_HOST')}")
        logger.info(f"MSSQL DB: {os.getenv('MSSQL_DATABASE')}")
        logger.info(f"MSSQL username: {os.getenv('MSSQL_USER')}")
        logger.info(f"MSSQL MCP Server running on http://0.0.0.0:{PORT}")
        logger.info(f"SSE endpoint: http://0.0.0.0:{PORT}/sse")
        logger.info(f"Health check: http://0.0.0.0:{PORT}/health")
        
        # Run without SSL
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=PORT, 
            log_level="info"
        )