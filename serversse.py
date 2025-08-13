import os
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
from fastapi import FastAPI

# Load environment variables
load_dotenv("mcp-chatbot/servers/server_mssql/mssql_mcp_server/.env4test")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mssql_mcp_server")

# Create MCP server and transport
mcp = FastMCP("MSSQL MCP Server", host="0.0.0.0", port=8008)
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
    config, connection_string = get_db_config()
    try:
        with connect(connection_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT TABLE_NAME, TABLE_SCHEMA
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_TYPE = 'BASE TABLE'
                    ORDER BY TABLE_SCHEMA, TABLE_NAME
                """)
                tables = []
                for row in cursor.fetchall():
                    tables.append({
                        "table_name": row[0],
                        "schema": row[1],
                        "full_name": f"{row[1]}.{row[0]}" if row[1] != 'dbo' else row[0]
                    })
                
                result = {
                    "total_tables": len(tables),
                    "tables": tables
                }
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Error as e:
        logger.error(f"Failed to list tables: {str(e)}")
        return [TextContent(type="text", text=f"Error listing tables: {str(e)}")]

@mcp.tool()
async def describe_table(table_name: str) -> list[TextContent]:
    """Get detailed information about a table including columns, data types, and constraints."""
    config, connection_string = get_db_config()
    try:
        with connect(connection_string) as conn:
            with conn.cursor() as cursor:
                # Get column information
                cursor.execute("""
                    SELECT 
                        COLUMN_NAME,
                        DATA_TYPE,
                        IS_NULLABLE,
                        COLUMN_DEFAULT,
                        CHARACTER_MAXIMUM_LENGTH,
                        NUMERIC_PRECISION,
                        NUMERIC_SCALE
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME = ?
                    ORDER BY ORDINAL_POSITION
                """, table_name)
                
                columns = []
                for row in cursor.fetchall():
                    column_info = {
                        "column_name": row[0],
                        "data_type": row[1],
                        "nullable": row[2] == 'YES',
                        "default_value": row[3],
                        "max_length": row[4],
                        "precision": row[5],
                        "scale": row[6]
                    }
                    columns.append(column_info)
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
                row_count = cursor.fetchone()[0]
                
                result = {
                    "table_name": table_name,
                    "row_count": row_count,
                    "columns": columns,
                    "total_columns": len(columns)
                }
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.error(f"Error describing table {table_name}: {str(e)}")
        return [TextContent(type="text", text=f"Error describing table: {str(e)}")]

@mcp.tool()
async def execute_sql(query: str) -> list[TextContent]:
    """Execute an SQL query on the MSSQL database. Supports SELECT, INSERT, UPDATE, DELETE, and other SQL commands."""
    config, connection_string = get_db_config()
    logger.info(f"Executing SQL query: {query}")
    
    try:
        with connect(connection_string) as conn:
            with conn.cursor() as cursor:
                # Handle special commands
                if query.strip().upper() == "SHOW TABLES":
                    cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME")
                    tables = [row[0] for row in cursor.fetchall()]
                    result = {
                        "command": "SHOW TABLES",
                        "tables": tables,
                        "count": len(tables)
                    }
                    return [TextContent(type="text", text=json.dumps(result, indent=2))]
                
                # Execute the query
                cursor.execute(query)
                
                # Handle SELECT queries
                if query.strip().upper().startswith("SELECT"):
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    rows_raw = cursor.fetchall()
                    
                    # Convert rows to serializable format
                    rows = []
                    for row in rows_raw:
                        if hasattr(row, '_asdict'):
                            # pyodbc Row object
                            row_dict = {}
                            for i, col_name in enumerate(columns):
                                row_dict[col_name] = serialize_row_data(row[i])
                            rows.append(row_dict)
                        else:
                            # Regular tuple/list
                            row_dict = {}
                            for i, col_name in enumerate(columns):
                                row_dict[col_name] = serialize_row_data(row[i])
                            rows.append(row_dict)
                    
                    result = {
                        "query": query,
                        "columns": columns,
                        "rows": rows,
                        "row_count": len(rows)
                    }
                    return [TextContent(type="text", text=json.dumps(result, indent=2))]
                
                # Handle non-SELECT queries (INSERT, UPDATE, DELETE, etc.)
                else:
                    conn.commit()
                    result = {
                        "query": query,
                        "rows_affected": cursor.rowcount,
                        "status": "success"
                    }
                    return [TextContent(type="text", text=json.dumps(result, indent=2))]
                    
    except Exception as e:
        logger.error(f"SQL Error executing '{query}': {str(e)}")
        error_result = {
            "query": query,
            "error": str(e),
            "status": "error"
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def get_table_sample(table_name: str, limit: int = 5) -> list[TextContent]:
    """Get a sample of records from a specific table."""
    config, connection_string = get_db_config()
    
    # Validate limit
    if limit <= 0 or limit > 1000:
        limit = 5
    
    try:
        with connect(connection_string) as conn:
            with conn.cursor() as cursor:
                # Use proper SQL Server syntax for limiting results
                query = f"SELECT TOP {limit} * FROM [{table_name}]"
                logger.info(f"Executing sample query: {query}")
                
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                rows_raw = cursor.fetchall()
                
                # Convert rows to serializable format
                rows = []
                for row in rows_raw:
                    row_dict = {}
                    for i, col_name in enumerate(columns):
                        row_dict[col_name] = serialize_row_data(row[i])
                    rows.append(row_dict)
                
                result = {
                    "table_name": table_name,
                    "sample_size": limit,
                    "columns": columns,
                    "rows": rows,
                    "actual_count": len(rows)
                }
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
                
    except Exception as e:
        logger.error(f"Error getting sample from table {table_name}: {str(e)}")
        error_result = {
            "table_name": table_name,
            "error": str(e),
            "status": "error"
        }
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

async def health_check(request):
    """Health check endpoint."""
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
            "driver": config["driver"]
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
    import uvicorn
    logger.info("Starting MSSQL MCP Server with SSE transport...")
    logger.info(f"MSSQL HOST: {os.getenv('MSSQL_HOST')}")
    logger.info(f"MSSQL DB: {os.getenv('MSSQL_DATABASE')}")
    logger.info(f"MSSQL username: {os.getenv('MSSQL_USER')}")
    logger.info("MSSQL MCP Server running on http://0.0.0.0:8008")
    logger.info("SSE endpoint: http://0.0.0.0:8008/sse")
    logger.info("Health check: http://0.0.0.0:8008/health")
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="info")