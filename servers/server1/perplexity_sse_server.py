import os
import json
import logging
import aiohttp
import hashlib
import time
import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from mcp.types import TextContent
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse

__version__ = "0.1.7"
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("perplexity_mcp_sse_server")

# Track server start time for uptime calculation
start_time = time.time()

# Create MCP server and transport
mcp = FastMCP("Perplexity MCP Server", host="0.0.0.0", port=8001)
transport = SseServerTransport("/messages/")

# In-memory cache for API responses
class APICache:
    def __init__(self, ttl_seconds=3600):  # 1 hour default TTL
        self.cache = {}
        self.ttl = ttl_seconds
    
    def _generate_key(self, query: str, **kwargs) -> str:
        """Generate a unique cache key based on query and parameters."""
        # Create a consistent string from all parameters
        params_str = json.dumps(kwargs, sort_keys=True)
        cache_input = f"{query}:{params_str}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def get(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired."""
        key = self._generate_key(query, **kwargs)
        
        if key in self.cache:
            cached_data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_data
            else:
                # Remove expired entry
                del self.cache[key]
                logger.info(f"Cache expired for query: {query[:50]}...")
        
        return None
    
    def set(self, query: str, response: Dict[str, Any], **kwargs):
        """Store response in cache."""
        key = self._generate_key(query, **kwargs)
        self.cache[key] = (response, time.time())
        logger.info(f"Cached response for query: {query[:50]}...")
    
    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        valid_entries = sum(1 for _, timestamp in self.cache.values() 
                          if current_time - timestamp < self.ttl)
        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "expired_entries": len(self.cache) - valid_entries,
            "ttl_seconds": self.ttl
        }

# Global cache instance
api_cache = APICache(ttl_seconds=1800)  # 30 minutes cache

# Health check cache
health_check_cache = {"last_check": 0, "result": None, "ttl": 300}  # 5 minutes

def validate_environment():
    """Validate required environment variables."""
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        logger.error("PERPLEXITY_API_KEY environment variable is required")
        raise ValueError("PERPLEXITY_API_KEY environment variable is required")
    
    model = os.getenv("PERPLEXITY_MODEL", "sonar")
    logger.info(f"Using Perplexity AI model: {model}")
    
    # List available models for reference
    available_models = {
        "sonar-deep-research": "128k context - Enhanced research capabilities",
        "sonar-reasoning-pro": "128k context - Advanced reasoning with professional focus", 
        "sonar-reasoning": "128k context - Enhanced reasoning capabilities",
        "sonar-pro": "200k context - Professional grade model",
        "sonar": "128k context - Default model",
        "r1-1776": "128k context - Alternative architecture"
    }
    
    logger.info("Available Perplexity models (set with PERPLEXITY_MODEL environment variable):")
    for model_name, description in available_models.items():
        marker = "→" if model_name == model else " "
        logger.info(f" {marker} {model_name}: {description}")
    
    return api_key, model

async def call_perplexity_api(query: str, recency: str = "month", **kwargs) -> Dict[str, Any]:
    """Call the Perplexity API with caching support."""
    
    # Check cache first
    cache_params = {"recency": recency, **kwargs}
    cached_response = api_cache.get(query, **cache_params)
    if cached_response:
        return cached_response
    
    # If not in cache, make API call
    api_key, model = validate_environment()
    
    url = "https://api.perplexity.ai/chat/completions"
    
    # Use provided model or default
    api_model = kwargs.get('model', model)
    max_tokens = kwargs.get('max_tokens', 512)
    temperature = kwargs.get('temperature', 0.2)
    
    payload = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": query},
        ],
        "max_tokens": str(max_tokens),
        "temperature": temperature,
        "top_p": 0.9,
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": recency,
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1,
        "return_citations": True,
        "search_context_size": "low",
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    logger.info(f"Making API request to Perplexity for query: {query[:50]}...")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            
            # Cache the response
            api_cache.set(query, data, **cache_params)
            
            return data

@mcp.tool()
async def perplexity_search_web(query: str, recency: str = "year") -> list[TextContent]:
    """Search the web using Perplexity AI with recency filtering and caching.
    
    Args:
        query: The search query to find information about
        recency: Filter results by how recent they are. Options: 'day' (last 24h), 
                'week' (last 7 days), 'month' (last 30 days), 'year' (last 365 days)
    
    Returns:
        TextContent with the search results and citations
    """
    # Validate recency parameter
    valid_recency = ["day", "week", "month", "year"]
    if recency not in valid_recency:
        logger.warning(f"Invalid recency '{recency}', defaulting to 'month'")
        recency = "month"
    
    logger.info(f"Executing Perplexity search - Query: '{query}', Recency: '{recency}'")
    
    try:
        data = await call_perplexity_api(query, recency)
        content = data["choices"][0]["message"]["content"]
        
        # Check if response came from cache
        cache_params = {"recency": recency}
        was_cached = api_cache.get(query, **cache_params) is not None
        
        # Format response with metadata and citations
        result = {
            "query": query,
            "recency_filter": recency,
            "model": os.getenv("PERPLEXITY_MODEL", "sonar"),
            "content": content,
            "usage": data.get("usage", {}),
            "citations": data.get("citations", []),
            "cached": was_cached
        }
        
        # Create formatted response
        formatted_response = f"**Query:** {query}\n"
        formatted_response += f"**Recency Filter:** {recency}\n"
        formatted_response += f"**Model:** {result['model']}\n"
        if was_cached:
            formatted_response += f"**Source:** Cached response\n"
        formatted_response += f"\n**Response:**\n{content}\n"
        
        if result["citations"]:
            formatted_response += "\n**Citations:**\n"
            for i, url in enumerate(result["citations"], 1):
                formatted_response += f"[{i}] {url}\n"
        
        if result["usage"] and not was_cached:
            formatted_response += f"\n**Token Usage:** {json.dumps(result['usage'])}"
        
        return [TextContent(type="text", text=formatted_response)]
        
    except aiohttp.ClientResponseError as e:
        error_msg = f"Perplexity API error: HTTP {e.status} - {e.message}"
        logger.error(error_msg)
        return [TextContent(type="text", text=f"Error: {error_msg}")]
    
    except Exception as e:
        error_msg = f"Error calling Perplexity API: {str(e)}"
        logger.error(error_msg)
        return [TextContent(type="text", text=f"Error: {error_msg}")]

@mcp.tool()
async def perplexity_advanced_search(
    query: str, 
    recency: str = "year",
    model: Optional[str] = None,
    max_tokens: int = 1024,
    temperature: float = 0.2
) -> list[TextContent]:
    """Advanced Perplexity search with custom parameters and caching.
    
    Args:
        query: The search query to find information about
        recency: Filter results by recency ('day', 'week', 'month', 'year')
        model: Override the default model (optional)
        max_tokens: Maximum tokens in response (default: 512)
        temperature: Response randomness (0.0-1.0, default: 0.2)
    
    Returns:
        TextContent with detailed search results and metadata
    """
    # Use provided model or fall back to environment/default
    if model is None:
        model = os.getenv("PERPLEXITY_MODEL", "sonar")
    
    # Validate parameters
    if recency not in ["day", "week", "month", "year"]:
        recency = "month"
    
    max_tokens = max(1, min(max_tokens, 2048))  # Clamp between 1 and 2048
    temperature = max(0.0, min(temperature, 1.0))  # Clamp between 0 and 1
    
    logger.info(f"Advanced Perplexity search - Query: '{query}', Model: '{model}', "
                f"Recency: '{recency}', Max tokens: {max_tokens}, Temperature: {temperature}")
    
    try:
        # Call API with caching support
        data = await call_perplexity_api(
            query, recency, 
            model=model, 
            max_tokens=max_tokens, 
            temperature=temperature
        )
        
        content = data["choices"][0]["message"]["content"]
        
        # Check if response came from cache
        cache_params = {
            "recency": recency, 
            "model": model, 
            "max_tokens": max_tokens, 
            "temperature": temperature
        }
        was_cached = api_cache.get(query, **cache_params) is not None
        
        # Create comprehensive result
        result = {
            "query": query,
            "model": model,
            "recency_filter": recency,
            "parameters": {
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            "content": content,
            "usage": data.get("usage", {}),
            "citations": data.get("citations", []),
            "finish_reason": data["choices"][0].get("finish_reason"),
            "response_metadata": {
                "timestamp": data.get("created"),
                "model_version": data.get("model"),
                "cached": was_cached
            }
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        error_msg = f"Error in advanced Perplexity search: {str(e)}"
        logger.error(error_msg)
        return [TextContent(type="text", text=f"Error: {error_msg}")]

# Add cache management endpoints
@mcp.tool()
async def clear_api_cache() -> list[TextContent]:
    """Clear the API response cache.
    
    Use this tool to clear cached responses when you need fresh data from the APIs.
    """
    cache_stats_before = api_cache.get_stats()
    api_cache.clear()
    
    result = {
        "message": "API cache cleared successfully",
        "entries_cleared": cache_stats_before["total_entries"],
        "cache_stats_before": cache_stats_before
    }
    
    return [TextContent(type="text", text=json.dumps(result, indent=2))]

@mcp.tool()
async def get_cache_stats() -> list[TextContent]:
    """Get current API cache statistics.
    
    Returns information about cached responses and cache performance.
    """
    stats = api_cache.get_stats()
    
    result = {
        "cache_statistics": stats,
        "description": {
            "total_entries": "Total number of cached responses",
            "valid_entries": "Non-expired cached responses",
            "expired_entries": "Expired cached responses (will be removed on next access)",
            "ttl_seconds": "Time-to-live for cache entries in seconds"
        }
    }
    
    return [TextContent(type="text", text=json.dumps(result, indent=2))]

async def health_check(request):
    """Health check endpoint WITHOUT external API calls - purely internal status."""
    try:
        current_time = time.time()
        
        # Check if we have a recent health check result (still cache for performance)
        if (health_check_cache["last_check"] > 0 and 
            current_time - health_check_cache["last_check"] < health_check_cache["ttl"]):
            logger.info("Using cached health check result")
            return JSONResponse(health_check_cache["result"])
        
        api_key = os.getenv("PERPLEXITY_API_KEY")
        model = os.getenv("PERPLEXITY_MODEL", "sonar")
        
        # Get cache stats (no external calls)
        cache_stats = api_cache.get_stats()
        
        # Determine status based on configuration only (NO API CALLS)
        if not api_key:
            status = "degraded"
            status_message = "PERPLEXITY_API_KEY not configured - server operational but API unavailable"
        else:
            status = "healthy"
            status_message = "Server operational - API key configured"
        
        result = {
            "status": status,
            "message": status_message,
            "version": __version__,
            "timestamp": datetime.datetime.now().isoformat(),
            "server_info": {
                "model": model,
                "api_key_configured": bool(api_key),
                "api_test_disabled": "Health checks do not test external APIs to avoid unnecessary calls",
                "uptime_seconds": int(current_time - start_time) if 'start_time' in globals() else 0
            },
            "cache_stats": cache_stats,
            "available_models": [
                "sonar-deep-research",
                "sonar-reasoning-pro", 
                "sonar-reasoning",
                "sonar-pro",
                "sonar",
                "r1-1776"
            ],
            "available_tools": [
                "perplexity_search_web",
                "perplexity_advanced_search", 
                "clear_api_cache",
                "get_cache_stats"
            ],
            "optimization_features": {
                "api_caching": True,
                "cache_ttl_seconds": api_cache.ttl,
                "health_check_caching": True,
                "external_api_calls_avoided": "Health checks do not call external APIs"
            }
        }
        
        # Cache the health check result
        health_check_cache["last_check"] = current_time
        health_check_cache["result"] = result
        
        return JSONResponse(result)
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        result = {
            "status": "unhealthy",
            "error": str(e),
            "version": __version__,
            "timestamp": datetime.datetime.now().isoformat(),
            "model": os.getenv("PERPLEXITY_MODEL", "sonar"),
            "api_key_configured": bool(os.getenv("PERPLEXITY_API_KEY")),
            "cache_stats": api_cache.get_stats() if 'api_cache' in globals() else {},
            "note": "Health check failed but no external API calls were made"
        }
        return JSONResponse(result)

async def handle_sse(request):
    """Handle SSE connections for MCP."""
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
    
    try:
        # Validate environment on startup
        validate_environment()
        
        logger.info(f"Starting Perplexity MCP Server v{__version__} with SSE transport and caching...")
        logger.info("Perplexity MCP Server running on http://0.0.0.0:8001")
        logger.info("SSE endpoint: http://0.0.0.0:8001/sse")
        logger.info("Health check: http://0.0.0.0:8001/health")
        logger.info("Available MCP tools:")
        logger.info("  - perplexity_search_web (cached)")
        logger.info("  - perplexity_advanced_search (cached)")
        logger.info("  - clear_api_cache (cache management)")
        logger.info("  - get_cache_stats (cache management)")
        logger.info(f"API response cache TTL: {api_cache.ttl} seconds")
        
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
        
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        exit(1)
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        exit(1)