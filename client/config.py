import os
import json
from dotenv import load_dotenv

load_dotenv()
env = os.getenv

# Model mapping - only OpenAI and Azure OpenAI
MODEL_OPTIONS = {
    'Azure OpenAI': 'gpt-4.1',  # Using the deployment from .env
    'OpenAI': 'gpt-4o',
}

# Streamlit defaults
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.1

# UI Configuration
TAB_CONFIG = {
    "chat": {
        "title": "💬 Chat",
        "description": "Main conversation interface"
    },
    "configuration": {
        "title": "⚙️ Configuration", 
        "description": "AI provider and model settings"
    },
    "connections": {
        "title": "🔌 Connections",
        "description": "MCP server management"
    },
    "tools": {
        "title": "🧰 Tools",
        "description": "Available MCP tools"
    }
}

# Load server configuration
config_path = os.path.join('.', 'servers_config.json')
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        SERVER_CONFIG = json.load(f)
else:
    # Default server configuration if file doesn't exist
    SERVER_CONFIG = {
        "mcpServers": {
            "Google Search": {
                "transport": "sse",
                "url": "http://mcpserver2:8002/sse",
                "timeout": 600,
                "headers": None,
                "sse_read_timeout": 900
            },
            "Perplexity Search": {
                "transport": "sse",
                "url": "http://mcpserver1:8001/sse", 
                "timeout": 600,
                "headers": None,
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