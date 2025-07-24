# Updated services/mcp_service.py

from typing import Dict, List, Union
import streamlit as st
import subprocess
import os
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage
from services.ai_service import create_llm_model
from utils.async_helpers import run_async


async def setup_mcp_client(server_config: Dict[str, Dict]) -> MultiServerMCPClient:
    """Initialize a MultiServerMCPClient with the provided server configuration."""
    client = MultiServerMCPClient(server_config)
    return await client.__aenter__()

async def get_tools_from_client(client: MultiServerMCPClient) -> List[BaseTool]:
    """Get tools from the MCP client."""
    return client.get_tools()

async def get_prompts_from_client(client: MultiServerMCPClient) -> List:
    """Get prompts from the MCP client."""
    try:
        # Get prompts using the client's list_prompts method
        prompts_response = await client._send_request("prompts/list", {})
        if prompts_response and "prompts" in prompts_response:
            return prompts_response["prompts"]
        return []
    except Exception as e:
        print(f"Error getting prompts: {e}")
        return []

async def get_resources_from_client(client: MultiServerMCPClient) -> List:
    """Get resources from the MCP client."""
    try:
        # Get resources using the client's list_resources method
        resources_response = await client._send_request("resources/list", {})
        if resources_response and "resources" in resources_response:
            return resources_response["resources"]
        return []
    except Exception as e:
        print(f"Error getting resources: {e}")
        return []

async def read_resource_from_client(client: MultiServerMCPClient, uri: str) -> str:
    """Read a specific resource from the MCP client."""
    try:
        # Read resource using the client
        resource_response = await client._send_request("resources/read", {
            "uri": uri
        })
        
        if resource_response and "contents" in resource_response:
            contents = resource_response["contents"]
            if isinstance(contents, list) and len(contents) > 0:
                # Return the first content item's text
                first_content = contents[0]
                if isinstance(first_content, dict) and "text" in first_content:
                    return first_content["text"]
                else:
                    return str(first_content)
            else:
                return str(contents)
        return ""
    except Exception as e:
        print(f"Error reading resource {uri}: {e}")
        return ""

async def use_prompt(client: MultiServerMCPClient, prompt_name: str, arguments: Dict = None):
    """Use a specific prompt from the MCP client."""
    try:
        if arguments is None:
            arguments = {}
        
        # Get the prompt using the client
        prompt_response = await client._send_request("prompts/get", {
            "name": prompt_name,
            "arguments": arguments
        })
        
        if prompt_response and "messages" in prompt_response:
            return prompt_response["messages"]
        return None
    except Exception as e:
        print(f"Error using prompt {prompt_name}: {e}")
        return None

async def run_agent(agent, messages: Union[str, List[BaseMessage]]) -> Dict:
    """Run the agent with the provided message(s)."""
    if isinstance(messages, str):
        # If it's a string, treat it as a simple message
        return await agent.ainvoke({"messages": [messages]})
    elif isinstance(messages, list):
        # If it's a list of messages, use them directly for conversation memory
        return await agent.ainvoke({"messages": messages})
    else:
        # Fallback for other types
        return await agent.ainvoke({"messages": str(messages)})

async def run_tool(tool, **kwargs):
    """Run a tool with the provided parameters."""
    return await tool.ainvoke(**kwargs)

def expand_env_vars(config: Dict) -> Dict:
    """Expand environment variables in server configuration."""
    if isinstance(config, dict):
        result = {}
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]  # Remove ${ and }
                result[key] = os.getenv(env_var, value)  # Use original value if env var not found
            elif isinstance(value, dict):
                result[key] = expand_env_vars(value)
            elif isinstance(value, list):
                result[key] = [expand_env_vars(item) if isinstance(item, dict) else item for item in value]
            else:
                result[key] = value
        return result
    return config

def prepare_server_config(servers: Dict[str, Dict]) -> Dict[str, Dict]:
    """Prepare server configuration with environment variable expansion and stdio setup."""
    prepared_config = {}
    
    for server_name, config in servers.items():
        expanded_config = expand_env_vars(config)
        
        if expanded_config.get("transport") == "stdio":
            # For stdio servers, we need to set up the command properly
            stdio_config = {
                "transport": "stdio",
                "command": expanded_config.get("command", "python"),
                "args": expanded_config.get("args", []),
                "env": expanded_config.get("env", {})
            }
            
            # Ensure the stdio server is accessible
            if stdio_config["args"] and stdio_config["args"][0] == "-m":
                # Convert module path to absolute path if needed
                module_path = stdio_config["args"][1]
                if module_path.startswith("mcp_servers."):
                    # This is our embedded server
                    app_root = Path(__file__).parent.parent
                    stdio_config["cwd"] = str(app_root)
            
            prepared_config[server_name] = stdio_config
        else:
            # SSE servers remain unchanged
            prepared_config[server_name] = expanded_config
    
    return prepared_config

def connect_to_mcp_servers():
    """Connect to all configured MCP servers (SSE and stdio)."""
    # Clean up existing client if any
    client = st.session_state.get("client")
    if client:
        try:
            run_async(client.__aexit__(None, None, None))
        except Exception as e:
            st.warning(f"Error closing previous client: {str(e)}")

    # Collect LLM config dynamically from session state
    params = st.session_state['params']
    llm_provider = params.get("model_id")
    try:
        llm = create_llm_model(llm_provider, temperature=params['temperature'], max_tokens=params['max_tokens'])
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        st.stop()
        return
    
    # Prepare server configuration with environment variable expansion
    prepared_servers = prepare_server_config(st.session_state.servers)
    
    # Log server configuration for debugging
    st.write("Debug: Prepared server configuration:")
    for name, config in prepared_servers.items():
        transport_type = config.get("transport", "unknown")
        if transport_type == "stdio":
            st.write(f"- {name}: stdio (command: {config.get('command')}, args: {config.get('args')})")
        else:
            st.write(f"- {name}: {transport_type} ({config.get('url', 'no url')})")
    
    # Setup new client with mixed transports
    try:
        st.session_state.client = run_async(setup_mcp_client(prepared_servers))
        st.session_state.tools = run_async(get_tools_from_client(st.session_state.client))
        
        # Get prompts and resources from MCP servers
        try:
            st.session_state.prompts = run_async(get_prompts_from_client(st.session_state.client))
        except:
            st.session_state.prompts = []
        
        try:
            st.session_state.resources = run_async(get_resources_from_client(st.session_state.client))
        except:
            st.session_state.resources = []
        
        st.session_state.agent = create_react_agent(llm, st.session_state.tools)
        
        # Log successful connections
        tool_count = len(st.session_state.tools)
        prompt_count = len(st.session_state.prompts)
        resource_count = len(st.session_state.resources)
        st.success(f"Successfully connected to {len(prepared_servers)} MCP servers with {tool_count} tools, {prompt_count} prompts, and {resource_count} resources")
        
        # Categorize tools for display with improved detection
        google_tools = []
        perplexity_tools = []
        company_tagging_tools = []
        other_tools = []
        
        for tool in st.session_state.tools:
            tool_name = tool.name.lower()
            tool_desc = tool.description.lower() if hasattr(tool, 'description') and tool.description else ""
            
            # Company Tagging tool detection - improved patterns
            if any(keyword in tool_name for keyword in [
                'search_show_categories', 'company_tagging', 'tag_companies', 
                'categorize', 'taxonomy', 'show_categories'
            ]) or any(keyword in tool_desc for keyword in [
                'company', 'categoriz', 'taxonomy', 'tag', 'show', 'exhibitor'
            ]):
                company_tagging_tools.append(tool.name)
            
            # Perplexity tool detection
            elif any(keyword in tool_name for keyword in [
                'perplexity_search_web', 'perplexity_advanced_search', 'perplexity'
            ]) or 'perplexity' in tool_desc:
                perplexity_tools.append(tool.name)
            
            # Google Search tool detection
            elif any(keyword in tool_name for keyword in [
                'google-search', 'read-webpage', 'google_search', 'webpage'
            ]) or (('google' in tool_name or 'search' in tool_name or 'webpage' in tool_name) 
                   and 'perplexity' not in tool_name):
                google_tools.append(tool.name)
            
            else:
                other_tools.append(tool.name)
        
        if google_tools:
            st.info(f"🔍 Google Search tools: {', '.join(google_tools)}")
        if perplexity_tools:
            st.info(f"🔮 Perplexity tools: {', '.join(perplexity_tools)}")
        if company_tagging_tools:
            st.info(f"📊 Company Tagging tools: {', '.join(company_tagging_tools)}")
        if other_tools:
            st.info(f"🔧 Other tools: {', '.join(other_tools)}")
        
        # Display prompts info
        if st.session_state.prompts:
            prompt_names = [prompt.get('name', 'unnamed') for prompt in st.session_state.prompts]
            st.info(f"📝 Available prompts: {', '.join(prompt_names)}")
        
        # Display resources info
        if st.session_state.resources:
            resource_names = [resource.get('name', 'unnamed') for resource in st.session_state.resources]
            st.info(f"📁 Available resources: {', '.join(resource_names)}")
            
    except Exception as e:
        st.error(f"Failed to connect to MCP servers: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        raise

def disconnect_from_mcp_servers():
    """Disconnect from all MCP servers and clean up session state."""
    # Clean up existing client if any and session state connections
    client = st.session_state.get("client")
    if client:
        try:
            run_async(client.__aexit__(None, None, None))    
        except Exception as e:
            st.warning(f"Error during disconnect: {str(e)}")
    else:
        st.info("No MCP connection to disconnect.")

    # Clean up session state
    st.session_state.client = None
    st.session_state.tools = []
    st.session_state.prompts = []
    st.session_state.resources = []
    st.session_state.agent = None

def test_stdio_server():
    """Test the stdio server independently."""
    try:
        # Test if we can run the stdio server
        app_root = Path(__file__).parent.parent
        server_path = app_root / "mcp_servers" / "company_tagging" / "server.py"
        
        if not server_path.exists():
            return False, f"Server file not found: {server_path}"
        
        # Try to import the server module
        import sys
        sys.path.insert(0, str(app_root))
        
        try:
            import mcp_servers.company_tagging.server
            return True, "Server module imported successfully"
        except ImportError as e:
            return False, f"Failed to import server module: {e}"
        
    except Exception as e:
        return False, f"Error testing stdio server: {e}"