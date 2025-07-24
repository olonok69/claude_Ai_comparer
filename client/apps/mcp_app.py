import os
import datetime
import streamlit as st
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from services.ai_service import get_response_stream
from services.mcp_service import run_agent, use_prompt
from services.chat_service import get_current_chat, _append_message_to_session, get_clean_conversation_memory
from utils.async_helpers import run_async
from utils.ai_prompts import make_system_prompt, make_main_prompt, get_server_connection_status, validate_company_tagging_requirements, get_research_strategy_description
import ui_components.sidebar_components as sd_components
from ui_components.main_components import display_tool_executions
from ui_components.tab_components import create_configuration_tab, create_connection_tab, create_tools_tab
from config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
import traceback


def check_authentication():
    """Check if user is authenticated and redirect if not."""
    if not st.session_state.get("authentication_status"):
        st.error("🔐 Authentication required. Please log in to access this application.")
        st.stop()


def is_company_tagging_request(user_text):
    """Check if the user request is for company tagging."""
    company_tagging_keywords = [
        'tag companies', 'tag company', 'categorize companies', 'categorize company',
        'company tagging', 'company categorization', 'trade show categories',
        'exhibitor categories', 'taxonomy', 'industry product pairs', 'tag the following'
    ]
    return any(keyword in user_text.lower() for keyword in company_tagging_keywords)


def extract_company_data_from_text(user_text):
    """Extract company data from user text for the prompt."""
    # Find the part of the text that contains company data
    lines = user_text.split('\n')
    company_data_lines = []
    
    for line in lines:
        # Look for lines that contain company information
        if any(keyword in line for keyword in ['CASEACCID', 'Account Name', 'Trading Name', 'Domain', 'Event']):
            company_data_lines.append(line.strip())
        elif line.strip() and '=' in line and any(company_data_lines):
            # Continue collecting if we're in a company data block
            company_data_lines.append(line.strip())
    
    return '\n'.join(company_data_lines) if company_data_lines else user_text


def display_server_status():
    """Display current server connection status."""
    server_status = get_server_connection_status()
    
    with st.container():
        st.markdown("### 🔧 Current Server Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if server_status["google_connected"]:
                st.success(f"🔍 Google Search: {server_status['tool_breakdown']['google']} tools")
            else:
                st.error("🔍 Google Search: Not connected")
        
        with col2:
            if server_status["perplexity_connected"]:
                st.success(f"🔮 Perplexity AI: {server_status['tool_breakdown']['perplexity']} tools")
            else:
                st.error("🔮 Perplexity AI: Not connected")
        
        with col3:
            if server_status["company_tagging_connected"]:
                st.success(f"📊 Company Tagging: {server_status['tool_breakdown']['company_tagging']} tools")
            else:
                st.error("📊 Company Tagging: Not connected")
        
        # Show research strategy
        if server_status["total_tools"] > 0:
            strategies = get_research_strategy_description()
            with st.expander("📋 Current Research Strategy", expanded=False):
                for strategy in strategies:
                    st.write(f"• {strategy}")
        else:
            st.warning("⚠️ No MCP servers connected. Please connect in the Connections tab.")


def display_company_tagging_requirements():
    """Display company tagging requirements and status."""
    is_valid, message = validate_company_tagging_requirements()
    
    if is_valid:
        if "Warning" in message:
            st.warning(f"⚠️ {message}")
        else:
            st.success(f"✅ {message}")
    else:
        st.error(f"❌ {message}")


def main():
    """Main application function with authentication check."""
    # Check authentication before proceeding
    check_authentication()
    
    # Initialize the title
    st.title("🔍 AI-Powered Search MCP Client - Multi-Engine Chat Interface")
    
    # Add user welcome message
    if st.session_state.get("name"):
        st.success(f"Welcome back, **{st.session_state['name']}**! 👋")
    
    # Create the main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "⚙️ Configuration", "🔌 Connections", "🧰 Tools"])
    
    # Sidebar with remaining components (CSM logo already added in app.py)
    with st.sidebar:
        # Add other sidebar components (logo is already at top from app.py)
        sd_components.create_user_info_sidebar()
        sd_components.create_history_chat_container()
        sd_components.create_sidebar_chat_buttons()

    # Chat Tab - Main conversation interface
    with tab1:
        # Check authentication again for this tab
        if not st.session_state.get("authentication_status"):
            st.warning("🔐 Please authenticate to access the chat interface.")
            return
        
        # Main chat interface
        st.header("💬 Chat with AI-Powered Search Agents")
        
        # Display current server status
        display_server_status()
        
        # Special section for company tagging
        server_status = get_server_connection_status()
        if server_status["company_tagging_connected"]:
            with st.expander("📊 Company Tagging Workflow", expanded=False):
                st.markdown("**Specialized workflow for company categorization:**")
                st.markdown("- Use phrases like 'tag companies' or 'categorize companies' to activate")
                st.markdown("- Provides systematic research and taxonomy-based categorization")
                st.markdown("- Supports multiple company analysis in a single request")
                
                display_company_tagging_requirements()
                
                # Show example usage
                st.markdown("**Example usage:**")
                st.code("""
Account Name = Microsoft Corporation
Trading Name = Microsoft
Domain = microsoft.com
Event = Cloud and AI Infrastructure

Account Name = Amazon Web Services
Trading Name = AWS
Domain = aws.amazon.com
Event = Cloud and AI Infrastructure

Tag these companies with the appropriate taxonomy categories.
                """, language="text")
        
        st.markdown("Ask questions to search the web using available MCP servers for comprehensive research and analysis.")
        
        messages_container = st.container(border=True, height=600)
        
        # Re-render previous messages
        if st.session_state.get('current_chat_id'):
            st.session_state["messages"] = get_current_chat(st.session_state['current_chat_id'])
            for m in st.session_state["messages"]:
                with messages_container.chat_message(m["role"]):
                    if "tool" in m and m["tool"]:
                        st.code(m["tool"], language='yaml')
                    if "content" in m and m["content"]:
                        st.markdown(m["content"])

        # Chat input
        user_text = st.chat_input("Ask a question, search the web, or tag companies using the specialized workflow")

        # Handle chat logic
        if user_text is not None:  # something submitted
            params = st.session_state.get('params')
            
            # Check if we have proper credentials loaded from environment
            selected_provider = params.get('model_id')
            credentials_valid = False
            
            if selected_provider == "OpenAI":
                credentials_valid = bool(os.getenv("OPENAI_API_KEY"))
            elif selected_provider == "Azure OpenAI":
                azure_keys = [
                    os.getenv("AZURE_API_KEY"),
                    os.getenv("AZURE_ENDPOINT"),
                    os.getenv("AZURE_DEPLOYMENT"),
                    os.getenv("AZURE_API_VERSION")
                ]
                credentials_valid = all(azure_keys)
            
            if not credentials_valid:
                err_mesg = f"❌ Missing credentials for {selected_provider}. Please check your .env file and configure in the Configuration tab."
                _append_message_to_session({"role": "assistant", "content": err_mesg})
                with messages_container.chat_message("assistant"):
                    st.markdown(err_mesg)
                st.rerun()

            # Handle question (if any text)
            if user_text:
                user_text_dct = {"role": "user", "content": user_text}
                _append_message_to_session(user_text_dct)
                with messages_container.chat_message("user"):
                    st.markdown(user_text)

                with st.spinner("Thinking…", show_time=True):
                    # Check if this is a company tagging request
                    if is_company_tagging_request(user_text):
                        # Validate company tagging requirements
                        is_valid, validation_message = validate_company_tagging_requirements()
                        
                        if not is_valid:
                            error_msg = f"❌ Company tagging not available: {validation_message}"
                            with messages_container.chat_message("assistant"):
                                st.markdown(error_msg)
                            _append_message_to_session({"role": "assistant", "content": error_msg})
                            st.rerun()
                            return
                        
                        # Handle company tagging with server-aware prompt
                        try:
                            if st.session_state.get("agent"):
                                # Use the updated prompt system that adapts to available servers
                                system_prompt = make_system_prompt()
                                main_prompt = make_main_prompt(user_text)
                                
                                # Create conversation memory for the agent
                                conversation_memory = get_clean_conversation_memory()
                                
                                # Add the company tagging prompt as the user message
                                conversation_memory.append(HumanMessage(content=main_prompt))
                                
                                # Run the agent with the specialized prompt
                                response = run_async(run_agent(st.session_state.agent, conversation_memory))
                                
                                # Extract and display the response (look for final markdown table)
                                assistant_response = None
                                if "messages" in response:
                                    for msg in response["messages"]:
                                        if isinstance(msg, AIMessage) and hasattr(msg, "content") and msg.content:
                                            assistant_response = str(msg.content)
                                            # Look for markdown table in the response
                                            if "|" in assistant_response and "Company Name" in assistant_response:
                                                # Extract just the table part
                                                lines = assistant_response.split('\n')
                                                table_lines = []
                                                in_table = False
                                                for line in lines:
                                                    if "|" in line and ("Company Name" in line or "Tech Industry" in line):
                                                        in_table = True
                                                    if in_table and "|" in line:
                                                        table_lines.append(line.strip())
                                                    elif in_table and not line.strip():
                                                        # Empty line might end the table
                                                        continue
                                                    elif in_table and "|" not in line.strip() and line.strip():
                                                        # Non-empty line without | ends the table
                                                        break
                                                
                                                if table_lines:
                                                    assistant_response = '\n'.join(table_lines)
                                            
                                            with messages_container.chat_message("assistant"):
                                                st.markdown(assistant_response)
                                            break
                                
                                if assistant_response:
                                    response_dct = {"role": "assistant", "content": assistant_response}
                                    _append_message_to_session(response_dct)
                                else:
                                    error_msg = "❌ Could not generate company tagging results. Please check the company data format and ensure required tools are available."
                                    with messages_container.chat_message("assistant"):
                                        st.markdown(error_msg)
                                    _append_message_to_session({"role": "assistant", "content": error_msg})
                            else:
                                error_msg = "❌ Company tagging requires connection to MCP servers. Please connect in the Connections tab."
                                with messages_container.chat_message("assistant"):
                                    st.markdown(error_msg)
                                _append_message_to_session({"role": "assistant", "content": error_msg})
                                
                        except Exception as e:
                            error_msg = f"❌ Error in company tagging: {str(e)}"
                            with messages_container.chat_message("assistant"):
                                st.markdown(error_msg)
                            _append_message_to_session({"role": "assistant", "content": error_msg})
                    
                    else:
                        # Handle normal conversation with server-aware prompts
                        system_prompt = make_system_prompt()
                        main_prompt = make_main_prompt(user_text)
                        
                        try:
                            # If agent is available, use it
                            if st.session_state.agent:
                                # Create conversation memory for the agent (clean version without tool messages)
                                conversation_memory = get_clean_conversation_memory()
                                
                                # Add the current user message to the conversation
                                conversation_memory.append(HumanMessage(content=user_text))
                                
                                try:
                                    # Run the agent with full conversation context
                                    response = run_async(run_agent(st.session_state.agent, conversation_memory))
                                    
                                    tool_output = None
                                    # Extract tool executions if available
                                    if "messages" in response:
                                        for msg in response["messages"]:
                                            # Look for AIMessage with tool calls
                                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                                for tool_call in msg.tool_calls:
                                                    # Find corresponding ToolMessage
                                                    tool_output = next(
                                                        (m.content for m in response["messages"] 
                                                            if isinstance(m, ToolMessage) and 
                                                            hasattr(m, 'tool_call_id') and
                                                            m.tool_call_id == tool_call['id']),
                                                        None
                                                    )
                                                    if tool_output:
                                                        st.session_state.tool_executions.append({
                                                            "tool_name": tool_call['name'],
                                                            "input": tool_call['args'],
                                                            "output": tool_output,
                                                            "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                            "user": st.session_state.get('username', 'Unknown')
                                                        })
                                    
                                    # Extract and display the response
                                    output = ""
                                    tool_count = 0
                                    assistant_response = None
                                    
                                    if "messages" in response:
                                        # Handle case where response contains error message
                                        if len(response["messages"]) == 1 and isinstance(response["messages"][0], dict):
                                            # This is likely our custom error response
                                            assistant_response = response["messages"][0].get("content", "")
                                            with messages_container.chat_message("assistant"):
                                                st.markdown(assistant_response)
                                        else:
                                            # Skip messages that were part of the input conversation
                                            new_messages = response["messages"][len(conversation_memory):]
                                            
                                            for msg in new_messages:
                                                if isinstance(msg, HumanMessage):
                                                    continue  # Skip human messages
                                                elif isinstance(msg, ToolMessage):
                                                    tool_count += 1
                                                    with messages_container.chat_message("assistant"):
                                                        tool_message = f"**ToolMessage - {tool_count} ({getattr(msg, 'name', 'Unknown')}):** \n" + str(msg.content)
                                                        st.code(tool_message, language='yaml')
                                                        # Store tool message separately - don't include in main conversation flow
                                                        _append_message_to_session({'role': 'assistant', 'tool': tool_message})
                                                elif isinstance(msg, AIMessage):
                                                    if hasattr(msg, "content") and msg.content:
                                                        assistant_response = str(msg.content)
                                                        output = assistant_response
                                                        with messages_container.chat_message("assistant"):
                                                            st.markdown(output)
                                    
                                    # Only add the final AI response to conversation memory
                                    if assistant_response:
                                        response_dct = {"role": "assistant", "content": assistant_response}
                                        _append_message_to_session(response_dct)
                                        
                                except Exception as agent_error:
                                    # Handle agent-specific errors
                                    error_message = f"⚠️ Agent execution error: {str(agent_error)}"
                                    if "recursion limit" in str(agent_error).lower():
                                        error_message = """⚠️ The agent encountered a complex task that couldn't be completed in the available steps. 

        **Possible solutions:**
        1. **Break down your request** into smaller, more specific questions
        2. **Be more specific** about what you want to achieve
        3. **Check MCP server connections** - try disconnecting and reconnecting in the Connections tab
        4. **Simplify the query** - avoid very complex multi-step requests

        **What happened:** The agent tried too many steps without reaching a conclusion, which usually means it got stuck in a loop or encountered tool execution issues."""
                                    
                                    with messages_container.chat_message("assistant"):
                                        st.markdown(error_message)
                                    _append_message_to_session({"role": "assistant", "content": error_message})
                                    
                            # Fall back to regular stream response if agent not available
                            else:
                                server_status = get_server_connection_status()
                                if server_status["total_tools"] > 0:
                                    connected_servers = []
                                    if server_status["google_connected"]:
                                        connected_servers.append("Google Search")
                                    if server_status["perplexity_connected"]:
                                        connected_servers.append("Perplexity AI")
                                    if server_status["company_tagging_connected"]:
                                        connected_servers.append("Company Tagging")
                                    
                                    server_list = ", ".join(connected_servers)
                                    st.warning(f"⚠️ You are connected to {server_list} but the agent is not initialized! Please reconnect in the Connections tab.")
                                else:
                                    st.warning("⚠️ No MCP servers connected! Please connect in the Connections tab.")
                                
                                response_stream = get_response_stream(
                                    main_prompt,
                                    llm_provider=st.session_state['params']['model_id'],
                                    system=system_prompt,
                                    temperature=st.session_state['params'].get('temperature', DEFAULT_TEMPERATURE),
                                    max_tokens=st.session_state['params'].get('max_tokens', DEFAULT_MAX_TOKENS), 
                                )         
                                with messages_container.chat_message("assistant"):
                                    response = st.write_stream(response_stream)
                                    response_dct = {"role": "assistant", "content": response}
                                    _append_message_to_session(response_dct)
                                    
                        except Exception as e:
                            response = f"⚠️ Something went wrong: {str(e)}"
                            st.error(response)
                            st.code(traceback.format_exc(), language="python")
                            _append_message_to_session({"role": "assistant", "content": response})
                            st.stop()
        
        # Display tool executions in the chat tab
        display_tool_executions()

    # Configuration Tab
    with tab2:
        check_authentication()  # Additional check for sensitive configurations
        create_configuration_tab()

    # Connections Tab  
    with tab3:
        check_authentication()  # Additional check for connections
        create_connection_tab()

    # Tools Tab
    with tab4:
        check_authentication()  # Additional check for tools
        create_tools_tab()