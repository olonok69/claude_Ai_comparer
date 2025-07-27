# Updated ui_components/sidebar_components.py

import streamlit as st
import os
from services.chat_service import create_chat, delete_chat, switch_chat


def create_sidebar_header_with_icon():
    """Create sidebar header with CSM icon and title at the very top."""
    # Check if CSM icon exists
    icon_path = os.path.join('.', 'icons', 'CSM.png')
    
    if os.path.exists(icon_path):
        # Create columns for better layout
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.image(icon_path, width=60)
        
        with col2:
            st.markdown("""
            <div style="padding-top: 10px;">
                <h3 style="margin: 0; color: #2F2E78;">Closer Still Media</h3>
                <p style="margin: 0; font-size: 12px; color: #666;">MCP Client</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Fallback if icon doesn't exist
        st.markdown("""
        <div class="icon-text-container">
            <span>🔍 CSM MCP Client</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Add separator
    st.markdown("---")


def categorize_tools_for_sidebar(tools):
    """Categorize tools by server type for sidebar display with improved detection."""
    google_search_tools = 0
    perplexity_tools = 0
    company_tagging_tools = 0
    other_tools = 0
    
    for tool in tools:
        tool_name = tool.name.lower()
        tool_desc = tool.description.lower() if hasattr(tool, 'description') and tool.description else ""
        
        # Company Tagging tool detection - improved patterns
        if any(keyword in tool_name for keyword in [
            'search_show_categories', 'company_tagging', 'tag_companies', 
            'categorize', 'taxonomy', 'show_categories'
        ]) or any(keyword in tool_desc for keyword in [
            'company', 'categoriz', 'taxonomy', 'tag', 'show', 'exhibitor'
        ]):
            company_tagging_tools += 1
        
        # Perplexity tool detection
        elif any(keyword in tool_name for keyword in [
            'perplexity_search_web', 'perplexity_advanced_search', 'perplexity'
        ]) or 'perplexity' in tool_desc:
            perplexity_tools += 1
        
        # Google Search tool detection
        elif any(keyword in tool_name for keyword in [
            'google-search', 'read-webpage', 'google_search', 'webpage'
        ]) or (('google' in tool_name or 'search' in tool_name or 'webpage' in tool_name) 
               and 'perplexity' not in tool_name):
            google_search_tools += 1
        
        else:
            other_tools += 1
    
    return google_search_tools, perplexity_tools, company_tagging_tools, other_tools


def create_history_chat_container():
    """Create the chat history container in the sidebar."""
    # Only show chat history if user is authenticated
    if not st.session_state.get("authentication_status"):
        return
    
    st.subheader("💬 Chat History")
    
    history_container = st.container(height=300, border=True)
    with history_container:
        chat_history_menu = [
                f"{chat['chat_name']}_::_{chat['chat_id']}"
                for chat in st.session_state["history_chats"]
            ]
        chat_history_menu = chat_history_menu[:50][::-1]
        
        if chat_history_menu:
            # Get current selection index
            current_selection = None
            for i, chat_option in enumerate(chat_history_menu):
                chat_id = chat_option.split("_::_")[1]
                if chat_id == st.session_state.get("current_chat_id"):
                    current_selection = i
                    break
            
            if current_selection is None:
                current_selection = 0
            
            selected_chat = st.radio(
                label="Select Chat",
                format_func=lambda x: x.split("_::_")[0] + '...' if "_::_" in x else x,
                options=chat_history_menu,
                label_visibility="collapsed",
                index=current_selection,
                key="chat_selector"
            )
            
            if selected_chat:
                selected_chat_id = selected_chat.split("_::_")[1]
                # Only switch if it's a different chat
                if selected_chat_id != st.session_state.get("current_chat_id"):
                    switch_chat(selected_chat_id)
                    st.rerun()


def create_sidebar_chat_buttons():
    """Create chat management buttons in the sidebar."""
    # Only show chat buttons if user is authenticated
    if not st.session_state.get("authentication_status"):
        return
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        create_chat_button = st.button(
            "🆕 New Chat", 
            use_container_width=True, 
            key="create_chat_button",
            help="Start a new conversation"
        )
        if create_chat_button:
            create_chat()
            st.rerun()

    with col2:
        delete_chat_button = st.button(
            "🗑️ Delete", 
            use_container_width=True, 
            key="delete_chat_button",
            help="Delete current conversation"
        )
        if delete_chat_button and st.session_state.get('current_chat_id'):
            delete_chat(st.session_state['current_chat_id'])
            st.rerun()

    # Quick stats
    st.markdown("---")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chats", len(st.session_state.get("history_chats", [])))
        with col2:
            current_messages = len(st.session_state.get("messages", []))
            st.metric("Messages", current_messages)

    # Enhanced status indicators
    st.markdown("---")
    st.markdown("**🔌 Connection Status:**")
    
    # Connection status
    if st.session_state.get("agent"):
        st.success("🟢 Connected to MCP")
    else:
        st.error("🔴 Not Connected")
    
    # Provider status  
    provider = st.session_state.get('params', {}).get('model_id', 'Not Set')
    st.info(f"🤖 Provider: {provider}")
    
    # Enhanced tools, prompts, and resources count
    tools_count = len(st.session_state.get('tools', []))
    prompts_count = len(st.session_state.get('prompts', []))
    resources_count = len(st.session_state.get('resources', []))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🧰 Tools", tools_count)
    with col2:
        st.metric("📝 Prompts", prompts_count)
    
    if resources_count > 0:
        st.metric("📁 Resources", resources_count)
    
    # Show detailed tool breakdown if tools are available
    if tools_count > 0:
        google_search_count, perplexity_count, company_tagging_count, other_count = categorize_tools_for_sidebar(st.session_state.get('tools', []))
        
        with st.container():
            st.markdown("**Tool Categories:**")
            
            # Create a compact display
            categories = []
            if google_search_count > 0:
                categories.append(f"🔍 Google: {google_search_count}")
            if perplexity_count > 0:
                categories.append(f"🔮 Perplexity: {perplexity_count}")
            if company_tagging_count > 0:
                categories.append(f"📊 Company: {company_tagging_count}")
            if other_count > 0:
                categories.append(f"🔧 Other: {other_count}")
            
            # Display categories in a compact format
            for category in categories:
                st.text(category)
    
    # Show prompts and resources breakdown if available
    if prompts_count > 0 or resources_count > 0:
        with st.container():
            st.markdown("**Additional Items:**")
            
            # Categorize prompts
            if prompts_count > 0:
                prompts = st.session_state.get('prompts', [])
                company_prompts = 0
                general_prompts = 0
                
                for prompt in prompts:
                    prompt_name = prompt.name.lower() if hasattr(prompt, 'name') else str(prompt).lower()
                    prompt_desc = prompt.description.lower() if hasattr(prompt, 'description') and prompt.description else ""
                    
                    if any(keyword in prompt_name for keyword in ['tag', 'company', 'categoriz', 'taxonomy', 'exhibitor']) or \
                       any(keyword in prompt_desc for keyword in ['company', 'categoriz', 'taxonomy', 'tag', 'exhibitor', 'trade show']):
                        company_prompts += 1
                    else:
                        general_prompts += 1
                
                if company_prompts > 0:
                    st.text(f"📝 Company Prompts: {company_prompts}")
                if general_prompts > 0:
                    st.text(f"📝 General Prompts: {general_prompts}")
            
            # Categorize resources
            if resources_count > 0:
                resources = st.session_state.get('resources', [])
                company_resources = 0
                general_resources = 0
                
                for resource in resources:
                    resource_name = resource.name.lower() if hasattr(resource, 'name') else str(resource).lower()
                    resource_desc = resource.description.lower() if hasattr(resource, 'description') and resource.description else ""
                    
                    if any(keyword in resource_name for keyword in ['categor', 'company', 'taxonomy', 'show', 'exhibitor', 'tag']) or \
                       any(keyword in resource_desc for keyword in ['company', 'categoriz', 'taxonomy', 'show', 'exhibitor', 'trade']):
                        company_resources += 1
                    else:
                        general_resources += 1
                
                if company_resources > 0:
                    st.text(f"📁 Company Resources: {company_resources}")
                if general_resources > 0:
                    st.text(f"📁 General Resources: {general_resources}")


def create_user_info_sidebar():
    """Create user information section in sidebar for authenticated users."""
    if st.session_state.get("authentication_status"):
        st.markdown("### 👤 User Information")
        
        with st.container(border=True):
            st.markdown(f"**Name:** {st.session_state.get('name', 'N/A')}")
            st.markdown(f"**Username:** {st.session_state.get('username', 'N/A')}")
            
            # Add session info
            import datetime
            if "login_time" not in st.session_state:
                st.session_state["login_time"] = datetime.datetime.now()
            
            session_time = datetime.datetime.now() - st.session_state["login_time"]
            hours, remainder = divmod(int(session_time.total_seconds()), 3600)
            minutes, _ = divmod(remainder, 60)
            
            st.markdown(f"**Session Time:** {hours:02d}:{minutes:02d}")
        
        st.markdown("---")


def create_complete_sidebar():
    """Create the complete sidebar with all components including CSM icon."""
    # Add the CSM icon header at the top
    create_sidebar_header_with_icon()
    
    # Show user info if authenticated
    create_user_info_sidebar()
    
    # Show chat history and controls if authenticated
    if st.session_state.get("authentication_status"):
        create_history_chat_container()
        create_sidebar_chat_buttons()


# Legacy functions maintained for backward compatibility but not used in new tab layout
def create_provider_select_widget():
    """Legacy function - moved to tab_components.py"""
    pass

def create_advanced_configuration_widget():
    """Legacy function - moved to tab_components.py"""
    pass

def create_mcp_connection_widget():
    """Legacy function - moved to tab_components.py"""
    pass

def create_mcp_tools_widget():
    """Legacy function - moved to tab_components.py"""
    pass