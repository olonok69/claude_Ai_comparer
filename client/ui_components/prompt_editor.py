import streamlit as st
import json
from utils.ai_prompts import load_prompts_config, save_prompts_config

def create_prompt_editor():
    """Create a prompt editor component in the sidebar."""
    with st.sidebar:
        with st.expander("🔧 Prompt Configuration", expanded=False):
            st.markdown("**Edit AI Prompts and Guidelines**")
            
            # Load current configuration
            config = load_prompts_config()
            
            # Create tabs for different sections
            tab1, tab2, tab3 = st.tabs(["System", "Templates", "Errors"])
            
            with tab1:
                st.subheader("System Prompts")
                
                # Main system prompt
                main_prompt = st.text_area(
                    "Main System Prompt",
                    value=config.get("system_prompt", {}).get("main", ""),
                    height=150,
                    help="Core instructions for the AI assistant"
                )
                
                # Neo4j specific guidelines
                neo4j_prompt = st.text_area(
                    "Neo4j Guidelines",
                    value=config.get("system_prompt", {}).get("neo4j_specific", ""),
                    height=200,
                    help="Specific instructions for Neo4j database operations"
                )
                
                # HubSpot specific guidelines
                hubspot_prompt = st.text_area(
                    "HubSpot Guidelines",
                    value=config.get("system_prompt", {}).get("hubspot_specific", ""),
                    height=200,
                    help="Specific instructions for HubSpot CRM operations"
                )
            
            with tab2:
                st.subheader("User Prompt Templates")
                
                # Neo4j query template
                neo4j_template = st.text_area(
                    "Neo4j Query Template",
                    value=config.get("user_prompt_templates", {}).get("neo4j_query", ""),
                    height=100,
                    help="Template for Neo4j-related user queries. Use {user_text} placeholder."
                )
                
                # HubSpot query template
                hubspot_template = st.text_area(
                    "HubSpot Query Template",
                    value=config.get("user_prompt_templates", {}).get("hubspot_query", ""),
                    height=100,
                    help="Template for HubSpot-related user queries. Use {user_text} placeholder."
                )
                
                # General query template
                general_template = st.text_area(
                    "General Query Template",
                    value=config.get("user_prompt_templates", {}).get("general_query", ""),
                    height=100,
                    help="Template for general user queries. Use {user_text} placeholder."
                )
            
            with tab3:
                st.subheader("Error Messages")
                
                error_config = config.get("error_handling", {})
                
                # Schema missing error
                schema_error = st.text_input(
                    "Neo4j Schema Missing",
                    value=error_config.get("neo4j_schema_missing", ""),
                    help="Message when Neo4j schema is needed but not retrieved"
                )
                
                # Flexible search error
                flexible_error = st.text_input(
                    "Flexible Search Needed",
                    value=error_config.get("neo4j_flexible_search", ""),
                    help="Message when search terms need to be more flexible"
                )
                
                # HubSpot context missing
                context_error = st.text_input(
                    "HubSpot Context Missing",
                    value=error_config.get("hubspot_context_missing", ""),
                    help="Message when HubSpot user context is needed"
                )
                
                # Data not found
                not_found_error = st.text_area(
                    "Data Not Found",
                    value=error_config.get("data_not_found", ""),
                    height=80,
                    help="Message when query returns no results"
                )
            
            # Save and Reset buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save Changes", use_container_width=True):
                    # Build new configuration
                    new_config = {
                        "system_prompt": {
                            "main": main_prompt,
                            "neo4j_specific": neo4j_prompt,
                            "hubspot_specific": hubspot_prompt
                        },
                        "user_prompt_templates": {
                            "neo4j_query": neo4j_template,
                            "hubspot_query": hubspot_template,
                            "general_query": general_template
                        },
                        "error_handling": {
                            "neo4j_schema_missing": schema_error,
                            "neo4j_flexible_search": flexible_error,
                            "hubspot_context_missing": context_error,
                            "data_not_found": not_found_error
                        },
                        # Preserve other sections from original config
                        "response_templates": config.get("response_templates", {})
                    }
                    
                    if save_prompts_config(new_config):
                        st.success("✅ Prompts saved successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save prompts")
            
            with col2:
                if st.button("🔄 Reset to Default", use_container_width=True):
                    if st.session_state.get('confirm_reset'):
                        # Reset to default configuration
                        import os
                        config_path = os.path.join(os.path.dirname(__file__), '..', 'prompts_config.json')
                        try:
                            os.remove(config_path)
                            st.success("✅ Reset to defaults!")
                            st.session_state['confirm_reset'] = False
                            st.rerun()
                        except:
                            st.error("❌ Failed to reset")
                    else:
                        st.session_state['confirm_reset'] = True
                        st.warning("⚠️ Click again to confirm reset")

def display_current_prompts():
    """Display current prompts for debugging/viewing."""
    with st.expander("👁️ View Current Prompts", expanded=False):
        config = load_prompts_config()
        
        st.subheader("Current Configuration")
        st.json(config)
        
        # Show rendered system prompt
        from utils.ai_prompts import make_system_prompt
        st.subheader("Rendered System Prompt")
        st.text_area(
            "Full System Prompt",
            value=make_system_prompt(),
            height=300,
            disabled=True
        )