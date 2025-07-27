import streamlit as st
from utils.ai_prompts import make_system_prompt, make_main_prompt

def show_prompt_debug():
    """Show debug information about prompts being used."""
    if st.session_state.get('show_prompt_debug', False):
        with st.expander("🔍 Prompt Debug Information", expanded=False):
            # Test prompt with a sample visitor query
            test_query = "give me the number of Visitors this year in neo4j"
            
            st.subheader("System Prompt")
            system_prompt = make_system_prompt()
            st.text_area("Current System Prompt", value=system_prompt, height=200, disabled=True)
            
            st.subheader("User Prompt (Test Query)")
            st.code(f'Test query: "{test_query}"')
            main_prompt = make_main_prompt(test_query)
            st.text_area("Generated Main Prompt", value=main_prompt, height=300, disabled=True)
            
            # Show keyword detection
            st.subheader("Keyword Detection")
            user_text_lower = test_query.lower()
            neo4j_keywords = [
                'neo4j', 'cypher', 'graph', 'node', 'relationship', 'database', 
                'schema', 'match', 'return', 'create', 'merge', 'delete',
                'visitors', 'visit', 'people', 'person', 'users', 'customer',
                'date', 'time', 'count', 'find all', 'show me', 'get all',
                'query', 'search in', 'database', 'data', 'how many', 'number of',
                'give me', 'tell me', 'show', 'list', 'find'
            ]
            
            detected_keywords = [kw for kw in neo4j_keywords if kw in user_text_lower]
            st.write(f"**Detected Neo4j Keywords:** {detected_keywords}")
            
            is_neo4j = any(keyword in user_text_lower for keyword in neo4j_keywords)
            st.write(f"**Classified as Neo4j Query:** {is_neo4j}")
            
            if not is_neo4j:
                st.error("❌ Query not being classified as Neo4j! This is why schema prompt isn't triggering.")
            else:
                st.success("✅ Query correctly classified as Neo4j")

def add_prompt_debug_toggle():
    """Add a toggle for prompt debugging."""
    if st.sidebar.checkbox("🔍 Show Prompt Debug", value=st.session_state.get('show_prompt_debug', False)):
        st.session_state['show_prompt_debug'] = True
    else:
        st.session_state['show_prompt_debug'] = False