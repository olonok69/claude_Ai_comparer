import streamlit as st
from typing import Optional, Dict, Any
import json
import datetime

class SchemaMemoryManager:
    """Manages schema information in conversation memory."""
    
    @staticmethod
    def has_neo4j_schema_in_memory() -> bool:
        """Check if Neo4j schema is available in current conversation memory."""
        current_chat_id = st.session_state.get('current_chat_id')
        if not current_chat_id:
            return False
            
        # Check if schema was retrieved in this conversation
        messages = st.session_state.get('messages', [])
        for message in messages:
            if (message.get('role') == 'assistant' and 
                'tool' in message and 
                'get_neo4j_schema' in str(message.get('tool', ''))):
                return True
        return False
    
    @staticmethod  
    def has_hubspot_context_in_memory() -> bool:
        """Check if HubSpot user context is available in current conversation memory."""
        current_chat_id = st.session_state.get('current_chat_id')
        if not current_chat_id:
            return False
            
        # Check if user details were retrieved in this conversation
        messages = st.session_state.get('messages', [])
        for message in messages:
            if (message.get('role') == 'assistant' and 
                'tool' in message and 
                'hubspot-get-user-details' in str(message.get('tool', ''))):
                return True
        return False
    
    @staticmethod
    def get_schema_status_prompt() -> str:
        """Get prompt addition based on current schema memory status."""
        neo4j_schema = SchemaMemoryManager.has_neo4j_schema_in_memory()
        hubspot_context = SchemaMemoryManager.has_hubspot_context_in_memory()
        
        status_parts = []
        
        if neo4j_schema:
            status_parts.append("✅ **Neo4j Schema Available**: You have already retrieved the database schema in this conversation. Use that information to build queries.")
        else:
            status_parts.append("🔧 **Neo4j Schema Required**: For any Neo4j query, you must call `get_neo4j_schema()` first.")
            
        if hubspot_context:
            status_parts.append("✅ **HubSpot Context Available**: You have already retrieved user details in this conversation. Use that information for operations.")
        else:
            status_parts.append("🔧 **HubSpot Context Required**: For any HubSpot operation, you must call `hubspot-get-user-details()` first.")
        
        return "\n".join(status_parts)
    
    @staticmethod
    def extract_schema_from_memory() -> Optional[str]:
        """Extract Neo4j schema information from conversation memory."""
        messages = st.session_state.get('messages', [])
        for message in reversed(messages):  # Start from most recent
            if (message.get('role') == 'assistant' and 
                'tool' in message and 
                'get_neo4j_schema' in str(message.get('tool', ''))):
                # Extract the schema content from the tool message
                tool_content = str(message.get('tool', ''))
                return tool_content
        return None
    
    @staticmethod
    def extract_hubspot_context_from_memory() -> Optional[str]:
        """Extract HubSpot context information from conversation memory."""
        messages = st.session_state.get('messages', [])
        for message in reversed(messages):  # Start from most recent
            if (message.get('role') == 'assistant' and 
                'tool' in message and 
                'hubspot-get-user-details' in str(message.get('tool', ''))):
                # Extract the context content from the tool message
                tool_content = str(message.get('tool', ''))
                return tool_content
        return None
    
    @staticmethod
    def create_context_aware_prompt(user_text: str, base_prompt: str) -> str:
        """Create a context-aware prompt that includes memory status."""
        schema_status = SchemaMemoryManager.get_schema_status_prompt()
        
        # Add memory context to the prompt
        enhanced_prompt = f"{base_prompt}\n\n**CONVERSATION MEMORY STATUS:**\n{schema_status}\n\n"
        
        # Add specific schema/context if available
        neo4j_schema = SchemaMemoryManager.extract_schema_from_memory()
        if neo4j_schema and ('neo4j' in user_text.lower() or 'database' in user_text.lower() or 'visitor' in user_text.lower()):
            enhanced_prompt += f"**AVAILABLE NEO4J SCHEMA:**\n```\n{neo4j_schema[:1000]}...\n```\nUse this schema information to build your query.\n\n"
        
        hubspot_context = SchemaMemoryManager.extract_hubspot_context_from_memory()
        if hubspot_context and ('hubspot' in user_text.lower() or 'crm' in user_text.lower() or 'contact' in user_text.lower()):
            enhanced_prompt += f"**AVAILABLE HUBSPOT CONTEXT:**\n```\n{hubspot_context[:1000]}...\n```\nUse this context for your operations.\n\n"
        
        return enhanced_prompt

def enhance_prompt_with_memory(user_text: str, base_prompt: str) -> str:
    """Main function to enhance prompts with memory context."""
    return SchemaMemoryManager.create_context_aware_prompt(user_text, base_prompt)