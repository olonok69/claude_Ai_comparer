# Updated AI prompts for Google Search and Perplexity MCP integration with server selection support

import streamlit as st

def get_available_tools_by_category():
    """Get available tools categorized by server type."""
    tools = st.session_state.get('tools', [])
    
    categorized = {
        "google": [],
        "perplexity": [],
        "company_tagging": []
    }
    
    for tool in tools:
        tool_name = tool.name.lower()
        tool_desc = tool.description.lower() if hasattr(tool, 'description') and tool.description else ""
        
        # Company Tagging tool detection
        if any(keyword in tool_name for keyword in [
            'search_show_categories', 'company_tagging', 'tag_companies', 
            'categorize', 'taxonomy', 'show_categories'
        ]) or any(keyword in tool_desc for keyword in [
            'company', 'categoriz', 'taxonomy', 'tag', 'show', 'exhibitor'
        ]):
            categorized["company_tagging"].append(tool)
        
        # Perplexity tool detection
        elif any(keyword in tool_name for keyword in [
            'perplexity_search_web', 'perplexity_advanced_search', 'perplexity'
        ]) or 'perplexity' in tool_desc:
            categorized["perplexity"].append(tool)
        
        # Google Search tool detection
        elif any(keyword in tool_name for keyword in [
            'google-search', 'read-webpage', 'google_search', 'webpage'
        ]) or (('google' in tool_name or 'search' in tool_name or 'webpage' in tool_name) 
               and 'perplexity' not in tool_name):
            categorized["google"].append(tool)
    
    return categorized

def make_system_prompt():
    """Create system prompt adapted to available servers."""
    available_tools = get_available_tools_by_category()
    
    # Build server status
    server_status = []
    if available_tools["google"]:
        server_status.append("Google Search operations (web search and webpage content extraction)")
    if available_tools["perplexity"]:
        server_status.append("Perplexity AI-powered search (intelligent web search with AI analysis)")
    if available_tools["company_tagging"]:
        server_status.append("Company categorization and taxonomy tools with STRICT taxonomy enforcement")
    
    if not server_status:
        server_status.append("No MCP servers connected")
    
    prompt = f"""
You are a helpful AI assistant with access to multiple search tools and specialized company tagging capabilities.

Your available tools include:
{chr(10).join(f"- {status}" for status in server_status)}

Core responsibilities:

1. **Understand the user's question** and provide helpful responses using your knowledge and available tools when needed.

2. **Use tools transparently** - Only use tools when you need current information from the web or when specifically asked to perform specialized tasks.

3. **For company tagging requests** - When a user asks to "tag companies" or mentions company categorization, use the specialized company tagging process with ABSOLUTE adherence to the existing taxonomy.

4. **Tool selection guidelines**:
   - Google Search tools: For finding specific information, extracting webpage content, comprehensive search results
   - Perplexity tools: For AI-powered research, intelligent analysis, current events
   - Company categorization tools: For accessing taxonomy data and when explicitly asked to tag/categorize companies

5. **CRITICAL for company tagging**: You MUST use ONLY the exact industry|product pairs that exist in the taxonomy. NEVER create new pairs or modify existing ones.

6. **Be direct and natural** - Don't over-explain tool usage unless specifically asked. Focus on providing the information the user needs.

**Important**: Only engage in the specialized company tagging workflow when users explicitly request company tagging, categorization, or mention analyzing companies for trade shows. When doing so, follow the strict taxonomy enforcement rules.
"""
    return prompt

def make_main_prompt(user_text):
    """Create a prompt adapted to available servers with strict taxonomy enforcement for company tagging."""
    
    # Get available tools
    available_tools = get_available_tools_by_category()
    
    # Check if this is a company tagging request
    company_tagging_keywords = [
        'tag companies', 'tag company', 'categorize companies', 'categorize company',
        'company tagging', 'company categorization', 'trade show categories',
        'exhibitor categories', 'taxonomy', 'industry product pairs', 'tag the following'
    ]
    
    is_company_tagging = any(keyword in user_text.lower() for keyword in company_tagging_keywords)
    
    if is_company_tagging:
        # Build research instructions based on available tools
        research_instructions = []
        
        if available_tools["google"]:
            research_instructions.append('   - Use google-search tool: If domain exists and is not empty, use "site:[domain] products services", otherwise use "[company name] products services technology"')
        
        if available_tools["perplexity"]:
            research_instructions.append('   - Use perplexity_search_web tool: "[company name] products services technology offerings"')
        
        if not research_instructions:
            research_instructions.append('   - No web search tools available - use available knowledge and company tagging tools only')
        
        # Build tool requirements
        tool_requirements = []
        if available_tools["google"] and available_tools["perplexity"]:
            tool_requirements.append("- MUST use both google-search AND perplexity_search_web for each company")
        elif available_tools["google"]:
            tool_requirements.append("- MUST use google-search tool for each company")
            tool_requirements.append("- **Google Search Strategy**: If domain exists and is not empty, use 'site:[domain] products services'. If no domain or empty domain, use '[company name] products services technology'")
        elif available_tools["perplexity"]:
            tool_requirements.append("- MUST use perplexity_search_web tool for each company")
        else:
            tool_requirements.append("- No web search tools available - proceed with taxonomy matching based on available information")
        
        # Build available tools status
        tools_status = []
        if available_tools["google"]:
            tools_status.append(f"✅ Google Search: {len(available_tools['google'])} tools available")
        else:
            tools_status.append("❌ Google Search: Not available")
        
        if available_tools["perplexity"]:
            tools_status.append(f"✅ Perplexity AI: {len(available_tools['perplexity'])} tools available")
        else:
            tools_status.append("❌ Perplexity AI: Not available")
        
        if available_tools["company_tagging"]:
            tools_status.append(f"✅ Company Tagging: {len(available_tools['company_tagging'])} tools available")
        else:
            tools_status.append("❌ Company Tagging: Not available")
        
        prompt = f"""
COMPANY TAGGING REQUEST DETECTED: {user_text}

AVAILABLE RESEARCH TOOLS:
{chr(10).join(tools_status)}

You must follow this EXACT process with ABSOLUTE adherence to the existing taxonomy:

MANDATORY STEP-BY-STEP PROCESS:

1. **FIRST AND MOST CRITICAL**: Use search_show_categories tool WITHOUT any filters to get the COMPLETE list of exact taxonomy pairs
   - This is MANDATORY before doing anything else
   - You MUST see all available pairs before proceeding

2. **Research Each Company** (if web search tools are available):
   - For EACH company in the data: Use available web search tools
   - Choose research name priority: Domain > Trading Name > Account Name
{chr(10).join(research_instructions)}
   - Identify what products/services the company actually offers

3. **STRICT Taxonomy Matching**:
   - Match company offerings ONLY to the EXACT pairs retrieved in step 1
   - Use industry and product names EXACTLY as they appear in the taxonomy
   - NEVER create new industry or product names
   - NEVER modify existing names (including punctuation like &, /, commas, spacing)
   - If no exact match exists, leave columns empty rather than inventing

4. **Final Output**:
   - Generate ONLY the markdown table with the specified columns
   - Use ONLY exact pairs from the taxonomy
   - Maximum 4 pairs per company
   - No explanations, context, or tool execution details

CRITICAL VALIDATION RULE:
Before using ANY industry|product pair in your output, verify it EXISTS in the taxonomy retrieved in step 1.

CRITICAL RULES FOR DIFFERENT SERVER CONFIGURATIONS:
{chr(10).join(tool_requirements)}
- MUST use search_show_categories to get complete exact taxonomy pairs before starting
- Use taxonomy pairs EXACTLY as written
- If no web search tools available, proceed with taxonomy matching based on company names and available information

EXAMPLES OF WHAT NOT TO DO (these are WRONG):
- "Platforms & Software | Freight Management" ✗ (not in taxonomy)
- "Platforms and Software | AI Applications" ✗ (changed & to "and")
- Creating any new industry or product names ✗

The user's company data: {user_text}

Begin with step 1: Use search_show_categories tool to get the complete exact taxonomy pairs.
"""
    else:
        # Build context about available tools for regular queries
        tool_context = []
        if available_tools["google"]:
            tool_context.append("Google Search tools for web research")
        if available_tools["perplexity"]:
            tool_context.append("Perplexity AI for intelligent web analysis")
        if available_tools["company_tagging"]:
            tool_context.append("Company taxonomy and categorization tools")
        
        if not tool_context:
            tool_context.append("No MCP tools currently available")
        
        prompt = f"""
User question: {user_text}

Available tools: {', '.join(tool_context)}

Please answer this question naturally. Use search tools if you need current information from the web, but focus on being helpful and direct in your response.
"""
    
    return prompt

def get_server_connection_status():
    """Get a summary of current server connections for debugging."""
    available_tools = get_available_tools_by_category()
    
    status = {
        "google_connected": len(available_tools["google"]) > 0,
        "perplexity_connected": len(available_tools["perplexity"]) > 0,
        "company_tagging_connected": len(available_tools["company_tagging"]) > 0,
        "total_tools": sum(len(tools) for tools in available_tools.values()),
        "tool_breakdown": {
            "google": len(available_tools["google"]),
            "perplexity": len(available_tools["perplexity"]),
            "company_tagging": len(available_tools["company_tagging"])
        }
    }
    
    return status

def validate_company_tagging_requirements():
    """Validate that required tools are available for company tagging."""
    available_tools = get_available_tools_by_category()
    
    # Company tagging always requires taxonomy access
    if not available_tools["company_tagging"]:
        return False, "Company Tagging MCP server not connected"
    
    # At least one search tool should be available for research
    if not available_tools["google"] and not available_tools["perplexity"]:
        return True, "Warning: No web search tools available. Company tagging will work with limited research capabilities."
    
    return True, "All required tools available for company tagging"

def get_research_strategy_description():
    """Get a description of the current research strategy based on available tools."""
    available_tools = get_available_tools_by_category()
    
    strategies = []
    
    if available_tools["google"]:
        strategies.append("Google Search: Domain-specific searches (site:domain.com) when domain available, otherwise company name searches")
    
    if available_tools["perplexity"]:
        strategies.append("Perplexity AI: Intelligent analysis of company products and services")
    
    if available_tools["company_tagging"]:
        strategies.append("Company Tagging: Access to complete taxonomy and categorization prompts")
    
    if not strategies:
        strategies.append("No MCP tools available - using AI knowledge only")
    
    return strategies