#!/usr/bin/env python3
"""
Script to update company_cli.py with enhanced prompts
This updates the main CLI to include all improvements from the validator
"""

import re
import shutil
from datetime import datetime

def update_company_cli_prompts(file_path="company_cli.py"):
    """Update the prompts in company_cli.py to include tech context and search improvements."""
    
    # Create backup
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # New enhanced create_research_prompt method
    new_prompt_method = '''    def create_research_prompt(self, companies_batch: List[Dict]) -> str:
        """Create research prompt for the batch with enhanced server-specific strategy and tech context."""
        company_data = self.format_companies_for_analysis(companies_batch)
        
        # Determine available search tools
        has_google = len(self.available_tools["google"]) > 0
        has_perplexity = len(self.available_tools["perplexity"]) > 0
        
        # Build research instructions based on available tools
        research_instructions = []
        
        if has_google and has_perplexity:
            # Both available - use both for comprehensive research
            research_instructions.extend([
                "   - Use google-search tool: If domain exists and is not empty, use \\"site:[domain] technology platform software cloud AI data\\", otherwise use \\"[company name] technology software platform services\\"",
                "   - Use perplexity_advanced_search tool with recency=\\"year\\": \\"[company name] [Industry if available] [Product/Service Type if available] technology platform software services\\"",
                "   - Combine insights from both searches for comprehensive understanding"
            ])
            tool_requirements = "- MUST use BOTH google-search AND perplexity_advanced_search for each company"
        elif has_perplexity:
            # Only Perplexity available
            research_instructions.append("   - Use perplexity_advanced_search tool with recency=\\"year\\": \\"[company name] [Industry if available] [Product/Service Type if available] technology platform software services\\"")
            tool_requirements = "- MUST use perplexity_advanced_search for each company with recency=\\"year\\""
        elif has_google:
            # Only Google available
            research_instructions.append("   - Use google-search tool: If domain exists and is not empty, use \\"site:[domain] technology platform software cloud AI data\\", otherwise use \\"[company name] technology software platform services\\"")
            tool_requirements = "- MUST use google-search tool for each company"
        else:
            # No search tools available
            research_instructions.append("   - No web search tools available - use available knowledge and company tagging tools only")
            tool_requirements = "- No web search tools available - proceed with taxonomy matching based on available information"
        
        return f"""You are a professional data analyst tasked with tagging exhibitor companies with accurate industry and product categories from our established taxonomy.

IMPORTANT CONTEXT: All companies in this dataset are TECHNOLOGY COMPANIES that exhibit at tech trade shows. They all operate in the technology sector, even if their names might not immediately suggest it.

COMPANY DATA TO ANALYZE:
{company_data}

MANDATORY RESEARCH PROCESS:

1. **Retrieve Complete Taxonomy** (ONCE ONLY):
   - Use search_show_categories tool without any filters to get all available categories

2. **For EACH Company - Enhanced Research Phase:**
   - Remember: These are ALL tech companies, so look for their technology products/services
   - Choose research name priority: Domain > Trading Name > Account Name
   - If Industry and Product/Service Type are available in the input data, incorporate them into your search queries
{chr(10).join(research_instructions)}
   - Focus on their SOFTWARE, PLATFORMS, AI, DATA, CLOUD, or INFRASTRUCTURE offerings
   - Identify what the company actually sells/offers in the TECH SECTOR

3. **Analysis Phase:**
   - Map company offerings to relevant shows (CAI, DOL, CCSE, BDAIW, DCW)
   - Match findings to EXACT taxonomy pairs from step 1
   - Select up to 4 (Industry | Product) pairs per company
   - Use pairs EXACTLY as they appear
   - Absolutely mandatory to use the taxonomy pairs from step 1. DO NOT create new pairs or modify existing ones.

4. **Output Requirements:**
   - Generate a markdown table with ONLY DATA ROWS (NO HEADER)
   - Use this exact format for each company:
   | Company Name | Trading Name | Tech Industry 1 | Tech Product 1 | Tech Industry 2 | Tech Product 2 | Tech Industry 3 | Tech Product 3 | Tech Industry 4 | Tech Product 4 |
   - Do NOT include the header row in your response
   - Do NOT provide additional text, explanations, or context
   - ONLY the data rows without header

IMPORTANT CONTEXT FOR MATCHING:
- These companies exhibit at: Cloud and AI Infrastructure (CAI), DevOps Live (DOL), Data Centre World (DCW), Cloud and Cyber Security Expo (CCSE), Big Data and AI World (BDAIW)
- Look for their TECHNOLOGY offerings, not their general business
- A logistics company might offer "Platforms & Software | Big Data & Analytics Tools" for supply chain
- A finance company might offer "Platforms & Software | AI Applications" for fintech
- A retail company might offer "Platforms & Software | Cloud Security Solutions" for e-commerce

SEARCH STRATEGY:
- Always append tech keywords: "technology software platform cloud AI data"
- For domain searches: "site:[domain] technology platform software"
- For company searches: "[Company Name] technology software platform services"

CRITICAL RULES:
- MUST use search_show_categories to get taxonomy before starting
{tool_requirements}
- If Industry and Product/Service Type are available in input data, USE THEM to enhance search queries
- Use taxonomy pairs EXACTLY as written
- Output ONLY data rows (no header row)
- Each row should have company data only

Begin the systematic analysis now."""'''
    
    # Find and replace the create_research_prompt method
    pattern = r'def create_research_prompt\(self.*?\n(?=    def\s|\nclass\s|\Z)'
    
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_prompt_method + '\n', content, flags=re.DOTALL)
        print("✅ Updated create_research_prompt method")
    else:
        print("⚠️  Could not find create_research_prompt method")
        print("You may need to manually update it")
    
    # Write the updated content
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated {file_path} with enhanced prompts")
    print("   - Added tech company context")
    print("   - Added tech keywords to search queries")
    print("   - Enhanced matching guidance")
    
    return backup_path

if __name__ == "__main__":
    import sys
    
    file_path = sys.argv[1] if len(sys.argv) > 1 else "company_cli.py"
    
    try:
        backup = update_company_cli_prompts(file_path)
        print(f"\n✅ Success! Original backed up to: {backup}")
        print("\nThe main company_cli.py now includes:")
        print("1. Tech company context for all companies")
        print("2. Enhanced search queries with tech keywords")
        print("3. Better guidance for non-obvious tech companies")
        print("4. Same strict taxonomy enforcement")
        print("\nYour second chunk of 8,000 companies will now benefit from these improvements!")
    except Exception as e:
        print(f"❌ Error updating file: {e}")
        import traceback
        traceback.print_exc()