#!/usr/bin/env python3
"""
Company Tagging MCP Server with stdio transport.
FIXED VERSION: Provides category taxonomy resources and company tagging prompts with strict taxonomy enforcement.
"""

import os
import json
import logging
import asyncio
import csv
from typing import Optional, Dict, Any, List
from pathlib import Path

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    Prompt,
    PromptMessage,
    TextContent,
    LoggingLevel,
)
import mcp.types as types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("company_tagging_mcp")

# Initialize server
server = Server("company-tagging-mcp")

def load_csv_data() -> List[Dict[str, str]]:
    """Load and parse the CSV file."""
    csv_path = Path(__file__).parent / "categories" / "classes.csv"
    
    if not csv_path.exists():
        logger.warning(f"CSV file not found at {csv_path}")
        return []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            data = list(csv_reader)
            logger.info(f"Loaded {len(data)} rows from CSV file")
            return data
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        return []

def format_categories_for_analysis() -> str:
    """Format categories data for company analysis and tagging with STRICT taxonomy enforcement."""
    csv_data = load_csv_data()
    
    if not csv_data:
        return "No category data available."
    
    # Create the complete list of EXACT pairs
    exact_pairs = []
    for row in csv_data:
        industry = row.get('Industry', '').strip()
        product = row.get('Product', '').strip()
        show = row.get('Show', '').strip()
        
        if industry and product and show:
            exact_pairs.append(f"{industry} | {product}")
    
    # Remove duplicates and sort
    unique_pairs = sorted(list(set(exact_pairs)))
    
    # Format for analysis with VERY CLEAR instructions
    formatted_output = f"""COMPLETE TAXONOMY - EXACT Industry | Product Pairs (USE THESE EXACTLY):

CRITICAL: You MUST use these pairs EXACTLY as written. Do NOT create new pairs or modify existing ones.

Total available pairs: {len(unique_pairs)}

EXACT PAIRS TO USE (copy these exactly, including all punctuation, spacing, and special characters):

"""
    
    # Add each pair with clear formatting
    for i, pair in enumerate(unique_pairs, 1):
        formatted_output += f"{i:2d}. {pair}\n"
    
    formatted_output += f"""

MANDATORY RULES:
1. Use ONLY the {len(unique_pairs)} pairs listed above
2. Copy the industry and product names EXACTLY as shown (including punctuation like &, /, commas)
3. Do NOT create new industry or product names
4. Do NOT modify spelling, spacing, or characters
5. If a company doesn't fit any existing pair, leave those columns empty
6. Maximum 4 pairs per company

EXAMPLES OF CORRECT USAGE:
- "Platforms & Software | AI Applications" ✓
- "Cloud and AI Infrastructure Services | Network-as-a-service" ✓
- "Data, Development & Platforms Security | Data Encryption & Security" ✓

EXAMPLES OF INCORRECT USAGE (DO NOT DO THIS):
- "Platforms & Software | Freight Management" ✗ (not in taxonomy)
- "Platforms and Software | AI Applications" ✗ (changed & to "and")
- "Data Development Platforms Security | Data Encryption Security" ✗ (removed punctuation)
"""
    
    return formatted_output

@server.list_resources()
async def handle_list_resources() -> list[Resource]:
    """List available resources."""
    resources = [
        Resource(
            uri="categories://all",
            name="All Categories",
            description="Complete list of show categories with industries and products",
            mimeType="application/json",
        ),
        Resource(
            uri="categories://shows",
            name="Categories by Show",
            description="Categories organized by show with statistics",
            mimeType="application/json",
        ),
        Resource(
            uri="categories://industries",
            name="Categories by Industry",
            description="Categories organized by industry",
            mimeType="application/json",
        ),
        Resource(
            uri="categories://for-analysis",
            name="Categories for Analysis",
            description="Categories formatted for company analysis and tagging with strict enforcement",
            mimeType="text/plain",
        ),
        Resource(
            uri="categories://exact-pairs",
            name="Exact Industry Product Pairs",
            description="Complete list of exact industry|product pairs that MUST be used",
            mimeType="text/plain",
        ),
        Resource(
            uri="categories://system-prompt",
            name="System Prompt for Company Tagging",
            description="Detailed system prompt with strict instructions for company categorization",
            mimeType="text/plain",
        ),
    ]
    
    # Add dynamic show-specific resources
    csv_data = load_csv_data()
    shows = set(row.get('Show', '') for row in csv_data if row.get('Show'))
    
    for show in shows:
        resources.append(
            Resource(
                uri=f"categories://shows/{show}",
                name=f"{show} Categories",
                description=f"Categories for {show} show",
                mimeType="application/json",
            )
        )
    
    return resources

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read a specific resource."""
    csv_data = load_csv_data()
    
    if uri == "categories://all":
        return json.dumps({
            "total_categories": len(csv_data),
            "categories": csv_data,
            "description": "Complete list of show categories with industries and products"
        }, indent=2)
    
    elif uri == "categories://shows":
        shows = {}
        for row in csv_data:
            show = row.get('Show', '').strip()
            industry = row.get('Industry', '').strip()
            product = row.get('Product', '').strip()
            
            if show and industry and product:
                if show not in shows:
                    shows[show] = []
                shows[show].append({
                    'industry': industry,
                    'product': product
                })
        
        summary = {
            "total_shows": len(shows),
            "shows": {}
        }
        
        for show, categories in shows.items():
            summary["shows"][show] = {
                "total_categories": len(categories),
                "industries": list(set(cat['industry'] for cat in categories)),
                "categories": categories
            }
        
        return json.dumps(summary, indent=2)
    
    elif uri == "categories://industries":
        industries = {}
        for row in csv_data:
            industry = row.get('Industry', '').strip()
            product = row.get('Product', '').strip()
            show = row.get('Show', '').strip()
            
            if industry and product:
                if industry not in industries:
                    industries[industry] = []
                industries[industry].append({
                    'product': product,
                    'show': show
                })
        
        summary = {
            "total_industries": len(industries),
            "industries": {}
        }
        
        for industry, products in industries.items():
            summary["industries"][industry] = {
                "total_products": len(products),
                "shows": list(set(prod['show'] for prod in products if prod['show'])),
                "products": list(set(prod['product'] for prod in products))
            }
        
        return json.dumps(summary, indent=2)
    
    elif uri == "categories://for-analysis":
        return format_categories_for_analysis()
    
    elif uri == "categories://exact-pairs":
        # Return just the exact pairs in a simple format
        csv_data = load_csv_data()
        exact_pairs = []
        for row in csv_data:
            industry = row.get('Industry', '').strip()
            product = row.get('Product', '').strip()
            if industry and product:
                exact_pairs.append(f"{industry} | {product}")
        
        unique_pairs = sorted(list(set(exact_pairs)))
        return "\n".join(unique_pairs)
    
    elif uri == "categories://system-prompt":
        return """You are a data analyst tasked with tagging exhibitor companies with the most accurate tags from an industry and product taxonomy structure. Your goal is accuracy and consistency.

TAXONOMY CONTEXT:
The taxonomy consists of pairs of (Industry and Product) connected with show codes for companies that might exhibit at technology shows: Big Data and AI World (BDAIW), DevOps Live (DOL), Data Centre World (DCW), Cloud and Cyber Security Expo (CCSE), and CAI (Cloud and AI Infrastructure).

CRITICAL TAXONOMY ENFORCEMENT:
1. You MUST use search_show_categories tool FIRST to get the complete list of exact pairs
2. You can ONLY use pairs that exist in the taxonomy - NO EXCEPTIONS
3. Copy industry and product names EXACTLY (including punctuation, spacing, special characters)
4. If a company doesn't match any existing pair, leave those columns empty
5. Do NOT create new industry or product names

IMPORTANT INSTRUCTIONS:
1. Focus only on products/services relevant to the target shows
2. Use web sources (Google and Perplexity tools) to research company offerings
3. Name priority: Domain > Trading Name > Company Name
4. Select up to 4 pairs of (Industry and Product)
5. Use pairs EXACTLY as they appear - no modifications to spelling, spacing, or characters

PROCESS:
1. Use search_show_categories tool to retrieve all available exact pairs (MANDATORY FIRST STEP)
2. Choose research name (Domain > Trading Name > Company Name)
3. Research company using web tools to identify products/services
4. Check relevant shows from Event attribute
5. Match findings to EXISTING taxonomy pairs ONLY
6. Generate structured output table

OUTPUT FORMAT:
Markdown table with columns: Company Name, Trading Name, Tech Industry 1, Tech Product 1, Tech Industry 2, Tech Product 2, Tech Industry 3, Tech Product 3, Tech Industry 4, Tech Product 4.

Do not provide additional information or context beyond the requested table."""
    
    elif uri.startswith("categories://shows/"):
        show_name = uri.split("/")[-1].upper().strip()
        
        show_categories = []
        for row in csv_data:
            if row.get('Show', '').upper().strip() == show_name:
                show_categories.append({
                    'industry': row.get('Industry', '').strip(),
                    'product': row.get('Product', '').strip()
                })
        
        if show_categories:
            return json.dumps({
                "show": show_name,
                "total_categories": len(show_categories),
                "industries": list(set(cat['industry'] for cat in show_categories)),
                "categories": show_categories
            }, indent=2)
        else:
            available_shows = list(set(row.get('Show', '') for row in csv_data if row.get('Show')))
            return json.dumps({
                "error": f"Show '{show_name}' not found",
                "available_shows": available_shows
            }, indent=2)
    
    else:
        raise ValueError(f"Unknown resource URI: {uri}")

@server.list_prompts()
async def handle_list_prompts() -> list[Prompt]:
    """List available prompts."""
    return [
        Prompt(
            name="tag_companies",
            description="Professional data analyst prompt for systematic company categorization using EXACT taxonomy pairs",
            arguments=[
                {
                    "name": "company_data",
                    "description": "Company information to analyze (Company Name, Trading Name, Domain, Event)",
                    "required": True
                },
                {
                    "name": "target_shows",
                    "description": "Comma-separated show codes (e.g., 'CAI,BDAIW,DOL')",
                    "required": False
                }
            ]
        )
    ]

@server.get_prompt()
async def handle_get_prompt(name: str, arguments: Optional[Dict[str, str]] = None) -> types.GetPromptResult:
    """Handle prompt requests."""
    if name != "tag_companies":
        raise ValueError(f"Unknown prompt: {name}")
    
    if not arguments:
        arguments = {}
    
    company_data = arguments.get("company_data", "")
    target_shows = arguments.get("target_shows", "")
    
    # Get the taxonomy data with strict enforcement
    taxonomy_data = format_categories_for_analysis()
    
    # Create the comprehensive prompt with MUCH STRICTER language
    prompt_content = f"""You are a professional data analyst tasked with tagging exhibitor companies with accurate industry and product categories from our established taxonomy.

COMPANY DATA TO ANALYZE:
{company_data}

TARGET SHOWS: {target_shows if target_shows else "All relevant shows (CAI, DOL, CCSE, BDAIW, DCW)"}

MANDATORY RESEARCH PROCESS:

1. **Retrieve Complete Taxonomy** (ABSOLUTE REQUIREMENT):
   - Use search_show_categories tool without any filters to get ALL available exact pairs
   - You MUST see the complete list before proceeding - this is NON-NEGOTIABLE

2. **For EACH Company - Research Phase:**
   - Choose research name: Domain > Trading Name > Company Name
   - Use google-search tool: "site:[domain] products services" 
   - Use perplexity_search_web tool: "[company name] products services technology offerings"
   - Identify what the company actually sells/offers

3. **Analysis Phase - STRICT TAXONOMY MATCHING:**
   - Map company offerings to relevant shows (CAI, DOL, CCSE, BDAIW, DCW)
   - Match findings to EXACT taxonomy pairs from step 1 ONLY
   - NEVER create new industry or product names
   - NEVER modify existing industry or product names (not even small changes like & to "and")
   - If no exact match exists, leave those columns EMPTY
   - Select up to 4 exact pairs per company

4. **Output Requirements:**
   - Generate ONLY a markdown table with these columns:
   | Company Name | Trading Name | Tech Industry 1 | Tech Product 1 | Tech Industry 2 | Tech Product 2 | Tech Industry 3 | Tech Product 3 | Tech Industry 4 | Tech Product 4 |
   - Use ONLY the exact industry|product pairs from the taxonomy
   - Do NOT provide any additional text, explanations, or context
   - Do NOT show research details or tool executions
   - ONLY the markdown table

CRITICAL RULES - ABSOLUTE REQUIREMENTS:
- MUST use search_show_categories to get exact taxonomy pairs before starting
- MUST use both google-search AND perplexity_search_web for each company
- MUST use taxonomy pairs EXACTLY as written (copy/paste accuracy required)
- MUST NOT create new industry or product names under any circumstances
- MUST NOT modify existing names (including punctuation, spacing, capitalization)
- If no exact match exists, leave columns empty rather than inventing names
- Output ONLY the markdown table, nothing else

VALIDATION CHECK:
Before outputting any industry|product pair, verify it exists in the taxonomy retrieved in step 1.

Begin the systematic analysis now."""

    return types.GetPromptResult(
        description="Company tagging analysis prompt with STRICT taxonomy enforcement",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=prompt_content
                )
            )
        ]
    )

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_show_categories",
            description="Search and filter show categories from the CSV data - MUST be used to get exact taxonomy pairs",
            inputSchema={
                "type": "object",
                "properties": {
                    "show_name": {
                        "type": "string",
                        "description": "Filter by specific show (CAI, DOL, CCSE, BDAIW, DCW)",
                    },
                    "industry_filter": {
                        "type": "string",
                        "description": "Filter by industry name (partial match)",
                    },
                    "product_filter": {
                        "type": "string",
                        "description": "Filter by product name (partial match)",
                    },
                    "return_exact_pairs": {
                        "type": "boolean",
                        "description": "Return formatted exact pairs for taxonomy validation (default: true)",
                        "default": True
                    },
                },
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
    """Handle tool calls."""
    if not arguments:
        arguments = {}
    
    if name == "search_show_categories":
        return await search_show_categories(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def search_show_categories(arguments: dict[str, Any]) -> list[types.TextContent]:
    """Search and filter show categories from the CSV data with exact pairs formatting."""
    csv_data = load_csv_data()
    
    if not csv_data:
        return [types.TextContent(type="text", text="No category data available.")]
    
    show_name = arguments.get("show_name")
    industry_filter = arguments.get("industry_filter")
    product_filter = arguments.get("product_filter")
    return_exact_pairs = arguments.get("return_exact_pairs", True)
    
    filtered_data = csv_data.copy()
    filters_applied = []
    
    # Apply show filter
    if show_name:
        show_name_upper = show_name.upper().strip()
        filtered_data = [row for row in filtered_data 
                        if row.get('Show', '').upper().strip() == show_name_upper]
        filters_applied.append(f"Show: {show_name}")
    
    # Apply industry filter
    if industry_filter:
        industry_lower = industry_filter.lower().strip()
        filtered_data = [row for row in filtered_data 
                        if industry_lower in row.get('Industry', '').lower()]
        filters_applied.append(f"Industry contains: {industry_filter}")
    
    # Apply product filter
    if product_filter:
        product_lower = product_filter.lower().strip()
        filtered_data = [row for row in filtered_data 
                        if product_lower in row.get('Product', '').lower()]
        filters_applied.append(f"Product contains: {product_filter}")
    
    # Create exact pairs from filtered data
    exact_pairs = []
    for row in filtered_data:
        industry = row.get('Industry', '').strip()
        product = row.get('Product', '').strip()
        if industry and product:
            exact_pairs.append(f"{industry} | {product}")
    
    # Remove duplicates and sort
    unique_pairs = sorted(list(set(exact_pairs)))
    
    if return_exact_pairs:
        # Return formatted exact pairs for easy copy-paste validation
        result_text = f"""EXACT INDUSTRY | PRODUCT PAIRS ({len(unique_pairs)} total):

USE THESE PAIRS EXACTLY AS WRITTEN (copy/paste accuracy required):

"""
        for i, pair in enumerate(unique_pairs, 1):
            result_text += f"{i:2d}. {pair}\n"
        
        result_text += f"""

CRITICAL REMINDERS:
- Use ONLY the {len(unique_pairs)} pairs listed above
- Copy industry and product names EXACTLY (including all punctuation, spacing, special characters)
- Do NOT create new pairs or modify existing ones
- If no exact match exists, leave columns empty

Filters applied: {', '.join(filters_applied) if filters_applied else 'None (all categories)'}
"""
        
        return [types.TextContent(type="text", text=result_text)]
    
    else:
        # Return structured JSON format
        result = {
            "filters_applied": filters_applied,
            "total_matches": len(filtered_data),
            "unique_pairs": len(unique_pairs),
            "original_total": len(csv_data),
            "exact_pairs": unique_pairs,
            "raw_matches": filtered_data
        }
        
        if not filtered_data:
            result["message"] = "No categories match the specified filters."
            result["available_shows"] = list(set(row.get('Show', '') for row in csv_data if row.get('Show')))
        
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

async def main():
    """Main entry point for the server."""
    # Run the server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="company-tagging-mcp",
                server_version="0.3.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())