#!/usr/bin/env python3
"""
Cleanup Script for Neo4j Test Nodes

This script deletes all test nodes created by 4_compare_processor.py and 4_simple_test_runner.py
to ensure a clean database before running tests.
"""

import os
import sys
from neo4j import GraphDatabase
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("cleanup_neo4j.log")
    ]
)
logger = logging.getLogger(__name__)

def cleanup_neo4j_test_nodes():
    """Clean up all test nodes created by comparison scripts."""
    print("🧹 Cleaning up Neo4j test nodes...")
    
    # Load environment variables for Neo4j connection
    config_path = "config/config_vet.yaml"
    env_file = "keys/.env"  # Default path to .env file
    
    load_dotenv(env_file)
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not uri or not username or not password:
        print("❌ Missing Neo4j credentials for cleanup")
        logger.error("Missing Neo4j credentials")
        return
    
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        prefixes_to_delete = ["Old", "New", "OldTest", "NewTest"]
        
        with driver.session() as session:
            # First count how many nodes we'll delete
            count_query = """
                MATCH (n) 
                WHERE any(label in labels(n) 
                    WHERE toLower(label) STARTS WITH toLower('Old') 
                    OR toLower(label) STARTS WITH toLower('New'))
                RETURN count(n) as count
            """
            result = session.run(count_query)
            record = result.single()
            node_count = record["count"] if record else 0
            
            print(f"Found {node_count} test nodes to delete")
            
            # Now delete them
            delete_query = """
                MATCH (n) 
                WHERE any(label in labels(n) 
                    WHERE toLower(label) STARTS WITH toLower('Old') 
                    OR toLower(label) STARTS WITH toLower('New'))
                DETACH DELETE n
            """
            result = session.run(delete_query)
            
            # Verify they're all gone
            result = session.run(count_query)
            record = result.single()
            remaining = record["count"] if record else 0
            
            if remaining == 0:
                print(f"✅ Successfully deleted all {node_count} test nodes")
            else:
                print(f"⚠️ Deleted some nodes, but {remaining} still remain")
        
        driver.close()
        
    except Exception as e:
        print(f"❌ Error cleaning up Neo4j test nodes: {str(e)}")
        logger.error(f"Error cleaning up Neo4j test nodes: {str(e)}", exc_info=True)

if __name__ == "__main__":
    cleanup_neo4j_test_nodes()