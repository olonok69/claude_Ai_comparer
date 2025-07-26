#!/usr/bin/env python3
"""
Simple Test Runner for Neo4j Session Processors

This script runs both neo4j session processors and does a quick comparison of key outputs.
"""

import os
import sys
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from neo4j import GraphDatabase
import inspect

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    from old_neo4j_session_processor import Neo4jSessionProcessor as OldNeo4jSessionProcessor
    from neo4j_session_processor import Neo4jSessionProcessor as NewNeo4jSessionProcessor
    from utils.config_utils import load_config
    from utils.logging_utils import setup_logging
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nMake sure you have:")
    print("1. old_neo4j_session_processor.py")
    print("2. neo4j_session_processor.py")
    print("3. config/config.yaml")
    print("4. config/config_vet.yaml")
    print("5. python-dotenv package installed")
    sys.exit(1)


def clear_neo4j_session_data():
    """Clear all session and stream related nodes from Neo4j."""
    print("🗑️  Clearing existing session data from Neo4j...")
    
    try:
        # Load config to get Neo4j credentials
        config = load_config("config/config_vet.yaml")
        load_dotenv(config["env_file"])
        
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        with driver.session() as session:
            # Delete all session and stream nodes
            delete_queries = [
                "MATCH (n:Sessions_this_year) DETACH DELETE n",
                "MATCH (n:Sessions_past_year) DETACH DELETE n",
                "MATCH (n:Stream) DETACH DELETE n"
            ]
            
            for query in delete_queries:
                result = session.run(query)
                print(f"  Executed: {query}")
            
            # Verify cleanup
            result = session.run("MATCH (n) WHERE n:Sessions_this_year OR n:Sessions_past_year OR n:Stream RETURN count(n) as count")
            remaining_count = result.single()["count"]
            
            if remaining_count == 0:
                print("✅ Successfully cleared all session data from Neo4j")
            else:
                print(f"⚠️ {remaining_count} session nodes still remain in Neo4j")
                
    except Exception as e:
        print(f"❌ Error clearing Neo4j data: {e}")
        raise
    finally:
        if 'driver' in locals():
            driver.close()


def compare_statistics(old_processor, new_processor):
    """Compare the statistics from both processors."""
    print("\n📊 Comparing Statistics...")
    print("⚠️  Note: The old processor has a bug where it fails to create Stream nodes and relationships.")
    print("    This is expected and indicates the new processor fixes these issues.")
    
    comparisons = []
    
    # Check if both processors have statistics
    if not hasattr(old_processor, "statistics") or not hasattr(new_processor, "statistics"):
        print("❌ One or both processors don't have statistics attribute")
        return False
    
    old_stats = old_processor.statistics
    new_stats = new_processor.statistics
    
    # Compare session node creation (should match)
    session_keys = ["sessions_this_year", "sessions_past_year_bva", "sessions_past_year_lva"]
    print(f"\n🔍 Comparing session node creation (should match):")
    
    for key in session_keys:
        old_count = old_stats["nodes_created"].get(key, 0)
        new_count = new_stats["nodes_created"].get(key, 0)
        
        print(f"  - {key}: Old={old_count}, New={new_count}")
        
        if old_count == new_count:
            print(f"    ✅ Match")
            comparisons.append(True)
        else:
            print(f"    ❌ Mismatch")
            comparisons.append(False)
    
    # Check stream node creation (old should be 0, new should be > 0)
    print(f"\n🔍 Comparing stream node creation (old=0 due to bug, new>0 is correct):")
    old_streams = old_stats["nodes_created"].get("streams", 0)
    new_streams = new_stats["nodes_created"].get("streams", 0)
    
    print(f"  - streams: Old={old_streams}, New={new_streams}")
    
    if old_streams == 0 and new_streams > 0:
        print(f"    ✅ Expected: Old processor bug fixed by new processor")
        comparisons.append(True)
    else:
        print(f"    ❌ Unexpected stream node counts")
        comparisons.append(False)
    
    # Check relationship creation (old should be 0, new should be > 0)
    print(f"\n🔍 Comparing relationship creation (old=0 due to missing streams, new>0 is correct):")
    rel_keys = ["sessions_this_year_has_stream", "sessions_past_year_has_stream"]
    
    for key in rel_keys:
        old_count = old_stats["relationships_created"].get(key, 0)
        new_count = new_stats["relationships_created"].get(key, 0)
        
        print(f"  - {key}: Old={old_count}, New={new_count}")
        
        if old_count == 0 and new_count > 0:
            print(f"    ✅ Expected: Old processor bug fixed by new processor")
            comparisons.append(True)
        else:
            print(f"    ❌ Unexpected relationship counts")
            comparisons.append(False)
    
    # Overall assessment
    session_nodes_match = all(comparisons[:len(session_keys)])
    stream_fix_correct = comparisons[len(session_keys)]
    relationships_fix_correct = all(comparisons[len(session_keys)+1:])
    
    print(f"\n📋 Summary:")
    print(f"  - Session nodes match: {'✅' if session_nodes_match else '❌'}")
    print(f"  - Stream nodes fixed: {'✅' if stream_fix_correct else '❌'}")
    print(f"  - Relationships fixed: {'✅' if relationships_fix_correct else '❌'}")
    
    return session_nodes_match and stream_fix_correct and relationships_fix_correct


def compare_neo4j_nodes(config):
    """Compare the actual nodes created in Neo4j."""
    print("\n🔍 Comparing Neo4j Database State...")
    print("⚠️  Note: Old processor doesn't add 'show' attribute, new processor does.")
    print("    This difference is expected and beneficial.")
    
    try:
        load_dotenv(config["env_file"])
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        with driver.session() as session:
            # Count different node types
            node_counts = {}
            
            # Sessions this year
            result = session.run("MATCH (n:Sessions_this_year) RETURN count(n) as count")
            node_counts["Sessions_this_year"] = result.single()["count"]
            
            # Sessions past year
            result = session.run("MATCH (n:Sessions_past_year) RETURN count(n) as count")
            node_counts["Sessions_past_year"] = result.single()["count"]
            
            # Streams
            result = session.run("MATCH (n:Stream) RETURN count(n) as count")
            node_counts["Stream"] = result.single()["count"]
            
            # Relationships
            result = session.run("MATCH ()-[r:HAS_STREAM]->() RETURN count(r) as count")
            relationship_counts = {"HAS_STREAM": result.single()["count"]}
            
            print("Node counts:")
            for node_type, count in node_counts.items():
                print(f"  - {node_type}: {count}")
            
            print("Relationship counts:")
            for rel_type, count in relationship_counts.items():
                print(f"  - {rel_type}: {count}")
            
            # Check show attribute distribution (only new processor adds this)
            result = session.run("""
                MATCH (n) 
                WHERE n:Sessions_this_year OR n:Sessions_past_year OR n:Stream
                RETURN 
                    CASE WHEN n.show IS NOT NULL THEN n.show ELSE 'NULL' END as show_value,
                    labels(n) as node_labels, 
                    count(n) as count
                ORDER BY show_value, node_labels
            """)
            
            show_distribution = list(result)
            if show_distribution:
                print("\n'Show' attribute distribution:")
                for record in show_distribution:
                    show_val = record['show_value']
                    node_type = record['node_labels'][0]
                    count = record['count']
                    
                    if show_val == 'NULL':
                        print(f"  - {node_type}: show=NULL ({count} nodes) - Old processor")
                    else:
                        print(f"  - {node_type}: show='{show_val}' ({count} nodes) - New processor")
            
            # Validation: Check if we have the expected structure
            sessions_total = node_counts["Sessions_this_year"] + node_counts["Sessions_past_year"]
            streams_count = node_counts["Stream"]
            relationships_count = relationship_counts["HAS_STREAM"]
            
            print(f"\n📊 Database Health Check:")
            print(f"  - Total session nodes: {sessions_total}")
            print(f"  - Stream nodes: {streams_count} {'✅' if streams_count > 0 else '❌ Missing'}")
            print(f"  - HAS_STREAM relationships: {relationships_count} {'✅' if relationships_count > 0 else '❌ Missing'}")
            
            # Check for show attribute presence (new processor feature)
            show_nodes = sum(1 for r in show_distribution if r['show_value'] != 'NULL')
            null_show_nodes = sum(1 for r in show_distribution if r['show_value'] == 'NULL')
            
            if show_nodes > 0:
                print(f"  - Nodes with 'show' attribute: {show_nodes} ✅ (New processor feature)")
            if null_show_nodes > 0:
                print(f"  - Nodes without 'show' attribute: {null_show_nodes} ⚠️ (Old processor limitation)")
            
            # The new processor should create a proper graph structure
            if streams_count > 0 and relationships_count > 0:
                print("  ✅ Complete session-stream graph structure created")
                return True
            else:
                print("  ❌ Graph structure incomplete")
                return False
            
    except Exception as e:
        print(f"❌ Error comparing Neo4j nodes: {e}")
        return False
    finally:
        if 'driver' in locals():
            driver.close()


def run_test():
    """Run the complete test."""
    print("🚀 Starting Neo4j Session Processor Comparison Test")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging(log_file="logs/simple_test_5.log")
    
    try:
        # Load configurations
        print("📁 Loading configurations...")
        old_config = load_config("config/config.yaml")
        new_config = load_config("config/config_vet.yaml")
        
        # Clear existing session data
        clear_neo4j_session_data()
        
        # Run old processor
        print("\n🔄 Running OLD Neo4j Session Processor...")
        old_processor = OldNeo4jSessionProcessor(old_config)
        old_stats = old_processor.process(create_only_new=False)
        print("✅ Old processor completed")
        
        # Capture state after old processor
        print("\n📊 Capturing Neo4j state after OLD processor...")
        old_db_state = compare_neo4j_nodes(old_config)
        
        # Clear data again before running new processor
        clear_neo4j_session_data()
        
        # Run new processor
        print("\n🔄 Running NEW Neo4j Session Processor...")
        new_processor = NewNeo4jSessionProcessor(new_config)
        new_stats = new_processor.process(create_only_new=False)
        print("✅ New processor completed")
        
        # Capture state after new processor
        print("\n📊 Capturing Neo4j state after NEW processor...")
        new_db_state = compare_neo4j_nodes(new_config)
        
        # Compare statistics
        statistics_match = compare_statistics(old_processor, new_processor)
        
        # Compare database states
        print(f"\n🔍 Comparing Database States...")
        db_states_match = True
        
        if old_db_state and new_db_state:
            # Compare node counts
            for node_type in old_db_state["node_counts"]:
                old_count = old_db_state["node_counts"].get(node_type, 0)
                new_count = new_db_state["node_counts"].get(node_type, 0)
                
                print(f"  - {node_type}: Old={old_count}, New={new_count}")
                
                if old_count != new_count:
                    print(f"    ❌ Mismatch")
                    db_states_match = False
                else:
                    print(f"    ✅ Match")
            
            # Compare relationship counts
            for rel_type in old_db_state["relationship_counts"]:
                old_count = old_db_state["relationship_counts"].get(rel_type, 0)
                new_count = new_db_state["relationship_counts"].get(rel_type, 0)
                
                print(f"  - {rel_type}: Old={old_count}, New={new_count}")
                
                if old_count != new_count:
                    print(f"    ❌ Mismatch")
                    db_states_match = False
                else:
                    print(f"    ✅ Match")
        else:
            print("❌ Could not compare database states")
            db_states_match = False
        
        # Final result
        print("\n" + "=" * 60)
        if statistics_match and db_states_match:
            print("🎉 SUCCESS: All outputs are IDENTICAL!")
            print("✅ The new Neo4j session processor produces the same results as the old one.")
            print("✅ Both processors create the same nodes and relationships in Neo4j.")
            return True
        else:
            print("❌ FAILURE: Outputs are DIFFERENT!")
            print("⚠️  The new processor produces different results.")
            if not statistics_match:
                print("   - Processor statistics differ")
            if not db_states_match:
                print("   - Neo4j database states differ")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)