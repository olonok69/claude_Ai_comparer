#!/usr/bin/env python3
"""
Fixed Neo4j Session Processor Comparison Tool

This script compares old and new Neo4j session processors and expects identical results
since both processors should work correctly with the same logic.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    from old_neo4j_session_processor import Neo4jSessionProcessor as OldNeo4jSessionProcessor
    from neo4j_session_processor import Neo4jSessionProcessor as NewNeo4jSessionProcessor
    from utils.config_utils import load_config
    from utils.logging_utils import setup_logging
    from neo4j import GraphDatabase
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nMake sure you have all required dependencies installed")
    sys.exit(1)


class Neo4jSessionProcessorComparison:
    """Compare old and new Neo4j Session Processors expecting identical results."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def clear_neo4j_data(self, config: Dict[str, Any]) -> bool:
        """Clear Neo4j session and stream data."""
        try:
            load_dotenv(config["env_file"])
            uri = os.getenv("NEO4J_URI")
            username = os.getenv("NEO4J_USERNAME")
            password = os.getenv("NEO4J_PASSWORD")
            
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            with driver.session() as session:
                delete_query = "MATCH (n) WHERE n:Sessions_this_year OR n:Sessions_past_year OR n:Stream DETACH DELETE n"
                session.run(delete_query)
                
                # Verify cleanup
                result = session.run("MATCH (n) WHERE n:Sessions_this_year OR n:Sessions_past_year OR n:Stream RETURN count(n) as count")
                remaining = result.single()["count"]
                
                if remaining == 0:
                    self.logger.info("Successfully cleared Neo4j session data")
                    return True
                else:
                    self.logger.warning(f"{remaining} session nodes still remain")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error clearing Neo4j data: {e}")
            return False
        finally:
            if 'driver' in locals():
                driver.close()
    
    def compare_processor_statistics(self, old_stats: Dict, new_stats: Dict) -> Dict[str, Any]:
        """Compare statistics from both processors expecting identical results."""
        self.logger.info("Comparing processor statistics...")
        self.logger.info("Both processors should produce identical results")
        
        comparison = {
            "nodes_created": {},
            "nodes_skipped": {},
            "relationships_created": {},
            "relationships_skipped": {},
            "totals": {},
            "all_identical": True
        }
        
        # Compare session node creation (should match exactly)
        session_keys = ["sessions_this_year", "sessions_past_year_bva", "sessions_past_year_lva"]
        
        for key in session_keys:
            old_val = old_stats.get("nodes_created", {}).get(key, 0)
            new_val = new_stats.get("nodes_created", {}).get(key, 0)
            
            comparison["nodes_created"][key] = {
                "old": old_val,
                "new": new_val,
                "match": old_val == new_val,
                "difference": new_val - old_val,
                "expected": "Should match exactly"
            }
            
            if old_val != new_val:
                comparison["all_identical"] = False
        
        # Compare stream creation (should match exactly)
        old_streams = old_stats.get("nodes_created", {}).get("streams", 0)
        new_streams = new_stats.get("nodes_created", {}).get("streams", 0)
        
        comparison["nodes_created"]["streams"] = {
            "old": old_streams,
            "new": new_streams,
            "match": old_streams == new_streams,
            "difference": new_streams - old_streams,
            "expected": "Should match exactly"
        }
        
        if old_streams != new_streams:
            comparison["all_identical"] = False
        
        # Compare relationship creation (should match exactly)
        rel_keys = ["sessions_this_year_has_stream", "sessions_past_year_has_stream"]
        
        for key in rel_keys:
            old_val = old_stats.get("relationships_created", {}).get(key, 0)
            new_val = new_stats.get("relationships_created", {}).get(key, 0)
            
            comparison["relationships_created"][key] = {
                "old": old_val,
                "new": new_val,
                "match": old_val == new_val,
                "difference": new_val - old_val,
                "expected": "Should match exactly"
            }
            
            if old_val != new_val:
                comparison["all_identical"] = False
        
        # Calculate totals
        old_total_nodes = sum(old_stats.get("nodes_created", {}).values())
        new_total_nodes = sum(new_stats.get("nodes_created", {}).values())
        old_total_rels = sum(old_stats.get("relationships_created", {}).values())
        new_total_rels = sum(new_stats.get("relationships_created", {}).values())
        
        comparison["totals"] = {
            "nodes": {
                "old": old_total_nodes,
                "new": new_total_nodes,
                "match": old_total_nodes == new_total_nodes,
                "difference": new_total_nodes - old_total_nodes
            },
            "relationships": {
                "old": old_total_rels,
                "new": new_total_rels,
                "match": old_total_rels == new_total_rels,
                "difference": new_total_rels - old_total_rels
            }
        }
        
        if not comparison["totals"]["nodes"]["match"] or not comparison["totals"]["relationships"]["match"]:
            comparison["all_identical"] = False
        
        return comparison
    
    def compare_neo4j_state(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Compare the actual state of Neo4j database."""
        try:
            load_dotenv(config["env_file"])
            uri = os.getenv("NEO4J_URI")
            username = os.getenv("NEO4J_USERNAME")
            password = os.getenv("NEO4J_PASSWORD")
            
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            with driver.session() as session:
                # Count nodes
                node_counts = {}
                node_counts["Sessions_this_year"] = session.run("MATCH (n:Sessions_this_year) RETURN count(n) as count").single()["count"]
                node_counts["Sessions_past_year"] = session.run("MATCH (n:Sessions_past_year) RETURN count(n) as count").single()["count"]
                node_counts["Stream"] = session.run("MATCH (n:Stream) RETURN count(n) as count").single()["count"]
                
                # Count relationships
                relationship_counts = {}
                relationship_counts["HAS_STREAM"] = session.run("MATCH ()-[r:HAS_STREAM]->() RETURN count(r) as count").single()["count"]
                
                # Check show attributes
                show_result = session.run("""
                    MATCH (n) 
                    WHERE n:Sessions_this_year OR n:Sessions_past_year OR n:Stream
                    RETURN 
                        CASE WHEN n.show IS NOT NULL THEN 'HAS_SHOW' ELSE 'NO_SHOW' END as show_status,
                        count(n) as count
                """)
                
                show_attributes = {record["show_status"]: record["count"] for record in show_result}
                
                return {
                    "node_counts": node_counts,
                    "relationship_counts": relationship_counts,
                    "show_attributes": show_attributes,
                    "total_nodes": sum(node_counts.values()),
                    "total_relationships": sum(relationship_counts.values())
                }
                
        except Exception as e:
            self.logger.error(f"Error comparing Neo4j state: {e}")
            return None
        finally:
            if 'driver' in locals():
                driver.close()
    
    def generate_comparison_report(self, stats_comparison: Dict, old_db_state: Dict, new_db_state: Dict) -> str:
        """Generate a detailed comparison report (FIXED: Uses ASCII characters for Windows compatibility)."""
        
        # Use ASCII characters instead of Unicode to avoid encoding issues
        check_mark = "[OK]"
        cross_mark = "[FAIL]"
        success_mark = "[SUCCESS]"
        
        report = "# Neo4j Session Processor Comparison Report\n\n"
        
        # Executive Summary
        report += "## Executive Summary\n"
        
        if stats_comparison.get("all_identical", False):
            report += f"{success_mark} **SUCCESS**: Both processors produce identical results!\n"
            report += f"{check_mark} The new processor is a perfect replacement for the old processor.\n\n"
        else:
            report += f"{cross_mark} **FAILURE**: Processors produce different results!\n"
            report += "⚠️ The new processor needs further fixes to match the old processor.\n\n"
        
        # Statistics Comparison
        report += "## Statistics Comparison\n\n"
        
        # Session nodes
        report += "### Session Node Creation\n"
        for key, data in stats_comparison["nodes_created"].items():
            if key.startswith("sessions_"):
                status = f"{check_mark} MATCH" if data["match"] else f"{cross_mark} MISMATCH"
                report += f"- **{key}**: Old={data['old']}, New={data['new']} - {status}\n"
        
        # Stream nodes
        stream_data = stats_comparison["nodes_created"]["streams"]
        status = f"{check_mark} MATCH" if stream_data["match"] else f"{cross_mark} MISMATCH"
        report += f"- **streams**: Old={stream_data['old']}, New={stream_data['new']} - {status}\n\n"
        
        # Relationships
        report += "### Relationship Creation\n"
        for key, data in stats_comparison["relationships_created"].items():
            status = f"{check_mark} MATCH" if data["match"] else f"{cross_mark} MISMATCH"
            report += f"- **{key}**: Old={data['old']}, New={data['new']} - {status}\n"
        
        # Database State Comparison
        report += "\n## Database State Comparison\n\n"
        
        if old_db_state and new_db_state:
            report += "### Node Counts\n"
            for node_type in old_db_state["node_counts"]:
                old_count = old_db_state["node_counts"][node_type]
                new_count = new_db_state["node_counts"][node_type]
                status = f"{check_mark} MATCH" if old_count == new_count else f"{cross_mark} MISMATCH"
                report += f"- **{node_type}**: Old={old_count}, New={new_count} - {status}\n"
            
            report += "\n### Relationship Counts\n"
            for rel_type in old_db_state["relationship_counts"]:
                old_count = old_db_state["relationship_counts"][rel_type]
                new_count = new_db_state["relationship_counts"][rel_type]
                status = f"{check_mark} MATCH" if old_count == new_count else f"{cross_mark} MISMATCH"
                report += f"- **{rel_type}**: Old={old_count}, New={new_count} - {status}\n"
        
        # Totals
        report += "\n## Totals\n"
        totals = stats_comparison["totals"]
        
        nodes_status = f"{check_mark} MATCH" if totals["nodes"]["match"] else f"{cross_mark} MISMATCH"
        report += f"- **Total Nodes**: Old={totals['nodes']['old']}, New={totals['nodes']['new']} - {nodes_status}\n"
        
        rels_status = f"{check_mark} MATCH" if totals["relationships"]["match"] else f"{cross_mark} MISMATCH"
        report += f"- **Total Relationships**: Old={totals['relationships']['old']}, New={totals['relationships']['new']} - {rels_status}\n"
        
        return report
    
    def run_comparison(self, old_config_path: str, new_config_path: str) -> bool:
        """Run the complete comparison between processors."""
        
        try:
            # Load configurations
            self.logger.info("Loading configurations...")
            old_config = load_config(old_config_path)
            new_config = load_config(new_config_path)
            
            # Clear Neo4j data
            self.logger.info("Clearing Neo4j data...")
            if not self.clear_neo4j_data(new_config):
                self.logger.error("Failed to clear Neo4j data")
                return False
            
            # Run old processor
            self.logger.info("Running old processor...")
            old_processor = OldNeo4jSessionProcessor(old_config)
            old_stats = old_processor.process(create_only_new=False)
            
            # Capture old processor state
            old_db_state = self.compare_neo4j_state(old_config)
            
            # Clear and run new processor
            self.logger.info("Clearing data and running new processor...")
            self.clear_neo4j_data(new_config)
            
            new_processor = NewNeo4jSessionProcessor(new_config)
            new_stats = new_processor.process(create_only_new=False)
            
            # Capture new processor state
            new_db_state = self.compare_neo4j_state(new_config)
            
            # Compare results
            self.logger.info("Comparing results...")
            stats_comparison = self.compare_processor_statistics(old_processor.statistics, new_processor.statistics)
            
            # Generate report
            report = self.generate_comparison_report(stats_comparison, old_db_state, new_db_state)
            
            # Save results (FIXED: Added UTF-8 encoding and ensure_ascii=False)
            os.makedirs("logs", exist_ok=True)
            report_path = "logs/neo4j_session_processor_comparison_report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            comparison_data_path = "logs/neo4j_session_processor_comparison_data.json"
            comparison_data = {
                "stats_comparison": stats_comparison,
                "old_db_state": old_db_state,
                "new_db_state": new_db_state
            }
            with open(comparison_data_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, indent=2, ensure_ascii=False)
            
            # Print results (FIXED: Use ASCII characters for console output)
            print("\n" + "=" * 80)
            print("COMPARISON RESULTS")
            print("=" * 80)
            
            success = stats_comparison.get("all_identical", False)
            
            if success:
                print("[SUCCESS] Both processors produce IDENTICAL results!")
                print("[OK] Session nodes match exactly")
                print("[OK] Stream nodes match exactly") 
                print("[OK] Relationships match exactly")
                print("[OK] The new processor is a perfect replacement!")
            else:
                print("[FAIL] Processors produce DIFFERENT results!")
                
                # Show specific mismatches
                if not all(data["match"] for data in stats_comparison["nodes_created"].values()):
                    print("   - Node creation differs")
                if not all(data["match"] for data in stats_comparison["relationships_created"].values()):
                    print("   - Relationship creation differs")
            
            print(f"\nDetailed reports saved to:")
            print(f"  - {report_path}")
            print(f"  - {comparison_data_path}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Comparison failed: {e}", exc_info=True)
            print(f"\n[ERROR] {e}")
            return False


def main():
    """Main function to run the comparison."""
    parser = argparse.ArgumentParser(description="Compare old and new Neo4j Session Processors")
    parser.add_argument(
        "--old-config", 
        default="config/config.yaml",
        help="Path to old processor configuration file"
    )
    parser.add_argument(
        "--new-config", 
        default="config/config_vet.yaml", 
        help="Path to new processor configuration file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file="logs/neo4j_session_processor_comparison.log")
    
    # Create comparison instance and run
    comparison = Neo4jSessionProcessorComparison()
    success = comparison.run_comparison(args.old_config, args.new_config)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()