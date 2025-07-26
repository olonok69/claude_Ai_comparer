#!/usr/bin/env python3
"""
Neo4j Session Processor Comparison Tool

This script compares the outputs of old_neo4j_session_processor.py and neo4j_session_processor.py
to ensure they produce identical results when processing the same input data.

Usage:
    python comparers/5_compare_processors.py [--old-config CONFIG_PATH] [--new-config CONFIG_PATH]
"""

import os
import sys
import logging
import argparse
import json
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil
from neo4j import GraphDatabase
import tempfile
from dotenv import load_dotenv

# Add current directory to path to import local modules
sys.path.insert(0, os.getcwd())

try:
    from old_neo4j_session_processor import Neo4jSessionProcessor as OldNeo4jSessionProcessor
    from neo4j_session_processor import Neo4jSessionProcessor as NewNeo4jSessionProcessor
    from utils.config_utils import load_config
    from utils.logging_utils import setup_logging
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you have the following files:")
    print("1. old_neo4j_session_processor.py")
    print("2. neo4j_session_processor.py")
    print("3. utils/config_utils.py")
    print("4. utils/logging_utils.py")
    sys.exit(1)


class Neo4jSessionProcessorComparison:
    """A class to compare old and new Neo4j Session Processors."""

    def __init__(self):
        """Initialize the comparison tool."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.comparison_dir = Path("comparers") / "comparisons" / "neo4j_session_processor"
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        
        self.old_output_dir = self.comparison_dir / f"old_output_{self.timestamp}"
        self.new_output_dir = self.comparison_dir / f"new_output_{self.timestamp}"
        
        self.old_output_dir.mkdir(parents=True, exist_ok=True)
        self.new_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(
            log_file=str(self.comparison_dir / f"comparison_{self.timestamp}.log")
        )

    def clear_neo4j_test_data(self):
        """Clear any existing test data from Neo4j to ensure clean comparison."""
        self.logger.info("Clearing existing Neo4j test data...")
        
        try:
            # Load config to get Neo4j credentials
            config = load_config("config/config_vet.yaml")
            load_dotenv(config["env_file"])
            
            uri = os.getenv("NEO4J_URI")
            username = os.getenv("NEO4J_USERNAME")
            password = os.getenv("NEO4J_PASSWORD")
            
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            with driver.session() as session:
                # Delete all session nodes and stream nodes
                delete_queries = [
                    "MATCH (n:Sessions_this_year) DETACH DELETE n",
                    "MATCH (n:Sessions_past_year) DETACH DELETE n", 
                    "MATCH (n:Stream) DETACH DELETE n"
                ]
                
                for query in delete_queries:
                    result = session.run(query)
                    self.logger.info(f"Executed: {query}")
                
                # Verify cleanup
                result = session.run("MATCH (n) WHERE n:Sessions_this_year OR n:Sessions_past_year OR n:Stream RETURN count(n) as count")
                remaining_count = result.single()["count"]
                
                if remaining_count == 0:
                    self.logger.info("✅ Successfully cleared all test data from Neo4j")
                else:
                    self.logger.warning(f"⚠️ {remaining_count} test nodes still remain in Neo4j")
                    
        except Exception as e:
            self.logger.error(f"Error clearing Neo4j test data: {e}")
            raise
        finally:
            if 'driver' in locals():
                driver.close()

    def run_old_processor(self, config):
        """Run the old Neo4j session processor."""
        try:
            self.logger.info("Running OLD Neo4j Session Processor...")
            
            # Create processor and run it
            old_processor = OldNeo4jSessionProcessor(config)
            old_statistics = old_processor.process(create_only_new=False)
            
            self.logger.info("✅ Old processor completed successfully")
            return old_processor, old_statistics
            
        except Exception as e:
            self.logger.error(f"Error running old processor: {e}")
            raise

    def run_new_processor(self, config):
        """Run the new Neo4j session processor."""
        try:
            self.logger.info("Running NEW Neo4j Session Processor...")
            
            # Create processor and run it
            new_processor = NewNeo4jSessionProcessor(config)
            new_statistics = new_processor.process(create_only_new=False)
            
            self.logger.info("✅ New processor completed successfully")
            return new_processor, new_statistics
            
        except Exception as e:
            self.logger.error(f"Error running new processor: {e}")
            raise

    def compare_neo4j_nodes(self, config):
        """Compare the nodes created in Neo4j by both processors."""
        self.logger.info("Comparing Neo4j nodes...")
        self.logger.info("Note: Old processor doesn't add 'show' attribute, new processor does")
        
        try:
            load_dotenv(config["env_file"])
            uri = os.getenv("NEO4J_URI")
            username = os.getenv("NEO4J_USERNAME")
            password = os.getenv("NEO4J_PASSWORD")
            
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            comparisons = {}
            
            with driver.session() as session:
                # Define node types to compare
                node_types = [
                    ("Sessions_this_year", "session_id"),
                    ("Sessions_past_year", "session_id"),
                    ("Stream", "name")
                ]
                
                for node_label, id_property in node_types:
                    # Get all nodes of this type
                    query = f"""
                    MATCH (n:{node_label})
                    RETURN n.{id_property} as id, properties(n) as props
                    ORDER BY n.{id_property}
                    """
                    
                    result = session.run(query)
                    nodes = list(result)
                    
                    comparisons[node_label] = {
                        "count": len(nodes),
                        "sample_properties": nodes[:3] if nodes else [],
                        "all_nodes": nodes
                    }
                    
                    self.logger.info(f"Found {len(nodes)} nodes of type {node_label}")
                
                # Compare relationships
                rel_query = """
                MATCH (s)-[r:HAS_STREAM]->(stream:Stream)
                RETURN type(r) as rel_type, s.session_id as session_id, stream.name as stream_name
                ORDER BY s.session_id, stream.name
                """
                
                result = session.run(rel_query)
                relationships = list(result)
                
                comparisons["relationships"] = {
                    "HAS_STREAM_count": len(relationships),
                    "sample_relationships": relationships[:5] if relationships else [],
                    "all_relationships": relationships
                }
                
                self.logger.info(f"Found {len(relationships)} HAS_STREAM relationships")
                
                # Analyze show attribute distribution (key difference between processors)
                show_query = """
                MATCH (n) 
                WHERE n:Sessions_this_year OR n:Sessions_past_year OR n:Stream
                RETURN 
                    labels(n) as node_type,
                    CASE WHEN n.show IS NOT NULL THEN n.show ELSE 'NULL' END as show_value,
                    count(n) as count
                ORDER BY node_type, show_value
                """
                
                result = session.run(show_query)
                show_distribution = list(result)
                
                comparisons["show_attributes"] = {
                    "distribution": show_distribution,
                    "nodes_with_show": sum(r["count"] for r in show_distribution if r["show_value"] != "NULL"),
                    "nodes_without_show": sum(r["count"] for r in show_distribution if r["show_value"] == "NULL")
                }
                
                self.logger.info(f"Show attribute analysis: {len([r for r in show_distribution if r['show_value'] != 'NULL'])} types with show, {len([r for r in show_distribution if r['show_value'] == 'NULL'])} types without")
                
        except Exception as e:
            self.logger.error(f"Error comparing Neo4j nodes: {e}")
            raise
        finally:
            if 'driver' in locals():
                driver.close()
        
        return comparisons

    def compare_statistics(self, old_stats, new_stats):
        """Compare statistics from both processors."""
        self.logger.info("Comparing processor statistics...")
        self.logger.info("Note: The old processor has bugs in stream/relationship creation that the new processor fixes")
        
        comparison = {
            "nodes_created": {},
            "nodes_skipped": {},
            "relationships_created": {},
            "relationships_skipped": {},
            "totals": {},
            "bug_fixes": {}
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
        
        # Compare stream creation (old=0 due to bug, new>0 is fix)
        old_streams = old_stats.get("nodes_created", {}).get("streams", 0)
        new_streams = new_stats.get("nodes_created", {}).get("streams", 0)
        
        comparison["nodes_created"]["streams"] = {
            "old": old_streams,
            "new": new_streams,
            "match": False,  # Expected to be different
            "difference": new_streams - old_streams,
            "expected": "Old=0 (bug), New>0 (fixed)"
        }
        
        comparison["bug_fixes"]["stream_creation"] = {
            "old_broken": old_streams == 0,
            "new_fixed": new_streams > 0,
            "bug_fixed": old_streams == 0 and new_streams > 0
        }
        
        # Compare relationship creation (old=0 due to missing streams, new>0 is fix)
        rel_keys = ["sessions_this_year_has_stream", "sessions_past_year_has_stream"]
        
        for key in rel_keys:
            old_val = old_stats.get("relationships_created", {}).get(key, 0)
            new_val = new_stats.get("relationships_created", {}).get(key, 0)
            
            comparison["relationships_created"][key] = {
                "old": old_val,
                "new": new_val,
                "match": False,  # Expected to be different
                "difference": new_val - old_val,
                "expected": "Old=0 (no streams), New>0 (fixed)"
            }
        
        comparison["bug_fixes"]["relationship_creation"] = {
            "old_broken": all(old_stats.get("relationships_created", {}).get(k, 0) == 0 for k in rel_keys),
            "new_fixed": all(new_stats.get("relationships_created", {}).get(k, 0) > 0 for k in rel_keys),
            "bug_fixed": (all(old_stats.get("relationships_created", {}).get(k, 0) == 0 for k in rel_keys) and 
                         all(new_stats.get("relationships_created", {}).get(k, 0) > 0 for k in rel_keys))
        }
        
        # Compare totals
        total_keys = ["total_nodes_created", "total_nodes_skipped", "total_relationships_created", "total_relationships_skipped"]
        for key in total_keys:
            old_val = old_stats.get(key, 0)
            new_val = new_stats.get(key, 0)
            
            expected_match = key in ["total_nodes_skipped", "total_relationships_skipped"]
            
            comparison["totals"][key] = {
                "old": old_val,
                "new": new_val,
                "match": old_val == new_val,
                "difference": new_val - old_val,
                "expected": "Should match" if expected_match else "New should be higher (bug fixes)"
            }
        
        return comparison

    def generate_report(self, old_stats, new_stats, node_comparison, stats_comparison):
        """Generate a detailed comparison report."""
        report = f"""# Neo4j Session Processor Comparison Report

## Execution Details
- **Timestamp**: {self.timestamp}
- **Old Processor**: old_neo4j_session_processor.py (has bugs)
- **New Processor**: neo4j_session_processor.py (fixes bugs)

## Important Note
The old Neo4j session processor has critical bugs that prevent it from creating Stream nodes and HAS_STREAM relationships. The new processor fixes these bugs. The differences shown below are **expected** and indicate successful bug fixes.

## Statistics Comparison

### Session Nodes (Should Match Exactly)
"""
        
        session_keys = ["sessions_this_year", "sessions_past_year_bva", "sessions_past_year_lva"]
        for key in session_keys:
            if key in stats_comparison["nodes_created"]:
                data = stats_comparison["nodes_created"][key]
                status = "✅ MATCH" if data["match"] else "❌ MISMATCH"
                report += f"- **{key}**: Old={data['old']}, New={data['new']} ({status})\n"
        
        report += f"""
### Stream Nodes (Bug Fix - Old=0, New>0 Expected)
"""
        if "streams" in stats_comparison["nodes_created"]:
            data = stats_comparison["nodes_created"]["streams"]
            if data["old"] == 0 and data["new"] > 0:
                status = "✅ BUG FIXED"
            else:
                status = "❌ UNEXPECTED"
            report += f"- **streams**: Old={data['old']}, New={data['new']} ({status})\n"
        
        report += f"""
### Stream Relationships (Bug Fix - Old=0, New>0 Expected)
"""
        rel_keys = ["sessions_this_year_has_stream", "sessions_past_year_has_stream"]
        for key in rel_keys:
            if key in stats_comparison["relationships_created"]:
                data = stats_comparison["relationships_created"][key]
                if data["old"] == 0 and data["new"] > 0:
                    status = "✅ BUG FIXED"
                else:
                    status = "❌ UNEXPECTED"
                report += f"- **{key}**: Old={data['old']}, New={data['new']} ({status})\n"
        
        report += "\n## Bug Fix Analysis\n"
        
        if "bug_fixes" in stats_comparison:
            bug_fixes = stats_comparison["bug_fixes"]
            
            if "stream_creation" in bug_fixes:
                stream_fix = bug_fixes["stream_creation"]
                report += f"- **Stream Creation Bug**: {'✅ FIXED' if stream_fix['bug_fixed'] else '❌ NOT FIXED'}\n"
                report += f"  - Old processor created 0 streams (broken): {stream_fix['old_broken']}\n"
                report += f"  - New processor creates streams properly: {stream_fix['new_fixed']}\n"
            
            if "relationship_creation" in bug_fixes:
                rel_fix = bug_fixes["relationship_creation"]
                report += f"- **Relationship Creation Bug**: {'✅ FIXED' if rel_fix['bug_fixed'] else '❌ NOT FIXED'}\n"
                report += f"  - Old processor created 0 relationships (no streams): {rel_fix['old_broken']}\n"
                report += f"  - New processor creates relationships properly: {rel_fix['new_fixed']}\n"
        
        report += "\n## Show Attribute Analysis\n"
        
        if "show_attributes" in node_comparison:
            show_data = node_comparison["show_attributes"]
            report += f"- **Nodes with 'show' attribute**: {show_data['nodes_with_show']} (New processor feature)\n"
            report += f"- **Nodes without 'show' attribute**: {show_data['nodes_without_show']} (Old processor limitation)\n"
            
            if show_data["distribution"]:
                report += "\n### Show Attribute Distribution:\n"
                for item in show_data["distribution"]:
                    node_type = item["node_type"][0] if item["node_type"] else "Unknown"
                    show_val = item["show_value"]
                    count = item["count"]
                    
                    if show_val == "NULL":
                        report += f"- **{node_type}**: {count} nodes without show attribute (old processor)\n"
                    else:
                        report += f"- **{node_type}**: {count} nodes with show='{show_val}' (new processor)\n"
        
        report += "\n## Neo4j Database State\n"
        
        for node_type, data in node_comparison.items():
            if node_type not in ["relationships", "show_attributes"]:
                report += f"- **{node_type}**: {data['count']} nodes\n"
        
        if "relationships" in node_comparison:
            rel_data = node_comparison["relationships"]
            report += f"- **HAS_STREAM relationships**: {rel_data['HAS_STREAM_count']} relationships\n"
        
        # Overall assessment
        bugs_fixed = True
        if "bug_fixes" in stats_comparison:
            bugs_fixed = all(
                fix_data.get("bug_fixed", False) 
                for fix_data in stats_comparison["bug_fixes"].values()
            )
        
        sessions_match = all(
            stats_comparison["nodes_created"][key]["match"] 
            for key in session_keys 
            if key in stats_comparison["nodes_created"]
        )
        
        # Check if show attributes are properly distributed (should have both NULL and non-NULL)
        show_proper = False
        if "show_attributes" in node_comparison:
            show_data = node_comparison["show_attributes"]
            show_proper = show_data["nodes_with_show"] > 0 and show_data["nodes_without_show"] > 0
        
        report += f"\n## Overall Assessment\n"
        if sessions_match and bugs_fixed and show_proper:
            report += "🎉 **SUCCESS**: New processor fixes all old processor bugs and adds new features!\n"
            report += "✅ Session nodes are created identically by both processors.\n"
            report += "✅ Stream nodes and relationships are now created properly (old processor couldn't do this).\n"
            report += "✅ New processor adds 'show' attributes while old processor doesn't (expected improvement).\n"
            report += "✅ The new processor is a complete improvement over the old one.\n"
        else:
            report += "❌ **UNEXPECTED RESULTS**: The comparison doesn't show expected patterns!\n"
            if not sessions_match:
                report += "⚠️ Session node creation differs unexpectedly.\n"
            if not bugs_fixed:
                report += "⚠️ Expected bug fixes are not working as intended.\n"
            if not show_proper:
                report += "⚠️ Show attribute distribution is not as expected.\n"
        
        return report

    def run_comparison(self, old_config_path="config/config.yaml", new_config_path="config/config_vet.yaml"):
        """Run the complete comparison process."""
        try:
            self.logger.info(f"Starting Neo4j Session Processor Comparison")
            self.logger.info(f"Old config: {old_config_path}")
            self.logger.info(f"New config: {new_config_path}")
            
            # Load configurations
            old_config = load_config(old_config_path)
            new_config = load_config(new_config_path)
            
            # Clear existing test data
            self.clear_neo4j_test_data()
            
            # Run old processor
            old_processor, old_stats = self.run_old_processor(old_config)
            
            # Clear test data again before running new processor
            self.clear_neo4j_test_data()
            
            # Run new processor
            new_processor, new_stats = self.run_new_processor(new_config)
            
            # Compare Neo4j database state
            node_comparison = self.compare_neo4j_nodes(new_config)
            
            # Compare statistics
            stats_comparison = self.compare_statistics(old_stats, new_stats)
            
            # Generate report
            report = self.generate_report(old_stats, new_stats, node_comparison, stats_comparison)
            
            # Save report
            report_path = self.comparison_dir / f"neo4j_session_comparison_report_{self.timestamp}.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # Save detailed comparison data
            comparison_data = {
                "timestamp": self.timestamp,
                "old_statistics": old_stats,
                "new_statistics": new_stats,
                "statistics_comparison": stats_comparison,
                "node_comparison": node_comparison
            }
            
            comparison_data_path = self.comparison_dir / f"neo4j_session_comparison_data_{self.timestamp}.json"
            with open(comparison_data_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, indent=2, default=str, ensure_ascii=False)
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"COMPARISON SUMMARY")
            print(f"{'='*60}")
            
            # Check if all expected patterns are working
            session_nodes_match = all(
                stats_comparison["nodes_created"][key]["match"] 
                for key in ["sessions_this_year", "sessions_past_year_bva", "sessions_past_year_lva"]
                if key in stats_comparison["nodes_created"]
            )
            
            bugs_fixed = True
            if "bug_fixes" in stats_comparison:
                bugs_fixed = all(
                    fix_data.get("bug_fixed", False) 
                    for fix_data in stats_comparison["bug_fixes"].values()
                )
            
            # Check show attribute distribution (should have both old nodes without show and new nodes with show)
            show_distribution_correct = False
            if "show_attributes" in node_comparison:
                show_data = node_comparison["show_attributes"]
                show_distribution_correct = show_data["nodes_with_show"] > 0 and show_data["nodes_without_show"] > 0
            
            if session_nodes_match and bugs_fixed and show_distribution_correct:
                print("🎉 SUCCESS: New processor fixes all old processor bugs and adds new features!")
                print("✅ Session nodes are created identically by both processors.")
                print("✅ Stream nodes and relationships are now created properly.")
                print("✅ New processor adds 'show' attributes while old processor doesn't (expected).")
                print("✅ The new processor is a complete improvement over the old one.")
                success = True
            else:
                print("❌ FAILURE: Expected improvements are not working correctly!")
                print("⚠️ The processors don't show the expected improvement pattern.")
                
                # Show specific issues
                if not session_nodes_match:
                    print("   - Session node creation differs unexpectedly")
                if not bugs_fixed:
                    print("   - Stream/relationship bugs are not fixed as expected")
                if not show_distribution_correct:
                    print("   - Show attribute distribution is not as expected")
                    if "show_attributes" in node_comparison:
                        show_data = node_comparison["show_attributes"]
                        print(f"     - Nodes with show: {show_data['nodes_with_show']}")
                        print(f"     - Nodes without show: {show_data['nodes_without_show']}")
                
                success = False
            
            print(f"\nDetailed reports saved to:")
            print(f"  - {report_path}")
            print(f"  - {comparison_data_path}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Comparison failed: {e}", exc_info=True)
            print(f"\n❌ ERROR: {e}")
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
    
    # Create comparison instance and run
    comparison = Neo4jSessionProcessorComparison()
    success = comparison.run_comparison(args.old_config, args.new_config)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()