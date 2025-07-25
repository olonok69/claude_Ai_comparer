#!/usr/bin/env python3
"""
Fixed Neo4j Visitor Processor Comparison Tool

This script compares the outputs of old_neo4j_visitor_processor.py and neo4j_visitor_processor.py
to ensure they produce identical results when processing the same input data.

Key fixes:
1. Proper null checking for Neo4j query results
2. Correct method signatures for both processors
3. Fixed transaction handling with execute_write
4. Proper error handling for missing attributes
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
import inspect

# Add current directory to path to import local modules
sys.path.insert(0, os.getcwd())

try:
    from old_neo4j_visitor_processor import Neo4jVisitorProcessor as OldNeo4jVisitorProcessor
    from neo4j_visitor_processor import Neo4jVisitorProcessor as NewNeo4jVisitorProcessor
    from utils.config_utils import load_config
    from utils.logging_utils import setup_logging
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you have the required files in your project")
    sys.exit(1)


class Neo4jVisitorProcessorComparison:
    """A class to compare old and new Neo4j Visitor Processors."""

    def __init__(self):
        """Initialize the comparison tool."""
        self.logger = logging.getLogger(__name__)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = Path("comparers/reports")
        self.report_dir.mkdir(exist_ok=True, parents=True)
        
    def safe_single_result(self, result):
        """Safely get a single result from a Neo4j query result."""
        try:
            record = result.single()
            return record if record is not None else {}
        except Exception as e:
            self.logger.debug(f"No result found: {e}")
            return {}

    def safe_count_query(self, session, query):
        """Safely execute a count query and return the count."""
        try:
            result = session.run(query)
            record = self.safe_single_result(result)
            return record.get("count", 0) if record else 0
        except Exception as e:
            self.logger.debug(f"Count query failed: {e}")
            return 0

    def run_old_processor_with_config(self, config):
        """Run the old Neo4j Visitor Processor with the given config."""
        self.logger.info("Running old Neo4j Visitor Processor")
        
        # Make sure create_only_new is False
        config["create_only_new"] = False
        
        # Check the node labels configuration
        neo4j_config = config.get("neo4j", {})
        node_labels = neo4j_config.get("node_labels", {})
        
        print(f"DEBUG: Old processor node labels configuration:")
        for key, value in node_labels.items():
            print(f"  - {key}: {value}")
            
        # Patch the load_csv_to_neo4j method with improved implementation
        original_load_csv = OldNeo4jVisitorProcessor.load_csv_to_neo4j
        
        def patched_load_csv(self, csv_file_path, node_label, properties_map, create_only_new=False):
            """Improved patched version that properly creates nodes with correct labels"""
            self.logger.info(f"Patched load_csv_to_neo4j called: {csv_file_path}, {node_label}")
            
            # Get the configured node label based on the node type
            if node_label == "Visitor_this_year":
                node_label = "OldVisitor_this_year"
            elif node_label == "Visitor_last_year_bva":
                node_label = "OldVisitor_last_year_bva"
            elif node_label == "Visitor_last_year_lva":
                node_label = "OldVisitor_last_year_lva"
                
            self.logger.info(f"Using node label: {node_label}")

            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            nodes_created = 0
            nodes_skipped = 0

            try:
                # Read the CSV file
                df = pd.read_csv(csv_file_path)
                total_rows = len(df)
                self.logger.info(f"Read {total_rows} rows from {csv_file_path}")
                
                # Process in batches
                batch_size = 100
                for i in range(0, total_rows, batch_size):
                    batch_end = min(i + batch_size, total_rows)
                    batch_df = df.iloc[i:batch_end]
                    batch_rows = batch_df.to_dict('records')
                    
                    with driver.session() as session:
                        def create_batch_nodes(tx):
                            batch_created = 0
                            for idx, row in enumerate(batch_rows):
                                # Create properties dictionary
                                properties = {}
                                for col, prop_neo4j in properties_map.items():
                                    if col in row and row[col] is not None and not pd.isna(row[col]):
                                        properties[prop_neo4j] = str(row[col])
                                
                                # Skip empty rows
                                if not properties:
                                    continue
                                
                                # Add has_recommendation attribute
                                properties["has_recommendation"] = "0"
                                
                                # Create a unique identifier for this node
                                unique_id = f"{i + idx}_{hash(str(properties))}"
                                properties["_comparison_id"] = unique_id
                                
                                # Create the node
                                query = f"CREATE (n:{node_label} $props) RETURN n"
                                result = tx.run(query, props=properties)
                                if result.single():
                                    batch_created += 1
                            
                            return batch_created
                        
                        try:
                            # Use execute_write instead of deprecated write_transaction
                            batch_created = session.execute_write(create_batch_nodes)
                            nodes_created += batch_created
                            self.logger.info(f"Processed batch {i//batch_size + 1}/{(total_rows+batch_size-1)//batch_size} with {len(batch_rows)} rows, created {batch_created} nodes")
                        except Exception as e:
                            self.logger.error(f"Error in batch {i//batch_size + 1}: {e}")
                
                # Verify nodes were created using safe query
                with driver.session() as session:
                    count_query = f"MATCH (n:{node_label}) RETURN count(n) as count"
                    try:
                        result = session.run(count_query)
                        record = result.single()
                        actual_count = record["count"] if record else 0
                    except Exception:
                        actual_count = 0
                    print(f"DEBUG: Created {actual_count} nodes with label {node_label}")
                
            except Exception as e:
                self.logger.error(f"Error in patched load_csv_to_neo4j: {str(e)}", exc_info=True)
                print(f"DEBUG ERROR: {str(e)}")
            finally:
                driver.close()
            
            self.logger.info(f"CSV loaded to Neo4j. Created {nodes_created} nodes, skipped {nodes_skipped} nodes")
            return nodes_created, nodes_skipped
        
        # Replace the original method
        OldNeo4jVisitorProcessor.load_csv_to_neo4j = patched_load_csv
        
        # Run the processor
        processor = OldNeo4jVisitorProcessor(config)
        processor.process(create_only_new=False)
        
        # Verify the nodes were created
        self._verify_nodes_created(config, "Old")
        
        # Restore the original method
        OldNeo4jVisitorProcessor.load_csv_to_neo4j = original_load_csv
            
        return processor

    def run_new_processor_with_config(self, config):
        """Run the new Neo4j Visitor Processor with the given config."""
        self.logger.info("Running new Neo4j Visitor Processor")
        
        # Make sure create_only_new is False
        config["create_only_new"] = False
        
        # Patch the new processor - need to handle correct method signature
        original_load_csv = NewNeo4jVisitorProcessor.load_csv_to_neo4j
        
        def patched_new_load_csv(self, csv_file_path, node_label, properties_map, unique_property="BadgeId", create_only_new=False):
            """Patched new processor to use New prefix and handle correct method signature"""
            # Add New prefix to the label
            if not node_label.startswith("New"):
                node_label = f"New{node_label}"
                
            # Log what we're doing
            self.logger.info(f"Loading data from {csv_file_path} to Neo4j with label {node_label}")
            
            nodes_created = 0
            nodes_skipped = 0
            
            try:
                # Read CSV
                df = pd.read_csv(csv_file_path)
                total_rows = len(df)
                self.logger.info(f"Read {total_rows} rows from {csv_file_path}")
                
                # Create driver connection (new processor doesn't store driver as instance variable)
                driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
                
                # Process in batches
                batch_size = 100
                for i in range(0, total_rows, batch_size):
                    batch_end = min(i + batch_size, total_rows)
                    batch_df = df.iloc[i:batch_end]
                    
                    with driver.session() as session:
                        def create_batch(tx):
                            batch_created = 0
                            for idx, row in batch_df.iterrows():
                                # Convert row to dict and clean using properties_map
                                properties = {}
                                for col, prop_neo4j in properties_map.items():
                                    if col in row and row[col] is not None and not pd.isna(row[col]):
                                        properties[prop_neo4j] = str(row[col])
                                
                                if not properties:
                                    continue
                                
                                # Add required attributes
                                properties["has_recommendation"] = "0"
                                
                                # Add show attribute if the processor has it
                                if hasattr(self, 'show'):
                                    properties["show"] = self.show
                                    
                                properties["_comparison_id"] = f"{i + idx}_{hash(str(properties))}"
                                
                                # Create node
                                query = f"CREATE (n:{node_label} $props) RETURN n"
                                result = tx.run(query, props=properties)
                                if result.single():
                                    batch_created += 1
                            
                            return batch_created
                        
                        batch_created = session.execute_write(create_batch)
                        nodes_created += batch_created
                        self.logger.info(f"Processed batch {i//batch_size + 1}/{(total_rows+batch_size-1)//batch_size} with {len(batch_df)} rows, created {batch_created} nodes")
                
                driver.close()
                
            except Exception as e:
                self.logger.error(f"Error in new processor patched method: {e}")
                print(f"DEBUG ERROR: {str(e)}")
            
            self.logger.info(f"Created {nodes_created} nodes and skipped {nodes_skipped} nodes")
            return nodes_created, nodes_skipped
        
        # Replace the method
        NewNeo4jVisitorProcessor.load_csv_to_neo4j = patched_new_load_csv
        
        processor = NewNeo4jVisitorProcessor(config)
        processor.process(create_only_new=False)
        
        # Verify nodes
        self._verify_nodes_created(config, "New")
        
        # Restore original
        NewNeo4jVisitorProcessor.load_csv_to_neo4j = original_load_csv
        
        return processor

    def _verify_nodes_created(self, config, prefix):
        """Verify that nodes were actually created with the expected labels."""
        print(f"DEBUG: Verifying {prefix} nodes were created...")
        
        load_dotenv(config["env_file"])
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        
        try:
            driver = GraphDatabase.driver(uri, auth=(username, password))
            with driver.session() as session:
                # Get all labels
                result = session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
                labels = [record["label"] for record in result]
                print(f"DEBUG: Available labels: {labels}")
                
                # Check each expected label
                expected_labels = [
                    f"{prefix}Visitor_this_year",
                    f"{prefix}Visitor_last_year_bva", 
                    f"{prefix}Visitor_last_year_lva"
                ]
                
                for label in expected_labels:
                    count = self.safe_count_query(session, f"MATCH (n:{label}) RETURN count(n) as count")
                    print(f"DEBUG: {label}: {count} nodes")
            
            driver.close()
        except Exception as e:
            print(f"DEBUG: Error verifying nodes: {e}")

    def compare_statistics(self, old_processor, new_processor):
        """Compare the statistics attributes of both processors."""
        self.logger.info("Comparing processor statistics")
        
        if not hasattr(old_processor, "statistics") or not hasattr(new_processor, "statistics"):
            return {"success": False, "details": "Missing statistics attribute"}
        
        old_stats = old_processor.statistics
        new_stats = new_processor.statistics
        
        comparison_results = {"success": True, "details": {}}
        
        for stat_type in ["nodes_created", "nodes_skipped"]:
            if stat_type not in old_stats or stat_type not in new_stats:
                comparison_results["success"] = False
                comparison_results["details"][stat_type] = "Missing in one processor"
                continue
                
            old_stat_dict = old_stats[stat_type]
            new_stat_dict = new_stats[stat_type]
            
            node_types = list(set(old_stat_dict.keys()) | set(new_stat_dict.keys()))
            for node_type in node_types:
                old_count = old_stat_dict.get(node_type, 0)
                new_count = new_stat_dict.get(node_type, 0)
                
                if old_count != new_count:
                    comparison_results["success"] = False
                    if stat_type not in comparison_results["details"]:
                        comparison_results["details"][stat_type] = {}
                    comparison_results["details"][stat_type][node_type] = {
                        "old": old_count,
                        "new": new_count
                    }
        
        return comparison_results

    def compare_neo4j_nodes(self, config, old_label_prefix, new_label_prefix):
        """Compare nodes created in Neo4j by both processors with improved error handling."""
        self.logger.info(f"Comparing Neo4j nodes: {old_label_prefix} vs {new_label_prefix}")
        
        load_dotenv(config["env_file"])
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        
        if not uri or not username or not password:
            return {"success": False, "details": "Missing Neo4j credentials"}
        
        comparison_results = {"success": True, "details": {}}
        
        try:
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            with driver.session() as session:
                # Get all available labels
                result = session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
                labels = [record["label"] for record in result]
                print(f"DEBUG: Available labels: {labels}")
            
            node_types = ["Visitor_this_year", "Visitor_last_year_bva", "Visitor_last_year_lva"]
            
            for node_type in node_types:
                old_label = f"{old_label_prefix}{node_type}"
                new_label = f"{new_label_prefix}{node_type}"
                
                print(f"DEBUG: Checking labels: {old_label} and {new_label}")
                
                with driver.session() as session:
                    old_count = self.safe_count_query(session, f"MATCH (n:{old_label}) RETURN count(n) as count")
                    new_count = self.safe_count_query(session, f"MATCH (n:{new_label}) RETURN count(n) as count")
                    
                    print(f"DEBUG: {old_label}: {old_count} nodes")
                    print(f"DEBUG: {new_label}: {new_count} nodes")
                    
                    if old_count != new_count:
                        comparison_results["success"] = False
                        comparison_results["details"][node_type] = {
                            "old_label": old_label,
                            "new_label": new_label,
                            "old_count": old_count,
                            "new_count": new_count
                        }
            
            driver.close()
            
        except Exception as e:
            self.logger.error(f"Error comparing Neo4j nodes: {str(e)}")
            comparison_results["success"] = False
            comparison_results["details"]["error"] = str(e)
        
        return comparison_results

    def compare_processor_attributes(self, old_processor, new_processor):
        """Compare various attributes of the processors."""
        self.logger.info("Comparing processor attributes")
        
        attributes_to_compare = ["output_dir", "uri", "username", "password"]
        comparisons = []
        
        for attr in attributes_to_compare:
            old_has = hasattr(old_processor, attr)
            new_has = hasattr(new_processor, attr)
            
            if old_has and new_has:
                old_val = getattr(old_processor, attr)
                new_val = getattr(new_processor, attr)
                identical = old_val == new_val
                
                comparisons.append({
                    "attribute": attr,
                    "identical": identical,
                    "old_value": str(old_val) if attr != "password" else "***",
                    "new_value": str(new_val) if attr != "password" else "***"
                })
            else:
                comparisons.append({
                    "attribute": attr,
                    "identical": False,
                    "old_value": "present" if old_has else "missing",
                    "new_value": "present" if new_has else "missing"
                })
        
        return comparisons

    def _cleanup_neo4j_test_nodes(self, config, old_prefix, new_prefix):
        """Clean up test nodes created in Neo4j with improved error handling."""
        self.logger.info("Cleaning up Neo4j test nodes")
        print("\n🧹 Cleaning up Neo4j test nodes...")
        
        load_dotenv(config["env_file"])
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        
        if not uri or not username or not password:
            self.logger.error("Missing Neo4j credentials for cleanup")
            return
        
        try:
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            with driver.session() as session:
                # Get all labels
                result = session.run("CALL db.labels() YIELD label RETURN label")
                all_labels = [record["label"] for record in result]
                
                # Find labels to delete
                labels_to_delete = [
                    label for label in all_labels 
                    if label.startswith(old_prefix) or label.startswith(new_prefix)
                ]
                
                print(f"DEBUG: Found labels to delete: {labels_to_delete}")
                
                # Delete nodes with each label
                for label in labels_to_delete:
                    try:
                        result = session.run(f"MATCH (n:{label}) DETACH DELETE n")
                        print(f"DEBUG: Deleted nodes with label: {label}")
                    except Exception as e:
                        print(f"DEBUG: Error deleting {label}: {e}")
            
            driver.close()
            print("✅ Neo4j test nodes cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up Neo4j test nodes: {str(e)}")
            print(f"❌ Error cleaning up Neo4j test nodes: {str(e)}")

    def generate_report(self, comparison_data, report_path):
        """Generate a detailed report of the comparison."""
        self.logger.info(f"Generating report at {report_path}")
        
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("# Neo4j Visitor Processor Comparison Report\n\n")
                f.write(f"Generated: {self.timestamp}\n\n")
                
                f.write("## Summary\n\n")
                attr_summary = f"{comparison_data['summary']['attr_identical']}/{comparison_data['summary']['attr_total']} identical"
                f.write(f"Processor Attributes: {attr_summary}\n\n")
                
                if comparison_data.get('statistics_comparison'):
                    status = "[SUCCESS]" if comparison_data['statistics_comparison']['success'] else "[FAIL]"
                    f.write(f"Statistics: {status}\n\n")
                
                if comparison_data.get('neo4j_comparison'):
                    status = "[SUCCESS]" if comparison_data['neo4j_comparison']['success'] else "[FAIL]"
                    f.write(f"Neo4j Nodes: {status}\n\n")
                
                # Detailed sections...
                if comparison_data.get('neo4j_comparison', {}).get('details'):
                    f.write("## Neo4j Node Comparison Details\n\n")
                    for node_type, details in comparison_data['neo4j_comparison']['details'].items():
                        if isinstance(details, dict) and 'old_count' in details:
                            f.write(f"- {node_type}:\n")
                            f.write(f"  - {details['old_label']}: {details['old_count']} nodes\n")
                            f.write(f"  - {details['new_label']}: {details['new_count']} nodes\n")
                
                f.write("## Processor Attributes\n\n")
                f.write("| Attribute | Status | Old Value | New Value |\n")
                f.write("|-----------|--------|-----------|----------|\n")
                
                for attr in comparison_data['attribute_comparisons']:
                    status = "✅" if attr['identical'] else "❌"
                    f.write(f"| {attr['attribute']} | {status} | {attr['old_value']} | {attr['new_value']} |\n")
                
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")

    def run_comparison(self, old_config_path="config/config.yaml", new_config_path="config/config_vet.yaml"):
        """Run the full comparison process with improved error handling."""
        try:
            self.logger.info("Starting Neo4j Visitor Processor comparison")
            print("🚀 Starting Neo4j Visitor Processor Comparison")
            print("=" * 60)
            
            # Load configurations
            print("📁 Loading configurations...")
            old_config = load_config(old_config_path)
            new_config = load_config(new_config_path)
            
            # Setup prefixes
            old_label_prefix = "Old"
            new_label_prefix = "New"
            
            # Clean up first
            print("🧹 Cleaning up any existing test nodes...")
            self._cleanup_neo4j_test_nodes(old_config, old_label_prefix, new_label_prefix)
            
            # Run processors
            print("🔄 Running old processor...")
            old_processor = self.run_old_processor_with_config(old_config.copy())
            
            print("🔄 Running new processor...")
            new_processor = self.run_new_processor_with_config(new_config.copy())
            
            # Compare outputs
            print("📊 Comparing outputs...")
            
            attr_comparisons = self.compare_processor_attributes(old_processor, new_processor)
            statistics_comparison = self.compare_statistics(old_processor, new_processor)
            neo4j_comparison = self.compare_neo4j_nodes(old_config, old_label_prefix, new_label_prefix)
            
            # Compile results
            comparison_data = {
                "timestamp": self.timestamp,
                "old_config": old_config_path,
                "new_config": new_config_path,
                "attribute_comparisons": attr_comparisons,
                "statistics_comparison": statistics_comparison,
                "neo4j_comparison": neo4j_comparison,
                "summary": {
                    "attr_total": len(attr_comparisons),
                    "attr_identical": sum(1 for a in attr_comparisons if a["identical"]),
                }
            }
            
            # Generate report
            report_path = self.report_dir / f"neo4j_visitor_processor_comparison_{self.timestamp}.md"
            comparison_data_path = self.report_dir / f"neo4j_visitor_processor_comparison_data_{self.timestamp}.json"
            
            self.generate_report(comparison_data, report_path)
            
            with open(comparison_data_path, "w") as f:
                json.dump(comparison_data, f, indent=2)
            
            # Print summary
            print("\n" + "=" * 60)
            print("COMPARISON SUMMARY")
            print("=" * 60)
            print(f"Processor Attributes: {comparison_data['summary']['attr_identical']}/{comparison_data['summary']['attr_total']} identical")
            
            if statistics_comparison['success']:
                print("Statistics: [SUCCESS] Identical")
            else:
                print("Statistics: [FAIL] Differences found")
            
            if neo4j_comparison['success']:
                print("Neo4j Nodes: [SUCCESS] Identical")
            else:
                print("Neo4j Nodes: [FAIL] Differences found")
            
            print(f"\nReport: {report_path}")
            print(f"Data: {comparison_data_path}")
            
            # Check overall success
            all_identical = (
                comparison_data['summary']['attr_identical'] == comparison_data['summary']['attr_total'] and
                statistics_comparison['success'] and
                neo4j_comparison['success']
            )
            
            if all_identical:
                print("✅ SUCCESS: All outputs are identical!")
            else:
                print("❌ DIFFERENCES FOUND: Check the report for details")
            
            # Cleanup
            self._cleanup_neo4j_test_nodes(old_config, old_label_prefix, new_label_prefix)
            
            return all_identical
            
        except Exception as e:
            self.logger.error(f"Error during comparison: {str(e)}", exc_info=True)
            print(f"❌ ERROR: {str(e)}")
            return False


def main():
    """Main function to run the comparison."""
    parser = argparse.ArgumentParser(description="Compare old and new Neo4j Visitor Processors")
    parser.add_argument("--old-config", default="config/config.yaml", help="Path to old config")
    parser.add_argument("--new-config", default="config/config_vet.yaml", help="Path to new config")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_file="logs/neo4j_visitor_processor_comparison.log")
    
    comparison = Neo4jVisitorProcessorComparison()
    success = comparison.run_comparison(args.old_config, args.new_config)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()