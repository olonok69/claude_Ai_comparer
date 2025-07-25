#!/usr/bin/env python3
"""
Simple Test Runner for Neo4j Visitor Processors

This script runs both neo4j visitor processors and does a quick comparison of key outputs.
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
    from old_neo4j_visitor_processor import Neo4jVisitorProcessor as OldNeo4jVisitorProcessor
    from neo4j_visitor_processor import Neo4jVisitorProcessor as NewNeo4jVisitorProcessor
    from utils.config_utils import load_config
    from utils.logging_utils import setup_logging
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nMake sure you have:")
    print("1. old_neo4j_visitor_processor.py")
    print("2. neo4j_visitor_processor.py")
    print("3. config/config.yaml")
    print("4. config/config_vet.yaml")
    print("5. python-dotenv package installed")
    sys.exit(1)


def compare_statistics(old_processor, new_processor):
    """Compare the statistics from both processors."""
    print("\n📊 Comparing Statistics...")
    
    comparisons = []
    
    # Check if both processors have statistics
    if not hasattr(old_processor, "statistics") or not hasattr(new_processor, "statistics"):
        print("❌ One or both processors don't have statistics attribute")
        return False
    
    old_stats = old_processor.statistics
    new_stats = new_processor.statistics
    
    # Compare nodes_created and nodes_skipped dictionaries
    for stat_type in ["nodes_created", "nodes_skipped"]:
        print(f"\n🔍 Comparing {stat_type}:")
        
        if stat_type not in old_stats or stat_type not in new_stats:
            print(f"❌ {stat_type} missing in one of the processors")
            comparisons.append(False)
            continue
        
        old_stat_dict = old_stats[stat_type]
        new_stat_dict = new_stats[stat_type]
        
        # Get all node types from both dictionaries
        node_types = list(set(old_stat_dict.keys()) | set(new_stat_dict.keys()))
        
        for node_type in node_types:
            old_count = old_stat_dict.get(node_type, 0)
            new_count = new_stat_dict.get(node_type, 0)
            
            print(f"  - {node_type}: Old={old_count}, New={new_count}")
            
            if old_count == new_count:
                print(f"    ✅ Match")
                comparisons.append(True)
            else:
                print(f"    ❌ Mismatch")
                comparisons.append(False)
    
    return all(comparisons)


def compare_neo4j_nodes(config, old_label_prefix, new_label_prefix):
    """Compare the nodes created in Neo4j by both processors."""
    print("\n🔍 Comparing Neo4j Nodes...")
    
    # Load environment variables for Neo4j connection
    load_dotenv(config["env_file"])
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not uri or not username or not password:
        print("❌ Missing Neo4j credentials")
        return False
    
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        node_types = ["Visitor_this_year", "Visitor_last_year_bva", "Visitor_last_year_lva"]
        comparisons = []
        
        for node_type in node_types:
            old_label = f"{old_label_prefix}{node_type}"
            new_label = f"{new_label_prefix}{node_type}"
            
            with driver.session() as session:
                # Get count of nodes with old label
                result = session.run(f"MATCH (n:{old_label}) RETURN count(n) AS count")
                record = result.single()
                old_count = record["count"] if record else 0
                
                # Get count of nodes with new label
                result = session.run(f"MATCH (n:{new_label}) RETURN count(n) AS count")
                record = result.single()
                new_count = record["count"] if record else 0
            
            print(f"  - {node_type}: Old={old_count} ({old_label}), New={new_count} ({new_label})")
            
            if old_count == new_count:
                print(f"    ✅ Match")
                comparisons.append(True)
            else:
                print(f"    ❌ Mismatch")
                comparisons.append(False)
        
        driver.close()
        return all(comparisons)
        
    except Exception as e:
        print(f"❌ Error comparing Neo4j nodes: {str(e)}")
        return False


def cleanup_neo4j_test_nodes(config, old_prefix, new_prefix):
    """Clean up test nodes created in Neo4j."""
    print("\n🧹 Cleaning up Neo4j test nodes...")
    
    # Load environment variables for Neo4j connection
    load_dotenv(config["env_file"])
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not uri or not username or not password:
        print("❌ Missing Neo4j credentials for cleanup")
        return
    
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        with driver.session() as session:
            # Delete nodes with old prefix (using case-insensitive match)
            result = session.run(f"""
                MATCH (n) 
                WHERE any(label in labels(n) 
                    WHERE toLower(label) STARTS WITH toLower('{old_prefix}'))
                DETACH DELETE n
            """)
            
            # Delete nodes with new prefix (using case-insensitive match)
            result = session.run(f"""
                MATCH (n) 
                WHERE any(label in labels(n) 
                    WHERE toLower(label) STARTS WITH toLower('{new_prefix}'))
                DETACH DELETE n
            """)
        
        driver.close()
        print("✅ Neo4j test nodes cleaned up successfully")
        
    except Exception as e:
        print(f"❌ Error cleaning up Neo4j test nodes: {str(e)}")


def run_test():
    """Run the simple comparison test."""
    print("🚀 Starting Neo4j Visitor Processor Comparison Test")
    print("=" * 60)
    print("ℹ️  This test assumes you've already run:")
    print("   python main.py --config config/config.yaml --only-steps 1,2,3")
    print("   python main.py --config config/config_vet.yaml --only-steps 1,2,3")
    print()
    
    # Setup logging
    logger = setup_logging(log_file="logs/simple_neo4j_visitor_test.log")
    
    try:
        # Load configurations
        print("📁 Loading configurations...")
        old_config = load_config("config/config.yaml")
        new_config = load_config("config/config_vet.yaml")
        
        # Setup Neo4j label prefixes for comparison
        old_label_prefix = "OldTest"
        new_label_prefix = "NewTest"
        
        # Modify configs to use different node labels to avoid conflicts
        old_config_copy = old_config.copy()
        new_config_copy = new_config.copy()
        
        # Create neo4j configuration if it doesn't exist
        if "neo4j" not in old_config_copy:
            old_config_copy["neo4j"] = {}
        if "neo4j" not in new_config_copy:
            new_config_copy["neo4j"] = {}
            
        # Create node_labels if it doesn't exist
        if "node_labels" not in old_config_copy["neo4j"]:
            old_config_copy["neo4j"]["node_labels"] = {}
        if "node_labels" not in new_config_copy["neo4j"]:
            new_config_copy["neo4j"]["node_labels"] = {}
        
        # Update node labels in configs
        old_config_copy["neo4j"]["node_labels"]["visitor_this_year"] = f"{old_label_prefix}Visitor_this_year"
        old_config_copy["neo4j"]["node_labels"]["visitor_last_year_bva"] = f"{old_label_prefix}Visitor_last_year_bva"
        old_config_copy["neo4j"]["node_labels"]["visitor_last_year_lva"] = f"{old_label_prefix}Visitor_last_year_lva"
        
        new_config_copy["neo4j"]["node_labels"]["visitor_this_year"] = f"{new_label_prefix}Visitor_this_year"
        new_config_copy["neo4j"]["node_labels"]["visitor_last_year_bva"] = f"{new_label_prefix}Visitor_last_year_bva" 
        new_config_copy["neo4j"]["node_labels"]["visitor_last_year_lva"] = f"{new_label_prefix}Visitor_last_year_lva"
        
        # Get the production output directories
        production_old_output = old_config.get("output_dir", "output")
        production_new_output = new_config.get("output_dir", "data/bva")
        
        print(f"📂 Production old output directory: {production_old_output}")
        print(f"📂 Production new output directory: {production_new_output}")
        
        # Check if required files exist in production directories
        required_files_template = [
            "output/df_reg_demo_this.csv",
            "output/df_reg_demo_last_bva.csv",
            "output/df_reg_demo_last_lva.csv"
        ]
        
        required_files_old = [os.path.join(production_old_output, file) for file in required_files_template]
        required_files_new = [os.path.join(production_new_output, file) for file in required_files_template]
        
        # Check if files exist
        missing_files = []
        for file_path in required_files_old + required_files_new:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print("❌ Missing required files from previous steps:")
            for file_path in missing_files:
                print(f"   - {file_path}")
            print("\n💡 Please run the previous steps first:")
            print("   python main.py --config config/config.yaml --only-steps 1,2,3")
            print("   python main.py --config config/config_vet.yaml --only-steps 1,2,3")
            return False
        
        print("✅ All required files found!")
        
        # First, clean up any existing test nodes
        cleanup_neo4j_test_nodes(old_config, old_label_prefix, new_label_prefix)
        
        # Run the old processor
        print("\n🔄 Running OLD Neo4j Visitor Processor...")
        
        # Examine old processor's code
        print("📋 Checking implementation of old processor...")
        
        # For the old processor, we need to patch the process method to use configured node labels
        original_process = OldNeo4jVisitorProcessor.process
        
        def patched_process(self, create_only_new=False):
            """Patched process method that uses configured node labels"""
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Starting Neo4j visitor data processing"
            )

            # Test the connection first
            if not self._test_connection():
                self.logger.error(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Cannot proceed with Neo4j upload due to connection failure"
                )
                return

            # Get node labels from config
            neo4j_config = self.config.get("neo4j", {})
            node_labels = neo4j_config.get("node_labels", {})
            
            # Process visitors from this year
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processing visitors from this year"
            )
            try:
                csv_file_path = os.path.join(self.output_dir, "df_reg_demo_this.csv")
                data = pd.read_csv(csv_file_path)
                properties_map = {col: col for col in data.columns}
                node_label = node_labels.get("visitor_this_year", "Visitor_this_year")

                nodes_created, nodes_skipped = self.load_csv_to_neo4j(
                    csv_file_path, node_label, properties_map, create_only_new
                )

                self.statistics["nodes_created"]["visitor_this_year"] = nodes_created
                self.statistics["nodes_skipped"]["visitor_this_year"] = nodes_skipped
            except Exception as e:
                self.logger.error(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error processing visitors from this year: {str(e)}"
                )

            # Process visitors from last year BVA
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processing visitors from last year BVA"
            )
            try:
                csv_file_path = os.path.join(self.output_dir, "df_reg_demo_last_bva.csv")
                data = pd.read_csv(csv_file_path)
                properties_map = {col: col for col in data.columns}
                node_label = node_labels.get("visitor_last_year_bva", "Visitor_last_year_bva")

                nodes_created, nodes_skipped = self.load_csv_to_neo4j(
                    csv_file_path, node_label, properties_map, create_only_new
                )

                self.statistics["nodes_created"]["visitor_last_year_bva"] = nodes_created
                self.statistics["nodes_skipped"]["visitor_last_year_bva"] = nodes_skipped
            except Exception as e:
                self.logger.error(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error processing visitors from last year BVA: {str(e)}"
                )

            # Process visitors from last year LVA
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processing visitors from last year LVA"
            )
            try:
                csv_file_path = os.path.join(self.output_dir, "df_reg_demo_last_lva.csv")
                data = pd.read_csv(csv_file_path)
                properties_map = {col: col for col in data.columns}
                node_label = node_labels.get("visitor_last_year_lva", "Visitor_last_year_lva")

                nodes_created, nodes_skipped = self.load_csv_to_neo4j(
                    csv_file_path, node_label, properties_map, create_only_new
                )

                self.statistics["nodes_created"]["visitor_last_year_lva"] = nodes_created
                self.statistics["nodes_skipped"]["visitor_last_year_lva"] = nodes_skipped
            except Exception as e:
                self.logger.error(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error processing visitors from last year LVA: {str(e)}"
                )

            # Log summary statistics
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j visitor upload summary:"
            )
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Visitors this year: {self.statistics['nodes_created']['visitor_this_year']} created, {self.statistics['nodes_skipped']['visitor_this_year']} skipped"
            )
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Visitors last year BVA: {self.statistics['nodes_created']['visitor_last_year_bva']} created, {self.statistics['nodes_skipped']['visitor_last_year_bva']} skipped"
            )
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Visitors last year LVA: {self.statistics['nodes_created']['visitor_last_year_lva']} created, {self.statistics['nodes_skipped']['visitor_last_year_lva']} skipped"
            )

            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j visitor data processing completed"
            )
        
        # Replace the original process method with our patched version
        import inspect
        OldNeo4jVisitorProcessor.process = patched_process
        
        # Now run the old processor with the patched method
        old_processor = OldNeo4jVisitorProcessor(old_config_copy)
        old_processor.process(create_only_new=False)
        print("✅ Old processor completed")
        
        # Restore the original process method
        OldNeo4jVisitorProcessor.process = original_process
        
        # Run the new processor
        print("\n🔄 Running NEW Neo4j Visitor Processor...")
        new_processor = NewNeo4jVisitorProcessor(new_config_copy)
        new_processor.process(create_only_new=False)
        print("✅ New processor completed")
        
        # Compare statistics
        statistics_identical = compare_statistics(old_processor, new_processor)
        
        # Compare Neo4j nodes
        nodes_identical = compare_neo4j_nodes(old_config, old_label_prefix, new_label_prefix)
        
        # Clean up test nodes
        cleanup_neo4j_test_nodes(old_config, old_label_prefix, new_label_prefix)
        
        # Final result
        print("\n" + "=" * 60)
        if statistics_identical and nodes_identical:
            print("🎉 SUCCESS: All outputs are IDENTICAL!")
            print("✅ The new Neo4j Visitor Processor produces the same results as the old one.")
            return True
        else:
            print("❌ FAILURE: Outputs are DIFFERENT!")
            print("⚠️  The new Neo4j Visitor Processor produces different results.")
            if not statistics_identical:
                print("   - Statistics differ")
            if not nodes_identical:
                print("   - Neo4j nodes differ")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)