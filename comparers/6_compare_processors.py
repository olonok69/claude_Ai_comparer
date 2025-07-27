#!/usr/bin/env python3
"""
Comparison Tool for Neo4j Job Stream Processors

This script compares the output of the old and new Neo4j job stream processors,
including Neo4j database state comparison.
"""

import os
import sys
import json
import logging
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    from old_neo4j_job_stream_processor import Neo4jJobStreamProcessor as OldNeo4jJobStreamProcessor
    from neo4j_job_stream_processor import Neo4jJobStreamProcessor as NewNeo4jJobStreamProcessor
    from utils.config_utils import load_config
    from utils.logging_utils import setup_logging
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)


class Neo4jJobStreamProcessorComparison:
    """Comprehensive comparison of Neo4j job stream processors"""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.comparison_dir = Path("comparisons") / "neo4j_job_stream"
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create temporary directories for outputs
        self.temp_dir = Path(tempfile.mkdtemp())
        self.old_output_dir = self.temp_dir / "old_output"
        self.new_output_dir = self.temp_dir / "new_output"
        self.old_output_dir.mkdir(exist_ok=True)
        self.new_output_dir.mkdir(exist_ok=True)

    def __del__(self):
        """Clean up temporary directories"""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def clear_neo4j_job_stream_relationships(self, config):
        """Clear existing job stream relationships from Neo4j"""
        try:
            load_dotenv(config["env_file"])
            uri = os.getenv("NEO4J_URI")
            username = os.getenv("NEO4J_USERNAME")
            password = os.getenv("NEO4J_PASSWORD")
            
            if not all([uri, username, password]):
                raise ValueError("Missing Neo4j credentials")
            
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            # Get configuration values
            neo4j_config = config.get("neo4j", {})
            show_name = neo4j_config.get("show_name", "unknown")
            node_labels = neo4j_config.get("node_labels", {})
            relationships = neo4j_config.get("relationships", {})
            
            visitor_label = node_labels.get("visitor_this_year", "Visitor_this_year")
            stream_label = node_labels.get("stream", "Stream")
            relationship_name = relationships.get("job_stream", "job_to_stream")
            
            with driver.session() as session:
                # Count relationships before deletion
                count_query = f"""
                MATCH (v:{visitor_label})-[r:{relationship_name}]->(s:{stream_label})
                WHERE v.show = $show_name AND s.show = $show_name
                RETURN COUNT(r) AS count
                """
                initial_count = session.run(count_query, show_name=show_name).single()["count"]
                
                # Delete relationships
                delete_query = f"""
                MATCH (v:{visitor_label})-[r:{relationship_name}]->(s:{stream_label})
                WHERE v.show = $show_name AND s.show = $show_name
                DELETE r
                RETURN COUNT(r) AS deleted
                """
                result = session.run(delete_query, show_name=show_name)
                deleted_count = result.single()["deleted"]
                
                self.logger.info(f"Cleared {deleted_count} job stream relationships for show {show_name} (initial count: {initial_count})")
                
            driver.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing Neo4j job stream relationships: {str(e)}")
            return False

    def run_old_processor_with_config(self, config):
        """Run the old processor with given config"""
        self.logger.info("Running old Neo4j job stream processor...")
        
        # Update config to use old output directory
        old_config = config.copy()
        old_config["output_dir"] = str(self.old_output_dir)
        
        # Clear existing relationships
        self.clear_neo4j_job_stream_relationships(old_config)
        
        processor = OldNeo4jJobStreamProcessor(old_config)
        processor.process(create_only_new=False)
        return processor

    def run_new_processor_with_config(self, config):
        """Run the new processor with given config"""
        self.logger.info("Running new Neo4j job stream processor...")
        
        # Update config to use new output directory
        new_config = config.copy()
        new_config["output_dir"] = str(self.new_output_dir)
        
        # Clear existing relationships
        self.clear_neo4j_job_stream_relationships(new_config)
        
        processor = NewNeo4jJobStreamProcessor(new_config)
        processor.process(create_only_new=False)
        return processor

    def compare_processor_statistics(self, old_processor, new_processor):
        """Compare statistics from both processors"""
        self.logger.info("Comparing processor statistics...")
        
        comparisons = {}
        
        if not hasattr(old_processor, "statistics") or not hasattr(new_processor, "statistics"):
            return {"error": "One or both processors don't have statistics attribute"}
        
        old_stats = old_processor.statistics
        new_stats = new_processor.statistics
        
        # Compare each statistic
        for stat_name in old_stats.keys():
            old_value = old_stats.get(stat_name, 0)
            new_value = new_stats.get(stat_name, 0)
            
            comparisons[stat_name] = {
                "old_value": old_value,
                "new_value": new_value,
                "identical": old_value == new_value,
                "difference": new_value - old_value if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)) else None
            }
        
        return comparisons

    def compare_neo4j_relationships(self, config):
        """Compare job stream relationships in Neo4j after both processors have run"""
        self.logger.info("Comparing Neo4j job stream relationships...")
        
        try:
            load_dotenv(config["env_file"])
            uri = os.getenv("NEO4J_URI")
            username = os.getenv("NEO4J_USERNAME")
            password = os.getenv("NEO4J_PASSWORD")
            
            if not all([uri, username, password]):
                raise ValueError("Missing Neo4j credentials")
            
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            # Get configuration values
            neo4j_config = config.get("neo4j", {})
            show_name = neo4j_config.get("show_name", "unknown")
            node_labels = neo4j_config.get("node_labels", {})
            relationships = neo4j_config.get("relationships", {})
            
            visitor_label = node_labels.get("visitor_this_year", "Visitor_this_year")
            stream_label = node_labels.get("stream", "Stream")
            relationship_name = relationships.get("job_stream", "job_to_stream")
            
            with driver.session() as session:
                # Count total relationships
                count_query = f"""
                MATCH (v:{visitor_label})-[r:{relationship_name}]->(s:{stream_label})
                WHERE v.show = $show_name AND s.show = $show_name
                RETURN COUNT(r) AS total_relationships
                """
                total_count = session.run(count_query, show_name=show_name).single()["total_relationships"]
                
                # Get relationship breakdown by job role
                breakdown_query = f"""
                MATCH (v:{visitor_label})-[r:{relationship_name}]->(s:{stream_label})
                WHERE v.show = $show_name AND s.show = $show_name
                RETURN v.job_role AS job_role, s.stream AS stream, COUNT(r) AS relationship_count
                ORDER BY v.job_role, s.stream
                """
                breakdown_result = session.run(breakdown_query, show_name=show_name)
                breakdown = [dict(record) for record in breakdown_result]
                
                # Get unique job roles and streams
                job_roles_query = f"""
                MATCH (v:{visitor_label})-[r:{relationship_name}]->(s:{stream_label})
                WHERE v.show = $show_name AND s.show = $show_name
                RETURN DISTINCT v.job_role AS job_role
                ORDER BY v.job_role
                """
                job_roles = [record["job_role"] for record in session.run(job_roles_query, show_name=show_name)]
                
                streams_query = f"""
                MATCH (v:{visitor_label})-[r:{relationship_name}]->(s:{stream_label})
                WHERE v.show = $show_name AND s.show = $show_name
                RETURN DISTINCT s.stream AS stream
                ORDER BY s.stream
                """
                streams = [record["stream"] for record in session.run(streams_query, show_name=show_name)]
                
            driver.close()
            
            return {
                "total_relationships": total_count,
                "unique_job_roles": len(job_roles),
                "unique_streams": len(streams),
                "job_roles": job_roles,
                "streams": streams,
                "breakdown": breakdown
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing Neo4j relationships: {str(e)}")
            return {"error": str(e)}

    def generate_report(self, old_config_path: str, new_config_path: str, 
                       stats_comparison: Dict[str, Any], 
                       neo4j_comparison: Dict[str, Any]) -> str:
        """Generate a detailed comparison report"""
        self.logger.info("Generating comparison report...")
        
        report = f"""
# Neo4j Job Stream Processor Comparison Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- Old Config: {old_config_path}
- New Config: {new_config_path}

## Summary
"""
        
        # Statistics comparison summary
        if "error" not in stats_comparison:
            identical_stats = sum(1 for comp in stats_comparison.values() if comp["identical"])
            total_stats = len(stats_comparison)
            report += f"- Statistics: {identical_stats}/{total_stats} identical\n"
        else:
            report += f"- Statistics: Error - {stats_comparison['error']}\n"
        
        # Neo4j comparison summary
        if "error" not in neo4j_comparison:
            report += f"- Total Job Stream Relationships: {neo4j_comparison['total_relationships']}\n"
            report += f"- Unique Job Roles: {neo4j_comparison['unique_job_roles']}\n"
            report += f"- Unique Streams: {neo4j_comparison['unique_streams']}\n"
        else:
            report += f"- Neo4j Comparison: Error - {neo4j_comparison['error']}\n"
        
        # Detailed statistics comparison
        if "error" not in stats_comparison:
            report += "\n## Statistics Comparison\n"
            for stat_name, comp in stats_comparison.items():
                status = "IDENTICAL" if comp["identical"] else "DIFFERENT"
                report += f"\n### {stat_name} {status}\n"
                report += f"- Old value: {comp['old_value']}\n"
                report += f"- New value: {comp['new_value']}\n"
                
                if comp["difference"] is not None:
                    report += f"- Difference: {comp['difference']}\n"
        
        # Neo4j relationship details
        if "error" not in neo4j_comparison:
            report += "\n## Neo4j Relationships\n"
            
            # Job roles
            if neo4j_comparison['job_roles']:
                report += f"\n### Job Roles ({len(neo4j_comparison['job_roles'])})\n"
                for job_role in neo4j_comparison['job_roles'][:10]:  # Show first 10
                    report += f"- {job_role}\n"
                if len(neo4j_comparison['job_roles']) > 10:
                    report += f"... and {len(neo4j_comparison['job_roles']) - 10} more\n"
            
            # Streams
            if neo4j_comparison['streams']:
                report += f"\n### Streams ({len(neo4j_comparison['streams'])})\n"
                for stream in neo4j_comparison['streams'][:10]:  # Show first 10
                    report += f"- {stream}\n"
                if len(neo4j_comparison['streams']) > 10:
                    report += f"... and {len(neo4j_comparison['streams']) - 10} more\n"
            
            # Relationship breakdown (top 20)
            if neo4j_comparison['breakdown']:
                report += f"\n### Top 20 Job Role -> Stream Relationships\n"
                for rel in neo4j_comparison['breakdown'][:20]:
                    report += f"- {rel['job_role']} -> {rel['stream']}: {rel['relationship_count']} relationships\n"
        
        return report

    def run_comparison(self, old_config_path: str = "config/config.yaml", 
                      new_config_path: str = "config/config_vet.yaml") -> bool:
        """Run the complete comparison between old and new processors"""
        self.logger.info("Starting Neo4j job stream processor comparison")
        print("🚀 Starting Neo4j Job Stream Processor Comparison")
        print("=" * 60)
        
        try:
            # Load configurations
            print("📁 Loading configurations...")
            old_config = load_config(old_config_path)
            new_config = load_config(new_config_path)
            
            # Run old processor
            print("🔄 Running old processor...")
            old_processor = self.run_old_processor_with_config(old_config)
            
            # Compare Neo4j state after old processor
            print("📊 Capturing Neo4j state after old processor...")
            old_neo4j_state = self.compare_neo4j_relationships(old_config)
            
            # Run new processor
            print("🔄 Running new processor...")
            new_processor = self.run_new_processor_with_config(new_config)
            
            # Compare Neo4j state after new processor
            print("📊 Capturing Neo4j state after new processor...")
            new_neo4j_state = self.compare_neo4j_relationships(new_config)
            
            # Compare processor statistics
            print("📈 Comparing processor statistics...")
            stats_comparison = self.compare_processor_statistics(old_processor, new_processor)
            
            # Compare final Neo4j states (should be identical if processors work the same)
            print("🔍 Comparing final Neo4j states...")
            neo4j_states_identical = (
                old_neo4j_state.get("total_relationships", 0) == new_neo4j_state.get("total_relationships", 0) and
                old_neo4j_state.get("unique_job_roles", 0) == new_neo4j_state.get("unique_job_roles", 0) and
                old_neo4j_state.get("unique_streams", 0) == new_neo4j_state.get("unique_streams", 0)
            )
            
            # Generate report
            print("📝 Generating comparison report...")
            report = self.generate_report(old_config_path, new_config_path, stats_comparison, new_neo4j_state)
            
            # Save report
            report_path = self.comparison_dir / f"comparison_report_{self.timestamp}.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # Save detailed comparison data
            comparison_data = {
                "timestamp": self.timestamp,
                "old_config": old_config_path,
                "new_config": new_config_path,
                "statistics_comparison": stats_comparison,
                "old_neo4j_state": old_neo4j_state,
                "new_neo4j_state": new_neo4j_state,
                "neo4j_states_identical": neo4j_states_identical,
                "summary": {
                    "stats_identical": sum(1 for comp in stats_comparison.values() if comp.get("identical", False)) if "error" not in stats_comparison else 0,
                    "stats_total": len(stats_comparison) if "error" not in stats_comparison else 0,
                    "neo4j_identical": neo4j_states_identical
                }
            }
            
            comparison_data_path = self.comparison_dir / f"comparison_data_{self.timestamp}.json"
            with open(comparison_data_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, indent=2, default=str, ensure_ascii=False)
            
            # Print summary
            print(f"\n{'='*60}")
            print("NEO4J JOB STREAM PROCESSOR COMPARISON SUMMARY")
            print(f"{'='*60}")
            
            if "error" not in stats_comparison:
                stats_identical = comparison_data["summary"]["stats_identical"]
                stats_total = comparison_data["summary"]["stats_total"]
                print(f"Statistics: {stats_identical}/{stats_total} identical")
            else:
                print(f"Statistics: Error - {stats_comparison['error']}")
            
            print(f"Neo4j States Identical: {'✅ Yes' if neo4j_states_identical else '❌ No'}")
            
            if "error" not in new_neo4j_state:
                print(f"Total Relationships Created: {new_neo4j_state['total_relationships']}")
                print(f"Unique Job Roles: {new_neo4j_state['unique_job_roles']}")
                print(f"Unique Streams: {new_neo4j_state['unique_streams']}")
            
            print(f"\n📁 Reports saved to: {self.comparison_dir}")
            print(f"📄 Detailed report: {report_path}")
            print(f"📊 Raw data: {comparison_data_path}")
            
            # Return success status
            return neo4j_states_identical and ("error" not in stats_comparison)
            
        except Exception as e:
            self.logger.error(f"Error in comparison: {str(e)}")
            print(f"❌ Error in comparison: {str(e)}")
            return False


def main():
    """Main function to run the comparison"""
    print("🔧 Neo4j Job Stream Processor Comparison Tool")
    print("=" * 50)
    
    comparison = Neo4jJobStreamProcessorComparison()
    
    # Run comparison with both config files
    success = comparison.run_comparison(
        old_config_path="config/config.yaml",
        new_config_path="config/config_vet.yaml"
    )
    
    if success:
        print("\n✅ Comparison completed successfully!")
        print("Both processors produced identical results.")
    else:
        print("\n❌ Comparison found differences!")
        print("Check the comparison report for details.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)