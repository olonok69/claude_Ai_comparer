#!/usr/bin/env python3
"""
Fixed Neo4j Session Processor

This processor creates session and stream nodes in Neo4j with proper relationship handling
that matches the old processor's logic exactly.
"""

import os
import sys
import json
import pandas as pd
import inspect
import logging
from neo4j import GraphDatabase
from dotenv import load_dotenv
from typing import Dict, Any, Tuple

class Neo4jSessionProcessor:
    """
    A processor for loading session data into Neo4j and creating relationships with streams.
    Fixed to match old processor's relationship creation logic exactly.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Neo4j Session Processor.

        Args:
            config: Configuration dictionary containing database settings and file paths
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv(config.get("env_file", "keys/.env"))
        
        # Neo4j connection details
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME") 
        self.password = os.getenv("NEO4J_PASSWORD")
        
        # Configuration
        self.output_dir = config.get("output_dir", "data/bva")
        
        # Event configuration for generic processing
        event_config = config.get("event", {})
        self.main_event_name = event_config.get("main_event_name", "bva")
        self.secondary_event_name = event_config.get("secondary_event_name", "lva")
        
        # Show name for node properties
        self.show_name = config.get("neo4j", {}).get("show_name", "bva")
        
        # Statistics tracking
        self.statistics = {
            "nodes_created": {
                "sessions_this_year": 0,
                "sessions_past_year_bva": 0,
                "sessions_past_year_lva": 0,
                "streams": 0
            },
            "nodes_skipped": {
                "sessions_this_year": 0,
                "sessions_past_year_bva": 0,
                "sessions_past_year_lva": 0,
                "streams": 0
            },
            "relationships_created": {
                "sessions_this_year_has_stream": 0,
                "sessions_past_year_has_stream": 0
            },
            "relationships_skipped": {
                "sessions_this_year_has_stream": 0,
                "sessions_past_year_has_stream": 0
            }
        }

        self.logger.info(f"Initialized Neo4j Session Processor for {self.main_event_name}")

    def _test_connection(self) -> bool:
        """Test connection to Neo4j database."""
        self.logger.info("Testing connection to Neo4j")
        
        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count")
                count = result.single()["count"]
                self.logger.info(f"Successfully connected to Neo4j. Database contains {count} nodes")
            driver.close()
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {str(e)}")
            return False

    def load_csv_to_neo4j(self, csv_file_path: str, node_label: str, properties_map: Dict[str, str], 
                         unique_property: str, create_only_new: bool = True) -> Tuple[int, int]:
        """
        Load CSV data into Neo4j as nodes.

        Args:
            csv_file_path: Path to the CSV file
            node_label: Label for the nodes to create
            properties_map: Mapping of CSV columns to Neo4j properties
            unique_property: Property to use for uniqueness checking
            create_only_new: If True, only create new nodes

        Returns:
            Tuple of (nodes_created, nodes_skipped)
        """
        self.logger.info(f"Loading CSV to Neo4j: {csv_file_path}")

        nodes_created = 0
        nodes_skipped = 0

        try:
            # Read CSV file
            data = pd.read_csv(csv_file_path)
            self.logger.info(f"Loaded CSV with {len(data)} rows")

            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

            with driver.session() as session:
                for index, row in data.iterrows():
                    try:
                        # Build properties dictionary
                        properties = {}
                        for csv_col, neo4j_prop in properties_map.items():
                            if csv_col in data.columns:
                                properties[neo4j_prop] = str(row[csv_col]) if pd.notna(row[csv_col]) else ""
                        
                        # Add show attribute
                        properties['show'] = self.show_name

                        # Check if node already exists (only if create_only_new is True)
                        if create_only_new:
                            check_query = f"""
                            MATCH (n:{node_label} {{{unique_property}: $unique_value}})
                            RETURN count(n) as count
                            """
                            result = session.run(check_query, unique_value=properties[unique_property])
                            exists = result.single()["count"] > 0

                            if exists:
                                nodes_skipped += 1
                                continue

                        # Create or merge node
                        if create_only_new:
                            # Use CREATE when we know it doesn't exist
                            query = f"""
                            CREATE (n:{node_label})
                            SET n += $properties
                            RETURN n
                            """
                        else:
                            # Use MERGE for upsert behavior
                            query = f"""
                            MERGE (n:{node_label} {{{unique_property}: $unique_value}})
                            SET n += $properties
                            RETURN n
                            """

                        if create_only_new:
                            session.run(query, properties=properties)
                        else:
                            session.run(query, 
                                      unique_value=properties[unique_property], 
                                      properties=properties)
                        
                        nodes_created += 1

                    except Exception as e:
                        self.logger.error(f"Error processing row {index}: {str(e)}")
                        continue

        except Exception as e:
            self.logger.error(f"Error loading CSV to Neo4j: {str(e)}")
            raise
        finally:
            if 'driver' in locals():
                driver.close()

        self.logger.info(f"Nodes created: {nodes_created}, skipped: {nodes_skipped}")
        return nodes_created, nodes_skipped

    def create_unique_streams_and_relationships(self, create_only_new: bool = True) -> Tuple[int, int, int, int]:
        """
        Create unique stream nodes and relationships between sessions and streams.
        Fixed to match old processor's logic exactly.

        Args:
            create_only_new: If True, only create nodes and relationships that don't exist

        Returns:
            Tuple: (streams_created, streams_skipped, relationships_created, relationships_skipped)
        """
        self.logger.info("Creating unique streams and relationships")

        streams_created = 0
        streams_skipped = 0
        tracked_created = 0
        tracked_skipped = 0

        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

            with driver.session() as session:
                # Get initial relationship count for verification
                initial_result = session.run(
                    "MATCH ()-[r:HAS_STREAM]->() RETURN count(r) as count"
                )
                initial_count = initial_result.single()["count"]

                # Get all unique streams from session nodes by splitting semicolon-separated values
                stream_query = """
                MATCH (s)
                WHERE s:Sessions_this_year OR s:Sessions_past_year
                AND s.stream IS NOT NULL AND s.stream <> ''
                RETURN DISTINCT s.stream as stream
                """
                
                result = session.run(stream_query)
                unique_streams = set()
                
                for record in result:
                    stream_value = record["stream"]
                    if stream_value:
                        # Split by semicolon and clean each stream
                        streams = [s.strip() for s in stream_value.split(";") if s.strip()]
                        unique_streams.update(streams)

                unique_streams = sorted(list(unique_streams))
                self.logger.info(f"Found {len(unique_streams)} unique streams")

                # Create stream nodes
                for stream_name in unique_streams:
                    if create_only_new:
                        # Check if stream node already exists
                        check_query = "MATCH (s:Stream {stream: $stream}) RETURN count(s) as count"
                        result = session.run(check_query, stream=stream_name)
                        if result.single()["count"] > 0:
                            streams_skipped += 1
                            continue

                    try:
                        # Create stream node with 'stream' property and show attribute (PRODUCTION STANDARD)
                        create_query = """
                        CREATE (s:Stream {stream: $stream, show: $show})
                        RETURN s
                        """
                        session.run(create_query, stream=stream_name, show=self.show_name)
                        streams_created += 1
                    except Exception as e:
                        self.logger.error(f"Error creating stream node '{stream_name}': {str(e)}")
                        continue

                # Create relationships between sessions and streams (FIXED LOGIC - matches old processor)
                tracked_created = 0
                
                # Get all sessions with their stream data
                session_query = """
                MATCH (s)
                WHERE s:Sessions_this_year OR s:Sessions_past_year
                AND s.stream IS NOT NULL AND s.stream <> ''
                RETURN s.session_id as session_id, s.stream as stream, 
                       CASE WHEN s:Sessions_this_year THEN 'Sessions_this_year' ELSE 'Sessions_past_year' END as session_type
                """
                
                session_results = session.run(session_query)
                
                for session_record in session_results:
                    session_id = session_record["session_id"]
                    session_type = session_record["session_type"]
                    streams_str = session_record["stream"]
                    
                    if not streams_str:
                        continue
                    
                    # Split stream string by semicolon and normalize (SAME AS OLD PROCESSOR)
                    stream_list = [
                        stream.strip().lower() for stream in streams_str.split(";")
                    ]
                    
                    for stream in stream_list:
                        if create_only_new:
                            # Check if relationship already exists
                            check_query = f"""
                                MATCH (s:{session_type} {{session_id: $session_id}})-[r:HAS_STREAM]->(st:Stream {{stream: $stream_name}})
                                RETURN count(r) as count
                            """
                            check_result = session.run(check_query, session_id=session_id, stream_name=stream).single()
                            
                            if check_result and check_result["count"] > 0:
                                tracked_skipped += 1
                                continue
                        
                        # Check if both session and stream nodes exist
                        nodes_exist_query = f"""
                            MATCH (s:{session_type} {{session_id: $session_id}})
                            MATCH (st:Stream {{stream: $stream_name}})
                            RETURN COUNT(*) > 0 AS exists
                        """
                        nodes_exist = session.run(nodes_exist_query, session_id=session_id, stream_name=stream).single()["exists"]
                        
                        if not nodes_exist:
                            # Skip if nodes don't exist (stream might not have been created due to case mismatch)
                            self.logger.warning(f"Cannot create relationship: Session {session_id} or Stream {stream} not found")
                            continue
                        
                        # Create relationship
                        create_query = f"""
                            MATCH (s:{session_type} {{session_id: $session_id}})
                            MATCH (st:Stream {{stream: $stream_name}})
                            CREATE (s)-[r:HAS_STREAM]->(st)
                            RETURN COUNT(r) AS created
                        """
                        
                        try:
                            create_result = session.run(create_query, session_id=session_id, stream_name=stream).single()
                            if create_result and create_result["created"] > 0:
                                tracked_created += 1
                        except Exception as e:
                            self.logger.warning(f"Error creating relationship for session {session_id} to stream {stream}: {str(e)}")

                # Get final relationship count for verification
                final_result = session.run(
                    "MATCH ()-[r:HAS_STREAM]->() RETURN count(r) as count"
                )
                final_count = final_result.single()["count"]

                # Calculate actual created relationships
                actual_created = final_count - initial_count

                self.logger.info(f"Relationships: Initial={initial_count}, Final={final_count}")
                self.logger.info(f"Tracked created={tracked_created}, Actual created={actual_created}")

                # Use actual count if there's a discrepancy
                if actual_created != tracked_created:
                    self.logger.warning(f"Relationship count discrepancy: Tracked={tracked_created}, Actual={actual_created}")
                    tracked_created = actual_created

        except Exception as e:
            self.logger.error(f"Error creating streams and relationships: {str(e)}")
            raise
        finally:
            if 'driver' in locals():
                driver.close()

        return streams_created, streams_skipped, tracked_created, tracked_skipped

    def process(self, create_only_new: bool = True) -> Dict[str, Any]:
        """
        Process session data and create Neo4j nodes and relationships.

        Args:
            create_only_new: If True, only create new nodes and relationships

        Returns:
            Dictionary containing processing statistics
        """
        self.logger.info("Starting Neo4j session and stream data processing")

        # Test connection first
        if not self._test_connection():
            self.logger.error("Cannot proceed with Neo4j processing due to connection failure")
            return self.statistics

        try:
            # Process sessions from this year
            self.logger.info("Processing sessions from this year")
            csv_file_path = os.path.join(self.output_dir, "output", "session_this_filtered_valid_cols.csv")
            
            if not create_only_new:
                # Delete existing Sessions_this_year nodes
                self.logger.info("Recreating all Sessions_this_year nodes")
                driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
                with driver.session() as session:
                    session.run("MATCH (n:Sessions_this_year) DETACH DELETE n")
                driver.close()
                self.logger.info("Deleted all existing Sessions_this_year nodes")

            data = pd.read_csv(csv_file_path)
            properties_map = {col: col for col in data.columns}
            node_label = "Sessions_this_year"

            nodes_created, nodes_skipped = self.load_csv_to_neo4j(
                csv_file_path, node_label, properties_map, "session_id", create_only_new
            )

            self.statistics["nodes_created"]["sessions_this_year"] = nodes_created
            self.statistics["nodes_skipped"]["sessions_this_year"] = nodes_skipped

            # Process sessions from last year (main event)
            self.logger.info(f"Processing sessions from last year {self.main_event_name}")
            csv_file_path = os.path.join(
                self.output_dir, f"output/session_last_filtered_valid_cols_{self.main_event_name}.csv"
            )
            data = pd.read_csv(csv_file_path)
            properties_map = {col: col for col in data.columns}
            node_label = "Sessions_past_year"

            nodes_created, nodes_skipped = self.load_csv_to_neo4j(
                csv_file_path, node_label, properties_map, "session_id", create_only_new
            )

            self.statistics["nodes_created"][f"sessions_past_year_{self.main_event_name}"] = nodes_created
            self.statistics["nodes_skipped"][f"sessions_past_year_{self.main_event_name}"] = nodes_skipped
            
            # Update backward compatible keys
            self.statistics["nodes_created"]["sessions_past_year_bva"] = nodes_created
            self.statistics["nodes_skipped"]["sessions_past_year_bva"] = nodes_skipped

            # Process sessions from last year (secondary event)
            self.logger.info(f"Processing sessions from last year {self.secondary_event_name}")
            csv_file_path = os.path.join(
                self.output_dir, f"output/session_last_filtered_valid_cols_{self.secondary_event_name}.csv"
            )
            data = pd.read_csv(csv_file_path)
            properties_map = {col: col for col in data.columns}
            node_label = "Sessions_past_year"

            nodes_created, nodes_skipped = self.load_csv_to_neo4j(
                csv_file_path, node_label, properties_map, "session_id", create_only_new
            )

            self.statistics["nodes_created"][f"sessions_past_year_{self.secondary_event_name}"] = nodes_created
            self.statistics["nodes_skipped"][f"sessions_past_year_{self.secondary_event_name}"] = nodes_skipped
            
            # Update backward compatible keys
            self.statistics["nodes_created"]["sessions_past_year_lva"] = nodes_created
            self.statistics["nodes_skipped"]["sessions_past_year_lva"] = nodes_skipped

            # Create unique streams and relationships
            self.logger.info("Creating unique streams and relationships")
            streams_created, streams_skipped, relationships_created, relationships_skipped = self.create_unique_streams_and_relationships(create_only_new)

            self.statistics["nodes_created"]["streams"] = streams_created
            self.statistics["nodes_skipped"]["streams"] = streams_skipped

            # For relationships, split between this year and past year approximately
            # This matches the old processor's statistics tracking
            total_this_year_sessions = self.statistics["nodes_created"]["sessions_this_year"]
            total_past_year_sessions = (self.statistics["nodes_created"]["sessions_past_year_bva"] + 
                                      self.statistics["nodes_created"]["sessions_past_year_lva"])
            total_sessions = total_this_year_sessions + total_past_year_sessions
            
            if total_sessions > 0:
                this_year_ratio = total_this_year_sessions / total_sessions
                self.statistics["relationships_created"]["sessions_this_year_has_stream"] = int(relationships_created * this_year_ratio)
                self.statistics["relationships_created"]["sessions_past_year_has_stream"] = (
                    relationships_created - self.statistics["relationships_created"]["sessions_this_year_has_stream"]
                )
            else:
                self.statistics["relationships_created"]["sessions_this_year_has_stream"] = 0
                self.statistics["relationships_created"]["sessions_past_year_has_stream"] = relationships_created

            self.statistics["relationships_skipped"]["sessions_this_year_has_stream"] = 0
            self.statistics["relationships_skipped"]["sessions_past_year_has_stream"] = relationships_skipped

            # Log summary
            self.logger.info("Neo4j session data processing summary:")
            total_nodes = (self.statistics["nodes_created"]["sessions_this_year"] + 
                          self.statistics["nodes_created"]["sessions_past_year_bva"] + 
                          self.statistics["nodes_created"]["sessions_past_year_lva"] + 
                          self.statistics["nodes_created"]["streams"])
            total_skipped = (self.statistics["nodes_skipped"]["sessions_this_year"] + 
                           self.statistics["nodes_skipped"]["sessions_past_year_bva"] + 
                           self.statistics["nodes_skipped"]["sessions_past_year_lva"] + 
                           self.statistics["nodes_skipped"]["streams"])
            
            self.logger.info(f"Total nodes created: {total_nodes}, skipped: {total_skipped}")
            
            total_relationships = (self.statistics["relationships_created"]["sessions_this_year_has_stream"] + 
                                 self.statistics["relationships_created"]["sessions_past_year_has_stream"])
            total_rel_skipped = (self.statistics["relationships_skipped"]["sessions_this_year_has_stream"] + 
                               self.statistics["relationships_skipped"]["sessions_past_year_has_stream"])
            
            self.logger.info(f"Total relationships created: {total_relationships}, skipped: {total_rel_skipped}")
            self.logger.info("Neo4j session data processing completed")

        except Exception as e:
            self.logger.error(f"Error in Neo4j session processing: {str(e)}")
            raise

        return self.statistics


def main():
    """Main function for testing."""
    import sys
    sys.path.insert(0, os.getcwd())
    
    from utils.config_utils import load_config
    from utils.logging_utils import setup_logging

    # Setup logging
    logger = setup_logging(log_file="logs/neo4j_session_processor.log")
    
    try:
        # Load configuration
        config = load_config("config/config_vet.yaml")
        
        # Create processor and run
        processor = Neo4jSessionProcessor(config)
        stats = processor.process(create_only_new=False)
        
        print("Processing completed successfully!")
        print(f"Statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()