import logging
import os
import pandas as pd
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv
import csv
import inspect


class Neo4jSessionProcessor:
    """
    A class to process session data and upload it to Neo4j database.
    This class is responsible for loading CSV files and creating session nodes,
    stream nodes, and relationships between them in Neo4j.
    
    This is a generic version that works with any event type configuration.
    """

    def __init__(self, config):
        """
        Initialize the Neo4j Session Processor.

        Args:
            config (dict): Configuration dictionary containing paths and settings
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initializing Neo4j Session Processor"
        )

        self.config = config
        self.output_dir = os.path.join(config["output_dir"], "output")

        # Get event configuration
        self.event_config = config.get("event", {})
        
        # Get event names
        self.main_event_name = self.event_config.get("main_event_name", "bva")
        self.secondary_event_name = self.event_config.get("secondary_event_name", "lva")
        
        # Get show name for node attribute
        self.show_name = self.main_event_name  # Use main event name as show identifier

        # Load the environment variables to get Neo4j credentials
        load_dotenv(config["env_file"])
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")

        if not self.uri or not self.username or not self.password:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Missing Neo4j credentials in .env file"
            )
            raise ValueError("Missing Neo4j credentials in .env file")

        # Initialize statistics dictionary with generic event names
        self.statistics = {
            "nodes_created": {
                "sessions_this_year": 0,
                f"sessions_past_year_{self.main_event_name}": 0,
                f"sessions_past_year_{self.secondary_event_name}": 0,
                "streams": 0,
            },
            "nodes_skipped": {
                "sessions_this_year": 0,
                f"sessions_past_year_{self.main_event_name}": 0,
                f"sessions_past_year_{self.secondary_event_name}": 0,
                "streams": 0,
            },
            "relationships_created": {
                "sessions_this_year_has_stream": 0,
                "sessions_past_year_has_stream": 0,
            },
            "relationships_skipped": {
                "sessions_this_year_has_stream": 0,
                "sessions_past_year_has_stream": 0,
            },
        }

        # Backward compatibility - add traditional BVA/LVA keys
        self.statistics["nodes_created"]["sessions_past_year_bva"] = 0
        self.statistics["nodes_created"]["sessions_past_year_lva"] = 0
        self.statistics["nodes_skipped"]["sessions_past_year_bva"] = 0
        self.statistics["nodes_skipped"]["sessions_past_year_lva"] = 0

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j Session Processor initialized successfully"
        )

    def _test_connection(self):
        """Test the connection to Neo4j database"""
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Testing connection to Neo4j"
        )

        try:
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) AS count")
                count = result.single()["count"]
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Successfully connected to Neo4j. Total nodes: {count}"
                )
                return True
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Failed to connect to Neo4j: {str(e)}"
            )
            return False

    def load_csv_to_neo4j(self, csv_file_path, node_label, properties_map, unique_property, create_only_new=True):
        """
        Load data from CSV file to Neo4j as nodes.

        Args:
            csv_file_path (str): Path to the CSV file
            node_label (str): Label for the Neo4j nodes
            properties_map (dict): Mapping of CSV columns to node properties
            unique_property (str): Property to use for uniqueness constraint
            create_only_new (bool): If True, only create nodes that don't exist

        Returns:
            tuple: (nodes_created, nodes_skipped)
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Loading {csv_file_path} to Neo4j"
        )

        nodes_created = 0
        nodes_skipped = 0

        try:
            # Read CSV file
            data = pd.read_csv(csv_file_path)
            
            # Add show attribute to all records
            data['show'] = self.show_name
            
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Loaded CSV with {len(data)} rows"
            )

            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

            with driver.session() as session:
                for index, row in data.iterrows():
                    try:
                        # Prepare properties dictionary
                        properties = {}
                        for csv_col, neo4j_prop in properties_map.items():
                            if csv_col in row:
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
                        self.logger.error(
                            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error processing row {index}: {str(e)}"
                        )
                        continue

        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error loading CSV to Neo4j: {str(e)}"
            )
            raise
        finally:
            if 'driver' in locals():
                driver.close()

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Nodes created: {nodes_created}, skipped: {nodes_skipped}"
        )

        return nodes_created, nodes_skipped

    def create_unique_streams_and_relationships(self, create_only_new=True):
        """
        Create unique stream nodes and relationships between sessions and streams.

        Args:
            create_only_new (bool): If True, only create nodes and relationships that don't exist

        Returns:
            tuple: (streams_created, streams_skipped, relationships_created, relationships_skipped)
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Creating unique streams and relationships"
        )

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

                # Get all unique streams from session nodes
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
                        # Split by ';' and clean each stream
                        streams = [s.strip() for s in str(stream_value).split(";")]
                        unique_streams.update([s for s in streams if s])

                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Found {len(unique_streams)} unique streams"
                )

                # Create stream nodes
                for stream_name in unique_streams:
                    try:
                        # Check if stream node already exists
                        if create_only_new:
                            check_query = """
                            MATCH (s:Stream {name: $stream_name})
                            RETURN count(s) as count
                            """
                            result = session.run(check_query, stream_name=stream_name)
                            exists = result.single()["count"] > 0

                            if exists:
                                streams_skipped += 1
                                continue

                        # Create stream node with show attribute
                        create_stream_query = """
                        MERGE (s:Stream {name: $stream_name})
                        SET s.show = $show_name
                        RETURN s
                        """
                        session.run(create_stream_query, stream_name=stream_name, show_name=self.show_name)
                        streams_created += 1

                    except Exception as e:
                        self.logger.error(
                            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating stream node '{stream_name}': {str(e)}"
                        )
                        continue

                # Create relationships between sessions and streams
                relationship_query = """
                MATCH (sess), (stream:Stream)
                WHERE (sess:Sessions_this_year OR sess:Sessions_past_year)
                AND sess.stream IS NOT NULL AND sess.stream <> ''
                AND (sess.stream = stream.name OR sess.stream CONTAINS (';' + stream.name) OR sess.stream CONTAINS (stream.name + ';') OR sess.stream CONTAINS (';' + stream.name + ';'))
                """

                if create_only_new:
                    relationship_query += """
                    AND NOT EXISTS((sess)-[:HAS_STREAM]->(stream))
                    """

                relationship_query += """
                CREATE (sess)-[:HAS_STREAM]->(stream)
                RETURN count(*) as created
                """

                result = session.run(relationship_query)
                tracked_created = result.single()["created"]

                # Get final relationship count for verification
                final_result = session.run(
                    "MATCH ()-[r:HAS_STREAM]->() RETURN count(r) as count"
                )
                final_count = final_result.single()["count"]

                # Calculate actual created count
                actual_created = final_count - initial_count

                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships: Initial={initial_count}, Final={final_count}"
                )
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Tracked created={tracked_created}, Actual created={actual_created}"
                )

                # Update the tracked count to match actual database changes if there's a discrepancy
                if actual_created != tracked_created:
                    self.logger.warning(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Discrepancy in relationship counts: Tracked={tracked_created}, Actual={actual_created}"
                    )
                    # Use the actual count from the database for accuracy
                    tracked_created = actual_created

        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating relationships: {str(e)}"
            )
            raise
        finally:
            if 'driver' in locals():
                driver.close()

        return streams_created, streams_skipped, tracked_created, tracked_skipped

    def process(self, create_only_new=True):
        """
        Process session data and upload to Neo4j.

        Args:
            create_only_new (bool): If True, only create nodes and relationships that don't exist.

        Returns:
            dict: Statistics about created and skipped nodes and relationships
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Starting Neo4j session and stream data processing"
        )

        # Test the connection first
        if not self._test_connection():
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Cannot proceed with Neo4j upload due to connection failure"
            )
            return self.statistics

        # Process sessions from this year
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processing sessions from this year"
        )
        try:
            csv_file_path = os.path.join(
                self.output_dir, "session_this_filtered_valid_cols.csv"
            )
            data = pd.read_csv(csv_file_path)
            properties_map = {col: col for col in data.columns}
            node_label = "Sessions_this_year"

            # For Sessions_this_year, always recreate all nodes (delete existing ones first)
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Recreating all Sessions_this_year nodes"
            )

            # Delete existing Sessions_this_year nodes and their relationships
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            try:
                with driver.session() as session:
                    # Delete all Sessions_this_year nodes and their relationships
                    delete_query = """
                    MATCH (s:Sessions_this_year)
                    DETACH DELETE s
                    """
                    session.run(delete_query)
                    self.logger.info(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Deleted all existing Sessions_this_year nodes"
                    )
            finally:
                driver.close()

            # Now create all nodes fresh (with create_only_new=False for this specific case)
            nodes_created, nodes_skipped = self.load_csv_to_neo4j(
                csv_file_path,
                node_label,
                properties_map,
                "session_id",
                create_only_new=False,
            )

            self.statistics["nodes_created"]["sessions_this_year"] = nodes_created
            self.statistics["nodes_skipped"]["sessions_this_year"] = nodes_skipped
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error processing sessions from this year: {str(e)}"
            )

        # Process sessions from last year main event
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processing sessions from last year {self.main_event_name}"
        )
        try:
            csv_file_path = os.path.join(
                self.output_dir, f"session_last_filtered_valid_cols_{self.main_event_name}.csv"
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
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error processing sessions from last year {self.main_event_name}: {str(e)}"
            )

        # Process sessions from last year secondary event
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processing sessions from last year {self.secondary_event_name}"
        )
        try:
            csv_file_path = os.path.join(
                self.output_dir, f"session_last_filtered_valid_cols_{self.secondary_event_name}.csv"
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
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error processing sessions from last year {self.secondary_event_name}: {str(e)}"
            )

        # Create unique streams and relationships
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Creating unique streams and relationships"
        )
        try:
            streams_created, streams_skipped, relationships_created, relationships_skipped = self.create_unique_streams_and_relationships(create_only_new)

            self.statistics["nodes_created"]["streams"] = streams_created
            self.statistics["nodes_skipped"]["streams"] = streams_skipped

            # For relationships, we combine this year and past year
            self.statistics["relationships_created"]["sessions_this_year_has_stream"] = relationships_created // 2 if relationships_created > 0 else 0
            self.statistics["relationships_created"]["sessions_past_year_has_stream"] = relationships_created - self.statistics["relationships_created"]["sessions_this_year_has_stream"]
            self.statistics["relationships_skipped"]["sessions_this_year_has_stream"] = relationships_skipped // 2 if relationships_skipped > 0 else 0
            self.statistics["relationships_skipped"]["sessions_past_year_has_stream"] = relationships_skipped - self.statistics["relationships_skipped"]["sessions_this_year_has_stream"]

        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating stream relationships: {str(e)}"
            )

        # Calculate totals for summary
        total_nodes_created = sum(self.statistics["nodes_created"].values())
        total_nodes_skipped = sum(self.statistics["nodes_skipped"].values())
        total_relationships_created = sum(
            self.statistics["relationships_created"].values()
        )
        total_relationships_skipped = sum(
            self.statistics["relationships_skipped"].values()
        )

        # Add totals to statistics
        self.statistics["total_nodes_created"] = total_nodes_created
        self.statistics["total_nodes_skipped"] = total_nodes_skipped
        self.statistics["total_relationships_created"] = total_relationships_created
        self.statistics["total_relationships_skipped"] = total_relationships_skipped

        # Log summary statistics
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j session data processing summary:"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Total nodes created: {total_nodes_created}, skipped: {total_nodes_skipped}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Total relationships created: {total_relationships_created}, skipped: {total_relationships_skipped}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j session data processing completed"
        )

        return self.statistics