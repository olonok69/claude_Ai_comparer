import logging
import os
import pandas as pd
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv
import inspect


class Neo4jJobStreamProcessor:
    """
    A class to process job role to stream mappings and create relationships in Neo4j.
    This class is responsible for creating relationships between visitor nodes and
    stream nodes based on job roles.
    """

    def __init__(self, config):
        """
        Initialize the Neo4j Job Stream Processor.

        Args:
            config (dict): Configuration dictionary containing paths and settings
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initializing Neo4j Job Stream Processor"
        )

        self.config = config
        self.input_dir = config["output_dir"]

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

        self.statistics = {
            "initial_relationship_count": 0,
            "final_relationship_count": 0,
            "relationships_created": 0,
            "relationships_skipped": 0,
            "relationships_not_found": 0,
            "stream_mappings_applied": 0,
            "visitor_nodes_processed": 0,
            "job_roles_processed": 0,
            "stream_matches_found": 0,
        }

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j Job Stream Processor initialized successfully"
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
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Successfully connected to Neo4j. Database contains {count} nodes"
                )
            driver.close()
            return True
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Failed to connect to Neo4j: {str(e)}"
            )
            return False

    def load_job_stream_mapping(self, mapping_file_path):
        """
        Load the job role to stream mapping from a CSV file.

        Args:
            mapping_file_path (str): Path to the CSV file containing the mapping.

        Returns:
            dict: Dictionary mapping job roles to streams.
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Loading job to stream mapping from: {mapping_file_path}"
        )

        try:
            map_job_stream = pd.read_csv(mapping_file_path)

            # Verify the CSV has the expected structure
            if "Job Role" not in map_job_stream.columns:
                self.logger.error(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - CSV file does not contain 'Job Role' column"
                )
                raise ValueError("CSV file does not contain 'Job Role' column")

            # Convert the DataFrame to a dictionary using job role as the index
            job_stream_mapping = json.loads(
                map_job_stream.set_index("Job Role").to_json(orient="index")
            )

            # Log some stats about the mapping
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Loaded mapping for {len(job_stream_mapping)} job roles"
            )

            return job_stream_mapping
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error loading job to stream mapping: {str(e)}"
            )
            raise

    def _map_stream_name(self, stream_name):
        """
        Apply mapping rules to stream names.

        Args:
            stream_name (str): Original stream name

        Returns:
            tuple: (mapped_stream_name, mapping_applied)
        """
        mapping_applied = False

        # Apply mapping rules
        # if stream_name == "geriatric medicine":
        #     mapped_stream_name = "internal medicine"
        #     mapping_applied = True
        # elif stream_name == "welfare":
        #     mapped_stream_name = "animal welfare"
        #     mapping_applied = True
        # else:
        mapped_stream_name = stream_name

        return mapped_stream_name, mapping_applied

    def _count_relationships(self, session):
        """Count the total number of job_to_stream relationships"""
        result = session.run(
            """
            MATCH (:Visitor_this_year)-[r:job_to_stream]->(:Stream)
            RETURN COUNT(r) AS count
            """
        )
        return result.single()["count"]

    def create_job_stream_relationships(self, job_stream_mapping, create_only_new=True):
        """
        Create relationships between visitor nodes and stream nodes based on job roles.
        Enhanced with improved reconciliation and stream name mapping.

        Args:
            job_stream_mapping (dict): Dictionary mapping job roles to streams.
            create_only_new (bool): If True, only create relationships that don't exist.

        Returns:
            dict: Statistics about the relationship creation process.
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Creating job to stream relationships with enhanced tracking"
        )

        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

        # First count the initial relationships
        with driver.session() as session:
            initial_count = self._count_relationships(session)
            self.statistics["initial_relationship_count"] = initial_count
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initial relationship count: {initial_count}"
            )

        # Batch processing function with enhanced tracking
        def batch_create_relationships(tx, job_stream_mapping, create_only_new):
            batch_stats = {
                "relationships_created": 0,
                "relationships_skipped": 0,
                "relationships_not_found": 0,
                "stream_mappings_applied": 0,
                "visitor_nodes_processed": 0,
                "job_roles_processed": 0,
                "stream_matches_found": 0,
            }

            # Get all unique job roles from the database that match our mapping
            # Fixed to handle both 'NA' and 'na' job roles
            # When create_only_new is True, only process visitors without recommendations
            additional_filter = ""
            if create_only_new:
                additional_filter = (
                    ' AND (v.has_recommendation IS NULL OR v.has_recommendation = "0")'
                )

            job_roles_query = f"""
            MATCH (v:Visitor_this_year)
            WHERE v.job_role IS NOT NULL 
              AND LOWER(v.job_role) <> 'na'
              AND v.job_role IN $job_roles
              {additional_filter}
            RETURN DISTINCT v.job_role as job_role
            """

            job_roles_result = tx.run(
                job_roles_query, job_roles=list(job_stream_mapping.keys())
            )
            matched_job_roles = [record["job_role"] for record in job_roles_result]
            batch_stats["job_roles_processed"] = len(matched_job_roles)

            # Process each job role
            for job_role in matched_job_roles:
                # Get applicable streams for this job role with mapping applied
                applicable_streams = []
                mapped_streams = []

                for stream_name, applies in job_stream_mapping[job_role].items():
                    if applies == "YES":
                        # Apply stream name mapping
                        mapped_stream_name, mapping_applied = self._map_stream_name(
                            stream_name
                        )

                        if mapping_applied:
                            batch_stats["stream_mappings_applied"] += 1
                            self.logger.info(
                                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Mapped stream name: '{stream_name}' to '{mapped_stream_name}'"
                            )

                        mapped_streams.append(mapped_stream_name)

                # Get unique streams after mapping
                applicable_streams = list(set(mapped_streams))
                batch_stats["stream_matches_found"] += len(applicable_streams)

                # Count visitors with this job role
                count_query = """
                MATCH (v:Visitor_this_year)
                WHERE v.job_role = $job_role
                RETURN count(v) AS visitor_count
                """
                visitor_count = tx.run(count_query, job_role=job_role).single()[
                    "visitor_count"
                ]
                batch_stats["visitor_nodes_processed"] += visitor_count

                # For each applicable stream, check if it exists
                for stream_name in applicable_streams:
                    stream_exists_query = """
                    MATCH (s:Stream)
                    WHERE s.stream = $stream_name
                    RETURN count(s) > 0 AS exists
                    """

                    stream_exists = tx.run(
                        stream_exists_query, stream_name=stream_name
                    ).single()["exists"]

                    if not stream_exists:
                        self.logger.warning(
                            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Stream node with name '{stream_name}' not found"
                        )
                        batch_stats["relationships_not_found"] += visitor_count
                        continue

                    # Create relationships for this job role and stream
                    if create_only_new:
                        # This version checks for existing relationships first
                        create_query = """
                        MATCH (v:Visitor_this_year)
                        WHERE v.job_role = $job_role
                        
                        MATCH (s:Stream)
                        WHERE s.stream = $stream_name
                        
                        WITH v, s
                        WHERE NOT exists((v)-[:job_to_stream]->(s))
                        
                        CREATE (v)-[r:job_to_stream]->(s)
                        RETURN count(r) AS created
                        """

                        result = tx.run(
                            create_query, job_role=job_role, stream_name=stream_name
                        )
                        created = result.single()["created"]

                        # Calculate how many would have been created without the WHERE NOT exists condition
                        potential = visitor_count
                        skipped = potential - created

                        batch_stats["relationships_created"] += created
                        batch_stats["relationships_skipped"] += skipped
                    else:
                        # This version creates all possible relationships without checking
                        create_query = """
                        MATCH (v:Visitor_this_year)
                        WHERE v.job_role = $job_role
                        
                        MATCH (s:Stream)
                        WHERE s.stream = $stream_name
                        
                        MERGE (v)-[r:job_to_stream]->(s)
                        RETURN count(r) AS created
                        """

                        result = tx.run(
                            create_query, job_role=job_role, stream_name=stream_name
                        )
                        created = result.single()["created"]
                        batch_stats["relationships_created"] += created

            return batch_stats

        try:
            with driver.session() as session:
                batch_stats = session.execute_write(
                    batch_create_relationships, job_stream_mapping, create_only_new
                )

                # Update our statistics
                self.statistics.update(batch_stats)

                # Now count the final relationships
                final_count = self._count_relationships(session)
                self.statistics["final_relationship_count"] = final_count

                # Calculate the actual relationships created based on database counts
                actual_created = final_count - initial_count

                # Check for discrepancies and log them
                if actual_created != self.statistics["relationships_created"]:
                    self.logger.warning(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Discrepancy detected! "
                        f"Database shows {actual_created} new relationships, but code tracked {self.statistics['relationships_created']}."
                    )
                    self.logger.warning(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - This may indicate concurrent database access or transaction issues."
                    )

                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initial relationship count: {initial_count}"
                )
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Final relationship count: {final_count}"
                )
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Job to stream relationships - created: {self.statistics['relationships_created']}, skipped: {self.statistics['relationships_skipped']}"
                )
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processed {self.statistics['visitor_nodes_processed']} visitor nodes with {self.statistics['job_roles_processed']} matching job roles"
                )

                if self.statistics["stream_mappings_applied"] > 0:
                    self.logger.info(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Stream name mappings applied: {self.statistics['stream_mappings_applied']}"
                    )
                    self.logger.info(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - - 'geriatric medicine' mapped to 'internal medicine'"
                    )
                    self.logger.info(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - - 'welfare' mapped to 'animal welfare'"
                    )

                if self.statistics["relationships_not_found"] > 0:
                    self.logger.warning(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships not created due to missing nodes: {self.statistics['relationships_not_found']}"
                    )
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating job to stream relationships: {str(e)}"
            )
            raise
        finally:
            driver.close()

        return self.statistics

    def print_reconciliation_report(self):
        """Print a detailed reconciliation report with all statistics"""
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Job to stream relationship reconciliation report:"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initial relationship count: {self.statistics['initial_relationship_count']}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Final relationship count: {self.statistics['final_relationship_count']}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - New relationships created: {self.statistics['relationships_created']}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships skipped (already exist): {self.statistics['relationships_skipped']}"
        )

        if self.statistics["relationships_not_found"] > 0:
            self.logger.warning(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships not created due to missing nodes: {self.statistics['relationships_not_found']}"
            )

        if self.statistics["stream_mappings_applied"] > 0:
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Stream name mappings applied: {self.statistics['stream_mappings_applied']}"
            )
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - - 'geriatric medicine' mapped to 'internal medicine'"
            )
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - - 'welfare' mapped to 'animal welfare'"
            )

        # Check for discrepancies
        actual_created = (
            self.statistics["final_relationship_count"]
            - self.statistics["initial_relationship_count"]
        )
        if actual_created != self.statistics["relationships_created"]:
            self.logger.warning(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - WARNING: Discrepancy detected! "
                f"Database shows {actual_created} new relationships, but code tracked {self.statistics['relationships_created']}."
            )
            self.logger.warning(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - This may indicate concurrent database access or transaction issues."
            )

    def process(self, create_only_new=True):
        """
        Process job to stream mappings and create relationships in Neo4j.

        Args:
            create_only_new (bool): If True, only create relationships that don't exist.
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Starting Neo4j job to stream relationship processing"
        )

        # Test the connection first
        if not self._test_connection():
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Cannot proceed with Neo4j processing due to connection failure"
            )
            return

        # Get the mapping file path
        mapping_file_path = os.path.join(self.input_dir, "job_to_stream.csv")

        try:
            # Load the mapping
            job_stream_mapping = self.load_job_stream_mapping(mapping_file_path)

            # Create the relationships
            self.create_job_stream_relationships(job_stream_mapping, create_only_new)

            # Print detailed reconciliation report
            self.print_reconciliation_report()

            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j job to stream relationship processing completed"
            )
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error in job to stream relationship processing: {str(e)}"
            )
            raise
