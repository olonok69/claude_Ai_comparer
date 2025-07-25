import logging
import os
import pandas as pd
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv
import inspect


class Neo4jSpecializationStreamProcessor:
    """
    A class to process specialization to stream mappings and create relationships in Neo4j.
    This class is responsible for creating relationships between visitor nodes and
    stream nodes based on practice specializations.
    """

    def __init__(self, config):
        """
        Initialize the Neo4j Specialization Stream Processor.

        Args:
            config (dict): Configuration dictionary containing paths and settings
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initializing Neo4j Specialization Stream Processor"
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

        # Initialize statistics structure
        self.statistics = {
            "initial_count": 0,
            "final_count": 0,
            "relationships_created": {
                "visitor_this_year": 0,
                "visitor_last_year_bva": 0,
                "visitor_last_year_lva": 0,
                "total": 0,
            },
            "relationships_skipped": {
                "visitor_this_year": 0,
                "visitor_last_year_bva": 0,
                "visitor_last_year_lva": 0,
                "total": 0,
            },
            "relationships_failed": {
                "visitor_this_year": 0,
                "visitor_last_year_bva": 0,
                "visitor_last_year_lva": 0,
                "total": 0,
            },
            "visitor_nodes_processed": {
                "visitor_this_year": 0,
                "visitor_last_year_bva": 0,
                "visitor_last_year_lva": 0,
                "total": 0,
            },
            "specializations_processed": 0,
            "specializations_mapped": 0,
            "stream_matches_found": 0,
        }

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j Specialization Stream Processor initialized successfully"
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

    def load_mappings(self):
        """
        Load the specialization mappings from CSV files.

        Returns:
            tuple: (map_specialization, specialization_stream_mapping)
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Loading specialization mappings"
        )

        try:
            # Load specialization mapping
            map_specialization = {
                "Wildlife": "Other",
                "Small Animal": "Companion Animal",
                "Mixed": "Other",
                "Dairy": "Farm",
                "Poultry": "Farm",
                "Cattle": "Farm",
                "Pigs": "Farm",
                "Sheep": "Farm",
            }

            # Load specialization to stream mapping
            specialization_stream_file = os.path.join(
                self.input_dir, "spezialization_to_stream.csv"
            )
            map_specialization_stream = pd.read_csv(specialization_stream_file)

            # Convert to dictionary
            specialization_stream_mapping = json.loads(
                map_specialization_stream.set_index("spezialization").to_json(
                    orient="index"
                )
            )

            # Log some stats about the mapping
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Loaded mapping for {len(specialization_stream_mapping)} specializations"
            )

            return map_specialization, specialization_stream_mapping
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error loading specialization mappings: {str(e)}"
            )
            raise

    def _count_relationships(self, session):
        """Count the total number of specialization_to_stream relationships"""
        result = session.run(
            """
            MATCH ()-[r:specialization_to_stream]->(:Stream)
            RETURN COUNT(r) AS count
            """
        )
        return result.single()["count"]

    def _print_reconciliation_report(self):
        """Print a detailed reconciliation report with all counts"""
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Specialization to stream relationship processing summary:"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initial relationship count: {self.statistics['initial_count']}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Final relationship count: {self.statistics['final_count']}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Actual new relationships (database): {self.statistics['final_count'] - self.statistics['initial_count']}"
        )

        for visitor_type in [
            "visitor_this_year",
            "visitor_last_year_bva",
            "visitor_last_year_lva",
        ]:
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - {visitor_type.capitalize()} nodes processed: {self.statistics['visitor_nodes_processed'][visitor_type]}"
            )
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - {visitor_type.capitalize()} relationships created: {self.statistics['relationships_created'][visitor_type]}"
            )
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - {visitor_type.capitalize()} relationships skipped: {self.statistics['relationships_skipped'][visitor_type]}"
            )
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - {visitor_type.capitalize()} relationships failed: {self.statistics['relationships_failed'][visitor_type]}"
            )

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Total visitor nodes processed: {self.statistics['visitor_nodes_processed']['total']}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Total specializations processed: {self.statistics['specializations_processed']}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Total specializations mapped: {self.statistics['specializations_mapped']}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Total stream matches found: {self.statistics['stream_matches_found']}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Total relationships created: {self.statistics['relationships_created']['total']}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Total relationships skipped: {self.statistics['relationships_skipped']['total']}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Total relationships failed: {self.statistics['relationships_failed']['total']}"
        )

        # Check for discrepancies
        actual_created = (
            self.statistics["final_count"] - self.statistics["initial_count"]
        )
        if actual_created != self.statistics["relationships_created"]["total"]:
            self.logger.warning(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - WARNING: Discrepancy detected! Database shows {actual_created} new relationships, but code tracked {self.statistics['relationships_created']['total']}."
            )
            self.logger.warning(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - This may indicate concurrent database access or transaction issues."
            )

    @staticmethod
    def _process_visitor_type(
        tx,
        node_label,
        specialization_field,
        map_specialization,
        specialization_stream_mapping,
        stats,
        create_only_new=True,
    ):
        """
        Process a specific type of visitor nodes to create specialization to stream relationships.

        Args:
            tx: Neo4j transaction
            node_label (str): Label of the visitor nodes (e.g., "Visitor_this_year")
            specialization_field (str): Name of the field containing specialization info
            map_specialization (dict): Dictionary mapping specialization names to standard format
            specialization_stream_mapping (dict): Dictionary mapping specializations to applicable streams
            stats (dict): Statistics dictionary to update
            create_only_new (bool): If True, only create relationships that don't exist (default: True)

        Returns:
            dict: Updated statistics dictionary
        """
        # Get all visitor nodes with valid specializations
        # When create_only_new is True, only process visitors without recommendations for Visitor_this_year
        additional_filter = ""
        if create_only_new and node_label == "Visitor_this_year":
            additional_filter = (
                ' AND (v.has_recommendation IS NULL OR v.has_recommendation = "0")'
            )

        query = f"""
        MATCH (v:{node_label})
        WHERE v.{specialization_field} IS NOT NULL 
        AND LOWER(v.{specialization_field}) <> 'na'
        {additional_filter}
        RETURN v, v.{specialization_field} as specializations, 
            v.BadgeId as badge_id, v.Email as email
        """

        results = tx.run(query)
        visitor_type_key = node_label.lower()

        for record in results:
            visitor = record["v"]
            specializations_text = record["specializations"]
            badge_id = record.get("badge_id")
            email = record.get("email")

            stats["visitor_nodes_processed"][visitor_type_key] += 1

            # Skip if BadgeId or Email is not available (needed for identification)
            if not badge_id or not email:
                continue

            # Skip processing if the specialization text is explicitly 'NA'
            if specializations_text.upper() == "NA":
                continue

            # Split specializations by semicolon
            specializations = specializations_text.split(";")

            # Process each specialization
            for spec in specializations:
                spec = spec.strip()

                # Map specialization if needed
                if spec in map_specialization:
                    mapped_spec = map_specialization[spec]
                    stats["specializations_mapped"] += 1
                else:
                    mapped_spec = spec

                # Check if this specialization exists in our mapping
                if mapped_spec in specialization_stream_mapping:
                    stream_dict = specialization_stream_mapping[mapped_spec]
                    stats["specializations_processed"] += 1

                    # For each stream that applies to this specialization
                    for stream_name, applies in stream_dict.items():
                        if applies == "YES":
                            stats["stream_matches_found"] += 1

                            # First find the Stream node separately to avoid Cartesian product
                            stream_query = """
                            MATCH (s:Stream)
                            WHERE LOWER(s.stream) = LOWER($stream_name)
                            RETURN s
                            """

                            stream_result = tx.run(
                                stream_query, stream_name=stream_name
                            )
                            stream_record = stream_result.single()

                            # Skip if stream doesn't exist
                            if not stream_record:
                                stats["relationships_failed"][visitor_type_key] += 1
                                continue

                            if create_only_new:
                                # Check if the relationship already exists
                                relationship_exists_query = f"""
                                MATCH (v:{node_label} {{BadgeId: $badge_id, Email: $email}})
                                MATCH (s:Stream)
                                WHERE LOWER(s.stream) = LOWER($stream_name)
                                RETURN EXISTS((v)-[:specialization_to_stream]->(s)) AS exists
                                """

                                relationship_exists = tx.run(
                                    relationship_exists_query,
                                    badge_id=badge_id,
                                    email=email,
                                    stream_name=stream_name,
                                ).single()["exists"]

                                if relationship_exists:
                                    stats["relationships_skipped"][
                                        visitor_type_key
                                    ] += 1
                                    continue

                            # Create the relationship (or MERGE if create_only_new=False)
                            if create_only_new:
                                create_rel_query = f"""
                                MATCH (v:{node_label} {{BadgeId: $badge_id, Email: $email}})
                                MATCH (s:Stream WHERE LOWER(s.stream) = LOWER($stream_name))
                                CREATE (v)-[r:specialization_to_stream]->(s)
                                RETURN r
                                """
                            else:
                                create_rel_query = f"""
                                MATCH (v:{node_label} {{BadgeId: $badge_id, Email: $email}})
                                MATCH (s:Stream WHERE LOWER(s.stream) = LOWER($stream_name))
                                MERGE (v)-[r:specialization_to_stream]->(s)
                                RETURN r
                                """

                            try:
                                tx.run(
                                    create_rel_query,
                                    badge_id=badge_id,
                                    email=email,
                                    stream_name=stream_name,
                                )
                                stats["relationships_created"][visitor_type_key] += 1
                            except Exception as e:
                                stats["relationships_failed"][visitor_type_key] += 1

        return stats

    def create_specialization_stream_relationships(
        self, map_specialization, specialization_stream_mapping, create_only_new=True
    ):
        """
        Create relationships between visitor nodes and stream nodes based on practice specializations.

        Args:
            map_specialization (dict): Dictionary mapping specialization names to standard format
            specialization_stream_mapping (dict): Dictionary mapping specializations to applicable streams
            create_only_new (bool): If True, only create relationships that don't exist (default: True)

        Returns:
            dict: Statistics about the relationship creation process
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Creating specialization to stream relationships (create_only_new={create_only_new})"
        )

        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

        try:
            with driver.session() as session:
                # Count initial relationships for reconciliation
                self.statistics["initial_count"] = self._count_relationships(session)
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initial relationship count: {self.statistics['initial_count']}"
                )

                # Process all three visitor types
                visitor_types = [
                    ("Visitor_this_year", "what_type_does_your_practice_specialise_in"),
                    ("Visitor_last_year_bva", "what_areas_do_you_specialise_in"),
                    ("Visitor_last_year_lva", "what_areas_do_you_specialise_in"),
                ]

                for node_label, specialization_field in visitor_types:
                    self.logger.info(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processing {node_label} nodes"
                    )

                    # Modify the process_visitor_type to include create_only_new parameter
                    def process_with_create_flag(tx):
                        return self._process_visitor_type(
                            tx,
                            node_label,
                            specialization_field,
                            map_specialization,
                            specialization_stream_mapping,
                            self.statistics,
                            create_only_new,
                        )

                    # Execute write transaction for this visitor type
                    session.execute_write(process_with_create_flag)

                # Calculate totals
                self.statistics["visitor_nodes_processed"]["total"] = sum(
                    self.statistics["visitor_nodes_processed"][k]
                    for k in [
                        "visitor_this_year",
                        "visitor_last_year_bva",
                        "visitor_last_year_lva",
                    ]
                )

                self.statistics["relationships_created"]["total"] = sum(
                    self.statistics["relationships_created"][k]
                    for k in [
                        "visitor_this_year",
                        "visitor_last_year_bva",
                        "visitor_last_year_lva",
                    ]
                )

                self.statistics["relationships_skipped"]["total"] = sum(
                    self.statistics["relationships_skipped"][k]
                    for k in [
                        "visitor_this_year",
                        "visitor_last_year_bva",
                        "visitor_last_year_lva",
                    ]
                )

                self.statistics["relationships_failed"]["total"] = sum(
                    self.statistics["relationships_failed"][k]
                    for k in [
                        "visitor_this_year",
                        "visitor_last_year_bva",
                        "visitor_last_year_lva",
                    ]
                )

                # Count final relationships for reconciliation
                self.statistics["final_count"] = self._count_relationships(session)
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Final relationship count: {self.statistics['final_count']}"
                )

        finally:
            driver.close()

        return self.statistics

    def process(self, create_only_new=True):
        """
        Process specialization to stream mappings and create relationships in Neo4j.

        Args:
            create_only_new (bool): If True, only create relationships that don't exist (default: True)
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Starting Neo4j specialization to stream relationship processing (create_only_new={create_only_new})"
        )

        # Test the connection first
        if not self._test_connection():
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Cannot proceed with Neo4j processing due to connection failure"
            )
            return

        try:
            # Load the mappings
            map_specialization, specialization_stream_mapping = self.load_mappings()

            # Create the relationships with the create_only_new parameter
            self.create_specialization_stream_relationships(
                map_specialization, specialization_stream_mapping, create_only_new
            )

            # Print reconciliation report
            self._print_reconciliation_report()

            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j specialization to stream relationship processing completed"
            )

            return self.statistics
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error in specialization to stream relationship processing: {str(e)}"
            )
            raise
