import logging
import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import csv
import inspect


class Neo4jVisitorProcessor:
    """
    A class to process visitor data and upload it to Neo4j database.
    This class is responsible for loading CSV files and creating visitor nodes in Neo4j.
    """

    def __init__(self, config):
        """
        Initialize the Neo4j Visitor Processor.

        Args:
            config (dict): Configuration dictionary containing paths and settings
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initializing Neo4j Visitor Processor"
        )

        self.config = config
        self.output_dir = os.path.join(config["output_dir"], "output")

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
            "nodes_created": {
                "visitor_this_year": 0,
                "visitor_last_year_bva": 0,
                "visitor_last_year_lva": 0,
            },
            "nodes_skipped": {
                "visitor_this_year": 0,
                "visitor_last_year_bva": 0,
                "visitor_last_year_lva": 0,
            },
        }

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j Visitor Processor initialized successfully"
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

    def load_csv_to_neo4j(
        self, csv_file_path, node_label, properties_map, create_only_new=True
    ):
        """
        Load data from a CSV file into Neo4j.

        Args:
            csv_file_path (str): Path to the CSV file.
            node_label (str): Label to apply to the nodes.
            properties_map (dict): A dictionary mapping CSV column names to Neo4j property names.
            create_only_new (bool): If True, only create nodes that don't exist in the database.

        Returns:
            tuple: (nodes_created, nodes_skipped)
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Loading CSV to Neo4j: {csv_file_path}"
        )

        nodes_created = 0
        nodes_skipped = 0

        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

        def create_nodes(
            tx, csv_file_path, node_label, properties_map, create_only_new
        ):
            nonlocal nodes_created, nodes_skipped

            with open(csv_file_path, "r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    properties = {
                        prop_neo4j: row[col]
                        for col, prop_neo4j in properties_map.items()
                        if col in row and row[col]
                    }

                    # Skip empty rows
                    if not properties:
                        continue

                    # Add has_recommendation attribute with default value "0"
                    properties["has_recommendation"] = "0"

                    # Use BadgeId as the unique identifier property
                    badge_id_key = None
                    for col, prop in properties_map.items():
                        if col.lower() == "badgeid" or prop.lower() == "badgeid":
                            badge_id_key = prop
                            break

                    if not badge_id_key or badge_id_key not in properties:
                        # Skip records without BadgeId
                        self.logger.warning(
                            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Skipping record without BadgeId"
                        )
                        continue

                    badge_id = properties[badge_id_key]

                    if create_only_new:
                        # Check if node already exists
                        check_query = f"MATCH (n:{node_label} {{{badge_id_key}: $badge_id}}) RETURN count(n) as count"
                        result = tx.run(check_query, badge_id=badge_id).single()

                        if result and result["count"] > 0:
                            nodes_skipped += 1
                            continue

                    # Create node
                    query = f"CREATE (n:{node_label} $properties)"
                    tx.run(query, properties=properties)
                    nodes_created += 1

        try:
            with driver.session() as session:
                session.execute_write(
                    create_nodes,
                    csv_file_path,
                    node_label,
                    properties_map,
                    create_only_new,
                )

            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - CSV loaded to Neo4j. Created {nodes_created} nodes, skipped {nodes_skipped} nodes"
            )
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error loading CSV to Neo4j: {str(e)}"
            )
            raise
        finally:
            driver.close()

        return nodes_created, nodes_skipped

    def process(self, create_only_new=True):
        """
        Process visitor data and upload to Neo4j.

        Args:
            create_only_new (bool): If True, only create nodes that don't exist in the database.
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Starting Neo4j visitor data processing"
        )

        # Test the connection first
        if not self._test_connection():
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Cannot proceed with Neo4j upload due to connection failure"
            )
            return

        # Process visitors from this year
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Processing visitors from this year"
        )
        try:
            csv_file_path = os.path.join(self.output_dir, "df_reg_demo_this.csv")
            data = pd.read_csv(csv_file_path)
            properties_map = {col: col for col in data.columns}
            node_label = "Visitor_this_year"

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
            node_label = "Visitor_last_year_bva"

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
            node_label = "Visitor_last_year_lva"

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
