import logging
import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import inspect


class Neo4jVisitorRelationshipProcessor:
    """
    A class to create relationships between visitors from this year and past events.
    This processor creates Same_Visitor relationships and attended_session relationships.
    """

    def __init__(self, config):
        """
        Initialize the Neo4j Visitor Relationship Processor.

        Args:
            config (dict): Configuration dictionary containing paths and settings
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initializing Neo4j Visitor Relationship Processor"
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
            "relationships_created": {
                "same_visitor_bva": 0,
                "same_visitor_lva": 0,
                "attended_session_bva": 0,
                "attended_session_lva": 0,
            },
            "relationships_skipped": {
                "same_visitor_bva": 0,
                "same_visitor_lva": 0,
                "attended_session_bva": 0,
                "attended_session_lva": 0,
            },
            "relationships_failed": {
                "same_visitor_bva": 0,
                "same_visitor_lva": 0,
                "attended_session_bva": 0,
                "attended_session_lva": 0,
            },
        }

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j Visitor Relationship Processor initialized successfully"
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

    def create_same_visitor_relationships_bva(self, create_only_new=True):
        """
        Creates Same_Visitor relationships for all Visitor_this_year nodes to Visitor_last_year_bva nodes.
        Tracks statistics about relationships created, skipped, and failed.

        Args:
            create_only_new (bool): If True, only create relationships that don't exist in the database.

        Returns:
            tuple: (relationships_created, relationships_skipped, relationships_failed)
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Creating Same_Visitor relationships for BVA"
        )

        relationship_count = 0
        skipped_count = 0
        failed_count = 0

        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

        try:
            # First, count existing relationships
            with driver.session() as session:
                initial_count_result = session.run(
                    """
                    MATCH (this_year:Visitor_this_year)-[r:Same_Visitor {type: 'bva'}]->(last_year:Visitor_last_year_bva)
                    RETURN COUNT(r) AS initial_count
                    """
                )
                initial_count = initial_count_result.single()["initial_count"]
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initial BVA Same_Visitor relationship count: {initial_count}"
                )

            # Check if relationships already exist and we only want to create new ones
            if create_only_new and initial_count > 0:
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Found {initial_count} existing Same_Visitor relationships for BVA. Skipping creation as create_only_new=True."
                )
                skipped_count = initial_count
                return relationship_count, skipped_count, failed_count

            # Get all Visitor_this_year nodes with valid BadgeId_last_year_bva
            # When create_only_new is True, only process visitors without recommendations
            additional_filter = ""
            if create_only_new:
                additional_filter = ' AND (this_year.has_recommendation IS NULL OR this_year.has_recommendation = "0")'

            with driver.session() as session:
                visitors_result = session.run(
                    f"""
                    MATCH (this_year:Visitor_this_year)
                    WHERE this_year.BadgeId_last_year_bva IS NOT NULL AND this_year.BadgeId_last_year_bva <> "NA"
                    {additional_filter}
                    RETURN this_year.BadgeId as this_year_id, this_year.BadgeId_last_year_bva as last_year_id
                    """
                )

                for record in visitors_result:
                    this_year_id = record["this_year_id"]
                    last_year_id = record["last_year_id"]

                    # Check if the relationship already exists
                    relationship_exists = session.run(
                        """
                        MATCH (this_year:Visitor_this_year {BadgeId: $this_year_id})-[:Same_Visitor {type: 'bva'}]->
                              (last_year:Visitor_last_year_bva {BadgeId: $last_year_id})
                        RETURN count(*) > 0 AS exists
                        """,
                        this_year_id=this_year_id,
                        last_year_id=last_year_id,
                    ).single()["exists"]

                    if not relationship_exists:
                        # Check if both nodes exist before creating relationship
                        nodes_exist = session.run(
                            """
                            MATCH (this_year:Visitor_this_year {BadgeId: $this_year_id})
                            MATCH (last_year:Visitor_last_year_bva {BadgeId: $last_year_id})
                            RETURN count(*) > 0 AS exists
                            """,
                            this_year_id=this_year_id,
                            last_year_id=last_year_id,
                        ).single()["exists"]

                        if nodes_exist:
                            try:
                                session.run(
                                    """
                                    MATCH (this_year:Visitor_this_year {BadgeId: $this_year_id})
                                    MATCH (last_year:Visitor_last_year_bva {BadgeId: $last_year_id})
                                    CREATE (this_year)-[:Same_Visitor {type: 'bva'}]->(last_year)
                                    """,
                                    this_year_id=this_year_id,
                                    last_year_id=last_year_id,
                                )
                                relationship_count += 1
                            except Exception as e:
                                failed_count += 1
                                self.logger.error(
                                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating Same_Visitor relationship for {this_year_id} -> {last_year_id}: {str(e)}"
                                )
                        else:
                            failed_count += 1
                            self.logger.warning(
                                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Could not create Same_Visitor relationship for {this_year_id} -> {last_year_id} - one or both nodes not found"
                            )
                    else:
                        skipped_count += 1

            # After creation, count final relationships to verify
            with driver.session() as session:
                final_count_result = session.run(
                    """
                    MATCH (this_year:Visitor_this_year)-[r:Same_Visitor {type: 'bva'}]->(last_year:Visitor_last_year_bva)
                    RETURN COUNT(r) AS final_count
                    """
                )
                final_count = final_count_result.single()["final_count"]

                # Calculate actual created count
                actual_created = final_count - initial_count

                if actual_created != relationship_count:
                    self.logger.warning(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - WARNING: Discrepancy detected! Database shows {actual_created} new relationships, but code tracked {relationship_count}."
                    )

        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating Same_Visitor relationships for BVA: {str(e)}"
            )
            raise
        finally:
            driver.close()

        # Log summary statistics
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - BVA Same_Visitor Relationship Summary:"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships created: {relationship_count}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships skipped: {skipped_count}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships failed: {failed_count}"
        )

        return relationship_count, skipped_count, failed_count

    def create_same_visitor_relationships_lva(self, create_only_new=True):
        """
        Creates Same_Visitor relationships for all Visitor_this_year nodes to Visitor_last_year_lva nodes.
        Tracks statistics about relationships created, skipped, and failed.

        Args:
            create_only_new (bool): If True, only create relationships that don't exist in the database.

        Returns:
            tuple: (relationships_created, relationships_skipped, relationships_failed)
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Creating Same_Visitor relationships for LVA"
        )

        relationship_count = 0
        skipped_count = 0
        failed_count = 0

        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

        try:
            # First, count existing relationships
            with driver.session() as session:
                initial_count_result = session.run(
                    """
                    MATCH (this_year:Visitor_this_year)-[r:Same_Visitor {type: 'lva'}]->(last_year:Visitor_last_year_lva)
                    RETURN COUNT(r) AS initial_count
                    """
                )
                initial_count = initial_count_result.single()["initial_count"]
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initial LVA Same_Visitor relationship count: {initial_count}"
                )

            # Check if relationships already exist and we only want to create new ones
            if create_only_new and initial_count > 0:
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Found {initial_count} existing Same_Visitor relationships for LVA. Skipping creation as create_only_new=True."
                )
                skipped_count = initial_count
                return relationship_count, skipped_count, failed_count

            # Get all Visitor_this_year nodes with valid BadgeId_last_year_lva
            with driver.session() as session:
                visitors_result = session.run(
                    """
                    MATCH (this_year:Visitor_this_year)
                    WHERE this_year.BadgeId_last_year_lva IS NOT NULL AND this_year.BadgeId_last_year_lva <> "NA"
                    RETURN this_year.BadgeId as this_year_id, this_year.BadgeId_last_year_lva as last_year_id
                    """
                )

                for record in visitors_result:
                    this_year_id = record["this_year_id"]
                    last_year_id = record["last_year_id"]

                    # Check if the relationship already exists
                    relationship_exists = session.run(
                        """
                        MATCH (this_year:Visitor_this_year {BadgeId: $this_year_id})-[:Same_Visitor {type: 'lva'}]->
                              (last_year:Visitor_last_year_lva {BadgeId: $last_year_id})
                        RETURN count(*) > 0 AS exists
                        """,
                        this_year_id=this_year_id,
                        last_year_id=last_year_id,
                    ).single()["exists"]

                    if not relationship_exists:
                        # Check if both nodes exist before creating relationship
                        nodes_exist = session.run(
                            """
                            MATCH (this_year:Visitor_this_year {BadgeId: $this_year_id})
                            MATCH (last_year:Visitor_last_year_lva {BadgeId: $last_year_id})
                            RETURN count(*) > 0 AS exists
                            """,
                            this_year_id=this_year_id,
                            last_year_id=last_year_id,
                        ).single()["exists"]

                        if nodes_exist:
                            try:
                                session.run(
                                    """
                                    MATCH (this_year:Visitor_this_year {BadgeId: $this_year_id})
                                    MATCH (last_year:Visitor_last_year_lva {BadgeId: $last_year_id})
                                    CREATE (this_year)-[:Same_Visitor {type: 'lva'}]->(last_year)
                                    """,
                                    this_year_id=this_year_id,
                                    last_year_id=last_year_id,
                                )
                                relationship_count += 1
                            except Exception as e:
                                failed_count += 1
                                self.logger.error(
                                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating Same_Visitor relationship for {this_year_id} -> {last_year_id}: {str(e)}"
                                )
                        else:
                            failed_count += 1
                            self.logger.warning(
                                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Could not create Same_Visitor relationship for {this_year_id} -> {last_year_id} - one or both nodes not found"
                            )
                    else:
                        skipped_count += 1

            # After creation, count final relationships to verify
            with driver.session() as session:
                final_count_result = session.run(
                    """
                    MATCH (this_year:Visitor_this_year)-[r:Same_Visitor {type: 'lva'}]->(last_year:Visitor_last_year_lva)
                    RETURN COUNT(r) AS final_count
                    """
                )
                final_count = final_count_result.single()["final_count"]

                # Calculate actual created count
                actual_created = final_count - initial_count

                if actual_created != relationship_count:
                    self.logger.warning(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - WARNING: Discrepancy detected! Database shows {actual_created} new relationships, but code tracked {relationship_count}."
                    )

        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating Same_Visitor relationships for LVA: {str(e)}"
            )
            raise
        finally:
            driver.close()

        # Log summary statistics
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - LVA Same_Visitor Relationship Summary:"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships created: {relationship_count}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships skipped: {skipped_count}"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships failed: {failed_count}"
        )

        return relationship_count, skipped_count, failed_count

    def create_attended_session_relationships_bva(self, create_only_new=True):
        """
        Creates attended_session relationships between Visitor_last_year_bva nodes
        and Sessions_past_year nodes. Tracks statistics about relationships created, skipped, and failed.

        Args:
            create_only_new (bool): If True, only create relationships that don't exist in the database.

        Returns:
            tuple: (relationships_created, relationships_skipped, relationships_failed)
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Creating attended_session relationships for BVA"
        )

        relationship_count = 0
        skipped_count = 0
        failed_count = 0

        # Load scan data for BVA from the previous year
        scan_file_path = os.path.join(self.output_dir, "scan_bva_past.csv")
        if not os.path.exists(scan_file_path):
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Scan file not found: {scan_file_path}"
            )
            return relationship_count, skipped_count, failed_count

        # Load the badge IDs of visitors that attended both years
        filter_file_path = os.path.join(self.output_dir, "df_reg_demo_last_bva.csv")
        if not os.path.exists(filter_file_path):
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Filter file not found: {filter_file_path}"
            )
            return relationship_count, skipped_count, failed_count

        try:
            # Load data
            scan_data = pd.read_csv(scan_file_path)
            filter_data = pd.read_csv(filter_file_path)

            # Get list of badge IDs to filter by
            badge_ids = list(filter_data["BadgeId"].unique())
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Loaded {len(badge_ids)} badge IDs for filtering scan data"
            )

            # Filter scan data to only include visitors in the filter list
            scan_data = scan_data[scan_data["Badge Id"].isin(badge_ids)]
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Filtered scan data contains {len(scan_data)} records"
            )

            if len(scan_data) == 0:
                self.logger.warning(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - No scan data records found after filtering"
                )
                return relationship_count, skipped_count, failed_count

            # Create relationships in Neo4j
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

            try:
                # First, count existing relationships
                with driver.session() as session:
                    initial_count_result = session.run(
                        """
                        MATCH (v:Visitor_last_year_bva)-[r:attended_session]->(s:Sessions_past_year)
                        RETURN COUNT(r) AS initial_count
                        """
                    )
                    initial_count = initial_count_result.single()["initial_count"]
                    self.logger.info(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initial BVA attended_session relationship count: {initial_count}"
                    )

                # Check if relationships already exist and we only want to create new ones
                if create_only_new and initial_count > 0:
                    self.logger.info(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Found {initial_count} existing attended_session relationships for BVA. Skipping creation as create_only_new=True."
                    )
                    skipped_count = initial_count
                    return relationship_count, skipped_count, failed_count

                # Process each row in the DataFrame
                with driver.session() as session:
                    for _, row in scan_data.iterrows():
                        visitor_badge_id = row["Badge Id"]
                        session_key_text = row["key_text"]
                        scan_date = str(row["Scan Date"])
                        file = row["File"]
                        seminar_name = row["Seminar Name"]

                        # Check if the relationship already exists
                        relationship_exists = session.run(
                            """
                            MATCH (v:Visitor_last_year_bva {BadgeId: $visitor_badge_id})-[:attended_session]->
                                  (s:Sessions_past_year {key_text: $session_key_text})
                            RETURN count(*) > 0 AS exists
                            """,
                            visitor_badge_id=visitor_badge_id,
                            session_key_text=session_key_text,
                        ).single()["exists"]

                        if not relationship_exists:
                            # Check if both nodes exist before creating relationship
                            nodes_exist = session.run(
                                """
                                MATCH (v:Visitor_last_year_bva {BadgeId: $visitor_badge_id})
                                MATCH (s:Sessions_past_year {key_text: $session_key_text})
                                RETURN count(*) > 0 AS exists
                                """,
                                visitor_badge_id=visitor_badge_id,
                                session_key_text=session_key_text,
                            ).single()["exists"]

                            if nodes_exist:
                                try:
                                    session.run(
                                        """
                                        MATCH (v:Visitor_last_year_bva {BadgeId: $visitor_badge_id})
                                        MATCH (s:Sessions_past_year {key_text: $session_key_text})
                                        CREATE (v)-[:attended_session {
                                            scan_date: $scan_date, 
                                            file: $file, 
                                            seminar_name: $seminar_name
                                        }]->(s)
                                        """,
                                        visitor_badge_id=visitor_badge_id,
                                        session_key_text=session_key_text,
                                        scan_date=scan_date,
                                        file=file,
                                        seminar_name=seminar_name,
                                    )
                                    relationship_count += 1
                                except Exception as e:
                                    failed_count += 1
                                    self.logger.error(
                                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating attended_session relationship for {visitor_badge_id} -> {session_key_text}: {str(e)}"
                                    )
                            else:
                                failed_count += 1
                                self.logger.warning(
                                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Could not create attended_session relationship for {visitor_badge_id} -> {session_key_text} - one or both nodes not found"
                                )
                        else:
                            skipped_count += 1

                # After creation, count final relationships to verify
                with driver.session() as session:
                    final_count_result = session.run(
                        """
                        MATCH (v:Visitor_last_year_bva)-[r:attended_session]->(s:Sessions_past_year)
                        RETURN COUNT(r) AS final_count
                        """
                    )
                    final_count = final_count_result.single()["final_count"]

                    # Calculate actual created count
                    actual_created = final_count - initial_count

                    if actual_created != relationship_count:
                        self.logger.warning(
                            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - WARNING: Discrepancy detected! Database shows {actual_created} new relationships, but code tracked {relationship_count}."
                        )

            finally:
                driver.close()

            # Log summary statistics
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - BVA Attended Session Relationship Summary:"
            )
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships created: {relationship_count}"
            )
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships skipped: {skipped_count}"
            )
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships failed: {failed_count}"
            )

        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating attended_session relationships for BVA: {str(e)}"
            )
            raise

        return relationship_count, skipped_count, failed_count

    def create_attended_session_relationships_lva(self, create_only_new=True):
        """
        Creates attended_session relationships between Visitor_last_year_lva nodes
        and Sessions_past_year nodes. Tracks statistics about relationships created, skipped, and failed.

        Args:
            create_only_new (bool): If True, only create relationships that don't exist in the database.

        Returns:
            tuple: (relationships_created, relationships_skipped, relationships_failed)
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Creating attended_session relationships for LVA"
        )

        relationship_count = 0
        skipped_count = 0
        failed_count = 0

        # Load scan data for LVA from the previous year
        scan_file_path = os.path.join(self.output_dir, "scan_lva_past.csv")
        if not os.path.exists(scan_file_path):
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Scan file not found: {scan_file_path}"
            )
            return relationship_count, skipped_count, failed_count

        # Load the badge IDs of visitors that attended both years
        filter_file_path = os.path.join(self.output_dir, "df_reg_demo_last_lva.csv")
        if not os.path.exists(filter_file_path):
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Filter file not found: {filter_file_path}"
            )
            return relationship_count, skipped_count, failed_count

        try:
            # Load data
            scan_data = pd.read_csv(scan_file_path)
            filter_data = pd.read_csv(filter_file_path)

            # Get list of badge IDs to filter by
            badge_ids = list(filter_data["BadgeId"].unique())
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Loaded {len(badge_ids)} badge IDs for filtering scan data"
            )

            # Filter scan data to only include visitors in the filter list
            scan_data = scan_data[scan_data["Badge Id"].isin(badge_ids)]
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Filtered scan data contains {len(scan_data)} records"
            )

            if len(scan_data) == 0:
                self.logger.warning(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - No scan data records found after filtering"
                )
                return relationship_count, skipped_count, failed_count

            # Create relationships in Neo4j
            driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

            try:
                # First, count existing relationships
                with driver.session() as session:
                    initial_count_result = session.run(
                        """
                        MATCH (v:Visitor_last_year_lva)-[r:attended_session]->(s:Sessions_past_year)
                        RETURN COUNT(r) AS initial_count
                        """
                    )
                    initial_count = initial_count_result.single()["initial_count"]
                    self.logger.info(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initial LVA attended_session relationship count: {initial_count}"
                    )

                # Check if relationships already exist and we only want to create new ones
                if create_only_new and initial_count > 0:
                    self.logger.info(
                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Found {initial_count} existing attended_session relationships for LVA. Skipping creation as create_only_new=True."
                    )
                    skipped_count = initial_count
                    return relationship_count, skipped_count, failed_count

                # Process each row in the DataFrame
                with driver.session() as session:
                    for _, row in scan_data.iterrows():
                        visitor_badge_id = row["Badge Id"]
                        session_key_text = row["key_text"]
                        scan_date = str(row["Scan Date"])
                        file = row["File"]
                        seminar_name = row["Seminar Name"]

                        # Check if the relationship already exists
                        relationship_exists = session.run(
                            """
                            MATCH (v:Visitor_last_year_lva {BadgeId: $visitor_badge_id})-[:attended_session]->
                                  (s:Sessions_past_year {key_text: $session_key_text})
                            RETURN count(*) > 0 AS exists
                            """,
                            visitor_badge_id=visitor_badge_id,
                            session_key_text=session_key_text,
                        ).single()["exists"]

                        if not relationship_exists:
                            # Check if both nodes exist before creating relationship
                            nodes_exist = session.run(
                                """
                                MATCH (v:Visitor_last_year_lva {BadgeId: $visitor_badge_id})
                                MATCH (s:Sessions_past_year {key_text: $session_key_text})
                                RETURN count(*) > 0 AS exists
                                """,
                                visitor_badge_id=visitor_badge_id,
                                session_key_text=session_key_text,
                            ).single()["exists"]

                            if nodes_exist:
                                try:
                                    session.run(
                                        """
                                        MATCH (v:Visitor_last_year_lva {BadgeId: $visitor_badge_id})
                                        MATCH (s:Sessions_past_year {key_text: $session_key_text})
                                        CREATE (v)-[:attended_session {
                                            scan_date: $scan_date, 
                                            file: $file, 
                                            seminar_name: $seminar_name
                                        }]->(s)
                                        """,
                                        visitor_badge_id=visitor_badge_id,
                                        session_key_text=session_key_text,
                                        scan_date=scan_date,
                                        file=file,
                                        seminar_name=seminar_name,
                                    )
                                    relationship_count += 1
                                except Exception as e:
                                    failed_count += 1
                                    self.logger.error(
                                        f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating attended_session relationship for {visitor_badge_id} -> {session_key_text}: {str(e)}"
                                    )
                            else:
                                failed_count += 1
                                self.logger.warning(
                                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Could not create attended_session relationship for {visitor_badge_id} -> {session_key_text} - one or both nodes not found"
                                )
                        else:
                            skipped_count += 1

                # After creation, count final relationships to verify
                with driver.session() as session:
                    final_count_result = session.run(
                        """
                        MATCH (v:Visitor_last_year_lva)-[r:attended_session]->(s:Sessions_past_year)
                        RETURN COUNT(r) AS final_count
                        """
                    )
                    final_count = final_count_result.single()["final_count"]

                    # Calculate actual created count
                    actual_created = final_count - initial_count

                    if actual_created != relationship_count:
                        self.logger.warning(
                            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - WARNING: Discrepancy detected! Database shows {actual_created} new relationships, but code tracked {relationship_count}."
                        )

            finally:
                driver.close()

            # Log summary statistics
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - LVA Attended Session Relationship Summary:"
            )
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships created: {relationship_count}"
            )
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships skipped: {skipped_count}"
            )
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Relationships failed: {failed_count}"
            )

        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating attended_session relationships for LVA: {str(e)}"
            )
            raise

        return relationship_count, skipped_count, failed_count

    def process(self, create_only_new=True):
        """
        Process visitor relationship data and upload to Neo4j.

        Args:
            create_only_new (bool): If True, only create relationships that don't exist in the database.
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Starting Neo4j visitor relationship processing"
        )

        # Test the connection first
        if not self._test_connection():
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Cannot proceed with Neo4j upload due to connection failure"
            )
            return

        # Create Same_Visitor relationships for BVA
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Creating Same_Visitor relationships for BVA"
        )
        try:
            relationships_created, relationships_skipped, relationships_failed = (
                self.create_same_visitor_relationships_bva(create_only_new)
            )
            self.statistics["relationships_created"][
                "same_visitor_bva"
            ] = relationships_created
            self.statistics["relationships_skipped"][
                "same_visitor_bva"
            ] = relationships_skipped
            self.statistics["relationships_failed"][
                "same_visitor_bva"
            ] = relationships_failed
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating Same_Visitor relationships for BVA: {str(e)}"
            )

        # Create Same_Visitor relationships for LVA
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Creating Same_Visitor relationships for LVA"
        )
        try:
            relationships_created, relationships_skipped, relationships_failed = (
                self.create_same_visitor_relationships_lva(create_only_new)
            )
            self.statistics["relationships_created"][
                "same_visitor_lva"
            ] = relationships_created
            self.statistics["relationships_skipped"][
                "same_visitor_lva"
            ] = relationships_skipped
            self.statistics["relationships_failed"][
                "same_visitor_lva"
            ] = relationships_failed
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating Same_Visitor relationships for LVA: {str(e)}"
            )

        # Create attended_session relationships for BVA
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Creating attended_session relationships for BVA"
        )
        try:
            relationships_created, relationships_skipped, relationships_failed = (
                self.create_attended_session_relationships_bva(create_only_new)
            )
            self.statistics["relationships_created"][
                "attended_session_bva"
            ] = relationships_created
            self.statistics["relationships_skipped"][
                "attended_session_bva"
            ] = relationships_skipped
            self.statistics["relationships_failed"][
                "attended_session_bva"
            ] = relationships_failed
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating attended_session relationships for BVA: {str(e)}"
            )

        # Create attended_session relationships for LVA
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Creating attended_session relationships for LVA"
        )
        try:
            relationships_created, relationships_skipped, relationships_failed = (
                self.create_attended_session_relationships_lva(create_only_new)
            )
            self.statistics["relationships_created"][
                "attended_session_lva"
            ] = relationships_created
            self.statistics["relationships_skipped"][
                "attended_session_lva"
            ] = relationships_skipped
            self.statistics["relationships_failed"][
                "attended_session_lva"
            ] = relationships_failed
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error creating attended_session relationships for LVA: {str(e)}"
            )

        # Log summary statistics
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j visitor relationship processing summary:"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Same_Visitor relationships for BVA: {self.statistics['relationships_created']['same_visitor_bva']} created, {self.statistics['relationships_skipped']['same_visitor_bva']} skipped, {self.statistics['relationships_failed']['same_visitor_bva']} failed"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Same_Visitor relationships for LVA: {self.statistics['relationships_created']['same_visitor_lva']} created, {self.statistics['relationships_skipped']['same_visitor_lva']} skipped, {self.statistics['relationships_failed']['same_visitor_lva']} failed"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - attended_session relationships for BVA: {self.statistics['relationships_created']['attended_session_bva']} created, {self.statistics['relationships_skipped']['attended_session_bva']} skipped, {self.statistics['relationships_failed']['attended_session_bva']} failed"
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - attended_session relationships for LVA: {self.statistics['relationships_created']['attended_session_lva']} created, {self.statistics['relationships_skipped']['attended_session_lva']} skipped, {self.statistics['relationships_failed']['attended_session_lva']} failed"
        )

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Neo4j visitor relationship processing completed"
        )
