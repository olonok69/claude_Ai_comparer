import logging
import os
import json
import inspect
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


class SessionEmbeddingProcessor:
    """
    A class to generate and store text embeddings for session nodes in Neo4j.
    This class is responsible for creating embeddings that incorporate session data
    and stream descriptions to enrich the search capabilities.
    """

    def __init__(self, config):
        """
        Initialize the Session Embedding Processor.

        Args:
            config (dict): Configuration dictionary containing paths and settings
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initializing Session Embedding Processor"
        )

        self.config = config
        self.input_dir = config["output_dir"]

        # Check if we should include stream descriptions in embeddings
        self.include_stream_descriptions = config.get("embeddings", {}).get(
            "include_stream_descriptions", False
        )
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Include stream descriptions in embeddings: {self.include_stream_descriptions}"
        )

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

        # Initialize embedding model
        self.model_name = config.get("embeddings", {}).get("model", "all-MiniLM-L6-v2")
        self.batch_size = config.get("embeddings", {}).get("batch_size", 100)

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Initializing embedding model: {self.model_name}"
        )
        self.model = SentenceTransformer(self.model_name)

        # Initialize statistics
        self.statistics = {
            "total_sessions_processed": 0,
            "sessions_with_embeddings": 0,
            "sessions_with_stream_descriptions": 0,
            "sessions_by_type": {"sessions_this_year": 0, "sessions_past_year": 0},
            "errors": 0,
        }

        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Session Embedding Processor initialized successfully"
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

    def _create_session_embedding(self, session, stream_descriptions=None):
        """
        Create text embedding for a session including optional stream descriptions.

        Args:
            session (dict): Session data dictionary
            stream_descriptions (list): List of stream descriptions

        Returns:
            list: List of embedding values
        """
        # Include title, synopsis_stripped, and theatre__name as required
        base_text = f"{session['title']} {session['synopsis_stripped']} {session['theatre__name']}"

        # Add key_text if available
        if session.get("key_text"):
            base_text = f"{base_text} {session['key_text']}"

        # Add stream names (even if we don't include descriptions)
        if session.get("stream"):
            base_text = f"{base_text} {session['stream']}"

        # Add stream descriptions if feature is enabled and descriptions are provided
        if (
            self.include_stream_descriptions
            and stream_descriptions
            and len(stream_descriptions) > 0
        ):
            stream_desc_text = " ".join(stream_descriptions)
            text = f"{base_text} {stream_desc_text}"
        else:
            text = base_text

        # Return the embedding as a list (will be converted to JSON)
        return self.model.encode(text).tolist()

    def compute_and_save_embeddings(self, create_only_new=True):
        """
        Compute embeddings for all sessions and save them directly to the nodes.

        Args:
            create_only_new (bool): If True, only create embeddings for nodes that don't already have them

        Returns:
            dict: Statistics about the embedding process
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Computing and saving session embeddings"
        )

        # Initialize statistics for this run
        stats = {
            "total_sessions_processed": 0,
            "sessions_with_embeddings": 0,
            "sessions_with_stream_descriptions": 0,
            "sessions_by_type": {"sessions_this_year": 0, "sessions_past_year": 0},
            "errors": 0,
        }

        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

        # Batch processing function
        def batch_compute_embeddings(tx, create_only_new, batch_size, model):
            batch_stats = {
                "total_sessions_processed": 0,
                "sessions_with_embeddings": 0,
                "sessions_with_stream_descriptions": 0,
                "sessions_by_type": {"sessions_this_year": 0, "sessions_past_year": 0},
            }

            # Fixed the WHERE clause syntax error
            filter_clause = (
                "WHERE (s.embedding IS NULL OR s.embedding = '') AND "
                if create_only_new
                else "WHERE "
            )

            query = f"""
            MATCH (s)
            {filter_clause} (s:Sessions_this_year OR s:Sessions_past_year)
            RETURN s.session_id as session_id, s.title as title, 
                s.stream as stream, s.synopsis_stripped as synopsis_stripped,
                s.theatre__name as theatre__name, labels(s)[0] as type,
                CASE WHEN s.key_text IS NOT NULL THEN s.key_text ELSE '' END as key_text
            """

            sessions = tx.run(query).data()
            batch_stats["total_sessions_processed"] = len(sessions)

            # Fetch all stream descriptions once (only if needed)
            stream_descriptions = {}
            if model.include_stream_descriptions:
                stream_query = """
                MATCH (s:Stream)
                RETURN s.stream as stream, s.description as description
                """
                stream_data = tx.run(stream_query).data()

                # Create a dictionary of stream descriptions for quick lookup
                stream_descriptions = {
                    s["stream"].lower(): s["description"]
                    for s in stream_data
                    if s.get("description")
                }

            # Process sessions in batches to avoid memory issues
            for i in range(0, len(sessions), batch_size):
                batch = sessions[i : i + batch_size]

                for session in batch:
                    # Update session type statistics
                    session_type = session["type"]
                    if session_type == "Sessions_this_year":
                        batch_stats["sessions_by_type"]["sessions_this_year"] += 1
                    else:
                        batch_stats["sessions_by_type"]["sessions_past_year"] += 1

                    # Process the stream field - split by semicolon and handle duplicates
                    session_streams = []
                    stream_descriptions_list = []

                    if session["stream"] and model.include_stream_descriptions:
                        # Split the stream string and strip whitespace
                        stream_list = [
                            stream.strip().lower()
                            for stream in session["stream"].split(";")
                        ]
                        # Remove duplicates by converting to set and back to list
                        stream_list = list(set(stream_list))

                        # Get the streams for embedding context
                        for stream in stream_list:
                            if (
                                stream in stream_descriptions
                                and stream_descriptions[stream]
                            ):
                                session_streams.append(stream)
                                stream_descriptions_list.append(
                                    stream_descriptions[stream]
                                )

                    if stream_descriptions_list:
                        batch_stats["sessions_with_stream_descriptions"] += 1

                    # Create embedding with the session data and stream descriptions
                    embedding = model._create_session_embedding(
                        session, stream_descriptions_list
                    )

                    # Save the embedding back to the node
                    update_query = """
                    MATCH (s)
                    WHERE s.session_id = $session_id
                    SET s.embedding = $embedding
                    """
                    tx.run(
                        update_query,
                        session_id=session["session_id"],
                        embedding=json.dumps(embedding),
                    )

                    batch_stats["sessions_with_embeddings"] += 1

            return batch_stats

        try:
            with driver.session() as session:
                # Pass self as the model to access _create_session_embedding method
                batch_stats = session.execute_write(
                    batch_compute_embeddings, create_only_new, self.batch_size, self
                )

                # Update our statistics
                stats.update(batch_stats)

                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Session embeddings - processed: {stats['total_sessions_processed']}, created: {stats['sessions_with_embeddings']}"
                )
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Sessions with stream descriptions: {stats['sessions_with_stream_descriptions']}"
                )
                self.logger.info(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Sessions by type - this year: {stats['sessions_by_type']['sessions_this_year']}, past year: {stats['sessions_by_type']['sessions_past_year']}"
                )
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error computing session embeddings: {str(e)}"
            )
            stats["errors"] += 1
            raise
        finally:
            driver.close()

        return stats

    def process(self, create_only_new=True):
        """
        Process session nodes in Neo4j and generate embeddings.

        Args:
            create_only_new (bool): If True, only create embeddings for nodes that don't already have them
        """
        self.logger.info(
            f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Starting session embedding processing"
        )

        # Test the connection first
        if not self._test_connection():
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Cannot proceed with Neo4j processing due to connection failure"
            )
            return

        try:
            # Compute and save embeddings
            stats = self.compute_and_save_embeddings(create_only_new)

            # Update the statistics
            self.statistics.update(stats)

            # Log summary
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Session embedding processing summary:"
            )
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Total sessions processed: {self.statistics['total_sessions_processed']}"
            )
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Sessions with embeddings created: {self.statistics['sessions_with_embeddings']}"
            )
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Sessions with stream descriptions: {self.statistics['sessions_with_stream_descriptions']}"
            )
            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Sessions by type - this year: {self.statistics['sessions_by_type']['sessions_this_year']}, past year: {self.statistics['sessions_by_type']['sessions_past_year']}"
            )

            if self.statistics.get("errors", 0) > 0:
                self.logger.warning(
                    f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Errors encountered during processing: {self.statistics['errors']}"
                )

            self.logger.info(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Session embedding processing completed"
            )
        except Exception as e:
            self.logger.error(
                f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno} - Error in session embedding processing: {str(e)}"
            )
            self.statistics["errors"] += 1
            raise
