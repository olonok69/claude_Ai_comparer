import os
import json
import string
import logging
import pandas as pd
import functools
from typing import Dict, List, Set, Any
from dotenv import load_dotenv, dotenv_values
from langchain_openai import ChatOpenAI, AzureChatOpenAI


# Configure logger
logger = logging.getLogger(__name__)


# Optional decorator for logging function call
def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Entering {func.__name__}")
        result = func(*args, **kwargs)
        logger.debug(f"Exiting {func.__name__}")
        return result

    return wrapper


class SessionProcessor:
    """Process session data for event analytics - Generic Version."""

    def __init__(self, config: Dict):
        """
        Initialize the processor with configuration.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.output_dir = config.get("output_dir", "output")

        # Get event configuration
        self.event_config = config.get("event", {})
        
        # Get event names
        self.main_event_name = self.event_config.get("main_event_name", "main")
        self.secondary_event_name = self.event_config.get("secondary_event_name", "secondary")

        # Initialize the use_cached_descriptions parameter from config (default is True)
        self.use_cached_descriptions = config.get("stream_processing", {}).get(
            "use_cached_descriptions", True
        )
        self.cache_file_path = os.path.join(
            self.output_dir, "output", "streams_cache.json"
        )

        # Initialize logger first, before any other operations that use it
        self.logger = logging.getLogger(__name__)

        # Load environment variables for API access
        self.env_file = config.get("env_file", "keys/.env")
        self.load_env_variables()

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "output"), exist_ok=True)

        # Get output file configurations with backward compatibility
        self.output_files = config.get("session_output_files", {})
        
        # Backward compatibility: if session_output_files section doesn't exist, use default names
        if not self.output_files:
            self.output_files = {
                "processed_sessions": {
                    "this_year": "session_this_filtered_valid_cols.csv",
                    "last_year_main": "session_last_filtered_valid_cols_bva.csv",
                    "last_year_secondary": "session_last_filtered_valid_cols_lva.csv"
                },
                "streams_catalog": "streams.json"
            }

        self.logger.info(
            f"Initialized SessionProcessor for {self.main_event_name} event with output to {self.output_dir}"
        )

        # Load map_vets and titles_to_remove from config (for veterinary-specific processing)
        self.map_vets = self.config.get("map_vets", {})
        self.titles_to_remove = self.config.get("titles_to_remove", [])

    def load_env_variables(self) -> None:
        """Load environment variables for API access."""
        try:
            status = load_dotenv(self.env_file)
            self.env_config = dotenv_values(self.env_file)

            # Check for either OpenAI or Azure OpenAI API credentials
            if not status:
                self.logger.warning("Failed to load environment variables")
                raise ValueError("Failed to load environment variables")

            # Check for OpenAI credentials
            if "OPENAI_API_KEY" not in self.env_config and (
                "AZURE_API_KEY" not in self.env_config
                or "AZURE_ENDPOINT" not in self.env_config
                or "AZURE_DEPLOYMENT" not in self.env_config
                or "AZURE_API_VERSION" not in self.env_config
            ):
                self.logger.warning("API credentials missing")
                raise ValueError("API credentials not found in environment variables")

            self.logger.info("Successfully loaded environment variables")
        except Exception as e:
            self.logger.error(
                f"Error loading environment variables: {e}", exc_info=True
            )
            raise

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Remove punctuation and spaces, and convert to lowercase.

        Args:
            text: Text to be cleaned

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        return "".join(char for char in text if char.isalnum()).lower()

    def load_session_data(self) -> None:
        """Load all session data files based on configuration."""
        try:
            # Load session data with backward compatibility
            session_files = self.config["session_files"]
            session_this_path = session_files["session_this"]
            
            # Handle backward compatibility for session file names
            # Try new generic names first, fall back to old names
            if "session_past_main" in session_files:
                session_past_path_main = session_files["session_past_main"]
            else:
                session_past_path_main = session_files["session_past_bva"]
                self.logger.info("Using legacy session_past_bva configuration")
            
            if "session_past_secondary" in session_files:
                session_past_path_secondary = session_files["session_past_secondary"]
            else:
                session_past_path_secondary = session_files["session_past_lva"]
                self.logger.info("Using legacy session_past_lva configuration")

            # Read session data
            self.session_this = pd.read_csv(session_this_path)
            self.session_past_main = pd.read_csv(session_past_path_main)
            self.session_past_secondary = pd.read_csv(session_past_path_secondary)
            
            # Create backward compatible aliases for old names
            self.session_past_bva = self.session_past_main
            self.session_past_lva = self.session_past_secondary

            self.logger.info(
                f"Loaded session data: {len(self.session_this)} records this year, "
                f"{len(self.session_past_main)} {self.main_event_name} records last year, "
                f"{len(self.session_past_secondary)} {self.secondary_event_name} records last year"
            )

        except Exception as e:
            self.logger.error(f"Error loading session data: {e}", exc_info=True)
            raise

    def filter_session_data(self) -> None:
        """Filter and clean session data."""
        # Convert titles_to_remove to lowercase for case-insensitive comparison
        titles_to_remove_lower = [title.lower() for title in self.titles_to_remove]
        try:
            # Apply filters for each dataset
            self.session_this_filtered = self.session_this.copy()
            self.session_last_filtered_main = self.session_past_main.copy()
            self.session_last_filtered_secondary = self.session_past_secondary.copy()
            
            # Create backward compatible aliases
            self.session_last_filtered_bva = self.session_last_filtered_main
            self.session_last_filtered_lva = self.session_last_filtered_secondary

            # Apply title filters if titles_to_remove is configured
            if self.titles_to_remove:
                # Filter out rows where the title EXACTLY matches any in titles_to_remove (case-insensitive)
                self.session_this_filtered = self.session_this_filtered[
                    ~self.session_this_filtered["title"]
                    .str.lower()
                    .isin(titles_to_remove_lower)
                ]

                self.session_last_filtered_main = self.session_last_filtered_main[
                    ~self.session_last_filtered_main["title"]
                    .str.lower()
                    .isin(titles_to_remove_lower)
                ]

                self.session_last_filtered_secondary = self.session_last_filtered_secondary[
                    ~self.session_last_filtered_secondary["title"]
                    .str.lower()
                    .isin(titles_to_remove_lower)
                ]
                
                # Update aliases
                self.session_last_filtered_bva = self.session_last_filtered_main
                self.session_last_filtered_lva = self.session_last_filtered_secondary

                self.logger.info(f"Applied title filters - removed titles: {self.titles_to_remove}")
            else:
                self.logger.info("No title filtering configured - proceeding without title filters")

            # Create text keys for matching
            self.session_this_filtered["key_text"] = self.session_this_filtered[
                "title"
            ].apply(self.clean_text)
            self.session_last_filtered_main["key_text"] = self.session_last_filtered_main[
                "title"
            ].apply(self.clean_text)
            self.session_last_filtered_secondary["key_text"] = self.session_last_filtered_secondary[
                "title"
            ].apply(self.clean_text)

            # Remove duplicates
            self.session_this_filtered = self.session_this_filtered.drop_duplicates(
                subset=["key_text"]
            )
            self.session_last_filtered_main = (
                self.session_last_filtered_main.drop_duplicates(subset=["key_text"])
            )
            self.session_last_filtered_secondary = (
                self.session_last_filtered_secondary.drop_duplicates(subset=["key_text"])
            )
            
            # Update aliases after all operations
            self.session_last_filtered_bva = self.session_last_filtered_main
            self.session_last_filtered_lva = self.session_last_filtered_secondary

            self.logger.info(
                f"Filtered session data: {len(self.session_this_filtered)} records this year, "
                f"{len(self.session_last_filtered_main)} {self.main_event_name} records last year, "
                f"{len(self.session_last_filtered_secondary)} {self.secondary_event_name} records last year"
            )

        except Exception as e:
            self.logger.error(f"Error filtering session data: {e}", exc_info=True)
            raise

    def select_relevant_columns(self) -> None:
        """Select and clean relevant columns from session data."""
        try:
            # Define the columns we want to keep - EXACT ORDER as old processor
            cols_to_keep = [
                "session_id",
                "date",
                "start_time", 
                "end_time",
                "theatre__name",
                "title",
                "stream",
                "synopsis_stripped",
                "sponsored_session",
                "sponsored_by",
                "key_text",
            ]

            # Filter columns for each dataset
            self.session_this_filtered_valid_cols = self.session_this_filtered[
                cols_to_keep
            ].copy()
            self.session_last_filtered_valid_cols_main = self.session_last_filtered_main[
                cols_to_keep
            ].copy()
            self.session_last_filtered_valid_cols_secondary = self.session_last_filtered_secondary[
                cols_to_keep
            ].copy()
            
            # Create backward compatible aliases
            self.session_last_filtered_valid_cols_bva = self.session_last_filtered_valid_cols_main
            self.session_last_filtered_valid_cols_lva = self.session_last_filtered_valid_cols_secondary

            # Fill NaN values with empty strings
            self.session_this_filtered_valid_cols = (
                self.session_this_filtered_valid_cols.fillna("")
            )
            self.session_last_filtered_valid_cols_main = (
                self.session_last_filtered_valid_cols_main.fillna("")
            )
            self.session_last_filtered_valid_cols_secondary = (
                self.session_last_filtered_valid_cols_secondary.fillna("")
            )
            
            # Update aliases
            self.session_last_filtered_valid_cols_bva = self.session_last_filtered_valid_cols_main
            self.session_last_filtered_valid_cols_lva = self.session_last_filtered_valid_cols_secondary

            self.logger.info("Selected relevant columns and cleaned data")

        except Exception as e:
            self.logger.error(f"Error selecting relevant columns: {e}", exc_info=True)
            raise

    def extract_unique_streams(self) -> None:
        """Extract unique stream values from all session datasets."""
        try:
            # Collect all stream values
            all_streams = set()

            # Extract streams from each dataset
            for df in [
                self.session_this_filtered_valid_cols,
                self.session_last_filtered_valid_cols_main,
                self.session_last_filtered_valid_cols_secondary,
            ]:
                for stream_cell in df["stream"]:
                    if pd.notna(stream_cell) and stream_cell:
                        # Split by ';' and clean each stream
                        streams = [s.strip() for s in str(stream_cell).split(";")]
                        all_streams.update(streams)

            # Remove empty strings and sort
            self.unique_streams = sorted([s for s in all_streams if s])
            
            # Also create 'streams' attribute for backward compatibility with old processor
            self.streams = self.unique_streams

            self.logger.info(f"Extracted {len(self.unique_streams)} unique streams")

        except Exception as e:
            self.logger.error(f"Error extracting unique streams: {e}", exc_info=True)
            raise

    def setup_language_model(self) -> None:
        """Set up the language model for generating stream descriptions."""
        try:
            # Check if we have OpenAI credentials
            if "OPENAI_API_KEY" in self.env_config:
                self.llm = ChatOpenAI(
                    api_key=self.env_config["OPENAI_API_KEY"],
                    model="gpt-4o-mini",
                    temperature=0.3,
                )
                self.logger.info("Initialized OpenAI language model")
            # Check if we have Azure OpenAI credentials
            elif all(
                key in self.env_config
                for key in [
                    "AZURE_API_KEY",
                    "AZURE_ENDPOINT",
                    "AZURE_DEPLOYMENT",
                    "AZURE_API_VERSION",
                ]
            ):
                self.llm = AzureChatOpenAI(
                    azure_endpoint=self.env_config["AZURE_ENDPOINT"],
                    azure_api_key=self.env_config["AZURE_API_KEY"],
                    azure_deployment=self.env_config["AZURE_DEPLOYMENT"],
                    api_version=self.env_config["AZURE_API_VERSION"],
                    temperature=0.3,
                )
                self.logger.info("Initialized Azure OpenAI language model")
            else:
                raise ValueError("No valid API credentials found for language model")

        except Exception as e:
            self.logger.error(f"Error setting up language model: {e}", exc_info=True)
            raise

    def load_cached_descriptions(self) -> bool:
        """Load cached stream descriptions from file."""
        try:
            if os.path.exists(self.cache_file_path):
                with open(self.cache_file_path, "r") as cache_file:
                    self.streams_catalog = json.load(cache_file)
                self.logger.info(
                    f"Loaded {len(self.streams_catalog)} cached stream descriptions"
                )
                return True
            else:
                self.logger.info(
                    "No cached descriptions found. Will generate new descriptions."
                )
                self.cached_descriptions = {}
                return False
        except Exception as e:
            self.logger.error(f"Error loading cached descriptions: {e}", exc_info=True)
            self.cached_descriptions = {}
            return False

    def save_cached_descriptions(self) -> None:
        """Save current stream descriptions to cache file."""
        try:
            with open(self.cache_file_path, "w") as cache_file:
                json.dump(self.streams_catalog, cache_file, indent=4)
                self.logger.info(
                    f"Saved {len(self.streams_catalog)} stream descriptions to cache"
                )
        except Exception as e:
            self.logger.error(f"Error saving cached descriptions: {e}", exc_info=True)

    def generate_stream_descriptions(self) -> None:
        """Generate or load descriptions for streams."""
        try:
            # Try to load cached descriptions first if caching is enabled
            if self.use_cached_descriptions and self.load_cached_descriptions():
                self.logger.info("Using cached stream descriptions")
                return

            # If we reach here, we need to generate new descriptions
            self.logger.info("Will generate new stream descriptions")
            self.streams_catalog = {}

        except Exception as e:
            self.logger.error(f"Error in generate_stream_descriptions: {e}", exc_info=True)
            raise

    def generate_stream_catalog(self) -> None:
        """Generate descriptions for all streams using the language model."""
        # Only generate if we don't have cached descriptions
        if self.use_cached_descriptions and hasattr(self, 'streams_catalog') and self.streams_catalog:
            self.logger.info("Using existing cached stream descriptions")
            return

        # If no LLM is set up or caching prevented setup, skip generation
        if not hasattr(self, "llm"):
            self.logger.info("No language model configured - creating empty catalog")
            self.streams_catalog = {}
            return

        try:
            self.logger.info("Generating new stream descriptions using language model")
            for stream in self.unique_streams:
                if stream not in self.streams_catalog:
                    prompt = f"""
                    Provide a brief description (2-3 sentences) for the following conference stream: "{stream}"
                    
                    Focus on what topics and sessions would typically be covered in this stream.
                    Keep the description professional and informative.
                    """

                    response = self.llm.invoke(prompt)
                    self.streams_catalog[stream] = {
                        "stream": stream,
                        "description": response.content.strip(),
                    }

                    self.logger.debug(f"Generated description for stream: {stream}")

            self.logger.info(f"Generated descriptions for {len(self.streams_catalog)} streams")
            
            # Save the newly generated descriptions to cache
            self.save_cached_descriptions()

        except Exception as e:
            self.logger.error(f"Error generating stream catalog: {e}", exc_info=True)
            # Create empty catalog as fallback
            self.streams_catalog = {}
            raise

    def save_streams_catalog(self) -> None:
        """Save the streams catalog to JSON file."""
        try:
            streams_output_file = self.output_files.get("streams_catalog", "streams.json")
            output_path = os.path.join(self.output_dir, "output", streams_output_file)
            with open(output_path, "w") as f:
                json.dump(list(self.streams_catalog.values()), f, indent=2)

            self.logger.info(f"Saved streams catalog to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving streams catalog: {e}", exc_info=True)
            raise

    def find_short_labels(self, input_set: Set[str]) -> List[str]:
        """
        Find labels with length of 5 characters or less.

        Args:
          input_set: A set of strings (labels).

        Returns:
          A list containing the labels from the input set with a length of 5 or less.
        """
        short_labels = [
            label for label in input_set if isinstance(label, str) and len(label) <= 5
        ]
        return short_labels

    def expand_sponsor_abbreviations(self) -> None:
        """Expand sponsor abbreviations to full names using event-specific mappings."""
        try:
            # Skip sponsor abbreviation expansion if no mapping is configured
            if not self.map_vets:
                self.logger.info("No sponsor abbreviation mapping configured - skipping sponsor expansion")
                return

            # Get all unique sponsor values
            list_this_year = set(
                list(self.session_this_filtered_valid_cols.sponsored_by.unique())
            )
            list_last_secondary = set(
                list(self.session_last_filtered_valid_cols_secondary.sponsored_by.unique())
            )
            list_last_main = set(
                list(self.session_last_filtered_valid_cols_main.sponsored_by.unique())
            )

            # Combine all sponsor values
            full_list_sponsors = list_this_year.union(list_last_secondary, list_last_main)

            # Find abbreviations (short labels)
            list_abbreviations = set(self.find_short_labels(full_list_sponsors))

            # Check for any missing abbreviations in our mapping
            map_keys = set(list(self.map_vets.keys()))
            missing_abbrevs = list_abbreviations.difference(map_keys)

            if missing_abbrevs:
                self.logger.warning(f"Found unmapped abbreviations: {missing_abbrevs}")

            # Replace abbreviations with full names
            self.session_last_filtered_valid_cols_secondary["sponsored_by"] = (
                self.session_last_filtered_valid_cols_secondary["sponsored_by"].replace(
                    self.map_vets
                )
            )
            self.session_last_filtered_valid_cols_main["sponsored_by"] = (
                self.session_last_filtered_valid_cols_main["sponsored_by"].replace(
                    self.map_vets
                )
            )
            self.session_this_filtered_valid_cols["sponsored_by"] = (
                self.session_this_filtered_valid_cols["sponsored_by"].replace(
                    self.map_vets
                )
            )
            
            # Update backward compatible aliases
            self.session_last_filtered_valid_cols_bva = self.session_last_filtered_valid_cols_main
            self.session_last_filtered_valid_cols_lva = self.session_last_filtered_valid_cols_secondary

            self.logger.info("Expanded sponsor abbreviations to full names")

        except Exception as e:
            self.logger.error(
                f"Error expanding sponsor abbreviations: {e}", exc_info=True
            )
            raise

    def save_processed_session_data(self) -> None:
        """Save processed session data to CSV files."""
        try:
            # Get output file names from config
            processed_files = self.output_files.get("processed_sessions", {})
            
            # Define output paths using configuration
            main_last_output_path = os.path.join(
                self.output_dir, "output", 
                processed_files.get("last_year_main", "session_last_filtered_valid_cols_bva.csv")
            )
            secondary_last_output_path = os.path.join(
                self.output_dir, "output", 
                processed_files.get("last_year_secondary", "session_last_filtered_valid_cols_lva.csv")
            )
            this_output_path = os.path.join(
                self.output_dir, "output", 
                processed_files.get("this_year", "session_this_filtered_valid_cols.csv")
            )

            # Save to CSV
            self.session_last_filtered_valid_cols_main.to_csv(
                main_last_output_path, index=False
            )
            self.session_last_filtered_valid_cols_secondary.to_csv(
                secondary_last_output_path, index=False
            )
            self.session_this_filtered_valid_cols.to_csv(this_output_path, index=False)

            self.logger.info(
                f"Saved processed session data to {self.output_dir}/output/"
            )
            self.logger.info(
                f"{self.main_event_name} last year sessions: {len(self.session_last_filtered_valid_cols_main)} records"
            )
            self.logger.info(
                f"{self.secondary_event_name} last year sessions: {len(self.session_last_filtered_valid_cols_secondary)} records"
            )
            self.logger.info(
                f"This year sessions: {len(self.session_this_filtered_valid_cols)} records"
            )

        except Exception as e:
            self.logger.error(
                f"Error saving processed session data: {e}", exc_info=True
            )
            raise

    def process(self) -> None:
        """Execute the full session data processing workflow."""
        self.logger.info("Starting session data processing workflow")

        self.load_session_data()
        self.filter_session_data()
        self.select_relevant_columns()
        self.extract_unique_streams()
        self.setup_language_model()
        self.generate_stream_descriptions()
        self.generate_stream_catalog()
        self.save_streams_catalog()
        self.expand_sponsor_abbreviations()
        self.save_processed_session_data()

        self.logger.info("Session data processing workflow completed successfully")