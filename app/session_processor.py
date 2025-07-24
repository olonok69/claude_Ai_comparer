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
    """Process session data for event analytics."""

    def __init__(self, config: Dict):
        """
        Initialize the processor with configuration.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.output_dir = config.get("output_dir", "output")

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

        self.logger.info(
            f"Initialized SessionProcessor with output to {self.output_dir}"
        )

        # Load map_vets and titles_to_remove from config
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
            # Load session data
            session_this_path = self.config["session_files"]["session_this"]
            session_past_path_bva = self.config["session_files"]["session_past_bva"]
            session_past_path_lva = self.config["session_files"]["session_past_lva"]

            # Read session data
            self.session_this = pd.read_csv(session_this_path)
            self.session_past_bva = pd.read_csv(session_past_path_bva)
            self.session_past_lva = pd.read_csv(session_past_path_lva)

            self.logger.info(
                f"Loaded session data: {len(self.session_this)} records this year, "
                f"{len(self.session_past_bva)} BVA records last year, "
                f"{len(self.session_past_lva)} LVA records last year"
            )

        except Exception as e:
            self.logger.error(f"Error loading session data: {e}", exc_info=True)
            raise

    def filter_session_data(self) -> None:
        """Filter and clean session data."""
        titles_to_remove_lower = [title.lower() for title in self.titles_to_remove]
        try:
            # Apply filters for each dataset
            self.session_this_filtered = self.session_this.copy()
            self.session_last_filtered_bva = self.session_past_bva.copy()
            self.session_last_filtered_lva = self.session_past_lva.copy()

            # Apply title filters
            # Filter out rows where the title EXACTLY matches any in titles_to_remove (case-insensitive)
            self.session_this_filtered = self.session_this_filtered[
                ~self.session_this_filtered["title"]
                .str.lower()
                .isin(titles_to_remove_lower)
            ]

            self.session_last_filtered_bva = self.session_last_filtered_bva[
                ~self.session_last_filtered_bva["title"]
                .str.lower()
                .isin(titles_to_remove_lower)
            ]

            self.session_last_filtered_lva = self.session_last_filtered_lva[
                ~self.session_last_filtered_lva["title"]
                .str.lower()
                .isin(titles_to_remove_lower)
            ]

            # Create text keys for matching
            self.session_this_filtered["key_text"] = self.session_this_filtered[
                "title"
            ].apply(self.clean_text)
            self.session_last_filtered_bva["key_text"] = self.session_last_filtered_bva[
                "title"
            ].apply(self.clean_text)
            self.session_last_filtered_lva["key_text"] = self.session_last_filtered_lva[
                "title"
            ].apply(self.clean_text)

            # Remove duplicates
            self.session_this_filtered = self.session_this_filtered.drop_duplicates(
                subset=["key_text"]
            )
            self.session_last_filtered_bva = (
                self.session_last_filtered_bva.drop_duplicates(subset=["key_text"])
            )
            self.session_last_filtered_lva = (
                self.session_last_filtered_lva.drop_duplicates(subset=["key_text"])
            )

            self.logger.info(
                f"Filtered session data: {len(self.session_this_filtered)} records this year, "
                f"{len(self.session_last_filtered_bva)} BVA records last year, "
                f"{len(self.session_last_filtered_lva)} LVA records last year"
            )

        except Exception as e:
            self.logger.error(f"Error filtering session data: {e}", exc_info=True)
            raise

    def select_relevant_columns(self) -> None:
        """Select and clean relevant columns from session data."""
        try:
            # Define columns to keep
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

            # Select columns
            self.session_last_filtered_valid_cols_bva = self.session_last_filtered_bva[
                cols_to_keep
            ]
            self.session_last_filtered_valid_cols_lva = self.session_last_filtered_lva[
                cols_to_keep
            ]
            self.session_this_filtered_valid_cols = self.session_this_filtered[
                cols_to_keep
            ]

            # Fill NaN values
            self.session_last_filtered_valid_cols_bva = (
                self.session_last_filtered_valid_cols_bva.fillna("No Data")
            )
            self.session_last_filtered_valid_cols_lva = (
                self.session_last_filtered_valid_cols_lva.fillna("No Data")
            )
            self.session_this_filtered_valid_cols = (
                self.session_this_filtered_valid_cols.fillna("No Data")
            )

            # Remove rows with "-" in title
            self.session_this_filtered_valid_cols = (
                self.session_this_filtered_valid_cols[
                    ~(self.session_this_filtered_valid_cols.title == "-")
                ]
            )
            self.session_last_filtered_valid_cols_bva = (
                self.session_last_filtered_valid_cols_bva[
                    ~(self.session_last_filtered_valid_cols_bva.title == "-")
                ]
            )
            self.session_last_filtered_valid_cols_lva = (
                self.session_last_filtered_valid_cols_lva[
                    ~(self.session_last_filtered_valid_cols_lva.title == "-")
                ]
            )

            self.logger.info(
                f"Selected relevant columns: {len(self.session_this_filtered_valid_cols)} records this year, "
                f"{len(self.session_last_filtered_valid_cols_bva)} BVA records last year, "
                f"{len(self.session_last_filtered_valid_cols_lva)} LVA records last year"
            )

            # Concatenate all sessions
            self.total_sessions = pd.concat(
                [
                    self.session_last_filtered_valid_cols_bva,
                    self.session_last_filtered_valid_cols_lva,
                    self.session_this_filtered_valid_cols,
                ],
                ignore_index=True,
            )

            self.logger.info(
                f"Created combined dataset with {len(self.total_sessions)} total sessions"
            )

        except Exception as e:
            self.logger.error(f"Error selecting relevant columns: {e}", exc_info=True)
            raise

    def extract_unique_streams(self) -> None:
        """Extract and generate a set of unique session streams."""
        try:
            # Get lists of streams from each dataset
            list_stream_this = list(
                self.session_this_filtered_valid_cols.stream.unique()
            )
            list_stream_last_bva = list(
                self.session_last_filtered_valid_cols_bva.stream.unique()
            )
            list_stream_last_lva = list(
                self.session_last_filtered_valid_cols_lva.stream.unique()
            )

            # Initialize streams set
            self.streams = set()

            # Add streams from each dataset
            self.streams = self.generate_streams(self.streams, list_stream_this)
            self.streams = self.generate_streams(self.streams, list_stream_last_bva)
            self.streams = self.generate_streams(self.streams, list_stream_last_lva)

            # Remove "no data" if present
            if "no data" in self.streams:
                self.streams.remove("no data")

            self.logger.info(f"Extracted {len(self.streams)} unique streams")

        except Exception as e:
            self.logger.error(f"Error extracting unique streams: {e}", exc_info=True)
            raise

    @staticmethod
    def generate_streams(streams: Set[str], list_streams: List[str]) -> Set[str]:
        """
        Extract individual streams from semicolon-separated lists.

        Args:
            streams: Set of already identified streams
            list_streams: List of stream strings that may contain multiple streams

        Returns:
            Updated set of streams
        """
        for ele in list_streams:
            if not isinstance(ele, str):
                continue

            for sub_ele in ele.split(";"):
                stream = sub_ele.lower().strip()
                streams.add(stream)
        return streams

    def generate_stream_descriptions(self) -> None:
        """Generate descriptions for each unique stream based on session data."""
        try:
            self.stream_descriptions = {}

            # Initialize the dictionary to hold descriptions for each stream
            self.stream_descriptions = {stream: "" for stream in self.streams}

            # Iterate over each row in the dataframe
            for _, row in self.total_sessions.iterrows():
                if not isinstance(row["stream"], str):
                    continue

                # Split the stream column for current row and process each sub-stream
                session_streams = [s.lower().strip() for s in row["stream"].split(";")]

                # Remove duplicates while preserving order
                unique_streams = []
                [
                    unique_streams.append(s)
                    for s in session_streams
                    if s not in unique_streams
                ]

                # Concatenate title and synopsis_stripped
                session_description = f"Title: {row['title']}.\nDescription: {row['synopsis_stripped']} \n\n "

                # Add session description to relevant streams
                for stream in unique_streams:
                    if stream in self.stream_descriptions:
                        self.stream_descriptions[stream] += session_description

            self.logger.info(
                f"Generated descriptions for {len(self.stream_descriptions)} streams"
            )

        except Exception as e:
            self.logger.error(
                f"Error generating stream descriptions: {e}", exc_info=True
            )
            raise

    def setup_language_model(self) -> None:
        """Set up language model for generating stream descriptions."""
        try:
            # Check which API credentials are available and initialize the appropriate LLM
            model_name = self.config.get("language_model", {}).get(
                "model", "gpt-4.1-mini"
            )
            temperature = self.config.get("language_model", {}).get("temperature", 0.5)
            top_p = self.config.get("language_model", {}).get("top_p", 0.9)

            # Check if we should use Azure OpenAI
            if all(
                key in self.env_config
                for key in [
                    "AZURE_API_KEY",
                    "AZURE_ENDPOINT",
                    "AZURE_DEPLOYMENT",
                    "AZURE_API_VERSION",
                ]
            ):
                self.logger.info("Setting up Azure OpenAI language model")
                self.llm = AzureChatOpenAI(
                    azure_endpoint=self.env_config["AZURE_ENDPOINT"],
                    azure_deployment=self.env_config["AZURE_DEPLOYMENT"],
                    api_key=self.env_config["AZURE_API_KEY"],
                    api_version=self.env_config["AZURE_API_VERSION"],
                    temperature=temperature,
                    top_p=top_p,
                )
            # Otherwise use regular OpenAI
            elif "OPENAI_API_KEY" in self.env_config:
                self.logger.info("Setting up OpenAI language model")
                self.llm = ChatOpenAI(
                    model=model_name,
                    openai_api_key=self.env_config["OPENAI_API_KEY"],
                    temperature=temperature,
                    top_p=top_p,
                )
            else:
                raise ValueError(
                    "No valid API credentials found in environment variables"
                )

            # Create prompt template
            self.system_prompt = """
            you are an assistant specialized in create a definition from a given category label. You will receive the title and sinopsip of diferent session of an event under that category
            and based on that information you will prepare a description of the category label
            """

            self.logger.info("Language model setup completed")

        except Exception as e:
            self.logger.error(f"Error setting up language model: {e}", exc_info=True)
            raise

    def generate_description(
        self, key: str, text: str, force_regenerate: bool = False
    ) -> str:
        """
        Generate a description for a stream using the language model.

        Args:
            key: Stream name
            text: Text containing session descriptions for the stream
            force_regenerate: Force regeneration of description even if cached (default: False)

        Returns:
            Generated description
        """
        try:
            if self.use_cached_descriptions and not force_regenerate:
                # Try to load from cache first
                if (
                    hasattr(self, "cached_descriptions")
                    and key in self.cached_descriptions
                ):
                    # Log using cached description
                    self.logger.info(f"Using cached description for stream '{key}'")
                    return self.cached_descriptions[key]

            # If not using cache or description not found in cache, generate new one
            from langchain_core.prompts import PromptTemplate

            prompt = PromptTemplate(
                input_variables=["key", "text"],
                template=self.system_prompt
                + """Produce a description of the category: {key} based on the title and descriptions of the folowing session events {text}.\n Produce a description in 3 or 4 sentences of that category""",
            )
            chain = prompt | self.llm
            ai_msg = chain.invoke({"key": key, "text": text})

            # Log summary of the description (first 50 chars)
            description = ai_msg.content
            summary = description[:50] + "..." if len(description) > 50 else description
            self.logger.info(f"Generated description for stream '{key}': {summary}")

            return description

        except Exception as e:
            self.logger.error(
                f"Error generating description for stream '{key}': {e}", exc_info=True
            )
            return f"Description generation failed for stream: {key}"

    def load_cached_descriptions(self) -> bool:
        """
        Load cached stream descriptions from file if they exist.

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.cache_file_path):
                with open(self.cache_file_path, "r") as cache_file:
                    self.cached_descriptions = json.load(cache_file)
                    self.logger.info(
                        f"Loaded {len(self.cached_descriptions)} stream descriptions from cache"
                    )
                    return True
            else:
                self.logger.info("No cache file found. Will generate new descriptions.")
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

    def generate_stream_catalog(self) -> None:
        """Generate descriptions for all streams using the language model."""
        try:
            # Initialize the catalog
            self.streams_catalog = {}

            # If using cache, try to load it first
            if self.use_cached_descriptions:
                cache_loaded = self.load_cached_descriptions()

                # If cache was loaded and we want to use it without generating new descriptions
                if cache_loaded:
                    # Find streams that are not in the cache
                    missing_streams = set(self.stream_descriptions.keys()) - set(
                        self.cached_descriptions.keys()
                    )

                    # Copy existing descriptions from cache
                    self.streams_catalog = self.cached_descriptions.copy()

                    # Only generate descriptions for missing streams
                    for stream in missing_streams:
                        self.logger.info(
                            f"Generating description for new stream: {stream}"
                        )
                        self.streams_catalog[stream] = self.generate_description(
                            key=stream,
                            text=self.stream_descriptions[stream],
                        )

                    self.logger.info(
                        f"Generated catalog with {len(self.streams_catalog)} stream descriptions "
                        f"({len(self.streams_catalog) - len(missing_streams)} from cache, {len(missing_streams)} newly generated)"
                    )

                    # Save the updated catalog to cache
                    self.save_cached_descriptions()
                    return

            # If not using cache or cache loading failed, generate all descriptions
            for stream in self.stream_descriptions.keys():
                self.logger.info(f"Generating description for stream: {stream}")
                self.streams_catalog[stream] = self.generate_description(
                    key=stream,
                    text=self.stream_descriptions[stream],
                )

            self.logger.info(
                f"Generated catalog with {len(self.streams_catalog)} stream descriptions"
            )

            # Save the newly generated catalog to cache
            self.save_cached_descriptions()

        except Exception as e:
            self.logger.error(f"Error generating stream catalog: {e}", exc_info=True)
            raise

    def save_streams_catalog(self) -> None:
        """Save the generated stream catalog to a JSON file."""
        try:
            output_path = os.path.join(self.output_dir, "output", "streams.json")

            with open(output_path, "w") as json_file:
                json.dump(self.streams_catalog, json_file, indent=4)

            self.logger.info(f"Saved stream catalog to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving stream catalog: {e}", exc_info=True)
            raise

    @staticmethod
    def find_short_labels(input_set: Set[str]) -> List[str]:
        """
        Finds a list of labels in a set that have 5 characters or less.

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
        """Expand sponsor abbreviations to full names."""
        try:
            # Get all unique sponsor values
            list_bva_this = set(
                list(self.session_this_filtered_valid_cols.sponsored_by.unique())
            )
            list_lva_last = set(
                list(self.session_last_filtered_valid_cols_lva.sponsored_by.unique())
            )
            list_bva_last = set(
                list(self.session_last_filtered_valid_cols_bva.sponsored_by.unique())
            )

            # Combine all sponsor values
            full_list_sponsors = list_bva_this.union(list_lva_last, list_bva_last)

            # Find abbreviations (short labels)
            list_abbreviations = set(self.find_short_labels(full_list_sponsors))

            # Check for any missing abbreviations in our mapping
            map_keys = set(list(self.map_vets.keys()))
            missing_abbrevs = list_abbreviations.difference(map_keys)

            if missing_abbrevs:
                self.logger.warning(f"Found unmapped abbreviations: {missing_abbrevs}")

            # Replace abbreviations with full names
            self.session_last_filtered_valid_cols_lva["sponsored_by"] = (
                self.session_last_filtered_valid_cols_lva["sponsored_by"].replace(
                    self.map_vets
                )
            )
            self.session_last_filtered_valid_cols_bva["sponsored_by"] = (
                self.session_last_filtered_valid_cols_bva["sponsored_by"].replace(
                    self.map_vets
                )
            )
            self.session_this_filtered_valid_cols["sponsored_by"] = (
                self.session_this_filtered_valid_cols["sponsored_by"].replace(
                    self.map_vets
                )
            )

            self.logger.info("Expanded sponsor abbreviations to full names")

        except Exception as e:
            self.logger.error(
                f"Error expanding sponsor abbreviations: {e}", exc_info=True
            )
            raise

    def save_processed_session_data(self) -> None:
        """Save processed session data to CSV files."""
        try:
            # Define output paths
            bva_last_output_path = os.path.join(
                self.output_dir, "output", "session_last_filtered_valid_cols_bva.csv"
            )
            lva_last_output_path = os.path.join(
                self.output_dir, "output", "session_last_filtered_valid_cols_lva.csv"
            )
            this_output_path = os.path.join(
                self.output_dir, "output", "session_this_filtered_valid_cols.csv"
            )

            # Save to CSV
            self.session_last_filtered_valid_cols_bva.to_csv(
                bva_last_output_path, index=False
            )
            self.session_last_filtered_valid_cols_lva.to_csv(
                lva_last_output_path, index=False
            )
            self.session_this_filtered_valid_cols.to_csv(this_output_path, index=False)

            self.logger.info(
                f"Saved processed session data to {self.output_dir}/output/"
            )
            self.logger.info(
                f"BVA last year sessions: {len(self.session_last_filtered_valid_cols_bva)} records"
            )
            self.logger.info(
                f"LVA last year sessions: {len(self.session_last_filtered_valid_cols_lva)} records"
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
        self.generate_stream_descriptions()
        self.setup_language_model()
        self.generate_stream_catalog()
        self.save_streams_catalog()
        self.expand_sponsor_abbreviations()
        self.save_processed_session_data()

        self.logger.info("Session data processing workflow completed successfully")
