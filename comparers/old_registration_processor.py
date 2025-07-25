import os
import json
import string
import logging
import pandas as pd
import functools
from typing import Dict, List, Set
from datetime import datetime
from pandas import json_normalize
from fuzzywuzzy import process
import re
from difflib import SequenceMatcher

# Configure logger
logger = logging.getLogger(__name__)


class DataLoader:
    """Handle loading data from different sources."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_json(self, file_path: str) -> List[Dict]:
        """
        Load data from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            List of dictionaries containing the data
        """
        self.logger.info(f"Loading JSON data from {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.logger.info(
                f"Successfully loaded {len(data)} records from {file_path}"
            )
            return data
        except Exception as e:
            self.logger.error(
                f"Error loading JSON data from {file_path}: {e}", exc_info=True
            )
            raise


# Optional decorator for logging function entry/exit
# Optional decorator for logging function entry/exit
def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Entering {func.__name__}")
        result = func(*args, **kwargs)
        logger.debug(f"Exiting {func.__name__}")
        return result

    return wrapper


class RegistrationProcessor:
    """Process registration data for event analytics."""

    def __init__(self, config: Dict):
        """
        Initialize the processor with configuration.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.output_dir = config.get("output_dir", "output")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "csv"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "output"), exist_ok=True)

        # Use existing logger instead of configuring a new one
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Initialized RegistrationProcessor with output to {self.output_dir}"
        )

    def load_data(self) -> None:
        """Load all registration data files based on configuration."""
        loader = DataLoader()

        # Load BVA registration data
        bva_file = self.config["input_files"]["bva_registration"]
        self.bva_reg_data = loader.load_json(bva_file)

        # Load LVS registration data
        lvs_file = self.config["input_files"]["lvs_registration"]
        self.lvs_reg_data = loader.load_json(lvs_file)

        # Load BVA demographic data
        bva_demo_file = self.config["input_files"]["bva_demographic"]
        self.bva_demo_data = loader.load_json(bva_demo_file)

        # Load LVS demographic data
        lvs_demo_file = self.config["input_files"]["lvs_demographic"]
        self.lvs_demo_data = loader.load_json(lvs_demo_file)

        # Convert to dataframes
        self.df_bva = pd.json_normalize(self.bva_reg_data)
        self.df_lvs = pd.json_normalize(self.lvs_reg_data)
        self.df_bva_demo = pd.json_normalize(self.bva_demo_data)
        self.df_lvs_demo = pd.json_normalize(self.lvs_demo_data)

        self.logger.info(
            f"Loaded {len(self.df_bva)} BVA registration records and {len(self.df_lvs)} LVS registration records"
        )
        self.logger.info(
            f"Loaded {len(self.df_bva_demo)} BVA demographic records and {len(self.df_lvs_demo)} LVS demographic records"
        )

    def save_initial_data(self) -> None:
        """Save the initial raw data to CSV files."""
        bva_csv = os.path.join(self.output_dir, "csv", "Registration_data_bva.csv")
        lvs_csv = os.path.join(self.output_dir, "csv", "Registration_data_lva.csv")
        bva_demo_csv = os.path.join(
            self.output_dir, "csv", "Registration_demographicdata_bva.csv"
        )
        lvs_demo_csv = os.path.join(
            self.output_dir, "csv", "Registration_demographicdata_lva.csv"
        )

        self.df_bva.to_csv(bva_csv, index=False)
        self.df_lvs.to_csv(lvs_csv, index=False)
        self.df_bva_demo.to_csv(bva_demo_csv, index=False)
        self.df_lvs_demo.to_csv(lvs_demo_csv, index=False)

        self.logger.info(f"Saved initial data to CSV files in {self.output_dir}/csv/")

    def preprocess_data(self) -> None:
        """Preprocess the registration data."""
        # Clean email and name fields - create copies to avoid warnings
        self.df_bva = self.df_bva.dropna(subset=["Email", "Forename", "Surname"]).copy()
        self.df_lvs = self.df_lvs.dropna(subset=["Email", "Forename", "Surname"]).copy()

        # Normalize job titles - use loc to avoid SettingWithCopyWarning
        self.df_bva.loc[:, "JobTitle"] = self.df_bva["JobTitle"].str.lower()
        self.df_bva.loc[:, "JobTitle"] = self.df_bva["JobTitle"].fillna("NA")

        # Save unique job titles for reference
        job_titles = pd.DataFrame(
            data=list(self.df_bva["JobTitle"].unique()), columns=["JobTitle"]
        )
        job_titles.to_csv(
            os.path.join(self.output_dir, "list_of_professions.csv"), index=False
        )

        # Preprocess demographic data
        # Remove punctuation from QuestionText
        self.df_bva_demo.loc[:, "QuestionText"] = self.df_bva_demo[
            "QuestionText"
        ].apply(self.remove_punctuation)
        self.df_lvs_demo.loc[:, "QuestionText"] = self.df_lvs_demo[
            "QuestionText"
        ].apply(self.remove_punctuation)

        # Create backup of AnswerText
        self.df_bva_demo.loc[:, "AnswerText_backup"] = self.df_bva_demo[
            "AnswerText"
        ].astype(str)
        self.df_lvs_demo.loc[:, "AnswerText_backup"] = self.df_lvs_demo[
            "AnswerText"
        ].astype(str)

        self.logger.info(
            f"Preprocessed data - BVA: {len(self.df_bva)} records, LVS: {len(self.df_lvs)} records"
        )
        self.logger.info(f"Found {len(job_titles)} unique job titles")

    @staticmethod
    def remove_punctuation(text):
        """
        Removes punctuation from a string, excluding hyphens and forward slashes.

        Args:
            text: Input string

        Returns:
            String with punctuation removed
        """
        if not isinstance(text, str):
            return text

        custom_punctuation = (
            string.punctuation.replace("-", "").replace("/", "") + "''" "…" + "â€™Â"
        )
        return text.translate(str.maketrans("", "", custom_punctuation))

    @staticmethod
    def calculate_date_difference(
        df: pd.DataFrame, date_column: str, given_date_str: str
    ) -> pd.DataFrame:
        """
        Calculate the number of days between registration date and event date.

        Args:
            df: DataFrame containing registration data
            date_column: Name of the column containing registration dates
            given_date_str: Event date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with an additional 'Days_since_registration' column
        """
        logger = logging.getLogger(__name__)

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_copy = df.copy()

        # Check if date_column exists in the DataFrame
        if date_column not in df_copy.columns:
            logger.warning(
                f"Date column '{date_column}' not found in DataFrame. Adding dummy values."
            )
            df_copy.loc[:, "Days_since_registration"] = 0
            return df_copy

        # Convert the given date to datetime
        given_date = datetime.strptime(given_date_str, "%Y-%m-%d")

        # Convert the column to datetime, with error handling
        try:
            # Handle potential NaN values
            mask = df_copy[date_column].notna()
            if not mask.any():
                logger.warning(
                    f"No valid dates found in '{date_column}'. Adding dummy values."
                )
                df_copy.loc[:, "Days_since_registration"] = 0
                return df_copy

            # Initialize Days_since_registration column with default value
            df_copy.loc[:, "Days_since_registration"] = 0

            # For each row with a non-null date, calculate days difference
            for idx in df_copy[mask].index:
                try:
                    date_val = df_copy.loc[idx, date_column]
                    # Make sure we have a proper datetime object
                    if not isinstance(date_val, datetime):
                        date_val = pd.to_datetime(date_val, errors="coerce")

                    if pd.notna(date_val):
                        # Calculate days difference directly without using .dt accessor
                        days_diff = (given_date - date_val).days
                        df_copy.loc[idx, "Days_since_registration"] = days_diff
                except Exception as e:
                    logger.warning(f"Error processing date for row {idx}: {e}")

        except Exception as e:
            logger.error(f"Error calculating date difference: {e}", exc_info=True)
            # Keep default values to continue processing

        return df_copy

    @staticmethod
    def create_unique_id(row: pd.Series) -> str:
        """
        Create a unique identifier using name and email.

        Args:
            row: DataFrame row containing Forename, Surname, and Email

        Returns:
            String concatenation of lowercase forename, surname and email
        """
        return (
            f"{row['Forename'].lower()}_{row['Surname'].lower()}_{row['Email'].lower()}"
        )

    @staticmethod
    def extract_email_domain(email: str) -> str:
        """
        Extract the domain part from an email address.

        Args:
            email: Email address

        Returns:
            Domain part of the email
        """
        return email.split("@")[-1]

    def flag_returning_visitors(
        self, df: pd.DataFrame, returning_visitors: Set[str]
    ) -> pd.DataFrame:
        """
        Flag visitors who attended the previous year's event.

        Args:
            df: DataFrame containing registration data
            returning_visitors: Set of unique IDs of returning visitors

        Returns:
            DataFrame with an additional 'assist_year_before' column
        """
        # Create a copy to avoid SettingWithCopyWarning
        df_copy = df.copy()

        # Use loc to set values
        df_copy.loc[:, "assist_year_before"] = df_copy.apply(
            lambda row: 1 if row["id_both_years"] in returning_visitors else 0, axis=1
        )
        return df_copy

    def split_and_process_bva_data(self) -> None:
        """Split BVA data by year and process it."""
        # Define valid badge types
        valid_badge_types = self.config.get(
            "valid_badge_types", ["Delegate", "Delegate - Group"]
        )
        shows_this_year = self.config.get("shows_this_year", ["BVA2025"])

        # Split registration data by year
        self.df_bva_this_year = self.df_bva[self.df_bva.ShowRef.isin(shows_this_year)]
        self.df_bva_last_year = self.df_bva[
            ~(self.df_bva.ShowRef.isin(shows_this_year))
        ]

        # Split demographic data by year
        self.df_bva_demo_this_year = self.df_bva_demo[
            self.df_bva_demo.showref.isin(shows_this_year)
        ]
        self.df_bva_demo_last_year = self.df_bva_demo[
            ~(self.df_bva_demo.showref.isin(shows_this_year))
        ]

        self.logger.info(
            f"Split BVA data: {len(self.df_bva_this_year)} records for this year, {len(self.df_bva_last_year)} records for last year"
        )
        self.logger.info(
            f"Split BVA demo data: {len(self.df_bva_demo_this_year)} records for this year, {len(self.df_bva_demo_last_year)} records for last year"
        )

        # Filter by valid badge types
        self.df_bva_this_year = self.df_bva_this_year[
            self.df_bva_this_year["BadgeType"].isin(valid_badge_types)
        ]
        self.df_bva_last_year = self.df_bva_last_year[
            self.df_bva_last_year["BadgeType"].isin(valid_badge_types)
        ]
        self.df_lvs_last_year = self.df_lvs[
            self.df_lvs["BadgeType"].isin(valid_badge_types)
        ]

        # Remove duplicates
        self.df_bva_this_year = self.df_bva_this_year.drop_duplicates(
            subset="BadgeId", keep="first"
        )
        self.df_bva_last_year = self.df_bva_last_year.drop_duplicates(
            subset="BadgeId", keep="first"
        )
        self.df_lvs_last_year = self.df_lvs_last_year.drop_duplicates(
            subset="BadgeId", keep="first"
        )

        self.logger.info(
            f"After filtering and deduplication: BVA this year: {len(self.df_bva_this_year)}, BVA last year: {len(self.df_bva_last_year)}, LVS last year: {len(self.df_lvs_last_year)}"
        )

        # Create ID for matching visitors across years
        # Use loc to avoid SettingWithCopyWarning
        for df_name in ["df_bva_this_year", "df_bva_last_year", "df_lvs_last_year"]:
            df = getattr(self, df_name)
            df_copy = df.copy()
            df_copy.loc[:, "id_both_years"] = df.apply(self.create_unique_id, axis=1)
            df_copy.loc[:, "Email_domain"] = df["Email"].apply(
                self.extract_email_domain
            )
            setattr(self, df_name, df_copy)

    def identify_returning_visitors(self) -> None:
        """Identify visitors who attended previous year's events."""
        # Get unique IDs for each year
        ids_bva_last_year = set(self.df_bva_last_year["id_both_years"].unique())
        ids_bva_this_year = set(self.df_bva_this_year["id_both_years"].unique())
        ids_lvs_last_year = set(self.df_lvs_last_year["id_both_years"].unique())

        # Find intersections
        self.bva_returning_visitors = ids_bva_this_year.intersection(ids_bva_last_year)
        self.lvs_returning_visitors = ids_bva_this_year.intersection(ids_lvs_last_year)

        # Union of all returning visitors
        self.all_returning_visitors = self.bva_returning_visitors.union(
            self.lvs_returning_visitors
        )

        self.logger.info(
            f"Identified {len(self.bva_returning_visitors)} returning visitors from BVA and {len(self.lvs_returning_visitors)} from LVS"
        )
        self.logger.info(
            f"Total unique returning visitors: {len(self.all_returning_visitors)}"
        )

        # Create dataframes for returning visitors
        self.df_bva_returning = self.df_bva_last_year[
            self.df_bva_last_year["id_both_years"].isin(self.bva_returning_visitors)
        ]
        self.df_lvs_returning = self.df_lvs_last_year[
            self.df_lvs_last_year["id_both_years"].isin(self.lvs_returning_visitors)
        ]

    def add_badge_history(self) -> None:
        """Add badge history information to current year data."""
        # Create index dataframes
        badge_last_year_bva = self.df_bva_returning[["BadgeId", "id_both_years"]].copy()
        badge_last_year_lvs = self.df_lvs_returning[["BadgeId", "id_both_years"]].copy()

        # Rename columns
        badge_last_year_bva.columns = ["BadgeId_last_year_bva", "id_both_years"]
        badge_last_year_lvs.columns = ["BadgeId_last_year_lva", "id_both_years"]

        # Merge with current year data
        self.df_bva_this_year = pd.merge(
            self.df_bva_this_year, badge_last_year_bva, on=["id_both_years"], how="left"
        )
        self.df_bva_this_year = pd.merge(
            self.df_bva_this_year, badge_last_year_lvs, on=["id_both_years"], how="left"
        )

        # Fill missing values - use loc to avoid SettingWithCopyWarning
        self.df_bva_this_year.loc[:, "BadgeId_last_year_bva"] = self.df_bva_this_year[
            "BadgeId_last_year_bva"
        ].fillna("NA")
        self.df_bva_this_year.loc[:, "BadgeId_last_year_lva"] = self.df_bva_this_year[
            "BadgeId_last_year_lva"
        ].fillna("NA")

        self.logger.info("Added badge history information to current year data")
        # Log the number of matched records for verification
        self.logger.info(
            f"Number of unique BadgeId_last_year_bva values: {len(self.df_bva_this_year['BadgeId_last_year_bva'].unique())}"
        )
        self.logger.info(
            f"Number of unique BadgeId_last_year_lva values: {len(self.df_bva_this_year['BadgeId_last_year_lva'].unique())}"
        )

    def calculate_event_dates(self) -> None:
        """Calculate days until event for all dataframes."""
        event_date_this_year = self.config.get("event_date_this_year", "2025-06-12")
        event_date_last_year = self.config.get("event_date_last_year", "2024-06-12")

        self.df_bva_this_year = self.calculate_date_difference(
            self.df_bva_this_year, "RegistrationDate", event_date_this_year
        )
        self.df_bva_last_year = self.calculate_date_difference(
            self.df_bva_last_year, "RegistrationDate", event_date_this_year
        )
        self.df_bva_returning = self.calculate_date_difference(
            self.df_bva_returning, "RegistrationDate", event_date_last_year
        )
        self.df_lvs_returning = self.calculate_date_difference(
            self.df_lvs_returning, "RegistrationDate", event_date_last_year
        )

        self.logger.info("Calculated days until event for all datasets")

    def flag_all_returning_visitors(self) -> None:
        """Flag all returning visitors in current year data."""
        self.df_bva_this_year = self.flag_returning_visitors(
            self.df_bva_this_year, self.all_returning_visitors
        )
        self.logger.info(
            f"Returning visitors: {self.df_bva_this_year['assist_year_before'].sum()} out of {len(self.df_bva_this_year)}"
        )

    def select_and_save_final_data(self) -> None:
        """Select relevant columns and save final processed data."""
        # Define columns to keep
        columns_this_year = [
            "Email",
            "Email_domain",
            "Company",
            "JobTitle",
            "Country",
            "BadgeType",
            "ShowRef",
            "BadgeId",
            "Source",
            "Days_since_registration",
            "assist_year_before",
            "BadgeId_last_year_bva",
            "BadgeId_last_year_lva",
        ]

        columns_last_year = [
            "Email",
            "Email_domain",
            "Company",
            "JobTitle",
            "Country",
            "BadgeType",
            "ShowRef",
            "BadgeId",
            "Source",
            "Days_since_registration",
        ]

        # Select columns
        df_this_year_final = self.df_bva_this_year[columns_this_year]
        df_last_year_final = self.df_bva_last_year[columns_last_year]
        df_bva_returning_final = self.df_bva_returning[columns_last_year]
        df_lvs_returning_final = self.df_lvs_returning[columns_last_year]

        # Save files
        output_files = {
            "this_year": os.path.join(
                self.output_dir, "csv", "Registration_data_bva_25_only_valid.csv"
            ),
            "last_year": os.path.join(
                self.output_dir, "csv", "Registration_data_bva_24_only_valid.csv"
            ),
            "bva_returning": os.path.join(
                self.output_dir, "csv", "Registration_data_bva_24_25_only_valid.csv"
            ),
            "lvs_returning": os.path.join(
                self.output_dir, "csv", "Registration_data_lva_24_25_only_valid.csv"
            ),
        }

        df_this_year_final.to_csv(output_files["this_year"], index=False)
        df_last_year_final.to_csv(output_files["last_year"], index=False)
        df_bva_returning_final.to_csv(output_files["bva_returning"], index=False)
        df_lvs_returning_final.to_csv(output_files["lvs_returning"], index=False)

        self.logger.info(f"Saved final processed data to {self.output_dir}/csv/")

    def save_demographic_data(self) -> None:
        """Save demographic data to CSV files."""
        self.df_bva_demo_this_year.to_csv(
            os.path.join(
                self.output_dir, "csv", "Registration_demographicdata_bva_25_raw.csv"
            ),
            index=False,
        )
        self.df_bva_demo_last_year.to_csv(
            os.path.join(
                self.output_dir, "csv", "Registration_demographicdata_bva_24_raw.csv"
            ),
            index=False,
        )
        self.df_lvs_demo.to_csv(
            os.path.join(
                self.output_dir, "csv", "Registration_demographicdata_lva_24_raw.csv"
            ),
            index=False,
        )

        self.logger.info("Saved demographic data to CSV files")

    def extract_specialization_data(self) -> None:
        """Extract and analyze specialization data."""
        # Get the questions to keep from config
        questions_to_keep = self.config.get("questions_to_keep", {})
        list_keep_this = questions_to_keep.get("current", [])
        list_keep_past = questions_to_keep.get("past", [])

        # Extract specialization data
        df_spe_this = self.df_bva_demo_this_year[
            self.df_bva_demo_this_year["QuestionText"] == list_keep_this[0]
        ]
        df_spe_last_bva = self.df_bva_demo_last_year[
            self.df_bva_demo_last_year["QuestionText"] == list_keep_past[0]
        ]
        df_spe_last_lvs = self.df_lvs_demo[
            self.df_lvs_demo["QuestionText"] == list_keep_past[0]
        ]

        # Extract unique specializations
        specializations_this = self.extract_unique_classes(
            df_spe_this["AnswerText"].unique()
        )
        specializations_last_bva = self.extract_unique_classes(
            df_spe_last_bva["AnswerText"].unique()
        )
        specializations_last_lvs = self.extract_unique_classes(
            df_spe_last_lvs["AnswerText"].unique()
        )

        self.logger.info(
            f"Extracted {len(specializations_this)} unique specializations for this year"
        )
        self.logger.info(
            f"Extracted {len(specializations_last_bva)} unique specializations for BVA last year"
        )
        self.logger.info(
            f"Extracted {len(specializations_last_lvs)} unique specializations for LVS last year"
        )

        # Save specializations to JSON
        specializations = {
            "this_year": specializations_this,
            "last_year_bva": specializations_last_bva,
            "last_year_lvs": specializations_last_lvs,
        }

        with open(
            os.path.join(self.output_dir, "output", "specializations.json"), "w"
        ) as f:
            json.dump(specializations, f, indent=4)

    @staticmethod
    def extract_unique_classes(original_list):
        """
        Extract unique classes from a list of semicolon-separated values.

        Args:
            original_list: List of strings with semicolon-separated values

        Returns:
            List of unique classes
        """
        # Set to store unique classes
        unique_classes = set()

        # Process each item in the list
        for item in original_list:
            if not isinstance(item, str):
                continue

            # Split by semicolon and add each class to the set
            classes = item.split(";")
            for cls in classes:
                unique_classes.add(cls.strip())

        # Convert set back to a list for the final result
        return list(unique_classes)

    def extract_job_roles(self) -> None:
        """Extract and analyze job roles."""
        # Extract job role data
        job_records_this = self.df_bva_demo_this_year[
            self.df_bva_demo_this_year["QuestionText"] == "Job Role"
        ]
        job_records_last_bva = self.df_bva_demo_last_year[
            self.df_bva_demo_last_year["QuestionText"] == "Job Role"
        ]
        job_records_last_lvs = self.df_lvs_demo[
            self.df_lvs_demo["QuestionText"] == "Job Role"
        ]

        # Extract unique job roles
        job_roles_this = set(list(job_records_this["AnswerText"].unique()))
        job_roles_last_bva = set(list(job_records_last_bva["AnswerText"].unique()))
        job_roles_last_lvs = set(list(job_records_last_lvs["AnswerText"].unique()))

        # Combine all job roles
        all_job_roles = list(
            job_roles_this.union(job_roles_last_bva).union(job_roles_last_lvs)
        )

        self.logger.info(f"Extracted {len(all_job_roles)} unique job roles")

        # Save job roles to JSON
        with open(os.path.join(self.output_dir, "output", "job_roles.json"), "w") as f:
            json.dump(all_job_roles, f, indent=4)

    def find_demographics_for_returning_visitors(self) -> None:
        """Find demographic data for returning visitors."""
        # Get badge IDs for returning visitors - store these in class variables
        # so they can be used later in the combine_demographic_with_registration method
        self.list_badgeid_24_25_bva = list(self.df_bva_returning["BadgeId"].unique())
        self.list_badgeid_24_25_lva = list(self.df_lvs_returning["BadgeId"].unique())

        self.logger.info(f"BVA returning visitors: {len(self.list_badgeid_24_25_bva)}")
        self.logger.info(f"LVA returning visitors: {len(self.list_badgeid_24_25_lva)}")

        # Filter demographic data for returning visitors
        df_demo_24_25_bva = self.df_bva_demo_last_year[
            self.df_bva_demo_last_year["BadgeId"].isin(self.list_badgeid_24_25_bva)
        ]
        df_demo_24_25_lva = self.df_lvs_demo[
            self.df_lvs_demo["BadgeId"].isin(self.list_badgeid_24_25_lva)
        ]

        # Save filtered demographic data
        df_demo_24_25_bva.to_csv(
            os.path.join(
                self.output_dir, "csv", "Registration_demographicdata_bva_24_25.csv"
            ),
            index=False,
        )
        df_demo_24_25_lva.to_csv(
            os.path.join(
                self.output_dir, "csv", "Registration_demographicdata_lva_24_25.csv"
            ),
            index=False,
        )

        self.logger.info(
            f"Found demographic data for {len(df_demo_24_25_bva)} returning BVA visitors"
        )
        self.logger.info(
            f"Found demographic data for {len(df_demo_24_25_lva)} returning LVS visitors"
        )

    def find_registration_with_demographic_data(self) -> None:
        """Find registration records that have demographic data."""
        # Get badge IDs with demographic data
        demo_badge_id_25 = list(self.df_bva_demo_this_year["BadgeId"].unique())
        demo_badge_id_24_bva = list(self.df_bva_demo_last_year["BadgeId"].unique())
        demo_badge_id_24_lva = list(self.df_lvs_demo["BadgeId"].unique())

        # Filter registration data for records with demographic data
        df_reg_25_wdemo_data = self.df_bva_this_year[
            self.df_bva_this_year["BadgeId"].isin(demo_badge_id_25)
        ]
        df_reg_24_wdemo_data_bva = self.df_bva_returning[
            self.df_bva_returning["BadgeId"].isin(demo_badge_id_24_bva)
        ]
        df_reg_24_wdemo_data_lva = self.df_lvs_returning[
            self.df_lvs_returning["BadgeId"].isin(demo_badge_id_24_lva)
        ]

        # Save filtered registration data
        df_reg_25_wdemo_data.to_csv(
            os.path.join(
                self.output_dir,
                "output",
                "Registration_data_with_demographicdata_bva_this.csv",
            ),
            index=False,
        )
        df_reg_24_wdemo_data_bva.to_csv(
            os.path.join(
                self.output_dir,
                "output",
                "Registration_data_with_demographicdata_bva_last.csv",
            ),
            index=False,
        )
        df_reg_24_wdemo_data_lva.to_csv(
            os.path.join(
                self.output_dir,
                "output",
                "Registration_data_with_demographicdata_lva_last.csv",
            ),
            index=False,
        )

        self.logger.info(
            f"Found {len(df_reg_25_wdemo_data)} registration records with demographic data for this year"
        )
        self.logger.info(
            f"Found {len(df_reg_24_wdemo_data_bva)} registration records with demographic data for BVA last year"
        )
        self.logger.info(
            f"Found {len(df_reg_24_wdemo_data_lva)} registration records with demographic data for LVS last year"
        )
        # Get unique badge IDs again after concatenating with demo data
        demo_badge_id_25 = list(df_reg_25_wdemo_data["BadgeId"].unique())
        demo_badge_id_24_bva = list(df_reg_24_wdemo_data_bva["BadgeId"].unique())
        demo_badge_id_24_lva = list(df_reg_24_wdemo_data_lva["BadgeId"].unique())
        # Store for later use
        self.df_reg_25_wdemo_data = df_reg_25_wdemo_data
        self.df_reg_24_wdemo_data_bva = df_reg_24_wdemo_data_bva
        self.df_reg_24_wdemo_data_lva = df_reg_24_wdemo_data_lva
        self.demo_badge_id_25 = demo_badge_id_25
        self.demo_badge_id_24_bva = demo_badge_id_24_bva
        self.demo_badge_id_24_lva = demo_badge_id_24_lva

    def concatenate_qa_registration_data(self, df_registration, list_badge_id_events):
        """
        Concatenate Registration Values of a Participant.

        Args:
            df_registration: Pandas Dataframe with values of Participants
            list_badge_id_events: List with all BadgeId from which we have Geographic Data

        Returns:
            List of dictionaries containing concatenated registration data
        """

        def create_string_from_dict(data):
            keys_to_exclude = ["ShowRef", "BadgeId"]
            result = []

            for key, value in data.items():
                if key not in keys_to_exclude:
                    if value is not None:
                        result.append(f"{key}: {value}")
                    else:
                        result.append(f"{key}: no_data")

            return ", ".join(result)

        reg_data_list = []
        for badge_ID in list_badge_id_events:
            df = df_registration[df_registration["BadgeId"] == badge_ID]
            df_d = json.loads(df.to_json(orient="records"))
            reg_data = {}
            for row in df_d:
                ID = "_".join([row.get("ShowRef"), row.get("BadgeId")])
                reg_data[ID] = create_string_from_dict(row)
            if len(reg_data.keys()) > 0:
                reg_data_list.append(reg_data)

        self.logger.info(
            f"Created {len(reg_data_list)} concatenated registration data entries"
        )
        return reg_data_list

    def process_registration_data(self) -> None:
        """Process and save concatenated registration data."""
        # Define columns to keep
        valid_columns = [
            "Email",
            "Email_domain",
            "Company",
            "JobTitle",
            "Country",
            "BadgeType",
            "ShowRef",
            "BadgeId",
            "Source",
            "Days_since_registration",
        ]

        # Create valid column dataframes
        df_reg_25_valid_columns = self.df_reg_25_wdemo_data[valid_columns]
        df_reg_24_25_bva_valid_columns = self.df_reg_24_wdemo_data_bva[valid_columns]
        df_reg_24_25_lva_valid_columns = self.df_reg_24_wdemo_data_lva[valid_columns]

        # Create concatenated registration data
        reg_data_list_this_year = self.concatenate_qa_registration_data(
            df_reg_25_valid_columns, self.demo_badge_id_25
        )
        reg_data_list_past_year_bva = self.concatenate_qa_registration_data(
            df_reg_24_25_bva_valid_columns, self.demo_badge_id_24_bva
        )
        reg_data_list_past_year_lva = self.concatenate_qa_registration_data(
            df_reg_24_25_lva_valid_columns, self.demo_badge_id_24_lva
        )

        # Save concatenated registration data
        with open(
            os.path.join(
                self.output_dir, "output", "bva_registration_data_this_year.json"
            ),
            "w",
        ) as f:
            json.dump(reg_data_list_this_year, f, indent=4)
        with open(
            os.path.join(
                self.output_dir, "output", "bva_registration_data_past_year.json"
            ),
            "w",
        ) as f:
            json.dump(reg_data_list_past_year_bva, f, indent=4)
        with open(
            os.path.join(
                self.output_dir, "output", "lva_registration_data_past_year.json"
            ),
            "w",
        ) as f:
            json.dump(reg_data_list_past_year_lva, f, indent=4)

        self.logger.info("Saved concatenated registration data to JSON files")

    def concatenate_qa_demographic_data(
        self, df_demographic, list_badge_id_events, list_keep
    ):
        """
        Concatenate Questions and Answers of a Participant.

        Args:
            df_demographic: Pandas Dataframe with values of Participants
            list_badge_id_events: List with all BadgeId from which we have Geographic Data
            list_keep: List of questions that should be kept

        Returns:
            List of dictionaries containing demographic data
        """
        demo_data = []

        for badge_ID in list_badge_id_events:
            df = df_demographic[df_demographic["BadgeId"] == badge_ID]
            df_d = json.loads(df.to_json(orient="records"))
            qa = {badge_ID: {}}
            for row in df_d:
                question = row.get("QuestionText")
                answer = row.get("AnswerText")

                if not question or not answer:
                    logging.warning(
                        f"Issue with records ID {badge_ID} NO ANSWER/QUESTION in demographic data"
                    )
                    continue
                if question in list_keep:
                    qq = question.lower().replace(" ", "_")
                    qa[badge_ID][qq] = answer
            if len(qa[badge_ID].keys()) > 0:
                demo_data.append(qa)
            else:
                logging.warning(f"No demo data for {badge_ID}")

        self.logger.info(f"Created {len(demo_data)} demographic data entries")
        return demo_data

    def process_demographic_data(self) -> None:
        """Process and save demographic data."""
        # Get the questions to keep from config
        questions_to_keep = self.config.get("questions_to_keep", {})
        list_keep_this = questions_to_keep.get("current", [])
        list_keep_past = questions_to_keep.get("past", [])

        # Process demographic data
        demo_data_this = self.concatenate_qa_demographic_data(
            self.df_bva_demo_this_year, self.demo_badge_id_25, list_keep_this
        )
        demo_data_last_bva = self.concatenate_qa_demographic_data(
            self.df_bva_demo_last_year, self.demo_badge_id_24_bva, list_keep_past
        )
        demo_data_last_lva = self.concatenate_qa_demographic_data(
            self.df_lvs_demo, self.demo_badge_id_24_lva, list_keep_past
        )

        # Save demographic data
        with open(
            os.path.join(self.output_dir, "output", "demographic_data_this.json"), "w"
        ) as f:
            json.dump(demo_data_this, f, indent=4)
        with open(
            os.path.join(self.output_dir, "output", "demographic_data_last_bva.json"),
            "w",
        ) as f:
            json.dump(demo_data_last_bva, f, indent=4)
        with open(
            os.path.join(self.output_dir, "output", "demographic_data_last_lva.json"),
            "w",
        ) as f:
            json.dump(demo_data_last_lva, f, indent=4)

        self.logger.info("Saved demographic data to JSON files")

    def process_with_demographics(self) -> None:
        """Execute the workflow for processing demographic data."""
        self.logger.info("Starting demographic data processing workflow")

        self.save_demographic_data()
        self.extract_specialization_data()
        self.extract_job_roles()
        self.find_demographics_for_returning_visitors()
        self.find_registration_with_demographic_data()
        self.process_registration_data()
        self.process_demographic_data()

        self.logger.info("Demographic data processing workflow completed successfully")

    def create_col_placeholders(self, df, list_to_keep):
        """
        Create placeholder columns in the dataframe for demographic data.

        Args:
            df: DataFrame to add columns to
            list_to_keep: List of column names to add

        Returns:
            DataFrame with placeholder columns added
        """
        df_copy = df.copy()
        for col in list_to_keep:
            qq = col.lower().replace(" ", "_")
            df_copy[qq] = "NA"
        return df_copy

    def create_democols_in_registration_data(self, df, demo_data, list_keep):
        """
        Populate columns with demographic data values.

        Args:
            df: DataFrame to populate
            demo_data: List of dictionaries containing demographic data
            list_keep: List of columns to populate

        Returns:
            DataFrame with populated demographic data
        """
        # Create new columns with placeholders
        df = self.create_col_placeholders(df, list_keep)

        # Populate with demo data
        for reg in demo_data:
            badgeid = list(reg.keys())[0]
            list_keys_this_reg = list(reg[badgeid].keys())
            for col in list_keys_this_reg:
                df.loc[df["BadgeId"] == badgeid, col] = reg[badgeid][col]

        self.logger.info(
            f"Populated demographic data columns for {len(demo_data)} records"
        )
        return df

    def process_job_roles(self, df):
        """
        Process and standardize job roles using rules and fuzzy matching.

        Args:
            df: DataFrame containing job role information

        Returns:
            DataFrame with processed job roles
        """

        # Make a copy to avoid modifying the original dataframe
        df_copy = df.copy()

        # Only process rows where job_role is "NA"
        mask = df_copy["job_role"] == "NA"

        # Define potential roles
        potential_roles = [
            "Student",
            "Other (please specify)",
            "Receptionist",
            "Head Nurse/Senior Nurse",
            "Vet/Vet Surgeon",
            "Practice Partner/Owner",
            "Academic",
            "Clinical or other Director",
            "Assistant Vet",
            "Vet/Owner",
            "Vet Nurse",
            "Locum Vet",
            "Practice Manager",
            "Locum RVN",
        ]

        # Apply each rule in sequence
        for idx in df_copy[mask].index:
            job_title = str(df_copy.loc[idx, "JobTitle"]).lower()

            # Rule 1-6: Check for specific strings in JobTitle
            if "surgeon" in job_title:
                df_copy.loc[idx, "job_role"] = "Vet/Vet Surgeon"
            elif "nurse" in job_title:
                df_copy.loc[idx, "job_role"] = "Vet Nurse"
            elif "rvn" in job_title:
                df_copy.loc[idx, "job_role"] = "Vet Nurse"
            elif "locum" in job_title:
                df_copy.loc[idx, "job_role"] = "Locum Vet"
            elif "student" in job_title:
                df_copy.loc[idx, "job_role"] = "Student"
            elif "assistant" in job_title:
                df_copy.loc[idx, "job_role"] = "Assistant Vet"
            else:
                # Rule 7: Use text similarity to find the best matching role

                # Clean the job title: remove common words that might interfere with matching
                cleaned_title = re.sub(r"\b(and|the|of|in|at|for)\b", "", job_title)

                # Find the role with highest similarity score
                best_match = None
                best_score = 0

                for role in potential_roles:
                    # Calculate similarity between job title and each potential role
                    role_lower = role.lower()

                    # Check for key terms in the role
                    role_terms = role_lower.split("/")
                    role_terms.extend(role_lower.split())

                    # Calculate max similarity with any term in the role
                    max_term_score = 0
                    for term in role_terms:
                        if len(term) > 2:  # Only consider meaningful terms
                            if term in cleaned_title:
                                term_score = 0.9  # High score for direct matches
                            else:
                                # Use sequence matcher for fuzzy matching
                                term_score = SequenceMatcher(
                                    None, cleaned_title, term
                                ).ratio()
                            max_term_score = max(max_term_score, term_score)

                    if max_term_score > best_score:
                        best_score = max_term_score
                        best_match = role

                # If similarity is above threshold, use the best match
                if best_score > 0.3:  # Adjustable threshold
                    df_copy.loc[idx, "job_role"] = best_match
                else:
                    # Default to "Other" if no good match
                    df_copy.loc[idx, "job_role"] = "Other (please specify)"

        # Final rule: Replace any occurrence of "Other" with "Other (please specify)"
        other_mask = df_copy["job_role"].str.contains(
            "Other", case=False, na=False
        ) & ~df_copy["job_role"].eq("Other (please specify)")
        df_copy.loc[other_mask, "job_role"] = "Other (please specify)"

        self.logger.info(f"Processed job roles for {mask.sum()} records")
        return df_copy

    def fill_missing_practice_types(self, df, practices, column):
        """
        Fill missing practice types using fuzzy matching with company names.

        Args:
            df: DataFrame with potentially missing practice types
            practices: DataFrame with company names and practice types
            column: Name of the column containing practice types

        Returns:
            DataFrame with filled practice types
        """

        # Create a copy of the input dataframe to avoid modifying the original
        df_copy = df.copy()

        # Identify rows where practice types are "NA"
        missing_idx = df_copy[column] == "NA"
        companies_to_match = df_copy.loc[missing_idx, "Company"].tolist()

        # Create a dictionary of company names from practices dataframe
        practice_types_dict = dict(
            zip(
                practices["Company Name"], practices["Main Type of Veterinary Practice"]
            )
        )

        # List of all company names in practices dataframe for fuzzy matching
        all_practice_companies = practices["Company Name"].tolist()

        match_count = 0
        # For each company with "NA" practice type, find the best match
        for idx, company in zip(df_copy.loc[missing_idx].index, companies_to_match):
            # Skip if company name is missing or empty
            if pd.isna(company) or company == "":
                continue

            # Find the best match using fuzzy matching
            best_match, score = process.extractOne(company, all_practice_companies)

            # Only update if match score is good enough
            if score >= 95:  # 95% match threshold
                df_copy.loc[idx, column] = practice_types_dict[best_match]
                match_count += 1

        self.logger.info(
            f"Filled {match_count} missing practice types using fuzzy matching"
        )
        return df_copy

    def combine_demographic_with_registration(self):
        """Process and combine demographic data with registration data."""
        self.logger.info("Starting to combine demographic data with registration data")

        # Get the questions to keep from config
        questions_to_keep = self.config.get("questions_to_keep", {})
        list_keep_this = questions_to_keep.get("current", [])
        list_keep_past = questions_to_keep.get("past", [])

        # Create dataframes with valid columns for registration data
        valid_columns = [
            "Email",
            "Email_domain",
            "Company",
            "JobTitle",
            "Country",
            "BadgeType",
            "ShowRef",
            "BadgeId",
            "Source",
            "Days_since_registration",
            "assist_year_before",  # Make sure to include this
            "BadgeId_last_year_bva",  # Add this
            "BadgeId_last_year_lva",  # Add this
        ]

        # Ensure we have all the columns in the dataframe
        for col in valid_columns:
            if col not in self.df_reg_25_wdemo_data.columns:
                self.logger.warning(f"Column {col} not in df_reg_25_wdemo_data")
                if col in [
                    "BadgeId_last_year_bva",
                    "BadgeId_last_year_lva",
                    "assist_year_before",
                ]:
                    self.logger.info(f"Adding missing column {col}")
                    self.df_reg_25_wdemo_data[col] = "NA"

        # Now select only the columns we need
        available_columns = [
            col for col in valid_columns if col in self.df_reg_25_wdemo_data.columns
        ]
        df_reg_25_valid_columns = self.df_reg_25_wdemo_data[available_columns].copy()

        # For past year data, we don't need the badge history columns
        past_valid_columns = [
            col
            for col in valid_columns
            if col
            not in [
                "BadgeId_last_year_bva",
                "BadgeId_last_year_lva",
                "assist_year_before",
            ]
        ]

        # CRITICAL CHANGE: For previous year visitors, we need to start with the original BVA and LVS registration data
        # rather than the filtered demo data. This ensures we include ALL visitors from previous years.

        # Create valid column dataframes for returning visitors
        # Use the returning visitors dataframes directly instead of the filtered ones with demo data
        df_reg_24_25_bva_valid_columns = self.df_bva_returning[
            past_valid_columns
        ].copy()
        df_reg_24_25_lva_valid_columns = self.df_lvs_returning[
            past_valid_columns
        ].copy()

        # Filter current year to only include badges with demo data
        df_reg_25_valid_columns = df_reg_25_valid_columns[
            df_reg_25_valid_columns["BadgeId"].isin(self.demo_badge_id_25)
        ]

        # Log initial counts before any filtering
        self.logger.info(
            f"Initial counts - BVA returning: {len(self.df_bva_returning)}, LVS returning: {len(self.df_lvs_returning)}"
        )

        # Log the lengths after filtering
        self.logger.info(
            f"After filtering: current year: {len(df_reg_25_valid_columns)}, BVA last year: {len(df_reg_24_25_bva_valid_columns)}, LVA last year: {len(df_reg_24_25_lva_valid_columns)}"
        )

        # Process demographic data
        demo_data_this = self.concatenate_qa_demographic_data(
            self.df_bva_demo_this_year, self.demo_badge_id_25, list_keep_this
        )

        # For past years, we still need demographic data, but we'll apply it to all returning visitors
        # not just those with demo data from previous year
        demo_data_last_bva = self.concatenate_qa_demographic_data(
            self.df_bva_demo_last_year, self.demo_badge_id_24_bva, list_keep_past
        )
        demo_data_last_lva = self.concatenate_qa_demographic_data(
            self.df_lvs_demo, self.demo_badge_id_24_lva, list_keep_past
        )

        self.logger.info(
            f"Processed demographic data. Created demographic data for {len(demo_data_this)} this year, {len(demo_data_last_bva)} last year BVA, and {len(demo_data_last_lva)} last year LVS"
        )
        # Create combined registration and demographic data
        df_reg_demo_this = self.create_democols_in_registration_data(
            df_reg_25_valid_columns, demo_data_this, list_keep_this
        )
        df_reg_demo_last_bva = self.create_democols_in_registration_data(
            df_reg_24_25_bva_valid_columns, demo_data_last_bva, list_keep_past
        )
        df_reg_demo_last_lva = self.create_democols_in_registration_data(
            df_reg_24_25_lva_valid_columns, demo_data_last_lva, list_keep_past
        )

        # Process job roles
        df_reg_demo_this = self.process_job_roles(df_reg_demo_this)
        df_reg_demo_last_bva = self.process_job_roles(df_reg_demo_last_bva)
        df_reg_demo_last_lva = self.process_job_roles(df_reg_demo_last_lva)
        self.logger.info("Processed job roles for all demographic data entries")
        self.logger.info("Length of df_reg_demo_this: " + str(len(df_reg_demo_this)))
        self.logger.info(
            "Length of df_reg_demo_last_bva: " + str(len(df_reg_demo_last_bva))
        )
        self.logger.info(
            "Length of df_reg_demo_last_lva: " + str(len(df_reg_demo_last_lva))
        )

        # Check for badge history columns
        if "BadgeId_last_year_bva" in df_reg_demo_this.columns:
            self.logger.info("BadgeId_last_year_bva column exists in final dataframe")
            self.logger.info(
                f"Number of unique BadgeId_last_year_bva values: {len(df_reg_demo_this['BadgeId_last_year_bva'].unique())}"
            )
        else:
            self.logger.warning(
                "BadgeId_last_year_bva column missing from final dataframe"
            )

        if "BadgeId_last_year_lva" in df_reg_demo_this.columns:
            self.logger.info("BadgeId_last_year_lva column exists in final dataframe")
            self.logger.info(
                f"Number of unique BadgeId_last_year_lva values: {len(df_reg_demo_this['BadgeId_last_year_lva'].unique())}"
            )
        else:
            self.logger.warning(
                "BadgeId_last_year_lva column missing from final dataframe"
            )

        # Load practice data for filling missing practice types
        practices_file = self.config.get("input_files", {}).get("practices", "")
        if practices_file:
            try:
                practices = pd.read_csv(practices_file)

                # Fill missing practice types
                df_reg_demo_this = self.fill_missing_practice_types(
                    df_reg_demo_this,
                    practices,
                    column="what_type_does_your_practice_specialise_in",
                )
                df_reg_demo_last_bva = self.fill_missing_practice_types(
                    df_reg_demo_last_bva,
                    practices,
                    column="what_areas_do_you_specialise_in",
                )
                df_reg_demo_last_lva = self.fill_missing_practice_types(
                    df_reg_demo_last_lva,
                    practices,
                    column="what_areas_do_you_specialise_in",
                )

                self.logger.info("Filled missing practice types using practices data")
            except Exception as e:
                self.logger.error(f"Error loading or processing practices data: {e}")

        # Save combined data
        df_reg_demo_this.to_csv(
            os.path.join(self.output_dir, "output", "df_reg_demo_this.csv"), index=False
        )
        df_reg_demo_last_bva.to_csv(
            os.path.join(self.output_dir, "output", "df_reg_demo_last_bva.csv"),
            index=False,
        )
        df_reg_demo_last_lva.to_csv(
            os.path.join(self.output_dir, "output", "df_reg_demo_last_lva.csv"),
            index=False,
        )
        self.logger.info(
            "Saved combined demographic and registration data to CSV files"
        )

        # Store for potential further use
        self.df_reg_demo_this = df_reg_demo_this
        self.df_reg_demo_last_bva = df_reg_demo_last_bva
        self.df_reg_demo_last_lva = df_reg_demo_last_lva

        self.logger.info("Finished combining demographic with registration data")

        # Return statistics
        return {
            "this_year": len(df_reg_demo_this),
            "last_year_bva": len(df_reg_demo_last_bva),
            "last_year_lva": len(df_reg_demo_last_lva),
        }

    def process(self) -> None:
        """Execute the full data processing workflow."""
        self.logger.info("Starting data processing workflow")

        self.load_data()
        self.save_initial_data()
        self.preprocess_data()
        self.split_and_process_bva_data()
        self.identify_returning_visitors()
        self.add_badge_history()
        self.calculate_event_dates()
        self.flag_all_returning_visitors()
        self.select_and_save_final_data()
        self.process_with_demographics()
        self.combine_demographic_with_registration()

        self.logger.info("Data processing workflow completed successfully")
