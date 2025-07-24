import os
import logging
import pandas as pd
import string
import json
from typing import Dict, List, Set, Tuple
import functools


class ScanProcessor:
    """Process scan data for event analytics."""

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
        os.makedirs(os.path.join(self.output_dir, "output"), exist_ok=True)

        # Use existing logger instead of configuring a new one
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized ScanProcessor with output to {self.output_dir}")

    def load_scan_data(self) -> None:
        """Load all scan data files based on configuration."""
        try:
            # Load session data
            session_this_path = self.config["scan_files"]["session_this"]
            session_past_path_bva = self.config["scan_files"]["session_past_bva"]
            session_past_path_lva = self.config["scan_files"]["session_past_lva"]

            # Load seminar reference data
            seminars_scan_reference_past_path_bva = self.config["scan_files"][
                "seminars_scan_reference_past_bva"
            ]
            seminars_scans_past_path_bva = self.config["scan_files"][
                "seminars_scans_past_bva"
            ]
            seminars_scan_reference_past_path_lva = self.config["scan_files"][
                "seminars_scan_reference_past_lva"
            ]
            seminars_scans_past_path_lva = self.config["scan_files"][
                "seminars_scans_past_lva"
            ]

            # Read session data
            self.session_this = pd.read_csv(session_this_path)
            self.session_past_bva = pd.read_csv(session_past_path_bva)
            self.session_past_lva = pd.read_csv(session_past_path_lva)

            # Read seminar reference data
            self.seminars_scan_reference_past_bva = pd.read_csv(
                seminars_scan_reference_past_path_bva
            )
            self.seminars_scans_past_bva = pd.read_csv(seminars_scans_past_path_bva)
            self.seminars_scan_reference_past_lva = pd.read_csv(
                seminars_scan_reference_past_path_lva
            )
            self.seminars_scans_past_lva = pd.read_csv(seminars_scans_past_path_lva)

            self.logger.info(
                f"Loaded session data: {len(self.session_this)} records this year, "
                f"{len(self.session_past_bva)} BVA records last year, "
                f"{len(self.session_past_lva)} LVA records last year"
            )

            self.logger.info(
                f"Loaded seminar scan data: {len(self.seminars_scans_past_bva)} BVA records, "
                f"{len(self.seminars_scans_past_lva)} LVA records"
            )

        except Exception as e:
            self.logger.error(f"Error loading scan data: {e}", exc_info=True)
            raise

    def load_registration_data(self) -> None:
        """Load processed registration data from previous step."""
        try:
            # Load registration data with demographics
            reg_data_last_bva_path = os.path.join(
                self.output_dir, "output", "df_reg_demo_last_bva.csv"
            )
            reg_data_last_lva_path = os.path.join(
                self.output_dir, "output", "df_reg_demo_last_lva.csv"
            )

            detailed_reg_data_last_bva_path = os.path.join(
                self.output_dir,
                "output",
                "Registration_data_with_demographicdata_bva_last.csv",
            )
            detailed_reg_data_last_lva_path = os.path.join(
                self.output_dir,
                "output",
                "Registration_data_with_demographicdata_lva_last.csv",
            )

            # Read registration data
            self.reg_data_last_bva = pd.read_csv(reg_data_last_bva_path)
            self.reg_data_last_lva = pd.read_csv(reg_data_last_lva_path)

            self.df_reg_24_wdemo_data_bva = pd.read_csv(detailed_reg_data_last_bva_path)
            self.df_reg_24_wdemo_data_lva = pd.read_csv(detailed_reg_data_last_lva_path)

            self.logger.info(
                f"Loaded registration data: {len(self.reg_data_last_bva)} BVA records, "
                f"{len(self.reg_data_last_lva)} LVA records"
            )

            self.logger.info(
                f"Loaded detailed registration data: {len(self.df_reg_24_wdemo_data_bva)} BVA records, "
                f"{len(self.df_reg_24_wdemo_data_lva)} LVA records"
            )

        except Exception as e:
            self.logger.error(f"Error loading registration data: {e}", exc_info=True)
            raise

    @staticmethod
    def clean_text(text):
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

    def enhance_seminar_data(self) -> None:
        """Enhance seminar data with seminar names and create keys for matching."""
        try:
            # Enhance BVA seminar data
            self.seminars_scans_past_enhanced_bva = pd.merge(
                self.seminars_scans_past_bva,
                self.seminars_scan_reference_past_bva[["Short Name", "Seminar Name"]],
                on=["Short Name"],
                how="left",
            )

            # Enhance LVA seminar data
            self.seminars_scans_past_enhanced_lva = pd.merge(
                self.seminars_scans_past_lva,
                self.seminars_scan_reference_past_lva[["Short Name", "Seminar Name"]],
                on=["Short Name"],
                how="left",
            )

            # Create keys for matching seminar names
            self.seminars_scans_past_enhanced_bva["key_text"] = (
                self.seminars_scans_past_enhanced_bva["Seminar Name"].apply(
                    self.clean_text
                )
            )
            self.seminars_scans_past_enhanced_lva["key_text"] = (
                self.seminars_scans_past_enhanced_lva["Seminar Name"].apply(
                    self.clean_text
                )
            )

            # Create keys for session titles
            self.session_past_bva["key_text"] = self.session_past_bva["title"].apply(
                self.clean_text
            )
            self.session_past_lva["key_text"] = self.session_past_lva["title"].apply(
                self.clean_text
            )

            self.logger.info(
                f"Enhanced seminar data with seminar names: {len(self.seminars_scans_past_enhanced_bva)} BVA records, "
                f"{len(self.seminars_scans_past_enhanced_lva)} LVA records"
            )

        except Exception as e:
            self.logger.error(f"Error enhancing seminar data: {e}", exc_info=True)
            raise

    def analyze_data_intersections(self) -> None:
        """Analyze intersections between different datasets."""
        try:
            # BVA seminar and session intersections
            bva_seminar_keys = set(
                list(self.seminars_scans_past_enhanced_bva["key_text"].unique())
            )
            bva_session_keys = set(list(self.session_past_bva["key_text"].unique()))
            bva_keys_intersection = bva_seminar_keys.intersection(bva_session_keys)

            # LVA seminar and session intersections
            lva_seminar_keys = set(
                list(self.seminars_scans_past_enhanced_lva["key_text"].unique())
            )
            lva_session_keys = set(list(self.session_past_lva["key_text"].unique()))
            lva_keys_intersection = lva_seminar_keys.intersection(lva_session_keys)

            # BVA registration and seminar badge ID intersections
            bva_reg_badges = set(list(self.reg_data_last_bva["BadgeId"].unique()))
            bva_seminar_badges = set(
                list(self.seminars_scans_past_enhanced_bva["Badge Id"].unique())
            )
            bva_badges_intersection = bva_reg_badges.intersection(bva_seminar_badges)

            # LVA registration and seminar badge ID intersections
            lva_reg_badges = set(list(self.reg_data_last_lva["BadgeId"].unique()))
            lva_seminar_badges = set(
                list(self.seminars_scans_past_enhanced_lva["Badge Id"].unique())
            )
            lva_badges_intersection = lva_reg_badges.intersection(lva_seminar_badges)

            # Log statistics
            self.logger.info(
                f"BVA seminar-session intersection: {len(bva_keys_intersection)} out of {len(bva_seminar_keys)} seminar keys and {len(bva_session_keys)} session keys"
            )
            self.logger.info(
                f"LVA seminar-session intersection: {len(lva_keys_intersection)} out of {len(lva_seminar_keys)} seminar keys and {len(lva_session_keys)} session keys"
            )
            self.logger.info(
                f"BVA registration-seminar badge intersection: {len(bva_badges_intersection)} out of {len(bva_reg_badges)} registration badges and {len(bva_seminar_badges)} seminar badges"
            )
            self.logger.info(
                f"LVA registration-seminar badge intersection: {len(lva_badges_intersection)} out of {len(lva_reg_badges)} registration badges and {len(lva_seminar_badges)} seminar badges"
            )

        except Exception as e:
            self.logger.error(f"Error analyzing data intersections: {e}", exc_info=True)
            raise

    def add_demographics_to_seminars(
        self, registration_df: pd.DataFrame, seminars_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add demographic data to seminar scans dataframe for matching badge IDs.

        Args:
            registration_df: DataFrame containing registration and demographic data
            seminars_df: DataFrame containing seminar attendance data

        Returns:
            DataFrame with demographic data added to seminar scans
        """
        try:
            # Create copies to avoid modifying original dataframes
            registration_df = registration_df.copy()
            seminars_df = seminars_df.copy()

            # Rename 'BadgeId' column in registration dataframe to match 'Badge Id' in seminars dataframe
            registration_df = registration_df.rename(columns={"BadgeId": "Badge Id"})

            # Select demographic columns to include
            demographic_columns = [
                "Badge Id",
                "Title",
                "Forename",
                "Surname",
                "Email",
                "Company",
                "JobTitle",
                "Country",
                "BadgeType",
                "Email_domain",
            ]

            # Check if columns exist and filter to available columns
            available_cols = [
                col for col in demographic_columns if col in registration_df.columns
            ]
            reg_slim = registration_df[available_cols]

            # Merge the seminars dataframe with the registration dataframe
            enhanced_seminars = pd.merge(
                seminars_df, reg_slim, on="Badge Id", how="left"
            )

            # Log statistics
            total_seminars = len(seminars_df)
            matched_seminars = enhanced_seminars["Email"].notna().sum()
            match_percentage = (
                (matched_seminars / total_seminars * 100) if total_seminars > 0 else 0
            )

            self.logger.info(f"Total seminar records: {total_seminars}")
            self.logger.info(
                f"Records with matching badge IDs: {matched_seminars} ({match_percentage:.2f}%)"
            )

            return enhanced_seminars

        except Exception as e:
            self.logger.error(
                f"Error adding demographics to seminars: {e}", exc_info=True
            )
            raise

    def enhance_seminar_data_with_demographics(self) -> None:
        """Enhance seminar data with demographic information."""
        try:
            # Add demographics to BVA seminar data
            self.enhanced_seminars_df_bva = self.add_demographics_to_seminars(
                self.df_reg_24_wdemo_data_bva, self.seminars_scans_past_enhanced_bva
            )

            # Add demographics to LVA seminar data
            self.enhanced_seminars_df_lva = self.add_demographics_to_seminars(
                self.df_reg_24_wdemo_data_lva, self.seminars_scans_past_enhanced_lva
            )

            self.logger.info("Successfully enhanced seminar data with demographics")

        except Exception as e:
            self.logger.error(
                f"Error enhancing seminar data with demographics: {e}", exc_info=True
            )
            raise

    def save_processed_data(self) -> None:
        """Save all processed data to CSV files."""
        try:
            # Save enhanced seminar scan data
            bva_scan_output_path = os.path.join(
                self.output_dir, "output", "scan_bva_past.csv"
            )
            lva_scan_output_path = os.path.join(
                self.output_dir, "output", "scan_lva_past.csv"
            )

            # Save enhanced seminar data with demographics
            bva_sessions_output_path = os.path.join(
                self.output_dir, "output", "sessions_visited_last_bva.csv"
            )
            lva_sessions_output_path = os.path.join(
                self.output_dir, "output", "sessions_visited_last_lva.csv"
            )

            # Save the data
            self.seminars_scans_past_enhanced_bva.to_csv(
                bva_scan_output_path, index=False
            )
            self.seminars_scans_past_enhanced_lva.to_csv(
                lva_scan_output_path, index=False
            )
            self.enhanced_seminars_df_bva.to_csv(bva_sessions_output_path, index=False)
            self.enhanced_seminars_df_lva.to_csv(lva_sessions_output_path, index=False)

            self.logger.info(f"Saved processed scan data to {self.output_dir}/output/")
            self.logger.info(
                f"BVA scan data: {len(self.seminars_scans_past_enhanced_bva)} records"
            )
            self.logger.info(
                f"LVA scan data: {len(self.seminars_scans_past_enhanced_lva)} records"
            )
            self.logger.info(
                f"BVA sessions with demographics: {len(self.enhanced_seminars_df_bva)} records"
            )
            self.logger.info(
                f"LVA sessions with demographics: {len(self.enhanced_seminars_df_lva)} records"
            )

        except Exception as e:
            self.logger.error(f"Error saving processed data: {e}", exc_info=True)
            raise

    def process(self) -> None:
        """Execute the full scan data processing workflow."""
        self.logger.info("Starting scan data processing workflow")

        self.load_scan_data()
        self.load_registration_data()
        self.enhance_seminar_data()
        self.analyze_data_intersections()
        self.enhance_seminar_data_with_demographics()
        self.save_processed_data()

        self.logger.info("Scan data processing workflow completed successfully")
