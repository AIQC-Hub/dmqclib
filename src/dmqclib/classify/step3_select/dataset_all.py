import polars as pl
from typing import Optional, List

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.prepare.step3_select.select_base import ProfileSelectionBase


class SelectDataSetAll(ProfileSelectionBase):
    """
    A subclass of :class:`ProfileSelectionBase` that defines negative
    and positive profiles from BO NRT + Cora test data.

    Main Steps:

      1. Select positive profiles: Those that have a QC flag of 4
         (indicating a "bad" measurement) in at least one of
         ``temp_qc``, ``psal_qc``, or ``pres_qc``.
      2. Select negative profiles: Those that have a QC flag of 1
         (indicating a "good" measurement) in each of
         ``temp_qc``, ``psal_qc``, ``pres_qc``, ``temp_qc_dm``,
         ``psal_qc_dm``, and ``pres_qc_dm``.
      3. Identify pairs by matching negative and positive profiles
         based on their proximity in time.
      4. Combine dataframes into a single DataFrame of selected profiles.
    """

    expected_class_name: str = "SelectDataSetAll"

    def __init__(
        self, config: ConfigBase, input_data: Optional[pl.DataFrame] = None
    ) -> None:
        """
        Initialize the dataset for selecting and labeling profiles.

        :param config: The dataset configuration object that includes
                       paths and parameters for the selection process.
        :type config: ConfigBase
        :param input_data: A Polars DataFrame containing the full set
                           of profiles from which to select positive
                           and negative examples.
        :type input_data: pl.DataFrame, optional
        """
        super().__init__(config, input_data=input_data)

        self.default_file_name: str = "selected_classify_profiles.parquet"
        self.output_file_name: str = self.config.get_full_file_name(
            "select", self.default_file_name
        )

        #: Column names used as unique identifiers for grouping or merging.
        self.key_col_names: List[str] = [
            "platform_code",
            "profile_no",
            "profile_timestamp",
            "longitude",
            "latitude",
        ]

    def select_all_profiles(self) -> None:
        """
        Select all profiles and label them as positives and negatives depending on QC flags.

        The resulting DataFrame is stored in :attr:`pos_profile_df`.
        """
        self.selected_profiles = (
            self.input_data.with_columns(
                pl.lit(0, dtype=pl.UInt32).alias("neg_profile_id"),
                pl.lit(0, dtype=pl.UInt32).alias("label"),
            )
            .select(
                pl.col("platform_code"),
                pl.col("profile_no"),
                pl.col("profile_timestamp"),
                pl.col("longitude"),
                pl.col("latitude"),
                pl.col("neg_profile_id"),
                pl.col("label"),
            )
            .unique(maintain_order=True)
            .with_row_index("profile_id", offset=1)
        )

    def label_profiles(self) -> None:
        """
        Select and label positive and negative datasets, then combine them
        into a single DataFrame in :attr:`selected_profiles`.
        """
        self.select_all_profiles()
