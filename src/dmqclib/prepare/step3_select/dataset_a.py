import polars as pl
from typing import Optional, List

from dmqclib.common.config.dataset_config import DataSetConfig
from dmqclib.prepare.step3_select.select_base import ProfileSelectionBase


class SelectDataSetA(ProfileSelectionBase):
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

    expected_class_name: str = "SelectDataSetA"

    def __init__(
        self, config: DataSetConfig, input_data: Optional[pl.DataFrame] = None
    ) -> None:
        """
        Initialize the dataset for selecting and labeling profiles.

        :param config: The dataset configuration object that includes
                       paths and parameters for the selection process.
        :type config: DataSetConfig
        :param input_data: A Polars DataFrame containing the full set
                           of profiles from which to select positive
                           and negative examples.
        :type input_data: pl.DataFrame, optional
        """
        super().__init__(config, input_data=input_data)

        #: Polars DataFrame containing positively-labeled profiles.
        self.pos_profile_df: Optional[pl.DataFrame] = None
        #: Polars DataFrame containing negatively-labeled profiles.
        self.neg_profile_df: Optional[pl.DataFrame] = None
        #: Column names used as unique identifiers for grouping or merging.
        self.key_col_names: List[str] = [
            "platform_code",
            "profile_no",
            "profile_timestamp",
            "longitude",
            "latitude",
        ]

    def select_positive_profiles(self) -> None:
        """
        Select profiles that have a QC flag of 4 in any of the
        specified QC columns, labeling them as "positive" (i.e.,
        containing errors).

        The resulting DataFrame is stored in :attr:`pos_profile_df`.
        """
        self.pos_profile_df = (
            self.input_data.filter(
                (pl.col("temp_qc") == 4)
                | (pl.col("psal_qc") == 4)
                | (pl.col("pres_qc") == 4)
            )
            .select(self.key_col_names)
            .unique(subset=self.key_col_names)
            .sort(["platform_code", "profile_no"])
            .with_row_index("profile_id", offset=1)
            .with_columns(
                pl.col("profile_timestamp").dt.ordinal_day().alias("pos_day_of_year")
            )
        )

    def select_negative_profiles(self) -> None:
        """
        Select profiles that have a QC flag of 1 in all specified QC columns,
        labeling them as "negative" (i.e., containing only good measurements).

        The resulting DataFrame is stored in :attr:`neg_profile_df`.
        """
        self.neg_profile_df = (
            self.input_data.group_by(self.key_col_names)
            .agg(
                [
                    pl.col("temp_qc").max().alias("max_temp_qc"),
                    pl.col("psal_qc").max().alias("max_psal_qc"),
                    pl.col("pres_qc").max().alias("max_pres_qc"),
                    pl.col("temp_qc_dm").max().alias("max_temp_qc_dm"),
                    pl.col("psal_qc_dm").max().alias("max_psal_qc_dm"),
                    pl.col("pres_qc_dm").max().alias("max_pres_qc_dm"),
                ]
            )
            .filter(
                (pl.col("max_temp_qc") == 1)
                & (pl.col("max_psal_qc") == 1)
                & (pl.col("max_pres_qc") == 1)
                & (pl.col("max_temp_qc_dm") == 1)
                & (pl.col("max_psal_qc_dm") == 1)
                & (pl.col("max_pres_qc_dm") == 1)
            )
            .select(self.key_col_names)
            .sort(["platform_code", "profile_no"])
            .with_row_index("profile_id", offset=self.pos_profile_df.shape[0] + 1)
            .with_columns(
                pl.col("profile_timestamp").dt.ordinal_day().alias("neg_day_of_year")
            )
        )

    def find_profile_pairs(self) -> None:
        """
        Identify the negative profile whose date is closest to each positive profile,
        to reduce the negative set to only those instances that are temporally
        near the positive set.
        """
        closest_neg_id = (
            self.pos_profile_df.join(self.neg_profile_df, how="cross", suffix="_neg")
            .with_columns(
                (pl.col("pos_day_of_year") - pl.col("neg_day_of_year"))
                .abs()
                .alias("day_diff")
            )
            .group_by("profile_id")
            .agg(
                pl.col("profile_id_neg")
                .sort_by(["day_diff", "profile_id"])
                .first()
                .alias("neg_profile_id")
            )
        )

        self.pos_profile_df = (
            self.pos_profile_df.join(closest_neg_id, on="profile_id", how="left")
            .with_columns(pl.lit(1).alias("label"))
            .drop("pos_day_of_year")
        )

        self.neg_profile_df = (
            self.neg_profile_df.filter(
                pl.col("profile_id").is_in(closest_neg_id["neg_profile_id"].to_list())
            )
            .with_columns(
                pl.lit(0, dtype=pl.UInt32).alias("neg_profile_id"),
                pl.lit(0).alias("label"),
            )
            .drop("neg_day_of_year")
        )

    def label_profiles(self) -> None:
        """
        Select and label positive and negative datasets, then combine them
        into a single DataFrame in :attr:`selected_profiles`.
        """
        self.select_positive_profiles()
        self.select_negative_profiles()
        self.find_profile_pairs()

        self.selected_profiles = self.pos_profile_df.vstack(self.neg_profile_df)
