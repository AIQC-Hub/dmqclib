import polars as pl
from typing import Optional, List

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.prepare.step3_select.select_base import ProfileSelectionBase


class SelectDataSetAll(ProfileSelectionBase):
    """
    A subclass of :class:`ProfileSelectionBase` that select all profiles from BO NRT + Cora test data.

    """

    expected_class_name: str = "SelectDataSetAll"

    def __init__(
        self, config: ConfigBase, input_data: Optional[pl.DataFrame] = None
    ) -> None:
        """
        Initialize an instance for selecting and labeling profiles.

        :param config: The configuration object specifying paths and
                       parameters for the selection process.
        :type config: ConfigBase
        :param input_data: An optional Polars DataFrame of all profiles
                           from which negative and positive examples are
                           to be selected. If not provided, it must be
                           assigned later.
        :type input_data: pl.DataFrame, optional
        """
        super().__init__(config, input_data=input_data)

        #: Default file name to which selected profiles are written.
        self.default_file_name: str = "selected_classify_profiles.parquet"

        #: Full path for the output file, resolved via the config.
        self.output_file_name: str = self.config.get_full_file_name(
            "select", self.default_file_name
        )

        #: Columns used as unique identifiers for grouping/merging
        #: (e.g., by platform or profile).
        self.key_col_names: List[str] = [
            "platform_code",
            "profile_no",
            "profile_timestamp",
            "longitude",
            "latitude",
        ]

    def select_all_profiles(self) -> None:
        """
        Select all profiles from the input data and assign initial
        identifiers for negative profiles and label columns.

        The resulting DataFrame is assigned to :attr:`selected_profiles`.
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
        Select and label positive and negative datasets before combining them
        into a single DataFrame in :attr:`selected_profiles`.
        """
        self.select_all_profiles()
