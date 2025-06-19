import polars as pl
from dmqclib.datasets.select.select_base import ProfileSelectionBase


class SelectDataSetA(ProfileSelectionBase):
    """
    SelectDataSetA inherits from ProfileSelectionBase and sets the 'expected_class_name' to 'SelectDataSetA'.
    """

    expected_class_name = "SelectDataSetA"

    def __init__(
        self,
        dataset_name: str,
        config_file: str = None,
        input_data: pl.DataFrame = None,
    ):
        super().__init__(dataset_name, config_file=config_file, input_data=input_data)

        self.pos_profile_df = None
        self.neg_profile_df = None

    def select_positive_profiles(self):
        """
        Select profiles with invalid value flags as positive profiles
        """
        self.pos_profile_df = (
            self.input_data
            .filter(
                (pl.col("temp_qc") == 4) |
                (pl.col("psal_qc") == 4) |
                (pl.col("pres_qc") == 4)
            )
            .select(["platform_code", "profile_no", "profile_timestamp", "longitude", "latitude"])
            .unique(subset=["platform_code", "profile_no", "profile_timestamp", "longitude", "latitude"])
            .with_row_index("profile_id", offset=1)
            .with_columns(
                pl.col("profile_timestamp").dt.ordinal_day().alias("pos_day_of_year")
            )
        )

    def select_negative_profiles(self):
        """
        Select profiles with all valid value flags as negative profiles
        """
        self.neg_profile_df = (
            self.input_data
            .group_by(["platform_code", "profile_no", "profile_timestamp", "longitude", "latitude"])
            .agg([
                pl.col("temp_qc").max().alias("max_temp_qc"),
                pl.col("psal_qc").max().alias("max_psal_qc"),
                pl.col("pres_qc").max().alias("max_pres_qc"),
                pl.col("temp_qc_dm").max().alias("max_temp_qc_dm"),
                pl.col("psal_qc_dm").max().alias("max_psal_qc_dm"),
                pl.col("pres_qc_dm").max().alias("max_pres_qc_dm"),
            ])
            .filter(
                (pl.col("max_temp_qc") == 1) &
                (pl.col("max_psal_qc") == 1) &
                (pl.col("max_pres_qc") == 1) &
                (pl.col("max_temp_qc_dm") == 1) &
                (pl.col("max_psal_qc_dm") == 1) &
                (pl.col("max_pres_qc_dm") == 1)
            )
            .select(["platform_code", "profile_no", "profile_timestamp", "longitude", "latitude"])
            .with_row_index("profile_id", offset=1)
            .with_columns(
                pl.col("profile_timestamp").dt.ordinal_day().alias("neg_day_of_year")
            )
        )

    def label_profiles(self):
        """
        Label profiles in terms of positive and negative candidates
        """
        pass

    def filter_profiles(self):
        """
        Filter profiles based on the labels
        """
        pass
