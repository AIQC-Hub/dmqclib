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
            self.input_data.filter(
                (pl.col("temp_qc") == 4)
                | (pl.col("psal_qc") == 4)
                | (pl.col("pres_qc") == 4)
            )
            .select(
                [
                    "platform_code",
                    "profile_no",
                    "profile_timestamp",
                    "longitude",
                    "latitude",
                ]
            )
            .unique(
                subset=[
                    "platform_code",
                    "profile_no",
                    "profile_timestamp",
                    "longitude",
                    "latitude",
                ]
            )
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
            self.input_data.group_by(
                [
                    "platform_code",
                    "profile_no",
                    "profile_timestamp",
                    "longitude",
                    "latitude",
                ]
            )
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
            .select(
                [
                    "platform_code",
                    "profile_no",
                    "profile_timestamp",
                    "longitude",
                    "latitude",
                ]
            )
            .with_row_index("profile_id", offset=self.pos_profile_df.shape[0] + 1)
            .with_columns(
                pl.col("profile_timestamp").dt.ordinal_day().alias("neg_day_of_year")
            )
        )

    def find_profile_pairs(self):
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
                .sort_by("day_diff")
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

    def label_profiles(self):
        self.select_positive_profiles()
        self.select_negative_profiles()
        self.find_profile_pairs()

        self.selected_profiles = self.pos_profile_df.vstack(self.neg_profile_df)
