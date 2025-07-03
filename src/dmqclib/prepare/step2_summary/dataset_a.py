import polars as pl

from dmqclib.config.dataset_config import DataSetConfig
from dmqclib.prepare.step2_summary.summary_base import SummaryStatsBase


class SummaryDataSetA(SummaryStatsBase):
    """
    A subclass of :class:`SummaryStatsBase` that calculates summary statistics
    for BO NRT + Cora test data.

    This class uses Polars to generate both global and per-profile statistics
    for a set of specified columns (:attr:`val_col_names`). It expects column names
    related to measurement attributes such as temperature, salinity, etc.,
    and also relies on certain profile identifiers (:attr:`profile_col_names`)
    to group data by platform and profile number.
    """

    expected_class_name: str = "SummaryDataSetA"

    def __init__(self, config: DataSetConfig, input_data: pl.DataFrame = None) -> None:
        """
        Initialize the dataset for summary statistics generation.

        :param config: The dataset configuration object containing paths
                       and parameters for summary stats.
        :type config: DataSetConfig
        :param input_data: A Polars DataFrame from which to calculate summary stats.
                           If None, data must be assigned later.
        :type input_data: pl.DataFrame, optional
        """
        super().__init__(config, input_data=input_data)

        #: List of numeric columns on which summary statistics will be computed.
        self.val_col_names = [
            "longitude",
            "latitude",
            "temp",
            "psal",
            "pres",
            "dist2coast",
            "bath",
        ]
        #: List of column names used as profile identifiers for group-by operations.
        self.profile_col_names = ["platform_code", "profile_no"]

    def calculate_global_stats(self, val_col_name: str) -> pl.DataFrame:
        """
        Calculate global summary statistics for a specific numeric column across all rows.

        :param val_col_name: The name of the column for which to calculate global stats.
        :type val_col_name: str
        :return: A Polars DataFrame containing global metrics (e.g., min, max, mean)
                 for the specified column, with placeholders for platform_code and
                 profile_no to align with the structure of per-profile stats.
        :rtype: pl.DataFrame
        """
        return (
            self.input_data.select(
                [
                    pl.col(val_col_name).min().cast(pl.Float64).alias("min"),
                    pl.col(val_col_name).max().cast(pl.Float64).alias("max"),
                    pl.col(val_col_name).mean().cast(pl.Float64).alias("mean"),
                    pl.col(val_col_name).median().cast(pl.Float64).alias("median"),
                    pl.col(val_col_name).quantile(0.25).cast(pl.Float64).alias("pct25"),
                    pl.col(val_col_name).quantile(0.75).cast(pl.Float64).alias("pct75"),
                    pl.col(val_col_name)
                    .quantile(0.025)
                    .cast(pl.Float64)
                    .alias("pct2.5"),
                    pl.col(val_col_name)
                    .quantile(0.975)
                    .cast(pl.Float64)
                    .alias("pct97.5"),
                    pl.col(val_col_name).std().cast(pl.Float64).alias("sd"),
                ]
            )
            .with_columns(
                pl.lit("all").alias("platform_code"),
                pl.lit(0).alias("profile_no"),
                pl.lit(val_col_name).alias("variable"),
            )
            .select(
                [
                    pl.col("platform_code"),
                    pl.col("profile_no"),
                    pl.col("variable"),
                    pl.col("min"),
                    pl.col("pct2.5"),
                    pl.col("pct25"),
                    pl.col("mean"),
                    pl.col("median"),
                    pl.col("pct75"),
                    pl.col("pct97.5"),
                    pl.col("max"),
                    pl.col("sd"),
                ]
            )
        )

    def calculate_profile_stats(
        self, grouped_df: pl.DataFrame, val_col_name: str
    ) -> pl.DataFrame:
        """
        Calculate per-profile summary statistics for a specific numeric column.

        :param grouped_df: A Polars DataFrame grouped by profile-identifying columns.
        :type grouped_df: pl.DataFrame
        :param val_col_name: The name of the column for which to calculate per-profile stats.
        :type val_col_name: str
        :return: A Polars DataFrame containing per-profile metrics (e.g., min, max, mean)
                 for the specified column.
        :rtype: pl.DataFrame
        """
        return (
            grouped_df.agg(
                [
                    pl.col(val_col_name).min().cast(pl.Float64).alias("min"),
                    pl.col(val_col_name).max().cast(pl.Float64).alias("max"),
                    pl.col(val_col_name).mean().cast(pl.Float64).alias("mean"),
                    pl.col(val_col_name).median().cast(pl.Float64).alias("median"),
                    pl.col(val_col_name).quantile(0.25).cast(pl.Float64).alias("pct25"),
                    pl.col(val_col_name).quantile(0.75).cast(pl.Float64).alias("pct75"),
                    pl.col(val_col_name)
                    .quantile(0.025)
                    .cast(pl.Float64)
                    .alias("pct2.5"),
                    pl.col(val_col_name)
                    .quantile(0.975)
                    .cast(pl.Float64)
                    .alias("pct97.5"),
                    pl.col(val_col_name).std().cast(pl.Float64).alias("sd"),
                ]
            )
            .with_columns(pl.lit(val_col_name).alias("variable"))
            .select(
                [
                    pl.col("platform_code"),
                    pl.col("profile_no"),
                    pl.col("variable"),
                    pl.col("min"),
                    pl.col("pct2.5"),
                    pl.col("pct25"),
                    pl.col("mean"),
                    pl.col("median"),
                    pl.col("pct75"),
                    pl.col("pct97.5"),
                    pl.col("max"),
                    pl.col("sd"),
                ]
            )
        )

    def calculate_stats(self) -> None:
        """
        Calculate summary statistics and store them in :attr:`summary_stats`.

        This method concatenates global statistics across all data rows and
        per-profile statistics grouped by :attr:`profile_col_names` for each
        column specified in :attr:`val_col_names`.
        """
        global_stats = pl.concat(
            [self.calculate_global_stats(x) for x in self.val_col_names]
        )
        grouped_df = self.input_data.group_by(self.profile_col_names)
        profile_stats = pl.concat(
            [self.calculate_profile_stats(grouped_df, x) for x in self.val_col_names]
        )

        self.summary_stats = global_stats.vstack(profile_stats)
