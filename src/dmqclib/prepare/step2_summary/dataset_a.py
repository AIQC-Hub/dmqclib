import polars as pl
from typing import List

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.prepare.step2_summary.summary_base import SummaryStatsBase


class SummaryDataSetA(SummaryStatsBase):
    """
    Subclass of SummaryStatsBase for calculating summary statistics
    for BO NRT + Cora test data using Polars.

    This class uses Polars to generate both global and per-profile statistics
    for a set of specified columns (:attr:`val_col_names`). It expects column names
    related to measurement attributes such as temperature, salinity, etc.,
    and also relies on certain profile identifiers (:attr:`profile_col_names`)
    to group data by platform and profile number.
    """

    expected_class_name: str = "SummaryDataSetA"

    def __init__(self, config: ConfigBase, input_data: pl.DataFrame = None) -> None:
        """
        Initialize SummaryDataSetA with configuration and optional input data.

        :param config: The dataset configuration object containing paths
                       and parameters for summary stats.
        :type config: ConfigBase
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
        #: List of columns defining the schema of output statistics.
        self.stats_col_names = [
            "platform_code",
            "profile_no",
            "variable",
            "min",
            "pct2.5",
            "pct25",
            "mean",
            "median",
            "pct75",
            "pct97.5",
            "max",
            "sd",
        ]
        #: List of columns used as identifiers for grouping data by profile.
        self.profile_col_names = ["platform_code", "profile_no"]

    def get_stats_expression(self, val_col_name: str) -> List:
        """
        Build a list of Polars expression objects used to compute summary
        statistics for a given numeric column.

        :param val_col_name: The name of the column for which to calculate global stats.
        :type val_col_name: str
        :return: A Polars DataFrame containing global metrics (e.g., min, max, mean)
                 for the specified column, with placeholders for platform_code and
                 profile_no to align with the structure of per-profile stats.
        :rtype: pl.DataFrame
        """
        return [
            pl.col(val_col_name).min().cast(pl.Float64).alias("min"),
            pl.col(val_col_name).max().cast(pl.Float64).alias("max"),
            pl.col(val_col_name).mean().cast(pl.Float64).alias("mean"),
            pl.col(val_col_name).median().cast(pl.Float64).alias("median"),
            pl.col(val_col_name).quantile(0.25).cast(pl.Float64).alias("pct25"),
            pl.col(val_col_name).quantile(0.75).cast(pl.Float64).alias("pct75"),
            pl.col(val_col_name).quantile(0.025).cast(pl.Float64).alias("pct2.5"),
            pl.col(val_col_name).quantile(0.975).cast(pl.Float64).alias("pct97.5"),
            pl.col(val_col_name).std().cast(pl.Float64).alias("sd"),
        ]

    def calculate_global_stats(self, val_col_name: str) -> pl.DataFrame:
        """
        Compute global summary statistics for a specified numeric column
        across all data rows.

        :param val_col_name: Name of the column for which to calculate global stats.
        :return: Polars DataFrame containing min, max, mean, etc., for the entire dataset,
                 along with placeholders for platform_code and profile_no to match
                 the structure of the per-profile statistics.
        """
        return (
            self.input_data.select(self.get_stats_expression(val_col_name))
            .with_columns(
                pl.lit("all").alias("platform_code"),
                pl.lit(0).alias("profile_no"),
                pl.lit(val_col_name).alias("variable"),
            )
            .select(self.stats_col_names)
        )

    def calculate_profile_stats(
        self, grouped_df: pl.DataFrame, val_col_name: str
    ) -> pl.DataFrame:
        """
        Compute per-profile summary statistics for a numeric column, given
        an already grouped DataFrame.

        :param grouped_df: A Polars DataFrame grouped by profile-identifying columns.
        :type grouped_df: pl.DataFrame
        :param val_col_name: The name of the column for which to calculate per-profile stats.
        :type val_col_name: str
        :return: A Polars DataFrame containing per-profile metrics (e.g., min, max, mean)
                 for the specified column.
        :rtype: pl.DataFrame
        """
        return (
            grouped_df.agg(self.get_stats_expression(val_col_name))
            .with_columns(pl.lit(val_col_name).alias("variable"))
            .select(self.stats_col_names)
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
