import numpy as np
import polars as pl
from typing import Optional, Dict

from dmqclib.common.base.feature_base import FeatureBase


class DayOfYearFeat(FeatureBase):
    """
    A feature-extraction class that derives day-of-year features
    from BO NRT + Cora test data.

    This class specifically leverages the ``profile_timestamp``
    column to generate a day-of-year value, optionally applying
    a sinusoidal transformation for cyclical encoding.
    """

    def __init__(
        self,
        target_name: Optional[str] = None,
        feature_info: Optional[Dict] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
        filtered_input: Optional[pl.DataFrame] = None,
        selected_rows: Optional[Dict[str, pl.DataFrame]] = None,
        summary_stats: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize the day-of-year feature extraction process.

        :param target_name: The name of the target variable used to index
                            :attr:`selected_rows`. Defaults to None.
        :type target_name: str, optional
        :param feature_info: A Polars DataFrame or dictionary describing feature
                             parameters, which may include a "convert" key
                             (e.g., "sine") for sinusoidal transformations,
                             defaults to None.
        :type feature_info: Dict, optional
        :param selected_profiles: A Polars DataFrame containing a subset
                                  of profiles relevant to feature extraction,
                                  defaults to None.
        :type selected_profiles: pl.DataFrame, optional
        :param filtered_input: (Unused in this feature class) A filtered Polars
                               DataFrame of input data for advanced merging,
                               defaults to None.
        :type filtered_input: pl.DataFrame, optional
        :param selected_rows: A dictionary of target-specific DataFrames, each
                            containing rows relevant to that target. Defaults to None.
        :type selected_rows: dict of (str to pl.DataFrame), optional
        :param summary_stats: (Unused in this feature class) A Polars DataFrame
                              containing statistical information for potential scaling,
                              defaults to None.
        :type summary_stats: pl.DataFrame, optional
        """
        super().__init__(
            target_name=target_name,
            feature_info=feature_info,
            selected_profiles=selected_profiles,
            filtered_input=filtered_input,
            selected_rows=selected_rows,
            summary_stats=summary_stats,
        )

    def extract_features(self) -> None:
        """
        Derive the day-of-year feature from the ``profile_timestamp`` column
        in :attr:`selected_profiles` and merge it with the target rows.

        Steps:
          1. Select columns ``row_id``, ``platform_code``, and ``profile_no``
             from :attr:`selected_rows[target_name]`.
          2. Join the subset with ``profile_timestamp`` from :attr:`selected_profiles`
             based on ``platform_code`` and ``profile_no``.
          3. Compute the day of year from ``profile_timestamp`` via
             Polars' .dt.ordinal_day().
          4. Remove columns no longer needed (i.e., the join keys and timestamp).
        """
        self.features = (
            self.selected_rows[self.target_name]
            .select(["row_id", "platform_code", "profile_no"])
            .join(
                self.selected_profiles.select(
                    ["platform_code", "profile_no", "profile_timestamp"]
                ),
                on=["platform_code", "profile_no"],
                maintain_order="left",
            )
            .with_columns(
                pl.col("profile_timestamp").dt.ordinal_day().alias("day_of_year"),
            )
            .drop(["platform_code", "profile_no", "profile_timestamp"])
        )

    def scale_first(self) -> None:
        """
        (Optional) Perform the initial scaling step.

        Currently, no transformations are applied to day-of-year values
        in this step, but it can be extended for outlier removal or
        other domain-specific logic.
        """
        pass  # pragma: no cover

    def scale_second(self) -> None:
        """
        Optionally apply a sinusoidal transformation to the day-of-year values.

        If ``"convert"`` is specified as ``"sine"`` in :attr:`feature_info`,
        transforms each day-of-year value into a sine-based cyclical feature
        in the range [0, 1]:

        day_of_year = ((sin(day_of_year * 2π / 365) + 1) / 2).
        """
        if "convert" in self.feature_info and self.feature_info["convert"] == "sine":
            self.features = self.features.with_columns(
                ((np.sin(pl.col("day_of_year") * 2 * np.pi / 365) + 1) / 2).alias(
                    "day_of_year"
                )
            )
