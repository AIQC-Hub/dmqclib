import polars as pl
from typing import Optional, Dict

from dmqclib.common.base.feature_base import FeatureBase


class BasicValues3PlusFlanks(FeatureBase):
    """
    A feature-extraction class for retrieving target values and their "flanking" values
    from BO NRT + Cora test data, extending :class:`FeatureBase`.

    The term "flanking values" refers to the concept of capturing neighboring observations
    around a specified index (e.g., observation_no) by shifting backward a specified amount.
    """

    def __init__(
        self,
        target_name: Optional[str] = None,
        feature_info: Optional[Dict] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
        filtered_input: Optional[pl.DataFrame] = None,
        target_rows: Optional[Dict[str, pl.DataFrame]] = None,
        summary_stats: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize an instance of BasicValues3PlusFlanks.

        :param target_name: The key identifying which target's rows to extract
                            features for from :attr:`target_rows`.
        :type target_name: str, optional
        :param feature_info: A dictionary containing feature-related parameters,
                             including a "stats" sub-dict with min/max info
                             and a "flank_up" integer specifying how many
                             upstream observations to retrieve, defaults to None.
        :type feature_info: dict, optional
        :param selected_profiles: A Polars DataFrame with selected profiles, typically
                                  used for further merges or lookups, defaults to None.
        :type selected_profiles: pl.DataFrame, optional
        :param filtered_input: A potentially filtered Polars DataFrame containing
                               full observed variables, defaults to None.
        :type filtered_input: pl.DataFrame, optional
        :param target_rows: A dictionary mapping target names to their respective
                            DataFrames of relevant rows, defaults to None.
        :type target_rows: dict of (str to pl.DataFrame), optional
        :param summary_stats: A Polars DataFrame of summary statistics
                              (unused in this subclass), defaults to None.
        :type summary_stats: pl.DataFrame, optional
        """
        super().__init__(
            target_name=target_name,
            feature_info=feature_info,
            selected_profiles=selected_profiles,
            filtered_input=filtered_input,
            target_rows=target_rows,
            summary_stats=summary_stats,
        )
        self._expanded_observations: Optional[pl.DataFrame] = None
        self._feature_wide: Optional[pl.DataFrame] = None

    def extract_features(self) -> None:
        """
        Initiate the multi-step process of creating the feature set in :attr:`features`.

        Steps:
          1. :meth:`_init_features` - Prepare a base DataFrame with essential columns
             (row_id, platform_code, profile_no).
          2. :meth:`_expand_observations` - Expand observations by adding rows for
             the specified number of "flank" steps (based on ``feature_info["flank_up"]``).
          3. For each column in ``feature_info["stats"]``, call:
             - :meth:`_pivot_features` to pivot the data for that column,
             - :meth:`_add_features` to join the pivoted data onto our feature table.
          4. :meth:`_clean_features` - Drop columns no longer needed.
        """
        self._init_features()
        self._expand_observations()
        for col_name in self.feature_info["stats"].keys():
            self._pivot_features(col_name)
            self._add_features()
        self._clean_features()

    def _init_features(self) -> None:
        """
        Initialize :attr:`features` by selecting core columns
        from :attr:`target_rows[target_name]`.
        """
        self.features = self.target_rows[self.target_name].select(
            ["row_id", "platform_code", "profile_no"]
        )

    def _expand_observations(self) -> None:
        """
        Generate a DataFrame with additional rows for each "flank" step.

        This expands each row in :attr:`target_rows[target_name]` by
        cross joining with a sequence from 0 to ``feature_info["flank_up"]``,
        then adjusts ``observation_no`` to shift backwards for each flank step.
        """
        self._expanded_observations = (
            self.target_rows[self.target_name]
            .select(["row_id", "platform_code", "profile_no", "observation_no"])
            .join(
                pl.DataFrame(
                    {"flank_seq": list(range(0, self.feature_info.get("flank_up") + 1))}
                ),
                how="cross",
            )
            .with_columns(
                (pl.col("observation_no") - pl.col("flank_seq")).alias("observation_no")
            )
            .with_columns(
                pl.when(pl.col("observation_no") < 1)
                .then(1)
                .otherwise(pl.col("observation_no"))
                .alias("observation_no")
            )
        )

    def _pivot_features(self, col_name: str) -> None:
        """
        Pivot the expanded observations to create columns for each flank step
        of the specified data column.

        :param col_name: The original data column to be pivoted (e.g., "temp").
        :type col_name: str
        """
        self._feature_wide = (
            self._expanded_observations.join(
                self.filtered_input.select(
                    pl.col("platform_code"),
                    pl.col("profile_no"),
                    pl.col("observation_no"),
                    pl.col(col_name).alias("value"),
                ),
                on=["platform_code", "profile_no", "observation_no"],
                maintain_order="left",
            )
            .with_columns(
                pl.concat_str(
                    [
                        pl.lit(f"{col_name}_up"),
                        pl.col("flank_seq").cast(pl.Utf8),
                    ],
                    separator="_",
                ).alias("col_name")
            )
            .drop(["observation_no", "flank_seq"])
            .pivot(
                "col_name",
                index=["row_id", "platform_code", "profile_no"],
                values="value",
            )
        )

    def _add_features(self) -> None:
        """
        Join the pivoted columns from :attr:`_feature_wide` onto :attr:`features`.
        """
        self.features = self.features.join(
            self._feature_wide,
            on=["row_id", "platform_code", "profile_no"],
            maintain_order="left",
        )

    def _clean_features(self) -> None:
        """
        Drop columns that are no longer needed in the final feature set.
        """
        self.features = self.features.drop(["platform_code", "profile_no"])

    def scale_first(self) -> None:
        """
        Apply a pre-feature-extraction scaling step on :attr:`filtered_input`
        using min-max scaling derived from :attr:`feature_info["stats"]`.

        This modifies :attr:`filtered_input` in place for each relevant column.
        """
        for col_name, v in self.feature_info["stats"].items():
            self.filtered_input = self.filtered_input.with_columns(
                ((pl.col(col_name) - v["min"]) / (v["max"] - v["min"])).alias(col_name)
            )

    def scale_second(self) -> None:
        """
        Apply a post-feature-extraction scaling step if needed.

        Currently, unimplemented; retains placeholders for additional
        scaling/normalization after feature pivoting and expansion.
        """
        pass  # pragma: no cover
