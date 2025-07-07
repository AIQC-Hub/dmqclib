import polars as pl
from typing import Optional, Dict

from dmqclib.common.base.feature_base import FeatureBase


class LocationFeat(FeatureBase):
    """
    A feature extraction class designed specifically for location-based fields
    (e.g., longitude, latitude) within the BO NRT + Cora dataset.

    This class uses the provided data frames to gather location-related fields
    and optionally apply scaling methods. It inherits from :class:`FeatureBase`
    which defines a generic feature extraction workflow.
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
        Initialize the location feature extractor with relevant data frames.

        :param target_name: The key for the target variable in :attr:`selected_rows`,
                            defaults to None.
        :type target_name: str, optional
        :param feature_info: A Polars DataFrame (or dictionary-like) describing
                             feature parameters, defaults to None.
        :type feature_info: Dict, optional
        :param selected_profiles: A Polars DataFrame containing a subset of profiles
                                  relevant to feature extraction, defaults to None.
        :type selected_profiles: pl.DataFrame, optional
        :param filtered_input: A filtered Polars DataFrame of input data,
                               potentially used for advanced merging or lookups,
                               defaults to None.
        :type filtered_input: pl.DataFrame, optional
        :param selected_rows: A dictionary keyed by target names, each mapping to
                            a Polars DataFrame of rows relevant for that target,
                            defaults to None.
        :type selected_rows: dict of (str to pl.DataFrame), optional
        :param summary_stats: A Polars DataFrame containing statistical
                              information that may aid in feature scaling,
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
        Gather and merge location columns (e.g., longitude and latitude) from
        :attr:`selected_profiles` into :attr:`selected_rows` to form the final
        feature set in :attr:`features`.

        Specifically:

          1. Selects columns like ``row_id``, ``platform_code``, and ``profile_no``
             from the DataFrame in :attr:`selected_rows[target_name]``.
          2. Joins this subset with corresponding columns from :attr:`selected_profiles`
             (including ``longitude`` and ``latitude``) on ``platform_code``
             and ``profile_no``.
          3. Drops those join columns from the final feature set, leaving
             ``row_id``, ``longitude``, and ``latitude`` among others.
        """
        self.features = (
            self.selected_rows[self.target_name]
            .select(["row_id", "platform_code", "profile_no"])
            .join(
                self.selected_profiles.select(
                    ["platform_code", "profile_no", "longitude", "latitude"]
                ),
                on=["platform_code", "profile_no"],
                maintain_order="left",
            )
            .drop(["platform_code", "profile_no"])
        )

    def scale_first(self) -> None:
        """
        (Optional) Initial scaling or normalization procedure.

        Currently, unimplemented for location features; can be extended
        if, for instance, location data requires some preprocessing
        or transformation steps.
        """
        pass  # pragma: no cover

    def scale_second(self) -> None:
        """
        Apply a second min-max scaling pass to each feature defined in :attr:`feature_info["stats"]`.

        This method expects :attr:`feature_info["stats"]` to be a dictionary of the form::

            {
                "longitude": {"min": ..., "max": ...},
                "latitude": {"min": ..., "max": ...},
                ...
            }
        """
        for k, v in self.feature_info["stats"].items():
            self.features = self.features.with_columns(
                ((pl.col(k) - v["min"]) / (v["max"] - v["min"])).alias(k)
            )
