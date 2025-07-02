import os
from typing import Dict

import polars as pl

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.common.loader.feature_loader import load_feature_class
from dmqclib.config.dataset_config import DataSetConfig


class ExtractFeatureBase(DataSetBase):
    """
    Base class to extract features
    """

    def __init__(
        self,
        config: DataSetConfig,
        input_data: pl.DataFrame = None,
        selected_profiles: pl.DataFrame = None,
        target_rows: pl.DataFrame = None,
        summary_stats: pl.DataFrame = None,
    ):
        super().__init__("extract", config)

        # Set member variables
        self.default_file_name = "{target_name}_features.parquet"
        self.output_file_names = self.config.get_target_file_names(
            "extract", self.default_file_name
        )
        self.input_data = input_data
        self.selected_profiles = selected_profiles
        if input_data is not None and selected_profiles is not None:
            self._filter_input()
        else:
            self.filtered_input = None
        self.target_rows = target_rows
        self.summary_stats = summary_stats
        self.feature_info = self.config.data["feature_param_set"]["params"]
        self.target_features = {}

    def _filter_input(self):
        self.filtered_input = self.input_data.join(
            (
                self.selected_profiles.select(
                    pl.col("platform_code"),
                    pl.col("profile_no"),
                )
            ),
            on=["platform_code", "profile_no"],
        )

    def process_targets(self):
        """
        Iterate all targets to generate features.
        """
        for k in self.config.get_target_names():
            self.extract_target_features(k)

    def extract_target_features(self, k):
        """
        Iterate all feature entries to generate features.
        """
        self.target_features[k] = (
            self.target_rows[k]
            .select(
                [
                    pl.col("label"),
                    pl.col("row_id"),
                    pl.col("profile_id"),
                    pl.col("pair_id"),
                    pl.col("platform_code"),
                    pl.col("profile_no"),
                    pl.col("observation_no"),
                ]
            )
            .join(
                pl.concat(
                    [self.extract_features(k, x) for x in self.feature_info],
                    how="align_left",
                ),
                on=["row_id"],
                maintain_order="left",
            )
        )

    def extract_features(self, target_name: str, feature_info: Dict) -> pl.DataFrame:
        """
        Extract target features.
        """
        ds = load_feature_class(
            target_name,
            feature_info,
            self.selected_profiles,
            self.filtered_input,
            self.target_rows,
            self.summary_stats,
        )

        ds.scale_first()
        ds.extract_features()
        ds.scale_second()

        return ds.features

    def write_target_features(self):
        """
        Write target_rows to parquet files
        """
        if len(self.target_features) == 0:
            raise ValueError("Member variable 'target_features' must not be empty.")

        for k, v in self.target_features.items():
            os.makedirs(os.path.dirname(self.output_file_names[k]), exist_ok=True)
            v.write_parquet(self.output_file_names[k])
