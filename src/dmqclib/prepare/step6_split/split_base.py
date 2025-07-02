import os
from abc import abstractmethod

import polars as pl

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.config.dataset_config import DataSetConfig


class SplitDataSetBase(DataSetBase):
    """
    Base class to identify training data rows
    """

    def __init__(self, config: DataSetConfig, target_features: pl.DataFrame = None):
        super().__init__("split", config)

        # Set member variables
        self.default_file_names = {
            "train": "{target_name}_train.parquet",
            "test": "{target_name}_test.parquet",
        }
        self.output_file_names = {
            k: self.config.get_target_file_names("split", v)
            for k, v in self.default_file_names.items()
        }
        self.target_features = target_features
        self.training_sets = {}
        self.test_sets = {}

        self.default_test_set_fraction = 0.1
        self.default_k_fold = 10

    def get_test_set_fraction(self) -> str:
        return (
            self.config.get_step_params("split").get(
                "test_set_fraction", self.default_test_set_fraction
            )
            or self.default_test_set_fraction
        )

    def get_k_fold(self) -> str:
        return (
            self.config.get_step_params("split").get("k_fold", self.default_k_fold)
            or self.default_k_fold
        )

    def process_targets(self):
        """
        Iterate all targets to locate training data rows.
        """
        for k in self.config.get_target_names():
            self.split_test_set(k)
            self.add_k_fold(k)
            self.drop_columns(k)

    @abstractmethod
    def split_test_set(self, target_name: str):
        pass  # pragma: no cover

    @abstractmethod
    def add_k_fold(self, target_name: str):
        pass  # pragma: no cover

    @abstractmethod
    def drop_columns(self, target_name: str):
        pass  # pragma: no cover

    def write_training_sets(self):
        """
        Write training sets to parquet files
        """
        if len(self.training_sets) == 0:
            raise ValueError("Member variable 'training_sets' must not be empty.")

        for k, v in self.training_sets.items():
            os.makedirs(
                os.path.dirname(self.output_file_names["train"][k]), exist_ok=True
            )
            v.write_parquet(self.output_file_names["train"][k])

    def write_test_sets(self):
        """
        Write test sets to parquet files
        """
        if len(self.test_sets) == 0:
            raise ValueError("Member variable 'test_sets' must not be empty.")

        for k, v in self.test_sets.items():
            os.makedirs(
                os.path.dirname(self.output_file_names["test"][k]), exist_ok=True
            )
            v.write_parquet(self.output_file_names["test"][k])

    def write_data_sets(self):
        """
        Write both training and test sets
        """

        self.write_test_sets()
        self.write_training_sets()
