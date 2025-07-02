import os

import polars as pl

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.config.training_config import TrainingConfig


class InputTrainingSetBase(DataSetBase):
    """
    Base class to import training data sets
    """

    def __init__(self, config: TrainingConfig):
        super().__init__("input", config)

        # Set member variables
        self.default_file_names = {
            "train": "{target_name}_train.parquet",
            "test": "{target_name}_test.parquet",
        }
        self.input_file_names = {
            k: self.config.get_target_file_names("input", v)
            for k, v in self.default_file_names.items()
        }
        self.training_sets = {}
        self.test_sets = {}

    def process_targets(self):
        """
        Iterate all targets to locate training data rows.
        """
        for k in self.config.get_target_names():
            self.read_training_set(k)
            self.read_test_sets(k)

    def read_training_set(self, target_name: str):
        """
        Read training set from parquet file
        """
        file_name = self.input_file_names["train"][target_name]
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File '{file_name}' does not exist.")
        self.training_sets[target_name] = pl.read_parquet(file_name)

    def read_test_sets(self, target_name: str):
        """
        Read test set from parquet files
        """
        file_name = self.input_file_names["test"][target_name]
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File '{file_name}' does not exist.")
        self.test_sets[target_name] = pl.read_parquet(file_name)
