import os
from abc import abstractmethod

import polars as pl

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.common.loader.model_loader import load_model_class
from dmqclib.config.training_config import TrainingConfig


class ValidationBase(DataSetBase):
    """
    Base class for validation classes.
    """

    def __init__(self, config: TrainingConfig, training_sets: pl.DataFrame = None):
        super().__init__("validate", config)

        # Set member variables
        self.default_file_names = {
            "result": "{target_name}_validation_result.tsv",
        }
        self.output_file_names = {
            k: self.config.get_target_file_names("validate", v)
            for k, v in self.default_file_names.items()
        }
        self.training_sets = training_sets

        self.base_model = None
        self.load_base_model()
        self.models = {}
        self.results = {}
        self.summarised_results = {}

    def load_base_model(self):
        self.base_model = load_model_class(self.config)

    def process_targets(self):
        """
        Iterate all targets to locate training data rows.
        """
        for k in self.config.get_target_names():
            self.validate(k)

    @abstractmethod
    def validate(self, target_name: str):
        """
        Validate models
        """
        pass  # pragma: no cover

    def write_results(self):
        """
        Write results
        """
        if self.results is None:
            raise ValueError("Member variable 'results' must not be empty.")

        for k, v in self.results.items():
            os.makedirs(
                os.path.dirname(self.output_file_names["result"][k]), exist_ok=True
            )
            v.write_csv(self.output_file_names["result"][k], separator="\t")
