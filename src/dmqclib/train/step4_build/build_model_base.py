import os
from abc import abstractmethod

import polars as pl

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.common.loader.model_loader import load_model_class
from dmqclib.config.training_config import TrainingConfig


class BuildModelBase(DataSetBase):
    """
    Base class for building models.
    """

    def __init__(
        self,
        config: TrainingConfig,
        training_sets: pl.DataFrame = None,
        test_sets: pl.DataFrame = None,
    ):
        super().__init__("build", config)

        # Set member variables
        self.default_file_name = "{target_name}_model.json"
        self.default_file_names = {
            "model": "{target_name}_model.joblib",
            "result": "{target_name}_test_result.tsv",
        }
        self.output_file_names = {
            k: self.config.get_target_file_names("build", v)
            for k, v in self.default_file_names.items()
        }
        self.training_sets = training_sets
        self.test_sets = test_sets

        self.base_model = None
        self.load_base_model()
        self.models = {}
        self.results = {}

    def load_base_model(self):
        self.base_model = load_model_class(self.config)

    def build_targets(self):
        """
        Iterate all targets to locate training data rows.
        """
        for k in self.config.get_target_names():
            self.build(k)
            if self.test_sets is not None and k in self.test_sets:
                self.test(k)

    def test_targets(self):
        """
        Iterate all targets to locate training data rows.
        """
        for k in self.config.get_target_names():
            if k not in self.models:
                raise ValueError(f"No valid model found for the variable '{k}'.")
            self.test(k)

    @abstractmethod
    def build(self, target_name: str):
        """
        Build models
        """
        pass  # pragma: no cover

    @abstractmethod
    def test(self, target_name: str):
        """
        Build models
        """
        pass  # pragma: no cover

    def write_results(self):
        """
        Write results
        """
        if len(self.results) == 0:
            raise ValueError("Member variable 'results' must not be empty.")

        for k, v in self.results.items():
            os.makedirs(
                os.path.dirname(self.output_file_names["result"][k]), exist_ok=True
            )
            v.write_csv(self.output_file_names["result"][k], separator="\t")

    def write_models(self):
        """
        Write models
        """
        if len(self.models) == 0:
            raise ValueError("Member variable 'built_models' must not be empty.")

        for k, v in self.models.items():
            os.makedirs(
                os.path.dirname(self.output_file_names["model"][k]), exist_ok=True
            )
            self.base_model.save_model(self.output_file_names["model"][k])

    def read_models(self):
        """
        Read models
        """

        for k, v in self.output_file_names["model"].items():
            if not os.path.exists(v):
                raise FileNotFoundError(f"File '{v}' does not exist.")

            self.load_base_model()
            self.base_model.load_model(v)
            self.models[k] = self.base_model
