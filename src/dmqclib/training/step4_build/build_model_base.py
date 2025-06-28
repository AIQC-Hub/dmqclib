import os
from abc import abstractmethod

import polars as pl

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.common.loader.model_loader import load_model_class
from dmqclib.utils.config import get_target_file_name
from dmqclib.utils.config import get_targets
from dmqclib.utils.path import build_full_data_path


class BuildModelBase(DataSetBase):
    """
    Base class for building models.
    """

    def __init__(
        self,
        dataset_name: str,
        config_file: str = None,
        training_sets: pl.DataFrame = None,
        test_sets: pl.DataFrame = None,
    ):
        super().__init__(
            "build",
            dataset_name,
            config_file=config_file,
            config_file_name="training.yaml",
        )

        # Set member variables
        self.config_file = config_file
        self.default_file_name = "{target_name}_model.json"
        self.default_file_names = {
            "model": "{target_name}_model.joblib",
            "result": "{target_name}_test_result.tsv",
        }
        self._build_output_file_names()
        self.training_sets = training_sets
        self.test_sets = test_sets

        self.base_model = None
        self.load_base_model()
        self.models = {}
        self.results = {}

    def _build_output_file_names(self):
        """
        Set the output files based on configuration entries.
        """
        targets = get_targets(self.dataset_info, "build", self.targets)
        self.output_file_names = {
            k1: {
                k2: build_full_data_path(
                    self.path_info,
                    self.dataset_info,
                    "build",
                    get_target_file_name(v1, k1, v2),
                )
                for k2, v2 in self.default_file_names.items()
            }
            for k1, v1 in targets.items()
        }

    def load_base_model(self):
        self.base_model = load_model_class(self.dataset_name, self.config_file)

    def build_targets(self):
        """
        Iterate all targets to locate training data rows.
        """
        targets = get_targets(self.dataset_info, "build", self.targets)
        for k in targets.keys():
            self.build(k)
            if self.test_sets is not None and k in self.test_sets:
                self.test(k)

    def test_targets(self):
        """
        Iterate all targets to locate training data rows.
        """
        targets = get_targets(self.dataset_info, "build", self.targets)
        for k in targets.keys():
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
                os.path.dirname(self.output_file_names[k]["result"]), exist_ok=True
            )
            v.write_csv(self.output_file_names[k]["result"], separator="\t")

    def write_models(self):
        """
        Write models
        """
        if len(self.models) == 0:
            raise ValueError("Member variable 'built_models' must not be empty.")

        for k, v in self.models.items():
            os.makedirs(
                os.path.dirname(self.output_file_names[k]["model"]), exist_ok=True
            )
            self.base_model.save_model(self.output_file_names[k]["model"])

    def read_models(self):
        """
        Read models
        """

        for k, v in self.output_file_names.items():
            if not os.path.exists(v["model"]):
                raise FileNotFoundError(f"The file '{v['model']}' does not exist.")

            self.load_base_model()
            self.base_model.load_model(v["model"])
            self.models[k] = self.base_model
