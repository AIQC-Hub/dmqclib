import os
from abc import abstractmethod

import polars as pl

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.config.training_config import TrainingConfig
from dmqclib.common.loader.model_loader import load_model_class
from dmqclib.utils.config import get_target_file_name
from dmqclib.utils.config import get_targets
from dmqclib.utils.path import build_full_data_path


class ValidationBase(DataSetBase):
    """
    Base class for validation classes.
    """

    def __init__(
        self,
        dataset_name: str,
        config: TrainingConfig = None,
        config_file: str = None,
        training_sets: pl.DataFrame = None,
    ):
        super().__init__(
            "validate",
            dataset_name,
            config=config,
            config_file=config_file,
            config_file_name="training.yaml",
        )

        # Set member variables
        self.config_file = config_file
        self.default_file_names = {
            "result": "{target_name}_validation_result.tsv",
        }
        self._build_output_file_names()
        self.training_sets = training_sets

        self.base_model = None
        self.load_base_model()
        self.models = {}
        self.results = {}
        self.summarised_results = {}

    def _build_output_file_names(self):
        """
        Set the output files based on configuration entries.
        """
        targets = get_targets(self.dataset_info, "validate", self.targets)
        self.output_file_names = {
            k1: {
                k2: build_full_data_path(
                    self.path_info,
                    self.dataset_info,
                    "validate",
                    get_target_file_name(v1, k1, v2),
                )
                for k2, v2 in self.default_file_names.items()
            }
            for k1, v1 in targets.items()
        }

    def load_base_model(self):
        self.base_model = load_model_class(self.dataset_name, self.config_file)

    def process_targets(self):
        """
        Iterate all targets to locate training data rows.
        """
        targets = get_targets(self.dataset_info, "validate", self.targets)
        for k in targets.keys():
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
                os.path.dirname(self.output_file_names[k]["result"]), exist_ok=True
            )
            v.write_csv(self.output_file_names[k]["result"], separator="\t")
