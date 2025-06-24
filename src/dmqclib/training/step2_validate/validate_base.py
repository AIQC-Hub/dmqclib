from abc import abstractmethod

import polars as pl

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.common.loader.model_loader import load_model_class
from dmqclib.utils.config import get_targets


class ValidationBase(DataSetBase):
    """
    Base class for validation classes.
    """

    def __init__(
        self,
        dataset_name: str,
        config_file: str = None,
        training_sets: pl.DataFrame = None,
    ):
        super().__init__(
            "validate",
            dataset_name,
            config_file=config_file,
            config_file_name="training.yaml",
        )

        base_model = load_model_class(dataset_name, config_file)

        # Set member variables
        self.training_sets = training_sets
        self.base_model = base_model
        self.built_models = {}
        self.results = {}
        self.summary = {}

    def process_targets(self):
        """
        Iterate all targets to locate training data rows.
        """
        targets = get_targets(self.dataset_info, "validate", self.targets)
        for k in targets.keys():
            self.validate(k)
            self.summarise(k)
            self.base_model.clear()

    @abstractmethod
    def validate(self, target_name: str):
        """
        Validate models
        """
        pass

    @abstractmethod
    def summarise(self, target_name: str):
        """
        Summarise results
        """
        pass
