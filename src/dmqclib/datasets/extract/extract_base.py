from abc import abstractmethod
from typing import Dict
import polars as pl
from dmqclib.datasets.base.dataset_base import DataSetBase
from dmqclib.utils.config import get_file_name_from_config
from dmqclib.utils.dataset_path import build_full_extract_path


class ExtractFeatureBase(DataSetBase):
    """
    Base class to extract features
    """

    def __init__(
        self,
        dataset_name: str,
        config_file: str = None,
        input_data: pl.DataFrame = None,
        target_rows: pl.DataFrame = None,
    ):
        super().__init__("extract", dataset_name, config_file=config_file)

        # Set member variables
        self._build_output_file_names()
        self.input_data = input_data
        self.target_rows = target_rows
        self.target_features = {}

    def _build_output_file_names(self):
        """
        Set the output files based on configuration entries.
        """
        folder_name = self.dataset_info["extract"].get("folder_name", "")

        self.output_file_names = {
            k: build_full_extract_path(
                self.path_info,
                folder_name,
                get_file_name_from_config(v, self.config_file_name),
            )
            for k, v in self.dataset_info["extract"]["targets"].items()
        }

    def process_targets(self):
        """
        Iterate all targets to locate training data rows.
        """
        for k, v in self.dataset_info["extract"]["targets"].items():
            self.extract_target_features(k, v)

    @abstractmethod
    def extract_target_features(self, target_name: str, target_value: Dict):
        """
        Extract target features.
        """
        pass

    def write_target_features(self):
        """
        Write target_rows to parquet files
        """
        if len(self.target_features) == 0:
            raise ValueError("Member variable 'target_features' must not be empty.")

        for k, v in self.target_features.items():
            v.write_parquet(self.output_file_names[k])
