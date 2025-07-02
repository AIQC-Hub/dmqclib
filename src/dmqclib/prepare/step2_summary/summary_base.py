import os
from abc import abstractmethod

import polars as pl

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.config.dataset_config import DataSetConfig


class SummaryStatsBase(DataSetBase):
    """
    Base class to calculate summary stats
    """

    def __init__(self, config: DataSetConfig, input_data: pl.DataFrame = None):
        super().__init__("summary", config)

        # Set member variables
        self.default_file_name = "summary_stats.tsv"
        self.output_file_name = self.config.get_full_file_name(
            "summary", self.default_file_name
        )
        self.input_data = input_data
        self.summary_stats = None

    @abstractmethod
    def calculate_stats(self):
        """
        Calculate summary stats.
        """
        pass  # pragma: no cover

    def write_summary_stats(self):
        """
        Write selected profiles to tsv file.
        """
        if self.summary_stats is None:
            raise ValueError("Member variable 'summary_stats' must not be empty.")

        os.makedirs(os.path.dirname(self.output_file_name), exist_ok=True)
        self.summary_stats.write_csv(self.output_file_name, separator="\t")
