import os
from abc import abstractmethod
from typing import Optional

import polars as pl

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.config.dataset_config import DataSetConfig


class SummaryStatsBase(DataSetBase):
    """
    An abstract base class for calculating and writing summary statistics.

    This class extends :class:`DataSetBase`, which provides configuration
    validation for the YAML descriptor. It introduces utilities to handle
    summary statistics, including writing them to a file.

    Subclasses must define or implement:

    - The :meth:`calculate_stats` method (abstract), for computing summary statistics.
    - An ``expected_class_name`` value if a subclass is to be instantiated
      (otherwise, :class:`DataSetBase` will raise a :class:`NotImplementedError`).
    """

    def __init__(
        self, config: DataSetConfig, input_data: Optional[pl.DataFrame] = None
    ) -> None:
        """
        Initialize the summary statistics base with a dataset configuration
        and optional DataFrame for input data.

        :param config: Configuration object that includes paths and parameters
                       for retrieving or storing summary statistics.
        :type config: DataSetConfig
        :param input_data: A Polars DataFrame holding the data upon which
                           statistics will be computed. Defaults to None.
        :type input_data: pl.DataFrame, optional
        :raises NotImplementedError: If the :attr:`expected_class_name` is not
                                     defined by a subclass (when actually instantiated).
        :raises ValueError: If the YAML's "base_class" does not match the
                            :attr:`expected_class_name`.
        """
        super().__init__("summary", config)

        # Prepare paths/filenames for output
        self.default_file_name: str = "summary_stats.tsv"
        self.output_file_name: str = self.config.get_full_file_name(
            "summary", self.default_file_name
        )
        self.input_data: Optional[pl.DataFrame] = input_data
        self.summary_stats: Optional[pl.DataFrame] = None

    @abstractmethod
    def calculate_stats(self) -> None:
        """
        Calculate summary statistics on :attr:`input_data` and assign results
        to :attr:`summary_stats`.

        Subclasses must implement the specific method for computing the
        statistics (e.g., aggregates, distributions). The results
        should be placed in :attr:`summary_stats`.
        """
        pass  # pragma: no cover

    def write_summary_stats(self) -> None:
        """
        Write the computed summary statistics to a TSV file.

        :raises ValueError: If :attr:`summary_stats` is None or has not
                            been assigned by :meth:`calculate_stats`.
        """
        if self.summary_stats is None:
            raise ValueError("Member variable 'summary_stats' must not be empty.")

        os.makedirs(os.path.dirname(self.output_file_name), exist_ok=True)
        self.summary_stats.write_csv(self.output_file_name, separator="\t")
