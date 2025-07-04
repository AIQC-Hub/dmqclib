import os
from abc import abstractmethod
from typing import Dict, Optional

import polars as pl

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.common.config.dataset_config import DataSetConfig


class LocatePositionBase(DataSetBase):
    """
    Abstract base class for locating and extracting target rows from a dataset.

    This class extends :class:`DataSetBase` to validate that the YAML configuration
    matches the expected structure and to provide a framework for operations related
    to identifying rows of interest (e.g., training data). Subclasses must implement:

    - The :meth:`locate_target_rows` method for per-target row identification.
    - Potentially define ``expected_class_name`` if this class is intended to be
      directly instantiated and matched against the YAML's ``base_class`` configuration.
    """

    def __init__(
        self,
        config: DataSetConfig,
        input_data: Optional[pl.DataFrame] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize the base class for locating position or training rows within a dataset.

        :param config: Configuration object for dataset paths and target definitions.
        :type config: DataSetConfig
        :param input_data: A Polars DataFrame containing the full dataset from which
                           target rows can be extracted, defaults to None.
        :type input_data: pl.DataFrame, optional
        :param selected_profiles: A Polars DataFrame containing pre-selected profiles
                                  or rows, if applicable, defaults to None.
        :type selected_profiles: pl.DataFrame, optional
        :raises NotImplementedError: If ``expected_class_name`` is not defined
                                     by a subclass and this class is directly instantiated.
        :raises ValueError: If the YAML's ``base_class`` does not match
                            the subclass's ``expected_class_name``.
        """
        super().__init__("locate", config)

        #: Default file name template for writing target rows (one file per target).
        self.default_file_name: str = "{target_name}_rows.parquet"

        #: Dictionary mapping each target name to the corresponding output Parquet file path.
        self.output_file_names: Dict[str, str] = self.config.get_target_file_names(
            "locate", self.default_file_name
        )

        #: An optional Polars DataFrame from which target rows will be extracted.
        self.input_data: Optional[pl.DataFrame] = input_data

        #: An optional Polars DataFrame of pre-selected profiles or rows that might
        #: be combined with the input data during the target-location process.
        self.selected_profiles: Optional[pl.DataFrame] = selected_profiles

        #: A dictionary to store the resulting target rows for each target as a Polars DataFrame.
        self.target_rows: Dict[str, pl.DataFrame] = {}

    def process_targets(self) -> None:
        """
        Iterate over all targets, calling :meth:`locate_target_rows` on each.

        The target definitions (names and other metadata) are retrieved from
        the configuration, and each target is processed in turn.
        Subclasses define the logic of :meth:`locate_target_rows`.
        """
        for target_name, target_info in self.config.get_target_dict().items():
            self.locate_target_rows(target_name, target_info)

    @abstractmethod
    def locate_target_rows(self, target_name: str, target_value: Dict) -> None:
        """
        Locate rows in :attr:`input_data` or :attr:`selected_profiles`
        relevant to a specific target.

        Implementations should identify the subset of rows matching
        the target criteria and store them in :attr:`target_rows` under
        the target name.

        :param target_name: Name of the target variable to process.
        :type target_name: str
        :param target_value: A dictionary of target metadata or criteria.
        :type target_value: Dict
        """
        pass  # pragma: no cover

    def write_target_rows(self) -> None:
        """
        Write the identified target rows to separate Parquet files.

        Each target's identified rows are written to a file path derived
        from :attr:`output_file_names`. The name is based on the target
        and a default pattern.

        :raises ValueError: If :attr:`target_rows` is empty.
        """
        if not self.target_rows:
            raise ValueError("Member variable 'target_rows' must not be empty.")

        for target_name, df in self.target_rows.items():
            file_path = self.output_file_names[target_name]
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.write_parquet(file_path)
