import os
from abc import abstractmethod
from typing import Optional

import polars as pl

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.config.dataset_config import DataSetConfig


class ProfileSelectionBase(DataSetBase):
    """
    Abstract base class for profile selection and group labeling.

    Inherits from :class:`DataSetBase` to leverage configuration handling and
    validation. Subclasses must define:

    - ``expected_class_name`` if they are intended to be instantiated (otherwise
      an error is raised).
    - A custom :meth:`label_profiles` method that implements profile selection
      and labeling logic.
    """

    def __init__(
        self, config: DataSetConfig, input_data: Optional[pl.DataFrame] = None
    ) -> None:
        """
        Initialize the profile selection base class with configuration and optional input data.

        :param config: A dataset configuration object that provides
                       file naming and folder paths.
        :type config: DataSetConfig
        :param input_data: Optional Polars DataFrame that serves as the
                           initial data for profile selection,
                           defaults to None.
        :type input_data: pl.DataFrame, optional
        :raises NotImplementedError: If ``expected_class_name`` is not defined by a subclass.
        :raises ValueError: If the YAML's "base_class" does not match the
                            subclass's ``expected_class_name``.
        """
        super().__init__("select", config)

        self.default_file_name: str = "selected_profiles.parquet"
        self.output_file_name: str = self.config.get_full_file_name(
            "select", self.default_file_name
        )
        self.input_data: Optional[pl.DataFrame] = input_data
        self.selected_profiles: Optional[pl.DataFrame] = None

    @abstractmethod
    def label_profiles(self) -> None:
        """
        Label profiles to identify positive and negative groups.

        Implementations should assign a DataFrame to :attr:`selected_profiles`
        indicating group labels (e.g., a "group_label" column).
        """
        pass  # pragma: no cover

    def write_selected_profiles(self) -> None:
        """
        Write the selected profiles to a Parquet file specified by :attr:`output_file_name`.

        :raises ValueError: If :attr:`selected_profiles` is None.
        """
        if self.selected_profiles is None:
            raise ValueError("Member variable 'selected_profiles' must not be empty.")

        os.makedirs(os.path.dirname(self.output_file_name), exist_ok=True)
        self.selected_profiles.write_parquet(self.output_file_name)
