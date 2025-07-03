from abc import abstractmethod
from typing import Optional

import polars as pl

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.config.dataset_config import DataSetConfig
from dmqclib.utils.file import read_input_file


class InputDataSetBase(DataSetBase):
    """
    Base class for input data loading classes. It extends
    :class:`DataSetBase` to provide logic for reading raw data
    from a specified input file and storing it locally.

    Subclasses must implement the abstract methods:

    - :meth:`select` (for selecting columns)
    - :meth:`filter` (for filtering rows)
    """

    def __init__(self, config: DataSetConfig) -> None:
        """
        Initialize the input dataset with a provided configuration.

        :param config: The dataset configuration object, which
                       includes the path and parameters for the dataset.
        :type config: DataSetConfig
        :raises NotImplementedError: If ``expected_class_name`` is not defined
                                     by a subclass of :class:`DataSetBase`.
        :raises ValueError: If the incoming YAML config does not match
                            this class's expected class name.
        """
        super().__init__("input", config)

        # Construct the full file name based on the config
        self.input_file_name: str = self.config.get_full_file_name(
            "input",
            default_file_name=self.config.data["input_file_name"],
            use_dataset_folder=False,
            folder_name_auto=False,
        )
        # Holds the loaded data; subclass usage can assume this is populated
        # after :meth:`read_input_data`.
        self.input_data: Optional[pl.DataFrame] = None

    def read_input_data(self) -> None:
        """
        Read input data into :attr:`input_data` based on configuration details.

        The file type and read options are extracted from the config,
        and :func:`read_input_file` is used to load the data from
        :attr:`input_file_name`.

        :raises FileNotFoundError: If the input file does not exist
                                   (depending on the behavior of
                                   :func:`read_input_file`).
        """
        input_file = self.input_file_name
        file_type = self.config.get_step_params("input").get("file_type")
        read_file_options = self.config.get_step_params("input").get(
            "read_file_options", {}
        )

        self.input_data = read_input_file(input_file, file_type, read_file_options)

    @abstractmethod
    def select_columns(self) -> None:
        """
        Select columns or features of interest within :attr:`input_data`.

        Subclasses should implement the exact columns or transformations
        to keep or rename, storing the modified DataFrame back into
        :attr:`input_data`.
        """
        pass  # pragma: no cover

    @abstractmethod
    def filter_rows(self) -> None:
        """
        Filter rows of :attr:`input_data` based on specified criteria.

        Subclasses should implement the row-level filtering logic,
        updating :attr:`input_data` with the filtered result.
        """
        pass  # pragma: no cover
