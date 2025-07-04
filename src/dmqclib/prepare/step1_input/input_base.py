from typing import Optional

import polars as pl

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.common.base.config_base import ConfigBase
from dmqclib.common.utils.file import read_input_file


class InputDataSetBase(DataSetBase):
    """
    Base class for input data loading. It extends :class:`DataSetBase` by adding
    mechanisms for reading raw data from a file, renaming columns, and filtering rows.

    Subclasses must implement or customize methods such as :meth:`rename_columns`
    and :meth:`filter_rows` to handle domain-specific requirements.
    """

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the input dataset with a given configuration.

        :param config: The ConfigBase object providing path and parameter details.
        :type config: ConfigBase
        :raises NotImplementedError: If the ``expected_class_name`` is not defined
                                     by a subclass of :class:`DataSetBase`.
        :raises ValueError: If the YAML config does not match this class's
                            expected class name.
        """
        super().__init__("input", config)

        #: The absolute or resolved file path from which data will be read.
        self.input_file_name: str = self.config.get_full_file_name(
            "input",
            default_file_name=self.config.data["input_file_name"],
            use_dataset_folder=False,
            folder_name_auto=False,
        )
        #: Polars DataFrame holding the loaded input data. Defaults to None
        #: until :meth:`read_input_data` is called.
        self.input_data: Optional[pl.DataFrame] = None

    def read_input_data(self) -> None:
        """
        Load data from the configured file into :attr:`input_data`.

        The method retrieves ``file_type`` and ``read_file_options`` from the config
        and uses :func:`read_input_file` to read the file specified by
        :attr:`input_file_name`.

        After reading the data, it optionally calls :meth:`rename_columns` and
        :meth:`filter_rows` to modify the DataFrame.

        :raises FileNotFoundError: If the specified file cannot be found.
        """
        input_file = self.input_file_name
        file_type = self.config.get_step_params("input").get("file_type")
        read_file_options = self.config.get_step_params("input").get(
            "read_file_options", {}
        )

        self.input_data = read_input_file(input_file, file_type, read_file_options)
        self.rename_columns()
        self.filter_rows()

    def rename_columns(self) -> None:
        """
        Rename columns in :attr:`input_data` using rename mappings from the config.

        If ``sub_steps.rename_columns`` is enabled and a ``rename_dict`` is present,
        columns will be renamed accordingly. Otherwise, the method does nothing.
        """
        if not self.config.get_step_params("input")["sub_steps"]["rename_columns"]:
            return None

        if "rename_dict" in self.config.get_step_params("input"):
            self.input_data = self.input_data.rename(
                self.config.get_step_params("input")["rename_dict"]
            )

        return None

    def filter_rows(self) -> None:
        """
        Filter rows in :attr:`input_data` based on year constraints or other rules.

        If ``sub_steps.filter_rows`` is enabled and relevant fields exist,
        it will either remove certain years via :meth:`remove_years` or keep
        only a specified set of years via :meth:`keep_years`.
        """
        if not self.config.get_step_params("input")["sub_steps"][
            "filter_rows"
        ] or "filter_method_dict" not in self.config.get_step_params("input"):
            return None

        if (
            "remove_years" in self.config.get_step_params("input")["filter_method_dict"]
            and len(
                self.config.get_step_params("input")["filter_method_dict"][
                    "remove_years"
                ]
            )
            > 0
        ):
            self.remove_years()

        if (
            "keep_years" in self.config.get_step_params("input")["filter_method_dict"]
            and len(
                self.config.get_step_params("input")["filter_method_dict"]["keep_years"]
            )
            > 0
        ):
            self.keep_years()

        return None

    def remove_years(self) -> None:
        """
        Remove data rows for years listed under ``remove_years`` in the config.

        Updates :attr:`input_data` by filtering out rows whose year is in
        the ``remove_years`` list.
        """
        years = self.config.get_step_params("input")["filter_method_dict"][
            "remove_years"
        ]
        self.input_data = (
            self.input_data.with_columns(
                pl.col("profile_timestamp").dt.year().alias("year")
            )
            .filter(~pl.col("year").is_in(years))
            .drop("year")
        )

    def keep_years(self) -> None:
        """
        Keep only data rows for years listed under ``keep_years`` in the config.

        Updates :attr:`input_data` by filtering in rows whose year is in
        the ``keep_years`` list.
        """
        years = self.config.get_step_params("input")["filter_method_dict"]["keep_years"]
        self.input_data = (
            self.input_data.with_columns(
                pl.col("profile_timestamp").dt.year().alias("year")
            )
            .filter(pl.col("year").is_in(years))
            .drop("year")
        )
