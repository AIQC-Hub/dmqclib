from abc import abstractmethod

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.config.dataset_config import DataSetConfig
from dmqclib.utils.file import read_input_file


class InputDataSetBase(DataSetBase):
    """
    Base class for input data loading classes.
    """

    def __init__(self, config: DataSetConfig):
        super().__init__("input", config)

        # Set member variables
        self.input_file_name = self.config.get_full_file_name(
            "input",
            default_file_name=self.config.data["input_file_name"],
            use_dataset_folder=False,
            folder_name_auto=False,
        )
        self.input_data = None

    def read_input_data(self):
        """
        Reads the input data specified by the dataset entry in configuration file.
        """
        input_file = self.input_file_name
        file_type = self.config.get_step_params("input").get("file_type")
        read_file_options = self.config.get_step_params("input").get(
            "read_file_options", {}
        )

        self.input_data = read_input_file(input_file, file_type, read_file_options)

    @abstractmethod
    def select(self):
        """
        Selects columns of the data frame in self.input_data.
        """
        pass  # pragma: no cover

    @abstractmethod
    def filter(self):
        """
        Filter rows of the data frame in self.input_data.
        """
        pass  # pragma: no cover
