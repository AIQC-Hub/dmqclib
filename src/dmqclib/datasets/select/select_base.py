from abc import abstractmethod
import polars as pl
from dmqclib.datasets.base.dataset_base import DataSetBase
from dmqclib.utils.config import get_file_name_from_config
from dmqclib.utils.dataset_path import build_full_select_path


class ProfileSelectionBase(DataSetBase):
    """
    Base class for profile selection and group labeling classes
    """

    def __init__(
        self,
        dataset_name: str,
        config_file: str = None,
        input_data: pl.DataFrame = None,
    ):
        super().__init__("select", dataset_name, config_file=config_file)

        # Set member variables
        self.__build_output_file_name()
        self.input_data = input_data
        self.selected_profiles = None

    def __build_output_file_name(self):
        """
        Set the output file based on configuration entries.
        """
        folder_name = self.dataset_info["select"].get("folder_name", "")
        file_name = get_file_name_from_config(
            self.dataset_info["select"], self.config_file_name
        )

        self.output_file_name = build_full_select_path(
            self.path_info, folder_name, file_name
        )

    @abstractmethod
    def label_profiles(self):
        """
        Label profiles to identify positive and negative groups.
        """
        pass

    def write_selected_profiles(self):
        """
        Write selected profiles to parquet file
        """
        if self.selected_profiles is None:
            raise ValueError("'selected_profiles' is empty.")

        self.selected_profiles.write_parquet(self.output_file_name)
