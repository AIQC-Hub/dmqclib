import polars as pl
from dmqclib.datasets.select.select_base import ProfileSelectionBase


class SelectDataSetA(ProfileSelectionBase):
    """
    SelectDataSetA inherits from ProfileSelectionBase and sets the 'expected_class_name' to 'SelectDataSetA'.
    """

    expected_class_name = "SelectDataSetA"

    def __init__(
        self,
        dataset_name: str,
        config_file: str = None,
        input_data: pl.DataFrame = None,
    ):
        super().__init__(dataset_name, config_file=config_file, input_data=input_data)

    def label_profiles(self):
        """
        Label profiles in terms of positive and negative candidates
        """
        pass

    def filter_profiles(self):
        """
        Filter profiles based on the labels
        """
        pass
