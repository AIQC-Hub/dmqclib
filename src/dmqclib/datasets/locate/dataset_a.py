import polars as pl
from dmqclib.datasets.locate.locate_base import LocatePositionBase


class LocateDataSetA(LocatePositionBase):
    """
    LocateDataSetA identifies training data chunks from BO NRT+Cora test data.
    """

    expected_class_name = "LocateDataSetA"

    def __init__(
        self,
        dataset_name: str,
        config_file: str = None,
        input_data: pl.DataFrame = None,
        selected_profiles: pl.DataFrame = None,
    ):
        super().__init__(
            dataset_name,
            config_file=config_file,
            input_data=input_data,
            selected_profiles=selected_profiles,
        )
