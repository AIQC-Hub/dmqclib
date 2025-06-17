from dmqclib.datasets.input.input_base import InputDataSetBase


class DataSetA(InputDataSetBase):
    """
    DataSetA inherits from DataSetBase and sets the 'expected_class_name' to 'DataSetA'.
    Any custom logic specific to DataSetA can go here.
    """

    expected_class_name = "DataSetA"

    def __init__(self, dataset_name: str, config_file: str = None):
        super().__init__(dataset_name, config_file=config_file)
        # If DataSetA had additional custom logic or fields, place them below.
        # For now, all logic is in the parent DataSetBase class.
