from dmqclib.datasets.input.input_base import InputDataSetBase


class InputDataSetA(InputDataSetBase):
    """
    InputDataSetA inherits from DataSetBase and sets the 'expected_class_name' to 'InputDataSetA'.
    """

    expected_class_name = "InputDataSetA"

    def __init__(self, dataset_name: str, config_file: str = None):
        super().__init__(dataset_name, config_file=config_file)

    def select(self):
        """
        Selects columns of the data frame in self.input_data
        """
        pass

    def filter(self):
        """
        Filter rows of the data frame in self.input_data
        """
        pass
