from dmqclib.config.dataset_config import DataSetConfig
from dmqclib.prepare.step1_input.input_base import InputDataSetBase


class InputDataSetA(InputDataSetBase):
    """
    InputDataSetA reads BO NRT+Cora test data.
    """

    expected_class_name = "InputDataSetA"

    def __init__(self, config: DataSetConfig):
        super().__init__(config)

    def select(self):
        """
        Selects columns of the data frame in self.input_data.
        """
        pass  # pragma: no cover

    def filter(self):
        """
        Filter rows of the data frame in self.input_data.
        """
        pass  # pragma: no cover
