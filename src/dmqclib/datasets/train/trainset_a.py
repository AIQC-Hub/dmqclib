from dmqclib.datasets.train.train_base import TrainingDataSetBase


class TrainingSetA(TrainingDataSetBase):
    """
    TrainingSetA inherits from TrainingDataSetBase and sets the 'expected_class_name' to 'TrainingSetA'.
    Any custom logic specific to TrainingSetA can go here.
    """

    expected_class_name = "TrainingSetA"

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
