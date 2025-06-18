from dmqclib.datasets.train.train_base import TrainingDataSetBin1Base


class TrainingSetA(TrainingDataSetBin1Base):
    """
    TrainingSetA inherits from TrainingDataSetBase and sets the 'expected_class_name' to 'TrainingSetA'.
    Any custom logic specific to TrainingSetA can go here.
    """

    expected_class_name = "TrainingSetA"

    def __init__(self, dataset_name: str, config_file: str = None):
        super().__init__(dataset_name, config_file=config_file)
