from dmqclib.config.training_config import TrainingConfig
from dmqclib.training.step1_input.input_base import InputTrainingSetBase


class InputTrainingSetA(InputTrainingSetBase):
    """
    InputTrainingSetA reads training and test sets for BO NRT+Cora test data.
    """

    expected_class_name = "InputTrainingSetA"

    def __init__(
        self,
        dataset_name: str,
        config: TrainingConfig = None,
        config_file: str = None,
    ):
        super().__init__(
            dataset_name,
            config=config,
            config_file=config_file,
        )
