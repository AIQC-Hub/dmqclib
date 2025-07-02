from dmqclib.config.training_config import TrainingConfig
from dmqclib.train.step1_input.input_base import InputTrainingSetBase


class InputTrainingSetA(InputTrainingSetBase):
    """
    InputTrainingSetA reads training and test sets for BO NRT+Cora test data.
    """

    expected_class_name = "InputTrainingSetA"

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
