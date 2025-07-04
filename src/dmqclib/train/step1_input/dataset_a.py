from dmqclib.common.config.training_config import TrainingConfig
from dmqclib.train.step1_input.input_base import InputTrainingSetBase


class InputTrainingSetA(InputTrainingSetBase):
    """
    A specialized input class for reading training and test sets
    for BO NRT + Cora test data.

    This class sets its :attr:`expected_class_name` to "InputTrainingSetA"
    so that config validation in the parent class matches the YAML's
    ``base_class`` value.
    """

    expected_class_name: str = "InputTrainingSetA"

    def __init__(self, config: TrainingConfig) -> None:
        """
        Initialize the specialized input training set class with the provided
        training configuration.

        :param config: A training configuration object containing paths,
                       file names, and target definitions.
        :type config: TrainingConfig
        """
        super().__init__(config)
