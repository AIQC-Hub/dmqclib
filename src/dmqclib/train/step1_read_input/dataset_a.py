"""
This module provides a specialized input class, :class:`InputTrainingSetA`,
designed for reading training and test datasets specific to Copernicus CTD data.
It extends :class:`dmqclib.train.step1_read_input.input_base.InputTrainingSetBase`
to handle particular data configuration and validation requirements.
"""

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.train.step1_read_input.input_base import InputTrainingSetBase


class InputTrainingSetA(InputTrainingSetBase):
    """
    A specialized input class for reading training and test sets
    for Copernicus CTD data.

    This class sets its :attr:`expected_class_name` to "InputTrainingSetA"
    so that config validation in the parent class matches the YAML's
    ``base_class`` value.
    """

    expected_class_name: str = "InputTrainingSetA"

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the specialized input training set class with the provided
        training configuration.

        :param config: A training configuration object containing paths,
                       file names, and target definitions.
        :type config: ConfigBase
        """
        super().__init__(config)
