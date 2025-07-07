from dmqclib.common.base.config_base import ConfigBase
from dmqclib.prepare.step1_read_input.input_base import InputDataSetBase


class InputDataSetAll(InputDataSetBase):
    """
    A subclass of :class:`InputDataSetBase` providing logic for reading
    BO NRT + Cora test data.

    This class sets the :attr:`expected_class_name` to ``InputDataSetAll``,
    ensuring the correct YAML configuration is matched for data loading.
    """

    expected_class_name: str = "InputDataSetAll"

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the input dataset with a specific configuration.

        :param config: A configuration object derived from :class:`ConfigBase`,
                       containing paths and parameters for retrieving
                       BO NRT + Cora test data.
        :type config: ConfigBase
        """
        super().__init__(config)
