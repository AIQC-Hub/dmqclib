from dmqclib.common.base.config_base import ConfigBase
from dmqclib.prepare.step1_input.input_base import InputDataSetBase


class InputDataSetA(InputDataSetBase):
    """
    A subclass of :class:`InputDataSetBase` providing specific
    logic to read BO NRT + Cora test data.

    This class sets the :attr:`expected_class_name` to match
    the YAML configuration, ensuring that the correct child
    class is being used for data loading.
    """

    expected_class_name: str = "InputDataSetA"

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the input dataset.

        :param config: The dataset configuration object, which includes
                       paths and parameters for retrieving BO NRT + Cora
                       test data.
        :type config: ConfigBase
        """
        super().__init__(config)
