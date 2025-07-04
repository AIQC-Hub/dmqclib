from dmqclib.config.dataset_config import DataSetConfig
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

    def __init__(self, config: DataSetConfig) -> None:
        """
        Initialize the input dataset.

        :param config: The dataset configuration object, which includes
                       paths and parameters for retrieving BO NRT + Cora
                       test data.
        :type config: DataSetConfig
        """
        super().__init__(config)

    def rename_columns(self) -> None:
        """
        Rename columns of interest within :attr:`input_data`.

        Subclasses should implement the rename columns
         storing the modified DataFrame back into
        :attr:`input_data`.
        """
        if not self.config.get_step_params("input")["sub_steps"]["rename_columns"]:
            return None

        if "rename_dict" not in self.config.get_step_params("input"):
            raise ValueError("Could not find 'rename_dict' in Input parameters")

        self.input_data = self.input_data.rename(
            self.config.get_step_params("input")["rename_dict"]
        )

    def select_columns(self) -> None:
        """
        Select columns of the data frame in :attr:`input_data`.

        Subclasses typically apply column-based transformations
        or filtering logic here. In this custom child class, further
        selection procedures may be added to tailor the data to
        modeling requirements.
        """
        pass  # pragma: no cover

    def filter_rows(self) -> None:
        """
        Filter rows of the data frame in :attr:`input_data`.

        Subclasses typically apply row-based filtering logic,
        such as removing invalid or incomplete entries. In this class,
        additional domain-specific rules might be placed here to refine
        the dataset based on business or scientific criteria.
        """
        pass  # pragma: no cover
