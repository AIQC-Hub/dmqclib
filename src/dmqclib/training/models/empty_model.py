from dmqclib.common.base.model_base import ModelBase


class EmptyModel(ModelBase):
    expected_class_name = "EmptyModel"

    def __init__(self, dataset_name: str, config_file: str = None):
        super().__init__(
            dataset_name,
            config_file=config_file,
        )

        self.training_data_set = None
        self.test_data_set = None
        self.model = None

    def build(self):
        """
        Build model
        """
        if self.training_data_set is None:
            raise ValueError("Member variable 'training_data_set' must not be empty.")

    def test(self):
        """
        Test model.
        """
        if self.model is None:
            raise ValueError("Member variable 'model' must not be empty.")

        if self.test_data_set is None:
            raise ValueError("Member variable 'training_data_set' must not be empty.")
