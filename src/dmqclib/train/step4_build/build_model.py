import polars as pl

from dmqclib.config.training_config import TrainingConfig
from dmqclib.train.step4_build.build_model_base import BuildModelBase


class BuildModel(BuildModelBase):
    """
    BuildModelBase builds models
    """

    expected_class_name = "BuildModel"

    def __init__(
        self,
        config: TrainingConfig,
        training_sets: pl.DataFrame = None,
        test_sets: pl.DataFrame = None,
    ):
        super().__init__(config, training_sets=training_sets, test_sets=test_sets)

    def build(self, target_name: str):
        """
        Build model
        """
        self.load_base_model()
        self.base_model.training_set = self.training_sets[target_name].drop("k_fold")
        self.base_model.build()
        self.models[target_name] = self.base_model

    def test(self, target_name: str):
        """
        Test model
        """
        self.base_model = self.models[target_name]
        self.base_model.test_set = self.test_sets[target_name]
        self.base_model.test()
        self.results[target_name] = self.base_model.result
