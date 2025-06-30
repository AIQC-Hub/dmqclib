import polars as pl

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.training.step2_validate.validate_base import ValidationBase


class KFoldValidation(ValidationBase):
    """
    KFoldValidation performs k-fold cross validation
    """

    expected_class_name = "KFoldValidation"

    def __init__(
        self,
        dataset_name: str,
        config: ConfigBase = None,
        config_file: str = None,
        training_sets: pl.DataFrame = None,
    ):
        super().__init__(
            dataset_name,
            config=config,
            config_file=config_file,
            training_sets=training_sets,
        )

        self.default_k_fold = 10

    def get_k_fold(self) -> str:
        if (
            "validate" in self.dataset_info
            and "k_fold" in self.dataset_info["validate"]
        ):
            k_fold = self.dataset_info["validate"].get("k_fold", self.default_k_fold)
        else:
            k_fold = self.default_k_fold

        return k_fold

    def validate(self, target_name: str):
        """
        Validate models
        """

        self.models[target_name] = list()
        results = list()

        k_fold = self.get_k_fold()
        for k in range(k_fold):
            self.load_base_model()
            self.base_model.k = k + 1
            self.base_model.training_set = (
                self.training_sets[target_name]
                .filter(pl.col("k_fold") != (k + 1))
                .drop("k_fold")
            )
            self.base_model.build()
            self.models[target_name].append(self.base_model)

            self.base_model.test_set = (
                self.training_sets[target_name]
                .filter(pl.col("k_fold") == (k + 1))
                .drop("k_fold")
            )
            self.base_model.test()
            results.append(self.base_model.result)

        self.results[target_name] = pl.concat(results)
