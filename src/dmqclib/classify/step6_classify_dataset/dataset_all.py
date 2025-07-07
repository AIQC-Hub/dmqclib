import polars as pl
from typing import Optional, Dict

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.train.step4_build_model.build_model_base import BuildModelBase


class ClassifyAll(BuildModelBase):
    """
    A subclass of :class:`BuildModelBase` that builds and tests models
    using provided training and test sets for each target.

    This class sets its :attr:`expected_class_name` to ``"ClassifyAll"``,
    which must match the YAML configuration’s ``base_class`` if you
    intend to instantiate it within that framework.

    .. note::

       The class-level docstring references “BuildModelBase builds models.”
       You may want to revise that note in your final documentation to
       accurately reflect this subclass’s role.
    """

    expected_class_name: str = "ClassifyAll"

    def __init__(
        self,
        config: ConfigBase,
        test_sets: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> None:
        """
        Initialize the BuildModel class with a training configuration,
        a dictionary of training sets, and optionally a dictionary
        of test sets.

        :param config: A training configuration object specifying paths,
                       parameters, and model-building directives.
        :type config: ConfigBase
        :param test_sets: A dictionary of test data keyed by target name,
                          each value being a Polars DataFrame. Defaults to None.
        :type test_sets: dict of (str to pl.DataFrame), optional
        """
        super().__init__(
            config, training_sets=None, test_sets=test_sets, step_name="classify"
        )

        #: Default names for model files and test reports,
        #: with placeholders for the target name.
        self.default_file_names: Dict[str, str] = {
            "report": "classify_report_{target_name}.tsv",
            "prediction": "classify_prediction_{target_name}.parquet",
        }
        self.default_model_file_name: str = "model_{target_name}.joblib"

        #: A dictionary mapping "model" or "result" to
        #: target-specific file paths.
        self.output_file_names: Dict[str, Dict[str, str]] = {
            k: self.config.get_target_file_names("classify", v)
            for k, v in self.default_file_names.items()
        }

        #: A dictionary mapping "model" to target-specific file paths.
        self.model_file_names: Dict[str, str] = self.config.get_target_file_names(
            "model", self.default_model_file_name, use_dataset_folder=False
        )

        self.drop_cols = ["row_id", "platform_code", "profile_no", "observation_no"]

        self.test_cols = [
            "row_id",
            "platform_code",
            "profile_no",
            "observation_no",
            "label",
        ]

    def build(self, target_name: str) -> None:
        """
        Build (train) a model for the specified target, storing it in :attr:`models`.

        :param target_name: The target variable name, used to index
                            :attr:`training_sets` and locate the training data.
        :type target_name: str
        """
        pass  # pragma: no cover

    def test(self, target_name: str) -> None:
        """
        Test the model for the given target, storing the results in :attr:`results`.

        This method:

          1. Retrieves the model from :attr:`models[target_name]`.
          2. Attaches the appropriate test set from :attr:`test_sets[target_name]`.
          3. Calls :meth:`base_model.test`.
          4. Stores the test results in :attr:`results[target_name]`.

        :param target_name: The target variable name, used to index
                            both :attr:`models` and :attr:`test_sets`.
        :type target_name: str
        """
        self.base_model = self.models[target_name]
        self.base_model.test_set = self.test_sets[target_name].drop(self.drop_cols)
        self.base_model.test()
        predictions = self.base_model.predictions
        self.predictions[target_name] = pl.concat(
            [
                self.test_sets[target_name].select(self.test_cols),
                predictions,
            ],
            how="horizontal",
        )
        self.reports[target_name] = self.base_model.report
