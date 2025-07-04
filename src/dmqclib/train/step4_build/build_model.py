import polars as pl
from typing import Optional, Dict

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.train.step4_build.build_model_base import BuildModelBase


class BuildModel(BuildModelBase):
    """
    A subclass of :class:`BuildModelBase` that builds and tests models
    using provided training and test sets for each target.

    This class sets its :attr:`expected_class_name` to ``"BuildModel"``,
    which must match the YAML configuration’s ``base_class`` if you
    intend to instantiate it within that framework.

    .. note::

       The class-level docstring references “BuildModelBase builds models.”
       You may want to revise that note in your final documentation to
       accurately reflect this subclass’s role.
    """

    expected_class_name: str = "BuildModel"

    def __init__(
        self,
        config: ConfigBase,
        training_sets: Optional[Dict[str, pl.DataFrame]] = None,
        test_sets: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> None:
        """
        Initialize the BuildModel class with a training configuration,
        a dictionary of training sets, and optionally a dictionary
        of test sets.

        :param config: A training configuration object specifying paths,
                       parameters, and model-building directives.
        :type config: ConfigBase
        :param training_sets: A dictionary of training data keyed by target name,
                              each value being a Polars DataFrame. Defaults to None.
        :type training_sets: dict of (str to pl.DataFrame), optional
        :param test_sets: A dictionary of test data keyed by target name,
                          each value being a Polars DataFrame. Defaults to None.
        :type test_sets: dict of (str to pl.DataFrame), optional
        """
        super().__init__(config, training_sets=training_sets, test_sets=test_sets)

    def build(self, target_name: str) -> None:
        """
        Build (train) a model for the specified target, storing it in :attr:`models`.

        This method:
          1. Reloads the base model via :meth:`load_base_model`.
          2. Attaches the training data for the target (dropping the ``k_fold`` column).
          3. Calls :meth:`base_model.build`.
          4. Stores the built model in :attr:`models[target_name]`.

        :param target_name: The target variable name, used to index
                            :attr:`training_sets` and locate the training data.
        :type target_name: str
        """
        self.load_base_model()
        self.base_model.training_set = self.training_sets[target_name].drop("k_fold")
        self.base_model.build()
        self.models[target_name] = self.base_model

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
        self.base_model.test_set = self.test_sets[target_name]
        self.base_model.test()
        self.results[target_name] = self.base_model.result
