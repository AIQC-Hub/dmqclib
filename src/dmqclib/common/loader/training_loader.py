import polars as pl
from typing import Optional, Dict, Type

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.common.loader.training_registry import BUILD_MODEL_REGISTRY
from dmqclib.common.loader.training_registry import INPUT_TRAINING_SET_REGISTRY
from dmqclib.common.loader.training_registry import MODEL_VALIDATION_REGISTRY
from dmqclib.common.config.training_config import TrainingConfig
from dmqclib.train.step1_input.input_base import InputTrainingSetBase
from dmqclib.train.step2_validate.validate_base import ValidationBase
from dmqclib.train.step4_build.build_model_base import BuildModelBase


def _get_train_class(
    config: TrainingConfig, step: str, registry: Dict[str, Type[DataSetBase]]
) -> Type[DataSetBase]:
    """
    Retrieve the dataset class constructor for a specified training step
    from the provided registry.

    Steps:

      1. Invoke :meth:`TrainingConfig.get_base_class(step)` to get the class name.
      2. Lookup the class in the given ``registry``.
      3. Return the uninstantiated class.

    :param config: A training configuration object, which should provide
                   a base class name for the given step.
    :type config: TrainingConfig
    :param step: A string indicating which step name to look up (e.g., "input", "validate", "build").
    :type step: str
    :param registry: A dictionary mapping class names (str) to class types
                     derived from :class:`DataSetBase`.
    :type registry: dict of (str to Type[DataSetBase])
    :raises ValueError: If the retrieved class name is not found in ``registry``.
    :return: The dataset class (uninstantiated) that is used for the specified step.
    :rtype: Type[DataSetBase]
    """
    class_name = config.get_base_class(step)
    dataset_class = registry.get(class_name)
    if not dataset_class:
        # Possibly a minor oversight: referencing "dataset_class" in the error
        # message rather than "class_name".
        raise ValueError(f"Unknown dataset class specified: {class_name}")

    return dataset_class


def load_step1_input_training_set(config: TrainingConfig) -> InputTrainingSetBase:
    """
    Retrieve and instantiate an :class:`InputTrainingSetBase` subclass
    for the "input" step, based on the YAML configuration.

    1. Extract the class name with :meth:`TrainingConfig.get_base_class("input")`.
    2. Retrieve the corresponding class from :data:`INPUT_TRAINING_SET_REGISTRY`.
    3. Instantiate the class and return it.

    :param config: The training configuration object containing a ``base_class``
                   entry under the "input" section.
    :type config: TrainingConfig
    :return: An instantiated object of a class that inherits from :class:`InputTrainingSetBase`.
    :rtype: InputTrainingSetBase
    """
    dataset_class = _get_train_class(config, "input", INPUT_TRAINING_SET_REGISTRY)
    return dataset_class(config)


def load_step2_model_validation_class(
    config: TrainingConfig, training_sets: Optional[pl.DataFrame] = None
) -> ValidationBase:
    """
    Retrieve and instantiate a :class:`ValidationBase` subclass for
    the "validate" step, based on the YAML configuration.

    Steps:
      1. Extract the class name with :meth:`TrainingConfig.get_base_class("validate")`.
      2. Retrieve the corresponding class from :data:`MODEL_VALIDATION_REGISTRY`.
      3. Instantiate the class, optionally passing the provided training sets.

    :param config: The training configuration object referencing a ``base_class``
                   under the "validate" section.
    :type config: TrainingConfig
    :param training_sets: A Polars DataFrame containing data for model validation,
                          defaults to None.
    :type training_sets: pl.DataFrame, optional
    :return: An instantiated object of a class that inherits from :class:`ValidationBase`.
    :rtype: ValidationBase
    """
    dataset_class = _get_train_class(config, "validate", MODEL_VALIDATION_REGISTRY)
    return dataset_class(config, training_sets=training_sets)


def load_step4_build_model_class(
    config: TrainingConfig,
    training_sets: Optional[pl.DataFrame] = None,
    test_sets: Optional[pl.DataFrame] = None,
) -> BuildModelBase:
    """
    Retrieve and instantiate a :class:`BuildModelBase` subclass for
    the "build" step, based on the YAML configuration.

    Steps:
      1. Extract the class name with :meth:`TrainingConfig.get_base_class("build")`.
      2. Retrieve the corresponding class from :data:`BUILD_MODEL_REGISTRY`.
      3. Instantiate the class, providing any training and test sets.

    :param config: The training configuration object referencing a ``base_class``
                   under the "build" section.
    :type config: TrainingConfig
    :param training_sets: A Polars DataFrame of training data, defaults to None.
    :type training_sets: pl.DataFrame, optional
    :param test_sets: A Polars DataFrame of test data, defaults to None.
    :type test_sets: pl.DataFrame, optional
    :return: An instantiated object of a class that inherits from :class:`BuildModelBase`.
    :rtype: BuildModelBase
    """
    dataset_class = _get_train_class(config, "build", BUILD_MODEL_REGISTRY)
    return dataset_class(
        config,
        training_sets=training_sets,
        test_sets=test_sets,
    )
