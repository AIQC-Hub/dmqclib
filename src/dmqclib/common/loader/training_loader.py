import polars as pl

from dmqclib.common.loader.training_registry import BUILD_MODEL_REGISTRY
from dmqclib.common.loader.training_registry import INPUT_TRAINING_SET_REGISTRY
from dmqclib.common.loader.training_registry import MODEL_VALIDATION_REGISTRY
from dmqclib.config.training_config import TrainingConfig
from dmqclib.train.step1_input.input_base import InputTrainingSetBase
from dmqclib.train.step2_validate.validate_base import ValidationBase
from dmqclib.train.step4_build.build_model_base import BuildModelBase


def load_step1_input_training_set(config: TrainingConfig) -> InputTrainingSetBase:
    """
    Given a dataset_name (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    class_name = config.get_base_class("input")
    dataset_class = INPUT_TRAINING_SET_REGISTRY.get(class_name)

    return dataset_class(config)


def load_step2_model_validation_class(
    config: TrainingConfig, training_sets: pl.DataFrame = None
) -> ValidationBase:
    """
    Given a dataset_name (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    class_name = config.get_base_class("validate")
    dataset_class = MODEL_VALIDATION_REGISTRY.get(class_name)

    return dataset_class(config, training_sets=training_sets)


def load_step4_build_model_class(
    config: TrainingConfig,
    training_sets: pl.DataFrame = None,
    test_sets: pl.DataFrame = None,
) -> BuildModelBase:
    """
    Given a dataset_name (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    class_name = config.get_base_class("build")
    dataset_class = BUILD_MODEL_REGISTRY.get(class_name)

    return dataset_class(
        config,
        training_sets=training_sets,
        test_sets=test_sets,
    )
