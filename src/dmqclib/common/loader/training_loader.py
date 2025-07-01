import polars as pl

from dmqclib.common.loader.training_registry import BUILD_MODEL_REGISTRY
from dmqclib.common.loader.training_registry import INPUT_TRAINING_SET_REGISTRY
from dmqclib.common.loader.training_registry import MODEL_VALIDATION_REGISTRY
from dmqclib.config.training_config import TrainingConfig
from dmqclib.training.step1_input.input_base import InputTrainingSetBase
from dmqclib.training.step2_validate.validate_base import ValidationBase
from dmqclib.training.step4_build.build_model_base import BuildModelBase


def load_step1_input_training_set(
    dataset_name: str, config: TrainingConfig
) -> InputTrainingSetBase:
    """
    Given a dataset_name (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    config.load_dataset_config(dataset_name)
    class_name = config.get_base_class("input")
    dataset_class = INPUT_TRAINING_SET_REGISTRY.get(class_name)

    return dataset_class(dataset_name, config)


def load_step2_model_validation_class(
    dataset_name: str, config: TrainingConfig, training_sets: pl.DataFrame = None
) -> ValidationBase:
    """
    Given a dataset_name (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    config.load_dataset_config(dataset_name)
    class_name = config.get_base_class("validate")
    dataset_class = MODEL_VALIDATION_REGISTRY.get(class_name)

    return dataset_class(dataset_name, config, training_sets=training_sets)


def load_step4_build_model_class(
    dataset_name: str,
    config: TrainingConfig,
    training_sets: pl.DataFrame = None,
    test_sets: pl.DataFrame = None,
) -> BuildModelBase:
    """
    Given a dataset_name (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    config.load_dataset_config(dataset_name)
    class_name = config.get_base_class("build")
    dataset_class = BUILD_MODEL_REGISTRY.get(class_name)

    return dataset_class(
        dataset_name,
        config,
        training_sets=training_sets,
        test_sets=test_sets,
    )
