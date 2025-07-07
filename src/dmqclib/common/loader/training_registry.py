"""
This module provides registries for training dataset, model validation,
and model-building classes. Each registry is a dictionary mapping
string keys from the YAML configuration to the corresponding Python classes.
"""

from typing import Dict, Type

from dmqclib.train.step1_read_input.dataset_a import InputTrainingSetA
from dmqclib.train.step2_validate_model.kfold_validation import KFoldValidation
from dmqclib.train.step4_build_model.build_model import BuildModel

from dmqclib.train.step1_read_input.input_base import InputTrainingSetBase
from dmqclib.train.step2_validate_model.validate_base import ValidationBase
from dmqclib.train.step4_build_model.build_model_base import BuildModelBase

#: A dictionary linking class names for the "step1_read_input" step
#: to their corresponding dataset classes.
INPUT_TRAINING_SET_REGISTRY: Dict[str, Type[InputTrainingSetBase]] = {
    "InputTrainingSetA": InputTrainingSetA,
}

#: A dictionary linking class names for the "step2_validate_model" step
#: to their corresponding validation classes.
MODEL_VALIDATION_REGISTRY: Dict[str, Type[ValidationBase]] = {
    "KFoldValidation": KFoldValidation,
}

#: A dictionary linking class names for the "step4_build_model" step
#: to their corresponding model-building classes.
BUILD_MODEL_REGISTRY: Dict[str, Type[BuildModelBase]] = {
    "BuildModel": BuildModel,
}
