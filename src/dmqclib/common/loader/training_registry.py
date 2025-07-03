"""
This module provides registries for training dataset, model validation,
and model-building classes. Each registry is a dictionary mapping
string keys from the YAML configuration to the corresponding Python classes.
"""

from typing import Dict, Type

from dmqclib.train.step1_input.dataset_a import InputTrainingSetA
from dmqclib.train.step2_validate.kfold_validation import KFoldValidation
from dmqclib.train.step4_build.build_model import BuildModel


#: A dictionary linking class names for the "input" step
#: to their corresponding dataset classes.
INPUT_TRAINING_SET_REGISTRY: Dict[str, Type[InputTrainingSetA]] = {
    "InputTrainingSetA": InputTrainingSetA,
}

#: A dictionary linking class names for the "validate" step
#: to their corresponding validation classes.
MODEL_VALIDATION_REGISTRY: Dict[str, Type[KFoldValidation]] = {
    "KFoldValidation": KFoldValidation,
}

#: A dictionary linking class names for the "build" step
#: to their corresponding model-building classes.
BUILD_MODEL_REGISTRY: Dict[str, Type[BuildModel]] = {
    "BuildModel": BuildModel,
}
