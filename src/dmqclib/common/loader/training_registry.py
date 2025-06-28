from dmqclib.training.step1_input.dataset_a import InputTrainingSetA
from dmqclib.training.step2_validate.kfold_validation import KFoldValidation
from dmqclib.training.step4_build.build_model import BuildModel

INPUT_TRAINING_SET_REGISTRY = {
    "InputTrainingSetA": InputTrainingSetA,
}

MODEL_VALIDATION_REGISTRY = {
    "KFoldValidation": KFoldValidation,
}

BUILD_MODEL_REGISTRY = {
    "BuildModel": BuildModel,
}
