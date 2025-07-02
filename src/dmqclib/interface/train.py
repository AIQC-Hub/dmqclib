from dmqclib.common.base.config_base import ConfigBase
from dmqclib.common.loader.training_loader import load_step1_input_training_set
from dmqclib.common.loader.training_loader import load_step2_model_validation_class
from dmqclib.common.loader.training_loader import load_step4_build_model_class


def train_and_evaluate(config: ConfigBase) -> None:
    ds_input = load_step1_input_training_set(config)
    ds_input.process_targets()

    ds_valid = load_step2_model_validation_class(config, ds_input.training_sets)
    ds_valid.process_targets()
    ds_valid.write_results()

    ds_build = load_step4_build_model_class(
        config, ds_input.training_sets, ds_input.test_sets
    )
    ds_build.build_targets()
    ds_build.test_targets()
    ds_build.write_results()
    ds_build.write_models()
