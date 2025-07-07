"""
Module that orchestrates the creation of a training dataset by sequentially
loading and processing data through multiple preparation steps.
"""

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.common.loader.dataset_loader import load_step1_input_dataset
from dmqclib.common.loader.dataset_loader import load_step2_summary_dataset
from dmqclib.common.loader.dataset_loader import load_step3_select_dataset
from dmqclib.common.loader.dataset_loader import load_step4_locate_dataset
from dmqclib.common.loader.dataset_loader import load_step5_extract_dataset
from dmqclib.common.loader.dataset_loader import load_step6_split_dataset


def create_training_dataset(config: ConfigBase) -> None:
    """
    Execute a series of steps to produce a training dataset, as defined
    by the provided configuration object.

    This function performs the following steps:

      1. Load and read the initial input data.
      2. Calculate and write summary statistics.
      3. Label and write selected profiles.
      4. Locate and write target rows.
      5. Extract and write target features.
      6. Split and write final data sets for training/validation purposes.

    :param config: A configuration object specifying the classes and parameters
                   for each step in the dataset preparation process.
    :type config: ConfigBase
    :return: None (the function performs I/O operations and does not return a value).
    :rtype: None

    Example Usage:
      >>> from dmqclib.common.base.config_base import ConfigBase
      >>> cfg = ConfigBase(...)
      >>> create_training_dataset(cfg)
    """
    ds_input = load_step1_input_dataset(config)
    ds_input.read_input_data()

    ds_summary = load_step2_summary_dataset(config, ds_input.input_data)
    ds_summary.calculate_stats()
    ds_summary.write_summary_stats()

    ds_select = load_step3_select_dataset(config, ds_input.input_data)
    ds_select.label_profiles()
    ds_select.write_selected_profiles()

    ds_locate = load_step4_locate_dataset(
        config, ds_input.input_data, ds_select.selected_profiles
    )
    ds_locate.process_targets()
    ds_locate.write_target_rows()

    ds_extract = load_step5_extract_dataset(
        config,
        ds_input.input_data,
        ds_select.selected_profiles,
        ds_locate.selected_rows,
        ds_summary.summary_stats,
    )
    ds_extract.process_targets()
    ds_extract.write_target_features()

    ds_split = load_step6_split_dataset(config, ds_extract.target_features)
    ds_split.process_targets()
    ds_split.write_data_sets()
