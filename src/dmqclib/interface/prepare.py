from dmqclib.common.base.config_base import ConfigBase
from dmqclib.common.loader.dataset_loader import load_step1_input_dataset
from dmqclib.common.loader.dataset_loader import load_step2_summary_dataset
from dmqclib.common.loader.dataset_loader import load_step3_select_dataset
from dmqclib.common.loader.dataset_loader import load_step4_locate_dataset
from dmqclib.common.loader.dataset_loader import load_step5_extract_dataset
from dmqclib.common.loader.dataset_loader import load_step6_split_dataset


def create_training_dataset(config: ConfigBase) -> None:
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
        ds_locate.target_rows,
        ds_summary.summary_stats,
    )
    ds_extract.process_targets()
    ds_extract.write_target_features()

    ds_split = load_step6_split_dataset(config, ds_extract.target_features)
    ds_split.process_targets()
    ds_split.write_data_sets()
