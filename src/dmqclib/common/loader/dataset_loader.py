import polars as pl

from dmqclib.common.loader.dataset_registry import EXTRACT_DATASET_REGISTRY
from dmqclib.common.loader.dataset_registry import INPUT_DATASET_REGISTRY
from dmqclib.common.loader.dataset_registry import LOCATE_DATASET_REGISTRY
from dmqclib.common.loader.dataset_registry import SELECT_DATASET_REGISTRY
from dmqclib.common.loader.dataset_registry import SPLIT_DATASET_REGISTRY
from dmqclib.common.loader.dataset_registry import SUMMARY_DATASET_REGISTRY
from dmqclib.config.dataset_config import DataSetConfig
from dmqclib.prepare.step1_input.input_base import InputDataSetBase
from dmqclib.prepare.step2_summary.summary_base import SummaryStatsBase
from dmqclib.prepare.step3_select.select_base import ProfileSelectionBase
from dmqclib.prepare.step4_locate.locate_base import LocatePositionBase
from dmqclib.prepare.step5_extract.extract_base import ExtractFeatureBase
from dmqclib.prepare.step6_split.split_base import SplitDataSetBase


def load_step1_input_dataset(config: DataSetConfig) -> InputDataSetBase:
    """
    Given a dataset_name (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    class_name = config.get_base_class("input")
    dataset_class = INPUT_DATASET_REGISTRY.get(class_name)

    return dataset_class(config)


def load_step2_summary_dataset(
    config: DataSetConfig, input_data: pl.DataFrame = None
) -> SummaryStatsBase:
    """
    Given a dataset_name (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    class_name = config.get_base_class("summary")
    dataset_class = SUMMARY_DATASET_REGISTRY.get(class_name)

    return dataset_class(config, input_data=input_data)


def load_step3_select_dataset(
    config: DataSetConfig, input_data: pl.DataFrame = None
) -> ProfileSelectionBase:
    """
    Given a dataset_name (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    class_name = config.get_base_class("select")
    dataset_class = SELECT_DATASET_REGISTRY.get(class_name)

    return dataset_class(config, input_data=input_data)


def load_step4_locate_dataset(
    config: DataSetConfig,
    input_data: pl.DataFrame = None,
    selected_profiles: pl.DataFrame = None,
) -> ExtractFeatureBase:
    """
    Given a dataset_name (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    class_name = config.get_base_class("locate")
    dataset_class = LOCATE_DATASET_REGISTRY.get(class_name)

    return dataset_class(
        config,
        input_data=input_data,
        selected_profiles=selected_profiles,
    )


def load_step5_extract_dataset(
    config: DataSetConfig,
    input_data: pl.DataFrame = None,
    selected_profiles: pl.DataFrame = None,
    target_rows: pl.DataFrame = None,
    summary_stats: pl.DataFrame = None,
) -> LocatePositionBase:
    """
    Given a dataset_name (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """

    class_name = config.get_base_class("extract")
    dataset_class = EXTRACT_DATASET_REGISTRY.get(class_name)

    return dataset_class(
        config,
        input_data=input_data,
        selected_profiles=selected_profiles,
        target_rows=target_rows,
        summary_stats=summary_stats,
    )


def load_step6_split_dataset(
    config: DataSetConfig, target_features: pl.DataFrame = None
) -> SplitDataSetBase:
    """
    Given a dataset_name (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    class_name = config.get_base_class("split")
    dataset_class = SPLIT_DATASET_REGISTRY.get(class_name)

    return dataset_class(
        config,
        target_features=target_features,
    )
