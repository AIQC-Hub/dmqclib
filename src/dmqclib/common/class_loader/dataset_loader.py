from typing import Dict
import polars as pl
from dmqclib.utils.config import read_config
from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.datasets.input.input_base import InputDataSetBase
from dmqclib.datasets.summary.summary_base import SummaryStatsBase
from dmqclib.datasets.select.select_base import ProfileSelectionBase
from dmqclib.datasets.locate.locate_base import LocatePositionBase
from dmqclib.datasets.extract.extract_base import ExtractFeatureBase
from dmqclib.datasets.split.split_base import SplitDataSetBase
from dmqclib.common.class_loader.dataset_registry import INPUT_DATASET_REGISTRY
from dmqclib.common.class_loader.dataset_registry import SUMMARY_DATASET_REGISTRY
from dmqclib.common.class_loader.dataset_registry import SELECT_DATASET_REGISTRY
from dmqclib.common.class_loader.dataset_registry import LOCATE_DATASET_REGISTRY
from dmqclib.common.class_loader.dataset_registry import EXTRACT_DATASET_REGISTRY
from dmqclib.common.class_loader.dataset_registry import SPLIT_DATASET_REGISTRY


def _get_dataset_info(label: str, config_file: str = None) -> Dict:
    config = read_config(config_file, "datasets.yaml")

    dataset_info = config.get(label)
    if dataset_info is None:
        raise ValueError(f"No dataset configuration found for label '{label}'")

    return dataset_info


def _get_class(dataset_info: Dict, data_type: str, registry: Dict) -> DataSetBase:
    class_name = dataset_info["base_class"].get(data_type)
    dataset_class = registry.get(class_name)
    if not dataset_class:
        raise ValueError(f"Unknown dataset class specified: {class_name}")

    return dataset_class


def load_input_dataset(label: str, config_file: str = None) -> InputDataSetBase:
    """
    Given a label (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    dataset_info = _get_dataset_info(label, config_file)
    dataset_class = _get_class(dataset_info, "input", INPUT_DATASET_REGISTRY)

    return dataset_class(label, config_file=config_file)


def load_summary_dataset(
    label: str, config_file: str = None, input_data: pl.DataFrame = None
) -> SummaryStatsBase:
    """
    Given a label (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    dataset_info = _get_dataset_info(label, config_file)
    dataset_class = _get_class(dataset_info, "summary", SUMMARY_DATASET_REGISTRY)

    return dataset_class(label, config_file=config_file, input_data=input_data)


def load_select_dataset(
    label: str, config_file: str = None, input_data: pl.DataFrame = None
) -> ProfileSelectionBase:
    """
    Given a label (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    dataset_info = _get_dataset_info(label, config_file)
    dataset_class = _get_class(dataset_info, "select", SELECT_DATASET_REGISTRY)

    return dataset_class(label, config_file=config_file, input_data=input_data)


def load_locate_dataset(
    label: str,
    config_file: str = None,
    input_data: pl.DataFrame = None,
    selected_profiles: pl.DataFrame = None,
) -> ExtractFeatureBase:
    """
    Given a label (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    dataset_info = _get_dataset_info(label, config_file)
    dataset_class = _get_class(dataset_info, "locate", LOCATE_DATASET_REGISTRY)

    return dataset_class(
        label,
        config_file=config_file,
        input_data=input_data,
        selected_profiles=selected_profiles,
    )


def load_extract_dataset(
    label: str,
    config_file: str = None,
    input_data: pl.DataFrame = None,
    selected_profiles: pl.DataFrame = None,
    target_rows: pl.DataFrame = None,
    summary_stats: pl.DataFrame = None,
) -> LocatePositionBase:
    """
    Given a label (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    dataset_info = _get_dataset_info(label, config_file)
    dataset_class = _get_class(dataset_info, "extract", EXTRACT_DATASET_REGISTRY)

    return dataset_class(
        label,
        config_file=config_file,
        input_data=input_data,
        selected_profiles=selected_profiles,
        target_rows=target_rows,
        summary_stats=summary_stats,
    )


def load_split_dataset(
    label: str,
    config_file: str = None,
    target_features: pl.DataFrame = None,
) -> SplitDataSetBase:
    """
    Given a label (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    dataset_info = _get_dataset_info(label, config_file)
    dataset_class = _get_class(dataset_info, "split", SPLIT_DATASET_REGISTRY)

    return dataset_class(
        label,
        config_file=config_file,
        target_features=target_features,
    )
