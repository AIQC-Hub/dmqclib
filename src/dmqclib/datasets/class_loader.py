import polars as pl
from typing import Dict
from dmqclib.utils.config import read_config
from dmqclib.datasets.input.input_base import InputDataSetBase
from dmqclib.datasets.registry import INPUT_DATASET_REGISTRY
from dmqclib.datasets.select.select_base import ProfileSelectionBase
from dmqclib.datasets.registry import SELECT_DATASET_REGISTRY


def _get_dataset_info(label: str, config_file: str = None) -> Dict:
    config = read_config(config_file, "datasets.yaml")

    dataset_info = config.get(label)
    if dataset_info is None:
        raise ValueError(f"No dataset configuration found for label '{label}'")

    return dataset_info


def load_input_dataset(label: str, config_file: str = None) -> InputDataSetBase:
    """
    Given a label (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    dataset_info = _get_dataset_info(label, config_file)

    class_name = dataset_info["input"].get("base_class")
    dataset_class = INPUT_DATASET_REGISTRY.get(class_name)
    if not dataset_class:
        raise ValueError(f"Unknown dataset class specified: {class_name}")

    return dataset_class(label, config_file=config_file)


def load_select_dataset(
    label: str, config_file: str = None, input_data: pl.DataFrame = None
) -> ProfileSelectionBase:
    """
    Given a label (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    dataset_info = _get_dataset_info(label, config_file)

    class_name = dataset_info["select"].get("base_class")
    dataset_class = SELECT_DATASET_REGISTRY.get(class_name)
    if not dataset_class:
        raise ValueError(f"Unknown dataset class specified: {class_name}")

    return dataset_class(label, config_file=config_file, input_data=input_data)
