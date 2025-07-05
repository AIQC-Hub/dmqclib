from typing import Dict, Optional, Type

import polars as pl

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.common.loader.classify_registry import (
    INPUT_CLASSIFY_REGISTRY,
    SUMMARY_CLASSIFY_REGISTRY,
    SELECT_CLASSIFY_REGISTRY,
    LOCATE_CLASSIFY_REGISTRY,
    EXTRACT_CLASSIFY_REGISTRY,
)
from dmqclib.common.config.dataset_config import DataSetConfig
from dmqclib.prepare.step1_input.input_base import InputDataSetBase
from dmqclib.prepare.step2_summary.summary_base import SummaryStatsBase
from dmqclib.prepare.step3_select.select_base import ProfileSelectionBase
from dmqclib.prepare.step4_locate.locate_base import LocatePositionBase
from dmqclib.prepare.step5_extract.extract_base import ExtractFeatureBase


def _get_prepare_class(
    config: DataSetConfig, 
    step: str, 
    registry: Dict[str, Type[DataSetBase]]
) -> Type[DataSetBase]:
    """
    Retrieve the class constructor from the specified registry for a given step.

    1. Obtain the class name by calling :meth:`DataSetConfig.get_base_class` on ``step``.
    2. Look up the class in the provided registry using the extracted name.
    3. Return the class (not an instantiated object).

    :param config: A dataset configuration object that contains the base class name 
                   for the requested step in the YAML.
    :type config: DataSetConfig
    :param step: The step name defined in the YAML (e.g., "input", "summary", or "select").
    :type step: str
    :param registry: A dictionary mapping class names to dataset class types 
                     inheriting from :class:`DataSetBase`.
    :type registry: dict of (str to Type[DataSetBase])
    :raises ValueError: If the class name from the configuration cannot be found 
                        in the given ``registry``.
    :return: The class constructor associated with the requested step.
    """
    class_name = config.get_base_class(step)
    dataset_class = registry.get(class_name)
    if not dataset_class:
        raise ValueError(f"Unknown classification class specified: {class_name}")

    return dataset_class


def load_classify_step1_input_dataset(config: DataSetConfig) -> InputDataSetBase:
    """
    Instantiate an :class:`InputDataSetBase`-derived class based on the configuration.

    Specifically:

    1. Fetches the class name from the config via :meth:`DataSetConfig.get_base_class("input")`.
    2. Looks up the class in :data:`INPUT_CLASSIFY_REGISTRY`.
    3. Instantiates and returns the class.

    :param config: The dataset configuration object, which includes 
                   a ``base_class`` field under "input" in the YAML file.
    :return: An instance of a class derived from :class:`InputDataSetBase`.
    """
    dataset_class = _get_prepare_class(config, "input", INPUT_CLASSIFY_REGISTRY)
    return dataset_class(config)


def load_classify_step2_summary_dataset(
    config: DataSetConfig, 
    input_data: Optional[pl.DataFrame] = None
) -> SummaryStatsBase:
    """
    Instantiate a :class:`SummaryStatsBase`-derived class based on the configuration.

    Specifically:

    1. Fetches the class name from the config via :meth:`DataSetConfig.get_base_class("summary")`.
    2. Looks up the class in :data:`SUMMARY_CLASSIFY_REGISTRY`.
    3. Instantiates and returns the class, optionally with an input dataset.

    :param config: The dataset configuration object referencing the "summary" step.
    :type config: DataSetConfig
    :param input_data: An optional Polars DataFrame for computing summary statistics.
    :rtype: SummaryStatsBase
    :return: An instance of a class derived from :class:`SummaryStatsBase`.
    """
    dataset_class = _get_prepare_class(config, "summary", SUMMARY_CLASSIFY_REGISTRY)
    return dataset_class(config, input_data=input_data)


def load_classify_step3_select_dataset(
    config: DataSetConfig, 
    input_data: Optional[pl.DataFrame] = None
) -> ProfileSelectionBase:
    """
    Instantiate a :class:`ProfileSelectionBase`-derived class based on the configuration.

    Specifically:

    1. Fetches the class name from the config via :meth:`DataSetConfig.get_base_class("select")`.
    2. Looks up the class in :data:`SELECT_CLASSIFY_REGISTRY`.
    3. Instantiates and returns the class, optionally with an input dataset.

    :param config: The dataset configuration object referencing the "select" step.
    :type config: DataSetConfig
    :param input_data: An optional Polars DataFrame for selecting profiles.
    :type input_data: pl.DataFrame, optional
    :return: An instance of a class derived from :class:`ProfileSelectionBase`.
    """
    dataset_class = _get_prepare_class(config, "select", SELECT_CLASSIFY_REGISTRY)
    return dataset_class(config, input_data=input_data)


def load_classify_step4_locate_dataset(
    config: DataSetConfig,
    input_data: Optional[pl.DataFrame] = None,
    selected_profiles: Optional[pl.DataFrame] = None,
) -> LocatePositionBase:
    """
    Instantiate a :class:`LocatePositionBase`-derived class based on the configuration.

    Specifically:

    1. Fetches the class name from the config via :meth:`DataSetConfig.get_base_class("locate")`.
    2. Looks up the class in :data:`LOCATE_CLASSIFY_REGISTRY`.
    3. Instantiates and returns the class, optionally with an input dataset 
       and previously selected profiles.

    :param config: The dataset configuration object referencing the "locate" step.
    :type config: DataSetConfig
    :param input_data: An optional Polars DataFrame containing the data from which 
                       location-based subsetting occurs.
    :type input_data: pl.DataFrame, optional
    :param selected_profiles: An optional Polars DataFrame containing already selected 
                              profiles that might be used for filtering additional rows.
    :type selected_profiles: pl.DataFrame, optional
    :return: An instance of a class derived from :class:`LocatePositionBase`.
    """
    dataset_class = _get_prepare_class(config, "locate", LOCATE_CLASSIFY_REGISTRY)
    return dataset_class(config, input_data=input_data, selected_profiles=selected_profiles)


def load_classify_step5_extract_dataset(
    config: DataSetConfig,
    input_data: Optional[pl.DataFrame] = None,
    selected_profiles: Optional[pl.DataFrame] = None,
    target_rows: Optional[pl.DataFrame] = None,
    summary_stats: Optional[pl.DataFrame] = None,
) -> ExtractFeatureBase:
    """
    Instantiate an :class:`ExtractFeatureBase`-derived class based on the configuration.

    Specifically:

    1. Fetches the class name from the config via :meth:`DataSetConfig.get_base_class("extract")`.
    2. Looks up the class in :data:`EXTRACT_CLASSIFY_REGISTRY`.
    3. Instantiates and returns the class, optionally with various intermediate datasets.

    :param config: The dataset configuration object referencing the "extract" step.
    :type config: DataSetConfig
    :param input_data: An optional Polars DataFrame containing the data from which
                       features will be extracted.
    :type input_data: pl.DataFrame, optional
    :param selected_profiles: An optional Polars DataFrame containing selected profiles,
                              if relevant to feature extraction.
    :type selected_profiles: pl.DataFrame, optional
    :param target_rows: An optional Polars DataFrame identifying rows relevant to each
                        target variable.
    :type target_rows: pl.DataFrame, optional
    :param summary_stats: An optional Polars DataFrame providing summary statistics that
                          might be used for feature scaling or reference.
    :type summary_stats: pl.DataFrame, optional
    :return: An instance of a class derived from :class:`ExtractFeatureBase`.
    """
    dataset_class = _get_prepare_class(config, "extract", EXTRACT_CLASSIFY_REGISTRY)
    return dataset_class(
        config,
        input_data=input_data,
        selected_profiles=selected_profiles,
        target_rows=target_rows,
        summary_stats=summary_stats,
    )
