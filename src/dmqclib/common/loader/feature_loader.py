from typing import Dict, Optional

import polars as pl

from dmqclib.common.base.feature_base import FeatureBase
from dmqclib.common.loader.feature_registry import FEATURE_REGISTRY


def load_feature_class(
    target_name: str,
    feature_info: Dict,
    selected_profiles: Optional[pl.DataFrame] = None,
    filtered_input: Optional[pl.DataFrame] = None,
    target_rows: Optional[pl.DataFrame] = None,
    summary_stats: Optional[pl.DataFrame] = None,
) -> FeatureBase:
    """
    Instantiate a feature extraction class using the specified feature registry.

    Steps:
      1. Retrieve the class name from the ``"feature"`` key in ``feature_info``.
      2. Look up the corresponding class in :data:`FEATURE_REGISTRY`.
      3. Instantiate the retrieved class using the supplied parameters.

    :param target_name: The target variable or dataset name for which features
                        will be extracted.
    :type target_name: str
    :param feature_info: A dictionary describing the feature extraction procedure,
                         which must at least include the key ``"feature"``
                         referencing the feature class name in :data:`FEATURE_REGISTRY`.
    :type feature_info: dict
    :param selected_profiles: A Polars DataFrame of selected profiles,
                              if applicable, defaults to None.
    :type selected_profiles: pl.DataFrame, optional
    :param filtered_input: A Polars DataFrame winnowed to relevant data
                           for advanced merging or lookups, defaults to None.
    :type filtered_input: pl.DataFrame, optional
    :param target_rows: A Polars DataFrame containing the rows or
                        observations for this target, defaults to None.
    :type target_rows: pl.DataFrame, optional
    :param summary_stats: A Polars DataFrame with summary statistics
                          for potential use in scaling or transformation,
                          defaults to None.
    :type summary_stats: pl.DataFrame, optional
    :return: An instance of the requested feature extraction class,
             inheriting from :class:`FeatureBase`.
    :rtype: FeatureBase
    :raises ValueError: If ``feature_info["feature"]`` is not found
                        in :data:`FEATURE_REGISTRY`.
    """
    class_name = feature_info.get("feature")
    feature_class = FEATURE_REGISTRY.get(class_name)
    if not feature_class:
        raise ValueError(f"Unknown feature class specified: {class_name}")

    return feature_class(
        target_name,
        feature_info,
        selected_profiles,
        filtered_input,
        target_rows,
        summary_stats,
    )
