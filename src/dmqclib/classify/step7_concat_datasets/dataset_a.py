import polars as pl
from typing import Optional, Dict

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.classify.step7_concat_datasets.concat_base import ConcatDatasetsBase


class ConcatDataSetA(ConcatDatasetsBase):
    """
    A subclass of :class:`ConcatDatasetsBase` to concatenate predictions and the input dataset

    This class sets its :attr:`expected_class_name` to ``"ConcatDataSetA"``,
    ensuring it is recognized in the YAML configuration as a valid
    extract class. It inherits the concatenation
    pipeline from :class:`ConcatDatasetsBase`.
    """

    expected_class_name: str = "ConcatDataSetA"

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
        predictions: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> None:
        """
        Initialize the feature extraction workflow for BO NRT + Cora data.

        :param config: A dataset configuration object that manages paths,
                       target definitions, and parameters for feature extraction.
        :type config: ConfigBase
        :param input_data: A Polars DataFrame containing all available data
                           for feature extraction, defaults to None.
        :type input_data: pl.DataFrame, optional
        :param predictions: A dictionary mapping each target to its respective
                            subset of predictions,
                            defaults to None.
        :type predictions: dict of str to pl.DataFrame, optional
        """
        super().__init__(
            config,
            input_data=input_data,
            predictions=predictions,
        )
