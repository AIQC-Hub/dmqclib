import polars as pl
from typing import Optional, Dict

from dmqclib.common.config.dataset_config import DataSetConfig
from dmqclib.prepare.step5_extract.extract_base import ExtractFeatureBase


class ExtractDataSetA(ExtractFeatureBase):
    """
    A subclass of :class:`ExtractFeatureBase` to extract features
    from BO NRT + Cora test data.

    This class sets its :attr:`expected_class_name` to ``"ExtractDataSetA"``,
    ensuring it is recognized in the YAML configuration as a valid
    extract class. It inherits the full feature extraction
    pipeline from :class:`ExtractFeatureBase`.
    """

    expected_class_name: str = "ExtractDataSetA"

    def __init__(
        self,
        config: DataSetConfig,
        input_data: Optional[pl.DataFrame] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
        target_rows: Optional[Dict[str, pl.DataFrame]] = None,
        summary_stats: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize the feature extraction workflow for BO NRT + Cora data.

        :param config: A dataset configuration object that manages paths,
                       target definitions, and parameters for feature extraction.
        :type config: DataSetConfig
        :param input_data: A Polars DataFrame containing all available data
                           for feature extraction, defaults to None.
        :type input_data: pl.DataFrame, optional
        :param selected_profiles: A Polars DataFrame containing specifically-selected
                                  profiles from the earlier steps, defaults to None.
        :type selected_profiles: pl.DataFrame, optional
        :param target_rows: A dictionary mapping each target to its respective
                            subset of rows for feature generation,
                            defaults to None.
        :type target_rows: dict of str to pl.DataFrame, optional
        :param summary_stats: A Polars DataFrame with summary statistics
                              that may guide scaling or normalization,
                              defaults to None.
        :type summary_stats: pl.DataFrame, optional
        """
        super().__init__(
            config,
            input_data=input_data,
            selected_profiles=selected_profiles,
            target_rows=target_rows,
            summary_stats=summary_stats,
        )
