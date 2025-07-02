import polars as pl

from dmqclib.config.dataset_config import DataSetConfig
from dmqclib.prepare.step5_extract.extract_base import ExtractFeatureBase


class ExtractDataSetA(ExtractFeatureBase):
    """
    ExtractDataSetA extracts features from BO NRT+Cora test data.
    """

    expected_class_name = "ExtractDataSetA"

    def __init__(
        self,
        config: DataSetConfig,
        input_data: pl.DataFrame = None,
        selected_profiles: pl.DataFrame = None,
        target_rows: pl.DataFrame = None,
        summary_stats: pl.DataFrame = None,
    ):
        super().__init__(
            config,
            input_data=input_data,
            selected_profiles=selected_profiles,
            target_rows=target_rows,
            summary_stats=summary_stats,
        )
