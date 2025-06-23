import polars as pl
from dmqclib.datasets.base.feature_base import FeatureBase


class ProfileSummaryStats5(FeatureBase):
    """
    ProfileSummaryStats5 extracts profile summary features from BO NRT+Cora test data.
    """

    def __init__(
        self,
        target_name: str = None,
        filtered_input: pl.DataFrame = None,
        target_rows: pl.DataFrame = None,
        summary_stats: pl.DataFrame = None,
        feature_info: pl.DataFrame = None,
    ):
        super().__init__(
            target_name,
            filtered_input,
            target_rows,
            summary_stats,
            feature_info,
        )

    def extract_features(self):
        """
        Extract features.
        """
        pass
