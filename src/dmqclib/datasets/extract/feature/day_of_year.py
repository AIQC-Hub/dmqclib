import polars as pl
from dmqclib.datasets.base.feature_base import FeatureBase


class DayOfYearFeat(FeatureBase):
    """
    DayOfYearFeat extracts day features from BO NRT+Cora test data.
    """

    def __init__(
        self,
        target_name: str = None,
        selected_profiles: pl.DataFrame = None,
        filtered_input: pl.DataFrame = None,
        target_rows: pl.DataFrame = None,
        summary_stats: pl.DataFrame = None,
        feature_info: pl.DataFrame = None,
    ):
        super().__init__(
            target_name,
            selected_profiles,
            filtered_input,
            target_rows,
            summary_stats,
            feature_info,
        )

    def extract_features(self):
        """
        Extract features.
        """
        self.features = self.target_rows[self.target_name]

    def scale_first(self):
        """
        Extract features.
        """
        pass

    def scale_second(self):
        """
        Extract features.
        """
        pass
