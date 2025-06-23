from abc import ABC, abstractmethod
import polars as pl


class FeatureBase(ABC):
    """
    Base class to extract features
    """

    def __init__(
        self,
        target_name: str = None,
        filtered_input: pl.DataFrame = None,
        target_rows: pl.DataFrame = None,
        summary_stats: pl.DataFrame = None,
        feature_info: pl.DataFrame = None,
    ):
        # Set member variables
        self.target_name = target_name
        self.filtered_input = filtered_input
        self.target_rows = target_rows
        self.summary_stats = summary_stats
        self.feature_info = feature_info
        self.features = {}

    @abstractmethod
    def extract_features(self):
        """
        Extract features.
        """
        pass
