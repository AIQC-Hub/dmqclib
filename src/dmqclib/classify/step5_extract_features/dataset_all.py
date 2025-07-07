import polars as pl
from typing import Optional, Dict

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.prepare.step5_extract_features.extract_base import ExtractFeatureBase


class ExtractDataSetAll(ExtractFeatureBase):
    """
    A subclass of :class:`ExtractFeatureBase` for extracting features
    from BO NRT + Cora test data.

    This class sets its :attr:`expected_class_name` to ``ExtractDataSetAll`` so
    that it matches the relevant YAML configuration. All feature extraction logic
    inherits from the parent class, :class:`ExtractFeatureBase`.
    """

    expected_class_name: str = "ExtractDataSetAll"

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
        target_rows: Optional[Dict[str, pl.DataFrame]] = None,
        summary_stats: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize the feature extraction process for BO NRT + Cora test data.

        :param config: A configuration object that manages paths, target definitions,
                       and parameters for feature extraction.
        :type config: ConfigBase
        :param input_data: An optional Polars DataFrame containing the complete dataset
                           for feature extraction. If not provided at initialization,
                           it should be assigned later.
        :type input_data: pl.DataFrame, optional
        :param selected_profiles: An optional Polars DataFrame of selected profiles
                                  from earlier steps. If not provided, it should be
                                  assigned later.
        :type selected_profiles: pl.DataFrame, optional
        :param target_rows: An optional dictionary mapping target names to respective
                            DataFrames containing the rows needed for feature generation.
                            If not provided, it should be assigned later.
        :type target_rows: dict of str to pl.DataFrame, optional
        :param summary_stats: An optional Polars DataFrame with summary statistics,
                              potentially used for scaling or normalization.
                              If not provided, it should be assigned later.
        :type summary_stats: pl.DataFrame, optional
        """
        super().__init__(
            config,
            input_data=input_data,
            selected_profiles=selected_profiles,
            target_rows=target_rows,
            summary_stats=summary_stats,
        )

        #: Default file naming pattern when writing feature files for each target.
        self.default_file_name: str = "extracted_features_classify_{target_name}.parquet"

        #: Dictionary mapping target names to the corresponding Parquet file paths.
        self.output_file_names: Dict[str, str] = self.config.get_target_file_names(
            "extract", self.default_file_name
        )

        #: Column names used for intermediate or reference purposes
        #: (e.g., linking positive and negative rows).
        self.work_col_names = [
            "profile_id",
            "pair_id",
            "platform_code",
            "profile_no",
            "observation_no",
        ]
