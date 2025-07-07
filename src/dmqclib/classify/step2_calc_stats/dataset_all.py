import polars as pl

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.prepare.step2_calc_stats.summary_base import SummaryStatsBase


class SummaryDataSetAll(SummaryStatsBase):
    """
    Subclass of :class:`SummaryStatsBase` for calculating summary statistics
    for BO NRT + Cora test data using Polars.

    Sets :attr:`expected_class_name` to ``SummaryDataSetAll`` to match
    the relevant YAML configuration.
    """

    expected_class_name: str = "SummaryDataSetAll"

    def __init__(self, config: ConfigBase, input_data: pl.DataFrame = None) -> None:
        """
        Initialize SummaryDataSetAll with the provided configuration and optional data.

        :param config: Configuration object containing paths and parameters for
                       generating summary statistics.
        :type config: ConfigBase
        :param input_data: Optional Polars DataFrame that can be used to
                           calculate the summary statistics. If not provided,
                           it should be assigned later before calling
                           statistic-related methods.
         :type input_data: pl.DataFrame, optional
        """
        super().__init__(config, input_data=input_data)

        #: Default output file name for summary statistics; can be overridden if necessary.
        self.default_file_name: str = "summary_stats_classify.tsv"

        #: The resolved absolute path for writing the summary statistics file,
        #: based on the configuration and self.default_file_name.
        self.output_file_name: str = self.config.get_full_file_name(
            "summary", self.default_file_name
        )
