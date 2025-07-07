import polars as pl

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.prepare.step2_calc_stats.summary_base import SummaryStatsBase


class SummaryDataSetA(SummaryStatsBase):
    """
    Subclass of SummaryStatsBase for calculating summary statistics
    for BO NRT + Cora test data using Polars.

    This class sets the :attr:`expected_class_name` to match
    the YAML configuration.
    """

    expected_class_name: str = "SummaryDataSetA"

    def __init__(self, config: ConfigBase, input_data: pl.DataFrame = None) -> None:
        """
        Initialize SummaryDataSetA with configuration and optional input data.

        :param config: The dataset configuration object containing paths
                       and parameters for summary stats.
        :type config: ConfigBase
        :param input_data: A Polars DataFrame from which to calculate summary stats.
                           If None, data must be assigned later.
        :type input_data: pl.DataFrame, optional
        """
        super().__init__(config, input_data=input_data)
