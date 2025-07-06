from typing import Dict, Optional

import polars as pl

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.prepare.step4_locate.locate_base import LocatePositionBase


class LocateDataSetAll(LocatePositionBase):
    """
    A subclass of :class:`LocatePositionBase` that locates all rows
    from BO NRT + Cora test data for training or evaluation purposes.

    This class assigns a default file naming scheme for target rows
    and uses configuration details (e.g., QC flags) to identify
    relevant data rows for each target.
    """

    expected_class_name: str = "LocateDataSetAll"

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
        selected_profiles: Optional[pl.DataFrame] = None,
    ) -> None:
        """
        Initialize the dataset with configuration, an optional input DataFrame,
        and an optional DataFrame of selected profiles.

        :param config: A configuration object specifying paths, parameters,
                       and target definitions for locating test data rows.
        :type config: ConfigBase
        :param input_data: An optional Polars DataFrame containing the full data
                           from which positive and negative rows will be derived.
                           If not provided, it should be set later.
        :type input_data: pl.DataFrame, optional
        :param selected_profiles: An optional Polars DataFrame containing profiles
                                  labeled as positive or negative. If not provided,
                                  it should be set later.
        :type selected_profiles: pl.DataFrame, optional
        """
        super().__init__(
            config, input_data=input_data, selected_profiles=selected_profiles
        )

        #: Default file name template for writing target rows (one file per target).
        self.default_file_name: str = "{target_name}_classify_rows.parquet"

        #: Dictionary mapping each target name to the corresponding output Parquet file path.
        self.output_file_names: Dict[str, str] = self.config.get_target_file_names(
            "locate", self.default_file_name
        )

    def select_all_rows(self, target_name: str, target_value: Dict) -> None:
        """
        Collect all rows for a specified target by applying
        flag-based labeling to each record.

        :param target_name: The name (key) of the target in the
                            configuration's target dictionary.
        :type target_name: str
        :param target_value: A dictionary of target metadata,
                             including the relevant QC flag variable name.
        :type target_value: Dict
        """
        flag_var_name = target_value["flag"]
        self.target_rows[target_name] = (
            self.input_data.with_row_index("row_id", offset=1)
            .with_columns(
                pl.lit(0, dtype=pl.UInt32).alias("profile_id"),
                pl.lit("").alias("pair_id"),
                pl.when(pl.col(flag_var_name).is_in([4]))
                .then(1)
                .otherwise(0)
                .alias("label"),
            )
            .select(
                pl.col("row_id"),
                pl.col("profile_id"),
                pl.col("platform_code"),
                pl.col("profile_no"),
                pl.col("observation_no"),
                pl.col("pres"),
                pl.col(flag_var_name).alias("flag"),
                pl.col("label"),
                pl.col("pair_id"),
            )
        )

    def locate_target_rows(self, target_name: str, target_value: Dict) -> None:
        """
        Locate target rows for training or evaluation.

        :param target_name: Name of the target variable.
        :param target_value: A dictionary of target metadata, including
                             the QC flag variable name used for labeling.
        """
        self.select_all_rows(target_name, target_value)
