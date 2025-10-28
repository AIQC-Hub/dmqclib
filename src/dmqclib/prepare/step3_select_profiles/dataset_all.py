"""Selects and labels oceanographic profiles based on QC flags.

This module defines the `SelectDataSetA` class, which is responsible for selecting
and labeling positive and negative oceanographic profiles from a given dataset.

It extends :class:`~.select_base.ProfileSelectionBase` to implement specific
criteria for identifying "bad" (positive) and "good" (negative) profiles
based on QC flags, and then pairs them temporally to construct a labeled dataset
suitable for quality control machine learning applications.
"""

import operator
from functools import reduce
from typing import Optional, List

import polars as pl

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.prepare.step3_select_profiles.select_base import ProfileSelectionBase


class SelectDataSetAll(ProfileSelectionBase):
    """Selects positive/negative profiles from Copernicus CTD data.

    This class implements a strategy for labeling oceanographic profiles as
    "positive" (bad) or "negative" (good) based on their quality control (QC)
    flags. The main steps are:

    1.  **Select Positive Profiles**: Identify profiles with at least one "bad"
        QC flag (e.g., a value of 4) in key sensor measurements.
    2.  **Select Negative Profiles**: Identify profiles where all measurements for
        all key sensors are "good" (e.g., a QC flag of 1).
    3.  **Combine Data**: Merge the labeled positive and negative profiles into a
        single DataFrame.

    :ivar expected_class_name: The expected name of the class, used for
        configuration validation.
    :vartype expected_class_name: str
    :ivar pos_profile_df: DataFrame containing positively-labeled profiles.
    :vartype pos_profile_df: Optional[polars.DataFrame]
    :ivar neg_profile_df: DataFrame containing negatively-labeled profiles.
    :vartype neg_profile_df: Optional[polars.DataFrame]
    :ivar key_col_names: Column names used as unique identifiers for profiles.
    :vartype key_col_names: List[str]
    """

    expected_class_name: str = "SelectDataSetAll"

    def __init__(
        self, config: ConfigBase, input_data: Optional[pl.DataFrame] = None
    ) -> None:
        """Initialize the selection and labeling process.

        :param config: The configuration object containing paths, parameters,
                       and QC flag definitions for the selection process.
        :type config: dmqclib.common.base.config_base.ConfigBase
        :param input_data: A Polars DataFrame containing the full set
                           of profiles from which to select examples. If None,
                           it is expected to be loaded by the base class.
        :type input_data: Optional[polars.DataFrame]
        """
        super().__init__(config=config, input_data=input_data)

        self.pos_profile_df: Optional[pl.DataFrame] = None
        self.neg_profile_df: Optional[pl.DataFrame] = None
        self.key_col_names: List[str] = [
            "platform_code",
            "profile_no",
            "profile_timestamp",
            "longitude",
            "latitude",
        ]

    def select_positive_profiles(self) -> None:
        """Select profiles with "bad" QC flags.

        A profile is considered "positive" (i.e., contains errors) if any of
        its measurements have a QC flag defined as a positive flag in the
        configuration (e.g., a flag of 4). The resulting unique profiles
        are stored in the :attr:`pos_profile_df` attribute.
        """
        conditions = reduce(
            operator.or_,
            [
                pl.col(param["flag"]).is_in(param.get("pos_flag_values", [4]))
                for param in self.config.get_target_dict().values()
            ],
        )

        self.pos_profile_df = (
            self.input_data.filter(conditions)
            .select(self.key_col_names)
            .unique()
            .sort(["platform_code", "profile_no"])
            .with_row_index("profile_id", offset=1)
            .with_columns(
                pl.lit(0, dtype=pl.UInt32).alias("neg_profile_id"),
                pl.lit(1, dtype=pl.UInt32).alias("label"),
            )
        )

    def select_negative_profiles(self) -> None:
        """Select profiles with consistently "good" QC flags.

        A profile is considered "negative" (i.e., contains only good data)
        if, for every monitored parameter (e.g., temperature, salinity),
        none of its measurements have a "bad" flag and at least one has a "good"
        flag. The resulting unique profiles are stored in the
        :attr:`neg_profile_df` attribute.
        """
        exprs = reduce(
            operator.and_,
            [
                (~pl.col(param["flag"]).is_in(param.get("pos_flag_values", [4])).any())
                & (pl.col(param["flag"]).is_in(param.get("neg_flag_values", [1])).any())
                for param in self.config.get_target_dict().values()
            ],
        )

        self.neg_profile_df = (
            self.input_data.filter(exprs.over(self.key_col_names))
            .select(self.key_col_names)
            .unique()
            .sort(["platform_code", "profile_no"])
            .with_row_index("profile_id", offset=self.pos_profile_df.shape[0] + 1)
            .with_columns(
                pl.lit(0, dtype=pl.UInt32).alias("neg_profile_id"),
                pl.lit(0, dtype=pl.UInt32).alias("label"),
            )
        )

    def label_profiles(self) -> None:
        """Execute the full profile selection and labeling workflow.

        This method orchestrates the process by calling, in order:

        1. :meth:`select_positive_profiles`
        2. :meth:`select_negative_profiles`

        The final combined DataFrame of labeled profiles is stored in the
        :attr:`selected_profiles` attribute of the base class.
        """
        self.select_positive_profiles()
        self.select_negative_profiles()

        self.selected_profiles = self.pos_profile_df.vstack(self.neg_profile_df)
