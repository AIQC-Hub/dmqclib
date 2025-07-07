"""
Module providing registry dictionaries that map dataset class names (str) to their
corresponding Python classes. These registries enable dynamic loading of
the correct class during each preparation step in the pipeline.
"""

from typing import Dict, Type

from dmqclib.prepare.step1_read_input.dataset_a import InputDataSetA
from dmqclib.prepare.step2_calc_stats.dataset_a import SummaryDataSetA
from dmqclib.prepare.step3_select_profiles.dataset_a import SelectDataSetA
from dmqclib.prepare.step4_select_rows.dataset_a import LocateDataSetA
from dmqclib.prepare.step5_extract_features.dataset_a import ExtractDataSetA
from dmqclib.prepare.step6_split_dataset.dataset_a import SplitDataSetA

from dmqclib.prepare.step1_read_input.input_base import InputDataSetBase
from dmqclib.prepare.step2_calc_stats.summary_base import SummaryStatsBase
from dmqclib.prepare.step3_select_profiles.select_base import ProfileSelectionBase
from dmqclib.prepare.step4_select_rows.locate_base import LocatePositionBase
from dmqclib.prepare.step5_extract_features.extract_base import ExtractFeatureBase
from dmqclib.prepare.step6_split_dataset.split_base import SplitDataSetBase

#: A registry mapping class names (used in YAML config) to the
#: actual Python classes for step1_input tasks.
INPUT_DATASET_REGISTRY: Dict[str, Type[InputDataSetBase]] = {
    "InputDataSetA": InputDataSetA,
}

#: A registry mapping class names (used in YAML config) to the
#: actual Python classes for step2_summary tasks.
SUMMARY_DATASET_REGISTRY: Dict[str, Type[SummaryStatsBase]] = {
    "SummaryDataSetA": SummaryDataSetA,
}

#: A registry mapping class names (used in YAML config) to the
#: actual Python classes for step3_select tasks.
SELECT_DATASET_REGISTRY: Dict[str, Type[ProfileSelectionBase]] = {
    "SelectDataSetA": SelectDataSetA,
}

#: A registry mapping class names (used in YAML config) to the
#: actual Python classes for step4_locate tasks.
LOCATE_DATASET_REGISTRY: Dict[str, Type[LocatePositionBase]] = {
    "LocateDataSetA": LocateDataSetA,
}

#: A registry mapping class names (used in YAML config) to the
#: actual Python classes for step5_extract tasks.
EXTRACT_DATASET_REGISTRY: Dict[str, Type[ExtractFeatureBase]] = {
    "ExtractDataSetA": ExtractDataSetA,
}

#: A registry mapping class names (used in YAML config) to the
#: actual Python classes for step6_split tasks.
SPLIT_DATASET_REGISTRY: Dict[str, Type[SplitDataSetBase]] = {
    "SplitDataSetA": SplitDataSetA,
}
