"""
Module providing registry dictionaries that map dataset class names (str) to their
corresponding Python classes. These registries enable dynamic loading of
the correct class during each preparation step in the pipeline.
"""

from typing import Dict, Type

from dmqclib.classify.step1_read_input.dataset_all import InputDataSetAll
from dmqclib.classify.step2_calc_stats.dataset_all import SummaryDataSetAll
from dmqclib.classify.step3_select_profiles.dataset_all import SelectDataSetAll
from dmqclib.classify.step4_select_rows.dataset_all import LocateDataSetAll
from dmqclib.classify.step5_extract_features.dataset_all import ExtractDataSetAll

from dmqclib.prepare.step1_read_input.input_base import InputDataSetBase
from dmqclib.prepare.step2_calc_stats.summary_base import SummaryStatsBase
from dmqclib.prepare.step3_select_profiles.select_base import ProfileSelectionBase
from dmqclib.prepare.step4_select_rows.locate_base import LocatePositionBase
from dmqclib.prepare.step5_extract_features.extract_base import ExtractFeatureBase

#: A registry mapping class names (used in YAML config) to the
#: actual Python classes for step1_input tasks.
INPUT_CLASSIFY_REGISTRY: Dict[str, Type[InputDataSetBase]] = {
    "InputDataSetAll": InputDataSetAll,
}

#: A registry mapping class names (used in YAML config) to the
#: actual Python classes for step2_summary tasks.
SUMMARY_CLASSIFY_REGISTRY: Dict[str, Type[SummaryStatsBase]] = {
    "SummaryDataSetAll": SummaryDataSetAll,
}

#: A registry mapping class names (used in YAML config) to the
#: actual Python classes for step3_select tasks.
SELECT_CLASSIFY_REGISTRY: Dict[str, Type[ProfileSelectionBase]] = {
    "SelectDataSetAll": SelectDataSetAll,
}

#: A registry mapping class names (used in YAML config) to the
#: actual Python classes for step4_locate tasks.
LOCATE_CLASSIFY_REGISTRY: Dict[str, Type[LocatePositionBase]] = {
    "LocateDataSetAll": LocateDataSetAll,
}

#: A registry mapping class names (used in YAML config) to the
#: actual Python classes for step5_extract tasks.
EXTRACT_CLASSIFY_REGISTRY: Dict[str, Type[ExtractFeatureBase]] = {
    "ExtractDataSetAll": ExtractDataSetAll,
}
