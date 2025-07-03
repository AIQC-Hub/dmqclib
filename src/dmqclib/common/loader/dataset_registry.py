"""
Module providing registry dictionaries that map dataset class names (str) to their
corresponding Python classes. These registries enable dynamic loading of
the correct class during each preparation step in the pipeline.
"""

from typing import Dict, Type

from dmqclib.prepare.step1_input.dataset_a import InputDataSetA
from dmqclib.prepare.step2_summary.dataset_a import SummaryDataSetA
from dmqclib.prepare.step3_select.dataset_a import SelectDataSetA
from dmqclib.prepare.step4_locate.dataset_a import LocateDataSetA
from dmqclib.prepare.step5_extract.dataset_a import ExtractDataSetA
from dmqclib.prepare.step6_split.dataset_a import SplitDataSetA


#: A registry mapping class names (used in YAML config) to the
#: actual Python classes for step1_input tasks.
INPUT_DATASET_REGISTRY: Dict[str, Type[InputDataSetA]] = {
    "InputDataSetA": InputDataSetA,
}

#: A registry mapping class names (used in YAML config) to the
#: actual Python classes for step2_summary tasks.
SUMMARY_DATASET_REGISTRY: Dict[str, Type[SummaryDataSetA]] = {
    "SummaryDataSetA": SummaryDataSetA,
}

#: A registry mapping class names (used in YAML config) to the
#: actual Python classes for step3_select tasks.
SELECT_DATASET_REGISTRY: Dict[str, Type[SelectDataSetA]] = {
    "SelectDataSetA": SelectDataSetA,
}

#: A registry mapping class names (used in YAML config) to the
#: actual Python classes for step4_locate tasks.
LOCATE_DATASET_REGISTRY: Dict[str, Type[LocateDataSetA]] = {
    "LocateDataSetA": LocateDataSetA,
}

#: A registry mapping class names (used in YAML config) to the
#: actual Python classes for step5_extract tasks.
EXTRACT_DATASET_REGISTRY: Dict[str, Type[ExtractDataSetA]] = {
    "ExtractDataSetA": ExtractDataSetA,
}

#: A registry mapping class names (used in YAML config) to the
#: actual Python classes for step6_split tasks.
SPLIT_DATASET_REGISTRY: Dict[str, Type[SplitDataSetA]] = {
    "SplitDataSetA": SplitDataSetA,
}
