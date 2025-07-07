"""
This module defines a registry of feature classes for preparing datasets in
the pipeline. Each key is a string identifier (e.g., "location") used to
reference the corresponding feature-extraction class.
"""

from typing import Dict, Type

from dmqclib.prepare.features.basic_values import BasicValues3PlusFlanks
from dmqclib.prepare.features.day_of_year import DayOfYearFeat
from dmqclib.prepare.features.location import LocationFeat
from dmqclib.prepare.features.profile_summary import ProfileSummaryStats5

from dmqclib.common.base.feature_base import FeatureBase

#: A dictionary mapping feature identifiers (str) to classes that inherit
#: from :class:`FeatureBase`. These classes are dynamically loaded based
#: on the "feature" key in a feature configuration dictionary.
FEATURE_REGISTRY: Dict[str, Type[FeatureBase]] = {
    "location": LocationFeat,
    "day_of_year": DayOfYearFeat,
    "profile_summary_stats5": ProfileSummaryStats5,
    "basic_values3_plus_flanks": BasicValues3PlusFlanks,
}
