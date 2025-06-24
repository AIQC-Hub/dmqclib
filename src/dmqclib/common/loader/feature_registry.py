from dmqclib.datasets.step5_extract.feature.location import LocationFeat
from dmqclib.datasets.step5_extract.feature.day_of_year import DayOfYearFeat
from dmqclib.datasets.step5_extract.feature.profile_summary import ProfileSummaryStats5
from dmqclib.datasets.step5_extract.feature.basic_values import BasicValues3PlusFlanks


FEATURE_REGISTRY = {
    "location": LocationFeat,
    "day_of_year": DayOfYearFeat,
    "profile_summary_stats5": ProfileSummaryStats5,
    "basic_values3_plus_flanks": BasicValues3PlusFlanks,
}
