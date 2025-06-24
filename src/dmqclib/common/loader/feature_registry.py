from dmqclib.datasets.extract.feature.location import LocationFeat
from dmqclib.datasets.extract.feature.day_of_year import DayOfYearFeat
from dmqclib.datasets.extract.feature.profile_summary import ProfileSummaryStats5
from dmqclib.datasets.extract.feature.basic_values import BasicValues3PlusFlanks


FEATURE_REGISTRY = {
    "location": LocationFeat,
    "day_of_year": DayOfYearFeat,
    "profile_summary_stats5": ProfileSummaryStats5,
    "basic_values3_plus_flanks": BasicValues3PlusFlanks,
}
