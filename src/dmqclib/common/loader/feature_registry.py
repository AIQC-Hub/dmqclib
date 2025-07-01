from dmqclib.prepare.features.basic_values import BasicValues3PlusFlanks
from dmqclib.prepare.features.day_of_year import DayOfYearFeat
from dmqclib.prepare.features.location import LocationFeat
from dmqclib.prepare.features.profile_summary import ProfileSummaryStats5

FEATURE_REGISTRY = {
    "location": LocationFeat,
    "day_of_year": DayOfYearFeat,
    "profile_summary_stats5": ProfileSummaryStats5,
    "basic_values3_plus_flanks": BasicValues3PlusFlanks,
}
