from dmqclib.datasets.input.dataset_a import InputDataSetA
from dmqclib.datasets.summary.dataset_a import SummaryDataSetA
from dmqclib.datasets.select.dataset_a import SelectDataSetA
from dmqclib.datasets.locate.dataset_a import LocateDataSetA
from dmqclib.datasets.extract.dataset_a import ExtractDataSetA
from dmqclib.datasets.split.dataset_a import SplitDataSetA

INPUT_DATASET_REGISTRY = {
    "InputDataSetA": InputDataSetA,
}

SUMMARY_DATASET_REGISTRY = {
    "SummaryDataSetA": SummaryDataSetA,
}

SELECT_DATASET_REGISTRY = {
    "SelectDataSetA": SelectDataSetA,
}

LOCATE_DATASET_REGISTRY = {
    "LocateDataSetA": LocateDataSetA,
}

EXTRACT_DATASET_REGISTRY = {
    "ExtractDataSetA": ExtractDataSetA,
}

SPLIT_DATASET_REGISTRY = {
    "SplitDataSetA": SplitDataSetA,
}
