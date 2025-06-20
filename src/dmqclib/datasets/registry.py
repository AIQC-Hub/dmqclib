from dmqclib.datasets.input.dataset_a import InputDataSetA
from dmqclib.datasets.select.dataset_a import SelectDataSetA
from dmqclib.datasets.locate.dataset_a import LocateDataSetA
from dmqclib.datasets.extract.dataset_a import ExtractDataSetA

INPUT_DATASET_REGISTRY = {
    "InputDataSetA": InputDataSetA,
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
