from dmqclib.utils.config import read_config
from dmqclib.datasets.registry import INPUT_DATASET_REGISTRY


def load_input_dataset(label: str, config_file: str = None):
    """
    Given a label (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """
    data = read_config(config_file, "datasets.yaml")

    dataset_config = data.get(label)
    if dataset_config is None:
        raise ValueError(f"No dataset configuration found for label '{label}'")

    class_name = dataset_config.get("input_class")
    dataset_class = INPUT_DATASET_REGISTRY.get(class_name)
    if not dataset_class:
        raise ValueError(f"Unknown dataset class specified: {class_name}")

    return dataset_class(label, config_file=config_file)
