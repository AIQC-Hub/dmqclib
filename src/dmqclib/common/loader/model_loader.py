from dmqclib.common.base.config_base import ConfigBase
from dmqclib.common.base.model_base import ModelBase
from dmqclib.common.loader.model_registry import MODEL_REGISTRY


def load_model_class(config: ConfigBase) -> ModelBase:
    """
    Given a label (e.g., 'NRT_BO_001'), look up the class specified in the
    YAML config and instantiate the appropriate class, returning it.
    """

    class_name = config.get_base_class("model")
    model_class = MODEL_REGISTRY.get(class_name)

    return model_class(config)
