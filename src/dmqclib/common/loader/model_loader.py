from typing import Optional, Type

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.common.base.model_base import ModelBase
from dmqclib.common.loader.model_registry import MODEL_REGISTRY


def load_model_class(config: ConfigBase) -> ModelBase:
    """
    Retrieve and instantiate a model class for the "model" step from the provided config.

    Steps:
      1. Fetch the class name from the config using config.get_base_class("model").
      2. Look up the corresponding class in MODEL_REGISTRY.
      3. Instantiate the class with the given config argument.

    :param config: A configuration object that includes a "base_class" entry
                   under the "model" step, specifying which model class to load.
    :type config: ConfigBase
    :return: An instantiated model object inheriting from ModelBase.
    :rtype: ModelBase
    :raises ValueError: If the retrieved model class name is not found in MODEL_REGISTRY.
    """
    class_name: str = config.get_base_class("model")
    model_class: Optional[Type[ModelBase]] = MODEL_REGISTRY.get(class_name)
    if not model_class:
        raise ValueError(f"Unknown model class specified: {class_name}")

    return model_class(config)
