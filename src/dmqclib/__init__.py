from importlib.metadata import version

from dmqclib.interface.config import write_config_template as write_config_template
from dmqclib.interface.config import read_config as read_config
from dmqclib.interface.prepare import create_training_dataset as create_training_dataset

__version__ = version("dmqclib")
