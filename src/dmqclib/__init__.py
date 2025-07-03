"""
This module serves as the primary interface for the dmqclib package,
exposing top-level functions for configuration handling, dataset preparation,
and training/evaluation routines. It also defines the package version
by querying the installed distribution metadata.

Re-Exported Functions:
  • read_config: Reads and parses a YAML configuration file.
  • write_config_template: Writes a YAML configuration template for either "prepare" or "train".
  • create_training_dataset: Orchestrates the creation of a training dataset via multiple steps.
  • train_and_evaluate: Executes the end-to-end training and evaluation process.

Attributes:
  __version__ (str): The package version, dynamically obtained from metadata.
"""

from importlib.metadata import version

from dmqclib.interface.config import read_config as read_config
from dmqclib.interface.config import write_config_template as write_config_template
from dmqclib.interface.prepare import create_training_dataset as create_training_dataset
from dmqclib.interface.train import train_and_evaluate as train_and_evaluate

__version__ = version("dmqclib")
