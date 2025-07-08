"""
Unit tests for the XGBoost model class, verifying its integration with
the dmqclib configuration system.
"""

import unittest
from pathlib import Path

from dmqclib.common.config.training_config import TrainingConfig
from dmqclib.train.models.xgboost import XGBoost


class TestXGBoost(unittest.TestCase):
    """
    A suite of tests verifying basic XGBoost model setup and functionality
    through the dmqclib configuration system.
    """

    def setUp(self):
        """
        Define the path to the training configuration file
        and select the appropriate dataset prior to each test.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")

    def test_init_class(self):
        """
        Check that initializing an XGBoost object sets default values correctly.
        """
        ds = XGBoost(self.config)
        self.assertEqual(ds.k, 0)
