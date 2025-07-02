import unittest
from pathlib import Path

from dmqclib.config.training_config import TrainingConfig
from dmqclib.train.models.xgboost import XGBoost


class TestXGBoost(unittest.TestCase):
    def setUp(self):
        """
        Called before each test method. We define the explicit path to
        the test data config file here for reuse.
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
        """Ensure initialization works as expected."""
        ds = XGBoost(self.config)
        self.assertEqual(ds.k, 0)
