import unittest
from pathlib import Path

from dmqclib.config.training_config import TrainingConfig
from dmqclib.training.models.xgboost import XGBoost


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

    def test_init_valid_dataset_name(self):
        """Ensure ExtractDataSetA constructs correctly with a valid label."""
        ds = XGBoost("NRT_BO_001", self.config)
        self.assertEqual(ds.dataset_name, "NRT_BO_001")

    def test_init_invalid_dataset_name(self):
        """Ensure ValueError is raised for an invalid label."""
        with self.assertRaises(ValueError):
            XGBoost("NON_EXISTENT_LABEL", self.config)
