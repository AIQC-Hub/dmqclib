import os
import unittest
from pathlib import Path

import dmqclib as dm
from dmqclib.config.dataset_config import DataSetConfig
from dmqclib.config.training_config import TrainingConfig


class TestTemplateConfig(unittest.TestCase):
    def setUp(self):
        """Set up test environment and define test data paths."""
        self.ds_config_template_file = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "temp_dataset_template.yaml"
        )

        self.train_config_template_file = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "temp_training_template.yaml"
        )

    def test_ds_config_template(self):
        dm.write_config_template(self.ds_config_template_file, "prepare")
        self.assertTrue(os.path.exists(self.ds_config_template_file))
        os.remove(self.ds_config_template_file)

    def test_train_config_template(self):
        dm.write_config_template(self.train_config_template_file, "train")
        self.assertTrue(os.path.exists(self.train_config_template_file))
        os.remove(self.train_config_template_file)


class TestReadConfig(unittest.TestCase):
    def setUp(self):
        """Set up test environment and define test data paths."""
        self.ds_config_file = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_001.yaml"
        )

        self.train_config_file = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )

    def test_ds_config(self):
        config = dm.read_config(self.ds_config_file, "prepare")
        self.assertIsInstance(config, DataSetConfig)

    def test_train_config(self):
        config = dm.read_config(self.train_config_file, "train")
        self.assertIsInstance(config, TrainingConfig)
