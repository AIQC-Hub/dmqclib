import os
import unittest
from pathlib import Path

from dmqclib.config.dataset_config import DataSetConfig
from dmqclib.config.training_config import TrainingConfig
from dmqclib.interface.config import read_config
from dmqclib.interface.config import write_config_template


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
        write_config_template(self.ds_config_template_file, "prepare")
        self.assertTrue(os.path.exists(self.ds_config_template_file))
        os.remove(self.ds_config_template_file)

    def test_train_config_template(self):
        write_config_template(self.train_config_template_file, "train")
        self.assertTrue(os.path.exists(self.train_config_template_file))
        os.remove(self.train_config_template_file)

    def test_config_template_with_invalid_module(self):
        with self.assertRaises(ValueError):
            write_config_template(self.ds_config_template_file, "prepare2")

    def test_config_template_with_invalid_path(self):
        with self.assertRaises(IOError):
            write_config_template("/abc" + str(self.ds_config_template_file), "prepare")


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
        config = read_config(self.ds_config_file, "prepare")
        self.assertIsInstance(config, DataSetConfig)

    def test_train_config(self):
        config = read_config(self.train_config_file, "train")
        self.assertIsInstance(config, TrainingConfig)

    def test_config_with_invalid_module(self):
        with self.assertRaises(ValueError):
            _ = read_config(self.ds_config_file, "prepare2")

    def test_config_with_invalid_path(self):
        with self.assertRaises(IOError):
            _ = read_config(str(self.ds_config_file) + "zzz", "prepare")
