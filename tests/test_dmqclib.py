import os
import shutil
import unittest
from pathlib import Path

import dmqclib as dm
from dmqclib.config.dataset_config import DataSetConfig
from dmqclib.config.training_config import TrainingConfig


class TestDMQCLibTemplateConfig(unittest.TestCase):
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


class TestDMQCLibReadConfig(unittest.TestCase):
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


class TestDMQCLibCreateTrainingDataSet(unittest.TestCase):
    def setUp(self):
        """Set up test environment and define test data paths."""
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_001.yaml"
        )
        self.config = DataSetConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.config.data["input_file_name"] = "nrt_cora_bo_test.parquet"
        self.test_data_location = Path(__file__).resolve().parent / "data" / "test"
        self.input_data_path = Path(__file__).resolve().parent / "data" / "input"
        self.config.data["path_info"] = {
            "name": "nrt_bo_001",
            "common": {"base_path": str(self.test_data_location)},
            "input": {"base_path": str(self.input_data_path), "step_folder_name": ""},
        }

    def test_create_training_data_set(self):
        dm.create_training_dataset(self.config)

        output_folder = (
            self.test_data_location / self.config.data["dataset_folder_name"]
        )

        self.assertTrue(
            os.path.exists(str(output_folder / "summary" / "summary_stats.tsv"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "select" / "selected_profiles.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "locate" / "temp_rows.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "locate" / "psal_rows.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "locate" / "pres_rows.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "extract" / "temp_features.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "extract" / "psal_features.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "extract" / "pres_features.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "split" / "temp_train.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "split" / "psal_train.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "split" / "pres_train.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "split" / "temp_test.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "split" / "psal_test.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "split" / "pres_test.parquet"))
        )

        shutil.rmtree(output_folder)


class TestDMQCCreateTrainingDataSet(unittest.TestCase):
    def setUp(self):
        """Set up test environment and define test data paths."""
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_training_001.yaml"
        )
        self.config = TrainingConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_location = Path(__file__).resolve().parent / "data" / "test"
        self.input_data_path = Path(__file__).resolve().parent / "data" / "training"
        self.config.data["path_info"] = {
            "name": "data_set_1",
            "common": {"base_path": str(self.test_data_location)},
            "input": {
                "base_path": str(self.input_data_path),
                "step_folder_name": "..",
            },
        }

    def test_train_and_evaluate(self):
        dm.train_and_evaluate(self.config)

        output_folder = (
            self.test_data_location / self.config.data["dataset_folder_name"]
        )

        self.assertTrue(
            os.path.exists(
                str(output_folder / "validate" / "temp_validation_result.tsv")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "validate" / "psal_validation_result.tsv")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "validate" / "pres_validation_result.tsv")
            )
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "build" / "temp_test_result.tsv"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "build" / "psal_test_result.tsv"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "build" / "pres_test_result.tsv"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "build" / "temp_model.joblib"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "build" / "psal_model.joblib"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "build" / "pres_model.joblib"))
        )

        shutil.rmtree(output_folder)
