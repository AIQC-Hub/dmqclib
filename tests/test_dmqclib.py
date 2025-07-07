"""
This module contains integration tests that exercise various dmqclib functionality:
1) Creating dataset and training configuration templates;
2) Reading existing configuration files;
3) Creating a training dataset for Argo-based data (prepare workflow);
4) Training and evaluating models (train workflow).
"""

import os
import shutil
import unittest
from pathlib import Path

import dmqclib as dm
from dmqclib.common.config.dataset_config import DataSetConfig
from dmqclib.common.config.training_config import TrainingConfig


class TestDMQCLibTemplateConfig(unittest.TestCase):
    """
    Tests for creating dataset and training configuration templates
    using the dmqclib library.
    """

    def setUp(self):
        """
        Prepare file paths for dataset and training configuration templates
        that will be written and removed during testing.
        """
        self.ds_config_template_file = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "temp_dataset_template.yaml"
        )
        self.config_train_set_template_file = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "temp_training_template.yaml"
        )

    def test_ds_config_template(self):
        """
        Check that a dataset (prepare) configuration template can be written
        to the specified path and then removed.
        """
        dm.write_config_template(self.ds_config_template_file, "prepare")
        self.assertTrue(os.path.exists(self.ds_config_template_file))
        os.remove(self.ds_config_template_file)

    def test_config_train_set_template(self):
        """
        Check that a training configuration template can be written
        to the specified path and then removed.
        """
        dm.write_config_template(self.config_train_set_template_file, "train")
        self.assertTrue(os.path.exists(self.config_train_set_template_file))
        os.remove(self.config_train_set_template_file)


class TestDMQCLibReadConfig(unittest.TestCase):
    """
    Tests for reading dataset (prepare) and training (train) configuration files
    using the dmqclib library.
    """

    def setUp(self):
        """
        Define paths to existing dataset and training configuration files
        used in subsequent read tests.
        """
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
        """
        Verify that reading a dataset configuration returns a DataSetConfig instance.
        """
        config = dm.read_config(self.ds_config_file, "prepare")
        self.assertIsInstance(config, DataSetConfig)

    def test_train_config(self):
        """
        Verify that reading a training configuration returns a TrainingConfig instance.
        """
        config = dm.read_config(self.train_config_file, "train")
        self.assertIsInstance(config, TrainingConfig)


class TestDMQCLibCreateTrainingDataSet(unittest.TestCase):
    """
    Tests for creating a training dataset (prepare workflow)
    using dmqclib to ensure that all expected outputs are generated.
    """

    def setUp(self):
        """
        Load dataset configuration, specify file names and paths,
        and prepare test output directories.
        """
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
        """
        Use dm.create_training_dataset to run all 'prepare' steps,
        then confirm the expected files and folders exist.
        """
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
            os.path.exists(str(output_folder / "locate" / "selected_rows_temp.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "locate" / "selected_rows_psal.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "locate" / "selected_rows_pres.parquet"))
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "extract" / "extracted_features_temp.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "extract" / "extracted_features_psal.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "extract" / "extracted_features_pres.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "split" / "train_set_temp.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "split" / "train_set_psal.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "split" / "train_set_pres.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "split" / "test_set_temp.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "split" / "test_set_psal.parquet"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "split" / "test_set_pres.parquet"))
        )

        shutil.rmtree(output_folder)


class TestDMQCCreateTrainingDataSet(unittest.TestCase):
    """
    Tests for the training workflow, ensuring training and evaluation
    produce expected validation/test results and models.
    """

    def setUp(self):
        """
        Load a TrainingConfig, define the input and output paths,
        and prepare directories for the train-and-evaluate steps.
        """
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
        """
        Run dm.train_and_evaluate on the loaded config,
        then verify that validation results and model files are generated.
        """
        dm.train_and_evaluate(self.config)

        output_folder = (
            self.test_data_location / self.config.data["dataset_folder_name"]
        )

        self.assertTrue(
            os.path.exists(
                str(output_folder / "validate" / "validation_report_temp.tsv")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "validate" / "validation_report_psal.tsv")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "validate" / "validation_report_pres.tsv")
            )
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "build" / "test_report_temp.tsv"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "build" / "test_report_psal.tsv"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "build" / "test_report_pres.tsv"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "model" / "model_temp.joblib"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "model" / "model_psal.joblib"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "model" / "model_pres.joblib"))
        )

        shutil.rmtree(output_folder)
