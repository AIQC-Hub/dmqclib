import os
import shutil
import unittest
from pathlib import Path

from dmqclib.config.dataset_config import DataSetConfig
from dmqclib.interface.prepare import create_training_dataset


class TestCreateTrainingDataSet(unittest.TestCase):
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
            "name": "data_set_1",
            "common": {"base_path": str(self.test_data_location)},
            "input": {"base_path": str(self.input_data_path), "step_folder_name": ""},
        }

    def test_create_training_data_set(self):
        create_training_dataset(self.config)

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
