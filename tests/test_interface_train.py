import os
import shutil
import unittest
from pathlib import Path

from dmqclib.config.training_config import TrainingConfig
from dmqclib.interface.train import train_and_evaluate


class TestCreateTrainingDataSet(unittest.TestCase):
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
        train_and_evaluate(self.config)

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
