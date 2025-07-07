import os
import shutil
import unittest
from pathlib import Path

from dmqclib.common.config.classify_config import ClassificationConfig
from dmqclib.interface.classify import classify_dataset


class TestClassifyDataSet(unittest.TestCase):
    """
    Tests for verifying that classify_dataset produces the
    expected directory structure and output files for classification processes.
    """

    def setUp(self):
        """
        Prepare the test environment by creating a DataSetConfig object,
        defining file paths, and updating the configuration with test input
        and output paths.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.config.data["input_file_name"] = "nrt_cora_bo_test.parquet"
        self.test_data_location = Path(__file__).resolve().parent / "data" / "test"
        self.data_path = Path(__file__).resolve().parent / "data"
        self.input_data_path = self.data_path / "input"
        self.config.data["path_info"] = {
            "name": "data_set_1",
            "common": {"base_path": str(self.test_data_location)},
            "input": {"base_path": str(self.input_data_path), "step_folder_name": ""},
            "model": {"base_path": str(self.data_path), "step_folder_name": "training"},
            "concat": {"step_folder_name": "classify"},
        }

    def test_classify_data_set(self):
        """
        Check that classify_dataset generates the expected folder
        hierarchy and files for summary, select, locate, extract, classify, and concat steps.
        """
        classify_dataset(self.config)

        output_folder = (
            self.test_data_location / self.config.data["dataset_folder_name"]
        )

        self.assertTrue(
            os.path.exists(
                str(output_folder / "summary" / "summary_stats_classify.tsv")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "select" / "selected_profiles_classify.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "locate" / "selected_rows_classify_temp.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "locate" / "selected_rows_classify_psal.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "locate" / "selected_rows_classify_pres.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(
                    output_folder
                    / "extract"
                    / "extracted_features_classify_temp.parquet"
                )
            )
        )
        self.assertTrue(
            os.path.exists(
                str(
                    output_folder
                    / "extract"
                    / "extracted_features_classify_psal.parquet"
                )
            )
        )
        self.assertTrue(
            os.path.exists(
                str(
                    output_folder
                    / "extract"
                    / "extracted_features_classify_pres.parquet"
                )
            )
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "classify" / "classify_prediction_temp.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "classify" / "classify_prediction_psal.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(
                str(output_folder / "classify" / "classify_prediction_pres.parquet")
            )
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "classify" / "classify_report_temp.tsv"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "classify" / "classify_report_psal.tsv"))
        )
        self.assertTrue(
            os.path.exists(str(output_folder / "classify" / "classify_report_pres.tsv"))
        )

        self.assertTrue(
            os.path.exists(str(output_folder / "classify" / "predictions.parquet"))
        )

        shutil.rmtree(output_folder)
