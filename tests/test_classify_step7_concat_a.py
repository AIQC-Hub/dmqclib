import os
import unittest
from pathlib import Path

import polars as pl

from dmqclib.common.loader.classify_loader import (
    load_classify_step1_input_dataset,
    load_classify_step2_summary_dataset,
    load_classify_step3_select_dataset,
    load_classify_step4_locate_dataset,
    load_classify_step5_extract_dataset,
    load_classify_step6_classify_dataset
)
from dmqclib.common.config.classify_config import ClassificationConfig
from dmqclib.classify.step7_concat_datasets.dataset_all import ConcatDataSetAll


class TestConcatPredicitons(unittest.TestCase):
    """
    A suite of tests ensuring that building, testing, and saving XGBoost models
    via BuildModel follows the expected configuration and data flows.
    """

    def setUp(self):
        """Set up test environment and load input, summary, select, and locate data."""
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")

        model_path = Path(__file__).resolve().parent / "data" / "training"
        self.model_file_names = {
            "temp": model_path / "model_temp.joblib",
            "psal": model_path / "model_psal.joblib",
            "pres": model_path / "model_pres.joblib",
        }

        self.prediction_file_name = (
            Path(__file__).resolve().parent
            / "data"
            / "classify"
            / "temp_predictions.parquet"
        )

        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )
        self.ds_input = load_classify_step1_input_dataset(self.config)
        self.ds_input.input_file_name = str(self.test_data_file)
        self.ds_input.read_input_data()

        self.ds_summary = load_classify_step2_summary_dataset(
            self.config, input_data=self.ds_input.input_data
        )
        self.ds_summary.calculate_stats()

        self.ds_select = load_classify_step3_select_dataset(
            self.config, input_data=self.ds_input.input_data
        )
        self.ds_select.label_profiles()

        self.ds_locate = load_classify_step4_locate_dataset(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )
        self.ds_locate.process_targets()

        self.ds_extract = load_classify_step5_extract_dataset(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
            selected_rows=self.ds_locate.selected_rows,
            summary_stats=self.ds_summary.summary_stats,
        )
        self.ds_extract.process_targets()

        self.ds_classify = load_classify_step6_classify_dataset(
            self.config,
            self.ds_extract.target_features
        )
        self.ds_classify.model_file_names = self.model_file_names
        self.ds_classify.read_models()
        self.ds_classify.test_targets()

    def test_step_name(self):
        """Check that the ConcatDataSetAll step name is correctly assigned."""
        ds = ConcatDataSetAll(self.config)
        self.assertEqual(ds.step_name, "concat")

    def test_output_file_names(self):
        """Verify that default output file names (model and results) are as expected."""
        ds = ConcatDataSetAll(self.config)

        self.assertEqual(
            "/path/to/concat_1/nrt_bo_001/concat_folder_1/predictions.parquet",
            str(ds.output_file_name),
        )

    def test_test_sets(self):
        """Check that training and test sets are loaded into BuildModel correctly."""
        ds = ConcatDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            predictions=self.ds_classify.predictions
        )

        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 19480)
        self.assertEqual(ds.input_data.shape[1], 30)

        self.assertIsInstance(ds.predictions["temp"], pl.DataFrame)
        self.assertEqual(ds.predictions["temp"].shape[0], 19480)
        self.assertEqual(ds.predictions["temp"].shape[1], 6)

        self.assertIsInstance(ds.predictions["psal"], pl.DataFrame)
        self.assertEqual(ds.predictions["psal"].shape[0], 19480)
        self.assertEqual(ds.predictions["psal"].shape[1], 6)

    def test_merge_predictions(self):
        """Confirm that building models populates the 'models' dictionary with XGBoost instances."""
        ds = ConcatDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            predictions=self.ds_classify.predictions
        )
        ds.merge_predictions()

        self.assertIsInstance(ds.merged_predictions, pl.DataFrame)
        self.assertEqual(ds.merged_predictions.shape[0], 19480)
        self.assertEqual(ds.merged_predictions.shape[1], 36)


    def test_write_predictions(self):
        """Check that the test reports are correctly written to file."""
        ds = ConcatDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            predictions=self.ds_classify.predictions
        )

        data_path = Path(__file__).resolve().parent / "data" / "classify"
        ds.output_file_name = str(data_path / "temp_predictions.parquet")

        ds.merge_predictions()
        ds.write_merged_predictions()

        self.assertTrue(os.path.exists(ds.output_file_name))

        os.remove(ds.output_file_name)

    def test_write_no_results(self):
        """Ensure ValueError is raised if write_results is called with no results available."""
        ds = ConcatDataSetAll(
            self.config,
            input_data=self.ds_input.input_data,
            predictions=self.ds_classify.predictions
        )

        with self.assertRaises(ValueError):
            ds.write_merged_predictions()
