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
)
from dmqclib.common.config.classify_config import ClassificationConfig
from dmqclib.train.models.xgboost import XGBoost
from dmqclib.classify.step6_classify.classify_dataset_all import ClassifyAll


class TestBuildModel(unittest.TestCase):
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
            "temp": model_path / "temp_model.joblib",
            "psal": model_path / "psal_model.joblib",
            "pres": model_path / "pres_model.joblib",
        }

        data_path = Path(__file__).resolve().parent / "data" / "classify"
        self.report_file_names = {
            "temp": data_path / "temp_temp_classify_report.tsv",
            "psal": data_path / "temp_psal_classify_report.tsv",
            "pres": data_path / "temp_pres_classify_report.tsv",
        }

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
            target_rows=self.ds_locate.target_rows,
            summary_stats=self.ds_summary.summary_stats,
        )
        self.ds_extract.process_targets()

    def test_step_name(self):
        """Check that the ClassifyAll step name is correctly assigned."""
        ds = ClassifyAll(self.config)
        self.assertEqual(ds.step_name, "classify")

    def test_output_file_names(self):
        """Verify that default output file names (model and results) are as expected."""
        ds = ClassifyAll(self.config)

        self.assertEqual(
            "/path/to/model_1/nrt_bo_001/model_folder_1/temp_model.joblib",
            str(ds.model_file_names["temp"]),
        )
        self.assertEqual(
            "/path/to/model_1/nrt_bo_001/model_folder_1/psal_model.joblib",
            str(ds.model_file_names["psal"]),
        )

        self.assertEqual(
            "/path/to/classify_1/nrt_bo_001/classify_folder_1/temp_classify_report.tsv",
            str(ds.output_file_names["report"]["temp"]),
        )
        self.assertEqual(
            "/path/to/classify_1/nrt_bo_001/classify_folder_1/psal_classify_report.tsv",
            str(ds.output_file_names["report"]["psal"]),
        )

    def test_base_model(self):
        """Ensure that the configured base model is an XGBoost instance."""
        ds = ClassifyAll(self.config)
        self.assertIsInstance(ds.base_model, XGBoost)

    def test_test_sets(self):
        """Check that training and test sets are loaded into BuildModel correctly."""
        ds = ClassifyAll(
            self.config,
            test_sets=self.ds_extract.target_features,
        )

        self.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.test_sets["temp"].shape[0], 19480)
        self.assertEqual(ds.test_sets["temp"].shape[1], 37)

        self.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.test_sets["psal"].shape[0], 19480)
        self.assertEqual(ds.test_sets["psal"].shape[1], 37)

    def test_read_models(self):
        """Confirm that building models populates the 'models' dictionary with XGBoost instances."""
        ds = ClassifyAll(
            self.config,
            test_sets=self.ds_extract.target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()

        self.assertIsInstance(ds.models["temp"], XGBoost)
        self.assertIsInstance(ds.models["psal"], XGBoost)
        self.assertIsInstance(ds.models["pres"], XGBoost)

    def test_test_with_xgboost(self):
        """Check that testing sets after model building populates the result columns."""
        ds = ClassifyAll(
            self.config,
            test_sets=self.ds_extract.target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()
        ds.test_targets()

        self.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.test_sets["temp"].shape[0], 19480)
        self.assertEqual(ds.test_sets["temp"].shape[1], 37)

        self.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.test_sets["psal"].shape[0], 19480)
        self.assertEqual(ds.test_sets["psal"].shape[1], 37)

    def test_test_without_model(self):
        """Ensure that testing without building models raises a ValueError."""
        ds = ClassifyAll(
            self.config,
            test_sets=self.ds_extract.target_features,
        )
        with self.assertRaises(ValueError):
            ds.test_targets()

    def test_write_reports(self):
        """Check that the test reports are correctly written to file."""
        ds = ClassifyAll(
            self.config,
            test_sets=self.ds_extract.target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.output_file_names["report"] = self.report_file_names
        ds.read_models()
        ds.test_targets()
        ds.write_reports()

        self.assertTrue(os.path.exists(ds.output_file_names["report"]["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["report"]["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["report"]["pres"]))

        os.remove(ds.output_file_names["report"]["temp"])
        os.remove(ds.output_file_names["report"]["psal"])
        os.remove(ds.output_file_names["report"]["pres"])

    def test_write_no_results(self):
        """Ensure ValueError is raised if write_results is called with no results available."""
        ds = ClassifyAll(
            self.config,
            test_sets=self.ds_extract.target_features,
        )
        with self.assertRaises(ValueError):
            ds.write_reports()

    def test_read_models_no_file(self):
        """Check that FileNotFoundError is raised if model files are missing."""
        ds = ClassifyAll(
            self.config,
            test_sets=self.ds_extract.target_features,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["temp"] = data_path / "model.joblib"
        ds.model_file_names["psal"] = data_path / "model.joblib"
        ds.model_file_names["pres"] = data_path / "model.joblib"

        with self.assertRaises(FileNotFoundError):
            ds.read_models()
