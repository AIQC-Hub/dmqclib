"""
This module contains unit tests for the `ClassifyAll` class,
focusing on its functionality for loading, testing, and saving
XGBoost models and their results within the classification workflow.
"""

import os
import pytest
from pathlib import Path

import polars as pl

from dmqclib.classify.step6_classify_dataset.dataset_all import ClassifyAll
from dmqclib.common.config.classify_config import ClassificationConfig
from dmqclib.common.loader.classify_loader import (
    load_classify_step1_input_dataset,
    load_classify_step2_summary_dataset,
    load_classify_step3_select_dataset,
    load_classify_step4_locate_dataset,
    load_classify_step5_extract_dataset,
)
from dmqclib.train.models.xgboost import XGBoost


class TestClassifyAllClass:
    """
    A suite of tests ensuring that the `ClassifyAll` step correctly loads models,
    tests them against input data, and saves classification reports and predictions.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment and load input, summary, select, and locate data."""
        config_path = Path(__file__).resolve().parent / "data" / "config"
        self.config_file = config_path / "test_classify_001.yaml"
        self.config = ClassificationConfig(self.config_file)
        self.config.select("NRT_BO_001")

        self.config_file_3 = config_path / "test_classify_003.yaml"
        self.config_3 = ClassificationConfig(self.config_file_3)
        self.config_3.select("NRT_BO_001")

    def test_step_name(self):
        """Check that the ClassifyAll step name is correctly assigned."""
        ds = ClassifyAll(self.config)
        assert ds.step_name == "classify"

    def test_output_file_names(self):
        """Verify that default output file names (model and results) are as expected."""
        ds = ClassifyAll(self.config)

        file_model = "/path/to/model_1/model_folder_1/model_{}.joblib"
        file_classify = (
            "/path/to/classify_1/nrt_bo_001/classify_folder_1/classify_report_{}.tsv"
        )
        assert file_model.format("temp") == str(ds.model_file_names["temp"])
        assert file_model.format("psal") == str(ds.model_file_names["psal"])
        assert file_model.format("pres") == str(ds.model_file_names["pres"])

        assert file_classify.format("temp") == str(
            ds.output_file_names["report"]["temp"]
        )
        assert file_classify.format("psal") == str(
            ds.output_file_names["report"]["psal"]
        )
        assert file_classify.format("pres") == str(
            ds.output_file_names["report"]["pres"]
        )

    def test_base_model(self):
        """Ensure that the configured base model is an XGBoost instance."""
        ds = ClassifyAll(self.config)
        assert isinstance(ds.base_model, XGBoost)

    def test_nthreads(self):
        ds = ClassifyAll(self.config)
        assert ds.base_model.model_params["n_jobs"] == -1

        ds_3 = ClassifyAll(self.config_3)
        assert ds_3.base_model.model_params["n_jobs"] == 2


class TestClassifyAll:
    """
    A suite of tests ensuring that the `ClassifyAll` step correctly loads models,
    tests them against input data, and saves classification reports and predictions.
    """

    def _setup_datasets(self):
        self.configs = []
        self.extracts = []
        for x in self.config_file_paths:
            config = ClassificationConfig(x)
            config.select("NRT_BO_001")

            ds_input = load_classify_step1_input_dataset(config)
            ds_input.input_file_name = str(self.test_data_file)
            ds_input.read_input_data()

            ds_summary = load_classify_step2_summary_dataset(
                config, input_data=ds_input.input_data
            )
            ds_summary.calculate_stats()

            ds_select = load_classify_step3_select_dataset(
                config, input_data=ds_input.input_data
            )
            ds_select.label_profiles()

            ds_locate = load_classify_step4_locate_dataset(
                config,
                input_data=ds_input.input_data,
                selected_profiles=ds_select.selected_profiles,
            )
            ds_locate.process_targets()

            ds_extract = load_classify_step5_extract_dataset(
                config,
                input_data=ds_input.input_data,
                selected_profiles=ds_select.selected_profiles,
                selected_rows=ds_locate.selected_rows,
                summary_stats=ds_summary.summary_stats,
            )
            ds_extract.process_targets()

            self.configs.append(config)
            self.extracts.append(ds_extract)

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment and load input, summary, select, and locate data."""
        config_path = Path(__file__).resolve().parent / "data" / "config"
        self.config_file_paths = [
            config_path / "test_classify_001.yaml",
            config_path / "test_classify_002.yaml",
            config_path / "test_classify_003.yaml",
        ]
        self.test_data_file = str(
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )
        self._setup_datasets()

        model_path = Path(__file__).resolve().parent / "data" / "training"
        self.model_file_names = {
            "temp": str(model_path / "model_temp.joblib"),
            "psal": str(model_path / "model_psal.joblib"),
            "pres": str(model_path / "model_pres.joblib"),
        }

        data_path = Path(__file__).resolve().parent / "data" / "classify"
        self.report_file_names = {
            "temp": str(data_path / "temp_classify_report_temp.tsv"),
            "psal": str(data_path / "temp_classify_report_psal.tsv"),
            "pres": str(data_path / "temp_classify_report_pres.tsv"),
        }
        self.prediction_file_names = {
            "temp": str(data_path / "temp_classify_prediction_temp.parquet"),
            "psal": str(data_path / "temp_classify_prediction_psal.parquet"),
            "pres": str(data_path / "temp_classify_prediction_pres.parquet"),
        }
        self.n_jobs = [-1, -1, 2]

    @pytest.mark.parametrize("idx", range(3))
    def test_test_sets(self, idx):
        """Check that test sets are loaded into ClassifyAll correctly."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )

        assert isinstance(ds.test_sets["temp"], pl.DataFrame)
        assert ds.test_sets["temp"].shape[0] == 19480
        assert ds.test_sets["temp"].shape[1] == 56

        assert isinstance(ds.test_sets["psal"], pl.DataFrame)
        assert ds.test_sets["psal"].shape[0] == 19480
        assert ds.test_sets["psal"].shape[1] == 56

        assert isinstance(ds.test_sets["pres"], pl.DataFrame)
        assert ds.test_sets["pres"].shape[0] == 19480
        assert ds.test_sets["pres"].shape[1] == 56

    @pytest.mark.parametrize("idx", range(3))
    def test_read_models(self, idx):
        """Confirm that reading models populates the 'models' dictionary with XGBoost instances."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()

        assert isinstance(ds.models["temp"], XGBoost)
        assert isinstance(ds.models["psal"], XGBoost)
        assert isinstance(ds.models["pres"], XGBoost)

    @pytest.mark.parametrize("idx", range(3))
    def test_with_xgboost(self, idx):
        """Check that testing targets after model loading populates the result columns."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()
        ds.test_targets()

        assert isinstance(ds.test_sets["temp"], pl.DataFrame)
        assert ds.test_sets["temp"].shape[0] == 19480
        assert ds.test_sets["temp"].shape[1] == 56

        assert isinstance(ds.test_sets["psal"], pl.DataFrame)
        assert ds.test_sets["psal"].shape[0] == 19480
        assert ds.test_sets["psal"].shape[1] == 56

        assert isinstance(ds.test_sets["pres"], pl.DataFrame)
        assert ds.test_sets["pres"].shape[0] == 19480
        assert ds.test_sets["pres"].shape[1] == 56

    @pytest.mark.parametrize("idx", range(3))
    def test_without_model(self, idx):
        """Ensure that testing without loaded models raises a ValueError."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        with pytest.raises(ValueError):
            ds.test_targets()

    @pytest.mark.parametrize("idx", range(3))
    def test_write_reports(self, idx):
        """Verify that test reports are correctly written to file."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.output_file_names["report"] = self.report_file_names
        ds.read_models()
        ds.test_targets()
        ds.write_reports()

        assert os.path.exists(ds.output_file_names["report"]["temp"])
        assert os.path.exists(ds.output_file_names["report"]["psal"])
        assert os.path.exists(ds.output_file_names["report"]["pres"])

        os.remove(ds.output_file_names["report"]["temp"])
        os.remove(ds.output_file_names["report"]["psal"])
        os.remove(ds.output_file_names["report"]["pres"])

    @pytest.mark.parametrize("idx", range(3))
    def test_write_no_results(self, idx):
        """Ensure ValueError is raised if write_reports is called without test results."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        with pytest.raises(ValueError):
            ds.write_reports()

    @pytest.mark.parametrize("idx", range(3))
    def test_read_models_no_file(self, idx):
        """Check that FileNotFoundError is raised if model files are missing during loading."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        data_path = Path(__file__).resolve().parent / "data" / "training"
        ds.model_file_names["temp"] = str(data_path / "model.joblib")
        ds.model_file_names["psal"] = str(data_path / "model.joblib")
        ds.model_file_names["pres"] = str(data_path / "model.joblib")

        with pytest.raises(FileNotFoundError):
            ds.read_models()

    @pytest.mark.parametrize("idx", range(3))
    def test_n_jobs(self, idx):
        """Confirm that reading models populates the 'models' dictionary with XGBoost instances."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.read_models()

        assert ds.models["temp"].model_params["n_jobs"] == self.n_jobs[idx]
        assert ds.models["psal"].model_params["n_jobs"] == self.n_jobs[idx]
        assert ds.models["pres"].model_params["n_jobs"] == self.n_jobs[idx]

        assert ds.models["temp"].model.n_jobs == self.n_jobs[idx]
        assert ds.models["psal"].model.n_jobs == self.n_jobs[idx]
        assert ds.models["pres"].model.n_jobs == self.n_jobs[idx]

    @pytest.mark.parametrize("idx", range(3))
    def test_write_predictions(self, idx):
        """Verify that test predictions are correctly written to file."""
        ds = ClassifyAll(
            self.configs[idx],
            test_sets=self.extracts[idx].target_features,
        )
        ds.model_file_names = self.model_file_names
        ds.output_file_names["prediction"] = self.prediction_file_names
        ds.read_models()
        ds.test_targets()
        ds.write_predictions()

        assert os.path.exists(ds.output_file_names["prediction"]["temp"])
        assert os.path.exists(ds.output_file_names["prediction"]["psal"])
        assert os.path.exists(ds.output_file_names["prediction"]["pres"])

        os.remove(ds.output_file_names["prediction"]["temp"])
        os.remove(ds.output_file_names["prediction"]["psal"])
        os.remove(ds.output_file_names["prediction"]["pres"])
