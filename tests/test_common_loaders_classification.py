import unittest
from pathlib import Path

import polars as pl

from dmqclib.common.config.classify_config import ClassificationConfig
from dmqclib.common.loader.classify_loader import (
    load_classify_step1_input_dataset,
    load_classify_step2_summary_dataset,
    load_classify_step3_select_dataset,
    load_classify_step4_locate_dataset,
    load_classify_step5_extract_dataset,
    load_classify_step6_classify_dataset,
)

from dmqclib.classify.step1_read_input.dataset_all import InputDataSetAll
from dmqclib.classify.step2_calc_stats.dataset_all import SummaryDataSetAll
from dmqclib.classify.step3_select_profiles.dataset_all import SelectDataSetAll
from dmqclib.classify.step4_select_rows.dataset_all import LocateDataSetAll
from dmqclib.classify.step5_extract_features.dataset_all import ExtractDataSetAll
from dmqclib.classify.step6_classify_dataset.dataset_all import ClassifyAll


class TestClassifyInputClassLoader(unittest.TestCase):
    """
    Tests related to loading the InputDataSetA class.
    """

    def setUp(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")

    def test_load_dataset_valid_config(self):
        """
        Check that load_step1_input_dataset returns an InputDataSetA instance with
        the expected step name.
        """
        ds = load_classify_step1_input_dataset(self.config)
        self.assertIsInstance(ds, InputDataSetAll)
        self.assertEqual(ds.step_name, "input")


class TestClassifySummaryClassLoader(unittest.TestCase):
    """
    Tests related to loading the SummaryDataSetA class.
    """

    def setUp(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def test_load_dataset_valid_config(self):
        """
        Check that load_step2_summary_dataset returns a SummaryDataSetA instance
        with the correct step name.
        """
        ds = load_classify_step2_summary_dataset(self.config)
        self.assertIsInstance(ds, SummaryDataSetAll)
        self.assertEqual(ds.step_name, "summary")

    def test_load_dataset_input_data(self):
        """
        Check that load_step2_summary_dataset sets input_data properly
        when provided.
        """
        ds_input = load_classify_step1_input_dataset(self.config)
        ds_input.input_file_name = str(self.test_data_file)
        ds_input.read_input_data()

        ds = load_classify_step2_summary_dataset(self.config, ds_input.input_data)
        self.assertIsInstance(ds, SummaryDataSetAll)
        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 19480)
        self.assertEqual(ds.input_data.shape[1], 30)


class TestClassifySelectClassLoader(unittest.TestCase):
    """
    Tests related to loading the SelectDataSetA class.
    """

    def setUp(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def test_load_dataset_valid_config(self):
        """
        Check that load_step3_select_dataset returns a SelectDataSetA instance
        with the correct step name.
        """
        ds = load_classify_step3_select_dataset(self.config)
        self.assertIsInstance(ds, SelectDataSetAll)
        self.assertEqual(ds.step_name, "select")

    def test_load_dataset_input_data(self):
        """
        Check that load_step3_select_dataset sets input_data properly
        when provided.
        """
        ds_input = load_classify_step1_input_dataset(self.config)
        ds_input.input_file_name = str(self.test_data_file)
        ds_input.read_input_data()

        ds = load_classify_step3_select_dataset(self.config, ds_input.input_data)
        self.assertIsInstance(ds, SelectDataSetAll)
        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 19480)
        self.assertEqual(ds.input_data.shape[1], 30)


class TestClassifyLocateClassLoader(unittest.TestCase):
    """
    Tests related to loading the LocateDataSetA class.
    """

    def setUp(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def test_load_dataset_valid_config(self):
        """
        Check that load_step4_locate_dataset returns a LocateDataSetA instance
        with the correct step name.
        """
        ds = load_classify_step4_locate_dataset(self.config)
        self.assertIsInstance(ds, LocateDataSetAll)
        self.assertEqual(ds.step_name, "locate")

    def test_load_dataset_input_data_and_profiles(self):
        """
        Check that load_step4_locate_dataset sets input_data and selected_profiles
        properly when provided.
        """
        ds_input = load_classify_step1_input_dataset(self.config)
        ds_input.input_file_name = str(self.test_data_file)
        ds_input.read_input_data()

        ds_select = load_classify_step3_select_dataset(self.config, ds_input.input_data)
        ds_select.label_profiles()

        ds = load_classify_step4_locate_dataset(
            self.config, ds_input.input_data, ds_select.selected_profiles
        )

        self.assertIsInstance(ds, LocateDataSetAll)

        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 19480)
        self.assertEqual(ds.input_data.shape[1], 30)

        self.assertIsInstance(ds.selected_profiles, pl.DataFrame)
        self.assertEqual(ds.selected_profiles.shape[0], 84)
        self.assertEqual(ds.selected_profiles.shape[1], 8)


class TestClassifyExtractClassLoader(unittest.TestCase):
    """
    Tests related to loading the ExtractDataSetA class.
    """

    def setUp(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def test_load_dataset_valid_config(self):
        """
        Check that load_step5_extract_dataset returns an ExtractDataSetA instance
        with the correct step name.
        """
        ds = load_classify_step5_extract_dataset(self.config)
        self.assertIsInstance(ds, ExtractDataSetAll)
        self.assertEqual(ds.step_name, "extract")

    def test_load_dataset_input_data_and_profiles(self):
        """
        Check that load_step5_extract_dataset sets input_data, selected_profiles,
        selected_rows, summary_stats, and filtered_input properly when provided.
        """
        ds_input = load_classify_step1_input_dataset(self.config)
        ds_input.input_file_name = str(self.test_data_file)
        ds_input.read_input_data()

        ds_select = load_classify_step3_select_dataset(self.config, ds_input.input_data)
        ds_select.label_profiles()

        ds_summary = load_classify_step2_summary_dataset(self.config, ds_input.input_data)
        ds_summary.calculate_stats()

        ds_locate = load_classify_step4_locate_dataset(
            self.config, ds_input.input_data, ds_select.selected_profiles
        )
        ds_locate.process_targets()

        ds = load_classify_step5_extract_dataset(
            self.config,
            ds_input.input_data,
            ds_select.selected_profiles,
            ds_locate.selected_rows,
            ds_summary.summary_stats,
        )

        self.assertIsInstance(ds, ExtractDataSetAll)

        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 19480)
        self.assertEqual(ds.input_data.shape[1], 30)

        self.assertIsInstance(ds.summary_stats, pl.DataFrame)
        self.assertEqual(ds.summary_stats.shape[0], 595)
        self.assertEqual(ds.summary_stats.shape[1], 12)

        self.assertIsInstance(ds.selected_profiles, pl.DataFrame)
        self.assertEqual(ds.selected_profiles.shape[0], 84)
        self.assertEqual(ds.selected_profiles.shape[1], 8)

        self.assertIsInstance(ds.filtered_input, pl.DataFrame)
        self.assertEqual(ds.filtered_input.shape[0], 19480)
        self.assertEqual(ds.filtered_input.shape[1], 30)

        self.assertIsInstance(ds.selected_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["temp"].shape[0], 19480)
        self.assertEqual(ds.selected_rows["temp"].shape[1], 9)

        self.assertIsInstance(ds.selected_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.selected_rows["psal"].shape[0], 19480)
        self.assertEqual(ds.selected_rows["psal"].shape[1], 9)


class TestClassifyClassifyClassLoader(unittest.TestCase):
    """
    Tests related to loading the SplitDataSetA class.
    """

    def setUp(self):
        """
        Define the path to the test config file and select a dataset
        prior to each test.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_classify_001.yaml"
        )
        self.config = ClassificationConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def test_load_dataset_valid_config(self):
        """
        Check that load_step6_split_dataset returns a ClassifyAll instance
        with the correct step name.
        """
        ds = load_classify_step6_classify_dataset(self.config)
        self.assertIsInstance(ds, ClassifyAll)
        self.assertEqual(ds.step_name, "classify")

    def test_load_dataset_input_data(self):
        """
        Check that load_step6_split_dataset sets target_features properly
        when provided, after all prior steps.
        """
        ds_input = load_classify_step1_input_dataset(self.config)
        ds_input.input_file_name = str(self.test_data_file)
        ds_input.read_input_data()

        ds_select = load_classify_step3_select_dataset(self.config, ds_input.input_data)
        ds_select.label_profiles()

        ds_summary = load_classify_step2_summary_dataset(self.config, ds_input.input_data)
        ds_summary.calculate_stats()

        ds_locate = load_classify_step4_locate_dataset(
            self.config, ds_input.input_data, ds_select.selected_profiles
        )
        ds_locate.process_targets()

        ds_extract = load_classify_step5_extract_dataset(
            self.config,
            ds_input.input_data,
            ds_select.selected_profiles,
            ds_locate.selected_rows,
            ds_summary.summary_stats,
        )
        ds_extract.process_targets()

        ds = load_classify_step6_classify_dataset(self.config, ds_extract.target_features)

        self.assertIsInstance(ds, ClassifyAll)

        self.assertIsInstance(ds.test_sets["temp"], pl.DataFrame)
        self.assertEqual(ds.test_sets["temp"].shape[0], 19480)
        self.assertEqual(ds.test_sets["temp"].shape[1], 41)

        self.assertIsInstance(ds.test_sets["psal"], pl.DataFrame)
        self.assertEqual(ds.test_sets["psal"].shape[0], 19480)
        self.assertEqual(ds.test_sets["psal"].shape[1], 41)
