import unittest
from pathlib import Path
import polars as pl
from dmqclib.datasets.class_loader.dataset_loader import load_input_dataset
from dmqclib.datasets.class_loader.dataset_loader import load_summary_dataset
from dmqclib.datasets.class_loader.dataset_loader import load_select_dataset
from dmqclib.datasets.class_loader.dataset_loader import load_locate_dataset
from dmqclib.datasets.class_loader.dataset_loader import load_extract_dataset
from dmqclib.datasets.extract.feature.location import LocationFeat


class TestFeatureLocation(unittest.TestCase):
    def setUp(self):
        """Set up test environment and load input and selected datasets."""
        self.config_file_path = str(
            Path(__file__).resolve().parent / "data" / "config" / "datasets.yaml"
        )
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )
        self.ds_input = load_input_dataset("NRT_BO_001", str(self.config_file_path))
        self.ds_input.input_file_name = str(self.test_data_file)
        self.ds_input.read_input_data()

        self.ds_summary = load_summary_dataset(
            "NRT_BO_001", str(self.config_file_path), self.ds_input.input_data
        )
        self.ds_summary.calculate_stats()

        self.ds_select = load_select_dataset(
            "NRT_BO_001", str(self.config_file_path), self.ds_input.input_data
        )
        self.ds_select.label_profiles()

        self.ds_locate = load_locate_dataset(
            "NRT_BO_001",
            str(self.config_file_path),
            self.ds_input.input_data,
            self.ds_select.selected_profiles,
        )
        self.ds_locate.process_targets()

        self.ds_extract = load_extract_dataset(
            "NRT_BO_001",
            str(self.config_file_path),
            self.ds_input.input_data,
            self.ds_select.selected_profiles,
            self.ds_locate.target_rows,
            self.ds_summary.summary_stats,
        )

    def test_init_arguments(self):
        """Ensure input data and selected profiles are read correctly."""
        feature_info = {
            "class": "location",
            "scales": {
                "longitude": {"min": 0, "max": 1},
                "latitude": {"min": 0, "max": 1},
            },
        }
        ds = LocationFeat(
            "temp",
            feature_info,
            self.ds_select.selected_profiles,
            self.ds_extract.filtered_input,
            self.ds_locate.target_rows,
            self.ds_summary.summary_stats,
        )

        self.assertIsInstance(ds.selected_profiles, pl.DataFrame)
        self.assertEqual(ds.selected_profiles.shape[0], 44)
        self.assertEqual(ds.selected_profiles.shape[1], 8)

        self.assertIsInstance(ds.filtered_input, pl.DataFrame)
        self.assertEqual(ds.filtered_input.shape[0], 9841)
        self.assertEqual(ds.filtered_input.shape[1], 30)

        self.assertIsInstance(ds.target_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.target_rows["temp"].shape[0], 128)
        self.assertEqual(ds.target_rows["temp"].shape[1], 11)

        self.assertIsInstance(ds.target_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.target_rows["psal"].shape[0], 140)
        self.assertEqual(ds.target_rows["psal"].shape[1], 11)

        self.assertIsInstance(ds.summary_stats, pl.DataFrame)
        self.assertEqual(ds.summary_stats.shape[0], 3528)
        self.assertEqual(ds.summary_stats.shape[1], 12)

    def test_location_features(self):
        """Ensure input data and selected profiles are read correctly."""
        feature_info = {
            "class": "location",
            "scales": {
                "longitude": {"min": 0, "max": 1},
                "latitude": {"min": 0, "max": 1},
            },
        }
        ds = LocationFeat(
            "temp",
            feature_info,
            self.ds_select.selected_profiles,
            self.ds_extract.filtered_input,
            self.ds_locate.target_rows,
            self.ds_summary.summary_stats,
        )

        ds.extract_features()

        self.assertIsInstance(ds.features, pl.DataFrame)
        self.assertEqual(ds.features.shape[0], 128)
        self.assertEqual(ds.features.shape[1], 3)


class TestProfileSummary(unittest.TestCase):
    def setUp(self):
        """Set up test environment and load input and selected datasets."""
        self.config_file_path = str(
            Path(__file__).resolve().parent / "data" / "config" / "datasets.yaml"
        )
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )
        self.ds_input = load_input_dataset("NRT_BO_001", str(self.config_file_path))
        self.ds_input.input_file_name = str(self.test_data_file)
        self.ds_input.read_input_data()

        self.ds_summary = load_summary_dataset(
            "NRT_BO_001", str(self.config_file_path), self.ds_input.input_data
        )
        self.ds_summary.calculate_stats()

        self.ds_select = load_select_dataset(
            "NRT_BO_001", str(self.config_file_path), self.ds_input.input_data
        )
        self.ds_select.label_profiles()

        self.ds_locate = load_locate_dataset(
            "NRT_BO_001",
            str(self.config_file_path),
            self.ds_input.input_data,
            self.ds_select.selected_profiles,
        )
        self.ds_locate.process_targets()

        self.ds_extract = load_extract_dataset(
            "NRT_BO_001",
            str(self.config_file_path),
            self.ds_input.input_data,
            self.ds_select.selected_profiles,
            self.ds_locate.target_rows,
            self.ds_summary.summary_stats,
        )

    def test_init_arguments(self):
        """Ensure input data and selected profiles are read correctly."""
        feature_info = {
            "class": "location",
            "scales": {
                "longitude": {"min": 0, "max": 1},
                "latitude": {"min": 0, "max": 1},
            },
        }
        ds = LocationFeat(
            "temp",
            feature_info,
            self.ds_select.selected_profiles,
            self.ds_extract.filtered_input,
            self.ds_locate.target_rows,
            self.ds_summary.summary_stats,
        )

        self.assertIsInstance(ds.selected_profiles, pl.DataFrame)
        self.assertEqual(ds.selected_profiles.shape[0], 44)
        self.assertEqual(ds.selected_profiles.shape[1], 8)

        self.assertIsInstance(ds.filtered_input, pl.DataFrame)
        self.assertEqual(ds.filtered_input.shape[0], 9841)
        self.assertEqual(ds.filtered_input.shape[1], 30)

        self.assertIsInstance(ds.target_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.target_rows["temp"].shape[0], 128)
        self.assertEqual(ds.target_rows["temp"].shape[1], 11)

        self.assertIsInstance(ds.target_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.target_rows["psal"].shape[0], 140)
        self.assertEqual(ds.target_rows["psal"].shape[1], 11)

        self.assertIsInstance(ds.summary_stats, pl.DataFrame)
        self.assertEqual(ds.summary_stats.shape[0], 3528)
        self.assertEqual(ds.summary_stats.shape[1], 12)

    def test_location_features(self):
        """Ensure input data and selected profiles are read correctly."""
        feature_info = {
            "class": "location",
            "scales": {
                "longitude": {"min": 0, "max": 1},
                "latitude": {"min": 0, "max": 1},
            },
        }
        ds = LocationFeat(
            "temp",
            feature_info,
            self.ds_select.selected_profiles,
            self.ds_extract.filtered_input,
            self.ds_locate.target_rows,
            self.ds_summary.summary_stats,
        )

        ds.extract_features()

        self.assertIsInstance(ds.features, pl.DataFrame)
        self.assertEqual(ds.features.shape[0], 128)
        self.assertEqual(ds.features.shape[1], 3)
