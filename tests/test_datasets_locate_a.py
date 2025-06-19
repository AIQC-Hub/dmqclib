import unittest
from pathlib import Path
import polars as pl
from dmqclib.datasets.input.dataset_a import InputDataSetA
from dmqclib.datasets.select.dataset_a import SelectDataSetA
from dmqclib.datasets.locate.dataset_a import LocateDataSetA


class TestLocateDataSetA(unittest.TestCase):
    def setUp(self):
        """
        Called before each test method. We define the explicit path to
        the test data config file here for reuse.
        """
        self.config_file_path = str(
            Path(__file__).resolve().parent / "data" / "config" / "datasets.yaml"
        )
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )
        self.ds_input = InputDataSetA("NRT_BO_001", str(self.config_file_path))
        self.ds_input.input_file_name = str(self.test_data_file)
        self.ds_input.read_input_data()

        self.ds_select = SelectDataSetA(
            "NRT_BO_001", str(self.config_file_path), self.ds_input.input_data
        )
        self.ds_select.label_profiles()

    def test_init_valid_label(self):
        """Test that we can properly construct a SelectDataSetA instance from the YAML."""
        ds = LocateDataSetA("NRT_BO_001", str(self.config_file_path))
        self.assertEqual(ds.dataset_name, "NRT_BO_001")

    def test_init_invalid_label(self):
        """Test that constructing SelectDataSetA with an invalid label raises ValueError."""
        with self.assertRaises(ValueError):
            LocateDataSetA("NON_EXISTENT_LABEL", str(self.config_file_path))

    def test_config_file(self):
        """Test that config file is properly set in the corresponding member variable"""
        ds = LocateDataSetA("NRT_BO_001", str(self.config_file_path))
        self.assertTrue("datasets.yaml" in ds.config_file_name)

    def test_output_file_names(self):
        """Test that config file is properly set in the corresponding member variable"""
        ds = LocateDataSetA("NRT_BO_001", str(self.config_file_path))
        self.assertEqual(
            "/path/to/data/nrt_bo_001/select/temp_positions.parquet",
            str(ds.output_file_names["temp"])
        )
        self.assertEqual(
            "/path/to/data/nrt_bo_001/select/psal_positions.parquet",
            str(ds.output_file_names["psal"])
        )

    def test_no_input_file_name(self):
        """Test that config file is properly set in the corresponding member variable"""
        with self.assertRaises(ValueError):
            _ = LocateDataSetA("NRT_BO_002", str(self.config_file_path))

    def test_input_data_and_selected_profiles(self):
        """
        Tests that data is read correctly when the dataset_config specifies the file type explicitly.
        """
        ds = LocateDataSetA(
            "NRT_BO_001", str(self.config_file_path), self.ds_input.input_data, self.ds_select.selected_profiles
        )

        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 132342)
        self.assertEqual(ds.input_data.shape[1], 30)

        self.assertIsInstance(ds.selected_profiles, pl.DataFrame)
        self.assertEqual(ds.selected_profiles.shape[0], 44)
        self.assertEqual(ds.selected_profiles.shape[1], 8)
