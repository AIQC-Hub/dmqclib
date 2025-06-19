import unittest
from pathlib import Path
import polars as pl
from dmqclib.datasets.input.dataset_a import InputDataSetA
from dmqclib.datasets.select.dataset_a import SelectDataSetA


class TestSelectDataSetA(unittest.TestCase):
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
        self.ds = InputDataSetA("NRT_BO_001", str(self.config_file_path))
        self.ds.input_file_name = str(self.test_data_file)
        self.ds.read_input_data()

    def test_init_valid_label(self):
        """Test that we can properly construct a SelectDataSetA instance from the YAML."""
        ds = SelectDataSetA("NRT_BO_001", str(self.config_file_path))
        self.assertEqual(ds.dataset_name, "NRT_BO_001")

    def test_init_invalid_label(self):
        """Test that constructing SelectDataSetA with an invalid label raises ValueError."""
        with self.assertRaises(ValueError):
            SelectDataSetA("NON_EXISTENT_LABEL", str(self.config_file_path))

    def test_config_file(self):
        """Test that config file is properly set in the corresponding member variable"""
        ds = SelectDataSetA("NRT_BO_001", str(self.config_file_path))
        self.assertTrue("datasets.yaml" in ds.config_file_name)

    def test_output_file_name(self):
        """Test that config file is properly set in the corresponding member variable"""
        ds = SelectDataSetA("NRT_BO_001", str(self.config_file_path))
        self.assertEqual(
            "/path/to/data/nrt_bo_001/select/selected_profiles.parquet",
            str(ds.output_file_name),
        )

    def test_no_input_file_name(self):
        """Test that config file is properly set in the corresponding member variable"""
        with self.assertRaises(ValueError):
            _ = SelectDataSetA("NRT_BO_002", str(self.config_file_path))

    def test_input_data(self):
        """
        Tests that data is read correctly when the dataset_config specifies the file type explicitly.
        """
        ds = SelectDataSetA(
            "NRT_BO_001", str(self.config_file_path), self.ds.input_data
        )

        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 132342)
        self.assertEqual(ds.input_data.shape[1], 30)

    def test_positive_profiles(self):
        """
        Tests that data is read correctly when the dataset_config specifies the file type explicitly.
        """
        ds = SelectDataSetA(
            "NRT_BO_001", str(self.config_file_path), self.ds.input_data
        )
        ds.select_positive_profiles()

        self.assertIsInstance(ds.pos_profile_df, pl.DataFrame)
        self.assertEqual(ds.pos_profile_df.shape[0], 25)
        self.assertEqual(ds.pos_profile_df.shape[1], 7)

    def test_negative_profiles(self):
        """
        Tests that data is read correctly when the dataset_config specifies the file type explicitly.
        """
        ds = SelectDataSetA(
            "NRT_BO_001", str(self.config_file_path), self.ds.input_data
        )
        ds.select_positive_profiles()
        ds.select_negative_profiles()

        self.assertIsInstance(ds.neg_profile_df, pl.DataFrame)
        self.assertEqual(ds.neg_profile_df.shape[0], 478)
        self.assertEqual(ds.neg_profile_df.shape[1], 7)

    def test_find_profile_pairs(self):
        """
        Tests that data is read correctly when the dataset_config specifies the file type explicitly.
        """
        ds = SelectDataSetA(
            "NRT_BO_001", str(self.config_file_path), self.ds.input_data
        )
        ds.select_positive_profiles()
        ds.select_negative_profiles()
        ds.find_profile_pairs()

        self.assertEqual(ds.pos_profile_df.shape[0], 25)
        self.assertEqual(ds.pos_profile_df.shape[1], 8)
        self.assertEqual(ds.neg_profile_df.shape[0], 19)
        self.assertEqual(ds.neg_profile_df.shape[1], 8)

    def test_label_profiles(self):
        """
        Tests that data is read correctly when the dataset_config specifies the file type explicitly.
        """
        ds = SelectDataSetA(
            "NRT_BO_001", str(self.config_file_path), self.ds.input_data
        )
        ds.label_profiles()

        self.assertEqual(ds.selected_profiles.shape[0], 44)
        self.assertEqual(ds.selected_profiles.shape[1], 8)
