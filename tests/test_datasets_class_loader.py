import unittest
from pathlib import Path
import polars as pl
from dmqclib.datasets.class_loader import load_input_dataset
from dmqclib.datasets.input.dataset_a import InputDataSetA
from dmqclib.datasets.class_loader import load_select_dataset
from dmqclib.datasets.select.dataset_a import SelectDataSetA


class TestInputClassLoader(unittest.TestCase):
    def setUp(self):
        """
        Called before each test method. We define the explicit path to
        the test data config file here for reuse.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent / "data" / "config" / "datasets.yaml"
        )

    def test_load_dataset_valid_label(self):
        """
        Test that load_dataset returns an instance of InputDataSetA for the known label.
        """
        ds = load_input_dataset("NRT_BO_001", str(self.config_file_path))
        self.assertIsInstance(ds, InputDataSetA)
        self.assertEqual(ds.dataset_name, "NRT_BO_001")

    def test_load_dataset_invalid_label(self):
        """
        Test that calling load_dataset with an invalid label raises a ValueError.
        """
        with self.assertRaises(ValueError):
            load_input_dataset("NON_EXISTENT_LABEL", str(self.config_file_path))


class TestSelectClassLoader(unittest.TestCase):
    def setUp(self):
        """
        Called before each test method. We define the explicit path to
        the test data config file here for reuse.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent / "data" / "config" / "datasets.yaml"
        )
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def test_load_dataset_valid_label(self):
        """
        Test that load_dataset returns an instance of SelectDataSetA for the known label.
        """
        ds = load_select_dataset("NRT_BO_001", str(self.config_file_path))
        self.assertIsInstance(ds, SelectDataSetA)
        self.assertEqual(ds.dataset_name, "NRT_BO_001")

    def test_load_dataset_input_data(self):
        """
        Test that load_dataset returns an instance of SelectDataSetA with correct input_data.
        """
        ds_input = load_input_dataset("NRT_BO_001", str(self.config_file_path))
        ds_input.input_file_name = str(self.test_data_file)
        ds_input.read_input_data()

        ds = load_select_dataset(
            "NRT_BO_001", str(self.config_file_path), ds_input.input_data
        )
        self.assertIsInstance(ds, SelectDataSetA)
        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 132342)
        self.assertEqual(ds.input_data.shape[1], 30)

    def test_load_dataset_invalid_label(self):
        """
        Test that calling load_dataset with an invalid label raises a ValueError.
        """
        with self.assertRaises(ValueError):
            load_select_dataset("NON_EXISTENT_LABEL", str(self.config_file_path))
