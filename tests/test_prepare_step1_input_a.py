import unittest
from pathlib import Path

import polars as pl

from dmqclib.config.dataset_config import DataSetConfig
from dmqclib.prepare.step1_input.dataset_a import InputDataSetA


class TestInputDataSetA(unittest.TestCase):
    def setUp(self):
        """Set up test environment and define test data paths."""
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_001.yaml"
        )
        self.config = DataSetConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def _get_input_data(self, file_type=None, read_file_options=None):
        """Helper to load input data with optional file type and options."""
        ds = InputDataSetA(self.config)
        ds.input_file_name = str(self.test_data_file)

        if file_type is not None:
            ds.config.data["step_param_set"]["steps"]["input"]["file_type"] = file_type

        if read_file_options is not None:
            ds.config.data["step_param_set"]["steps"]["input"]["read_file_options"] = (
                read_file_options
            )

        ds.read_input_data()
        return ds.input_data

    def test_step_name(self):
        """Ensure the step name is set correctly."""
        ds = InputDataSetA(self.config)
        self.assertEqual(ds.step_name, "input")

    def test_input_file_name(self):
        """Ensure the input file name is set correctly."""
        ds = InputDataSetA(self.config)
        self.assertEqual(
            "/path/to/input_1/input_folder_1/nrt_cora_bo_test.parquet",
            str(ds.input_file_name),
        )

    def test_read_input_data_with_explicit_type(self):
        """Ensure data is read correctly with explicit file type."""
        df = self._get_input_data(file_type="parquet", read_file_options={})

        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 132342)
        self.assertEqual(df.shape[1], 30)

    def test_read_input_data_infer_type(self):
        """Ensure data is read correctly with inferred file type."""
        df = self._get_input_data(file_type=None, read_file_options={})

        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 132342)
        self.assertEqual(df.shape[1], 30)

    def test_read_input_data_missing_options(self):
        """Ensure reading works with missing 'input_file_options' key."""
        df = self._get_input_data(file_type="parquet", read_file_options=None)

        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 132342)
        self.assertEqual(df.shape[1], 30)

    def test_read_input_data_file_not_found(self):
        """Ensure FileNotFoundError is raised for non-existent files."""
        ds = InputDataSetA(self.config)
        ds.input_file_name = str(self.test_data_file) + "_not_found"

        with self.assertRaises(FileNotFoundError):
            ds.read_input_data()

    def test_read_input_data_with_extra_options(self):
        """Ensure extra options can be passed while reading data."""
        df = self._get_input_data(
            file_type="parquet", read_file_options={"n_rows": 100}
        )

        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(df.shape[1], 30)
