import unittest
from pathlib import Path
import polars as pl
from dmqclib.datasets.input.dataset_a import InputDataSetA


class TestInputDataSetA(unittest.TestCase):
    def setUp(self):
        """
        Called before each test method. We define the explicit path to
        the test data config file here for reuse.
        """
        self.explicit_config_file_path = (
            Path(__file__).resolve().parent / "data" / "config" / "datasets.yaml"
        )
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def _get_input_data(self, file_type=None, options=None):
        ds = InputDataSetA("NRT_BO_001", str(self.explicit_config_file_path))
        ds.input_file_name = str(self.test_data_file)

        if file_type is not None:
            ds.dataset_info["input"]["file_type"] = file_type

        if options is not None:
            ds.dataset_info["input"]["options"] = options

        ds.read_input_data()

        return ds.input_data

    def test_init_valid_label(self):
        """Test that we can properly construct a InputDataSetA instance from the YAML."""
        ds = InputDataSetA("NRT_BO_001", str(self.explicit_config_file_path))
        self.assertEqual(ds.dataset_name, "NRT_BO_001")

    def test_init_invalid_label(self):
        """Test that constructing InputDataSetA with an invalid label raises ValueError."""
        with self.assertRaises(ValueError):
            InputDataSetA("NON_EXISTENT_LABEL", str(self.explicit_config_file_path))

    def test_config_file(self):
        """Test that config file is properly set in the corresponding member variable"""
        ds = InputDataSetA("NRT_BO_001", str(self.explicit_config_file_path))
        self.assertTrue("datasets.yaml" in ds.config_file_name)

    def test_input_file_name(self):
        """Test that config file is properly set in the corresponding member variable"""
        ds = InputDataSetA("NRT_BO_001", str(self.explicit_config_file_path))
        self.assertEqual(
            "/path/to/data/input/nrt_cora_bo_test.parquet", str(ds.input_file_name)
        )

    def test_no_input_file_name(self):
        """Test that config file is properly set in the corresponding member variable"""
        with self.assertRaises(ValueError):
            _ = InputDataSetA("NRT_BO_002", str(self.explicit_config_file_path))

    def test_read_input_data_with_explicit_type(self):
        """
        Tests that data is read correctly when the dataset_config specifies the file type explicitly.
        """
        df = self._get_input_data(file_type="parquet", options={})

        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 132342)
        self.assertEqual(df.shape[1], 30)

    def test_read_input_data_infer_type(self):
        """
        Tests that data is read correctly when file_type is not provided (auto-detection).
        """
        df = self._get_input_data(file_type=None, options={})

        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 132342)
        self.assertEqual(df.shape[1], 30)

    def test_read_input_data_missing_options(self):
        """
        Tests that passing no 'input_file_options' key (or None) defaults to empty dict,
        so reading still works as intended.
        """
        df = self._get_input_data(file_type="parquet", options=None)

        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 132342)
        self.assertEqual(df.shape[1], 30)

    def test_read_input_data_unsupported_file_type(self):
        """
        Tests that an unsupported file type raises a ValueError.
        """
        ds = InputDataSetA("NRT_BO_001", str(self.explicit_config_file_path))
        ds.input_file_name = str(self.test_data_file)
        ds.dataset_info["input"]["file_type"] = "foo"
        ds.dataset_info["input"]["options"] = {}

        with self.assertRaises(ValueError) as context:
            ds.read_input_data()
        self.assertIn("Unsupported file_type 'foo'", str(context.exception))

    def test_read_input_data_file_not_found(self):
        """
        Tests that reading a non-existent file raises FileNotFoundError.
        """
        ds = InputDataSetA("NRT_BO_001", str(self.explicit_config_file_path))
        ds.input_file_name = str(self.test_data_file) + "_not_found"
        ds.dataset_info["input"]["file_type"] = "parquet"
        ds.dataset_info["input"]["options"] = {}

        with self.assertRaises(FileNotFoundError):
            ds.read_input_data()

    def test_read_input_data_with_extra_options(self):
        """
        Demonstrates passing extra options
        """
        df = self._get_input_data(file_type="parquet", options={"n_rows": 100})

        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(df.shape[1], 30)
