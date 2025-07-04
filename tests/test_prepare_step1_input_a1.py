import unittest
from pathlib import Path

import polars as pl

from dmqclib.config.dataset_config import DataSetConfig
from dmqclib.prepare.step1_input.dataset_a import InputDataSetA


class TestInputDataSetARename(unittest.TestCase):
    """
    A suite of unit tests for verifying input data handling in the InputDataSetA class.
    Ensures that data reading, file name resolution, and DataFrame properties behave
    as expected under various configurations.
    """

    def setUp(self):
        """
        Set up test environment by loading a test configuration and specifying paths
        to test files (config file and input data).
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        self.config = DataSetConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )

    def _get_input_data(self, config, file_type=None, read_file_options=None):
        """
        Helper method for loading input data using optional file type and reading options.
        Returns a Polars DataFrame.
        """
        ds = InputDataSetA(config)
        ds.input_file_name = str(self.test_data_file)

        if file_type is not None:
            ds.config.data["step_param_set"]["steps"]["input"]["file_type"] = file_type

        if read_file_options is not None:
            ds.config.data["step_param_set"]["steps"]["input"]["read_file_options"] = (
                read_file_options
            )

        ds.read_input_data()
        return ds.input_data

    def test_rename(self):
        """
        Test reading data from a Parquet file with an explicitly
        specified file_type in the configuration.
        """
        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options={}
        )

        self.assertFalse("filename" in df.columns)
        self.assertTrue("filename_new" in df.columns)

    def test_rename_with_incorrect_param(self):
        """
        Test reading data from a Parquet file with an explicitly
        specified file_type in the configuration.
        """
        del self.config.get_step_params("input")["rename_dict"]

        with self.assertRaises(ValueError):
            _ = self._get_input_data(
                self.config, file_type="parquet", read_file_options={}
            )
