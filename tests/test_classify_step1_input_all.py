import unittest
from pathlib import Path

import polars as pl

from dmqclib.common.config.classify_config import ClassificationConfig
from dmqclib.classify.step1_read_input.dataset_all import InputDataSetAll


class TestInputDataSetAll(unittest.TestCase):
    """
    Tests for verifying input data reading and resolution in the InputDataSetAll class.
    Ensures data is loaded as expected from Parquet files, file names are resolved
    properly, and property checks are correct.
    """

    def setUp(self):
        """
        Set up the test configuration objects and specify test data file paths.
        """
        self.config_file_path = (
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

    def _get_input_data(self, config, file_type=None, read_file_options=None):
        """
        Helper method that loads input data into a Polars DataFrame, optionally
        setting the file type and read options in the config before reading.
        """
        ds = InputDataSetAll(config)
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
        """
        Confirm that InputDataSetA instances declare their step name as "input".
        """
        ds = InputDataSetAll(self.config)
        self.assertEqual(ds.step_name, "input")

    def test_input_file_name(self):
        """
        Ensure the input file name is determined correctly based on
        the loaded configuration.
        """
        ds = InputDataSetAll(self.config)
        self.assertEqual(
            "/path/to/input_1/input_folder_1/nrt_cora_bo_test.parquet",
            str(ds.input_file_name),
        )

    def test_read_input_data_with_explicit_type(self):
        """
        Verify that data is read from a Parquet file when a 'parquet' file_type
        is explicitly specified in the config.
        """
        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options={}
        )
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 19480)
        self.assertEqual(df.shape[1], 30)

    def test_read_input_data_infer_type(self):
        """
        Verify that data reading automatically infers the file type
        (from extension) when file_type is not explicitly set.
        """
        df = self._get_input_data(self.config, file_type=None, read_file_options={})
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 19480)
        self.assertEqual(df.shape[1], 30)

    def test_read_input_data_missing_options(self):
        """
        Confirm that data can be read successfully when no additional
        file reading options are provided.
        """
        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options=None
        )
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 19480)
        self.assertEqual(df.shape[1], 30)

    def test_read_input_data_file_not_found(self):
        """
        Ensure that attempting to read a non-existent input file
        raises a FileNotFoundError.
        """
        ds = InputDataSetAll(self.config)
        ds.input_file_name = str(self.test_data_file) + "_not_found"

        with self.assertRaises(FileNotFoundError):
            ds.read_input_data()

    def test_read_input_data_with_extra_options(self):
        """
        Verify that additional reading options (e.g., read only 100 rows)
        can be passed and applied to the data loading process.
        """
        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options={"n_rows": 100}
        )
        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(df.shape[1], 30)
