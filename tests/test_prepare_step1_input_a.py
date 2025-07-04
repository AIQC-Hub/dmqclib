import unittest
from pathlib import Path

import polars as pl

from dmqclib.config.dataset_config import DataSetConfig
from dmqclib.prepare.step1_input.dataset_a import InputDataSetA


class TestInputDataSetA(unittest.TestCase):
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

        self.config_file_path2 = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        self.config2 = DataSetConfig(str(self.config_file_path2))
        self.config2.select("NRT_BO_001")

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

    def test_step_name(self):
        """
        Verify that the step name property is set to "input".
        """
        ds = InputDataSetA(self.config)
        self.assertEqual(ds.step_name, "input")

    def test_input_file_name(self):
        """
        Verify that the input file name is correctly set and resolved
        based on the configuration.
        """
        ds = InputDataSetA(self.config)
        self.assertEqual(
            "/path/to/input_1/input_folder_1/nrt_cora_bo_test.parquet",
            str(ds.input_file_name),
        )

    def test_read_input_data_with_explicit_type(self):
        """
        Test reading data from a Parquet file with an explicitly
        specified file_type in the configuration.
        """
        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options={}
        )

        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 132342)
        self.assertEqual(df.shape[1], 30)

    def test_read_input_data_infer_type(self):
        """
        Test reading data while allowing file type to be inferred
        from the file extension.
        """
        df = self._get_input_data(self.config, file_type=None, read_file_options={})

        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 132342)
        self.assertEqual(df.shape[1], 30)

    def test_read_input_data_missing_options(self):
        """
        Test reading data without specifying any additional file
        reading options in the config.
        """
        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options=None
        )

        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 132342)
        self.assertEqual(df.shape[1], 30)

    def test_read_input_data_file_not_found(self):
        """
        Test the behavior when the specified input file does not exist,
        expecting a FileNotFoundError.
        """
        ds = InputDataSetA(self.config)
        ds.input_file_name = str(self.test_data_file) + "_not_found"

        with self.assertRaises(FileNotFoundError):
            ds.read_input_data()

    def test_read_input_data_with_extra_options(self):
        """
        Test reading data with extra options (e.g., limiting the number of rows).
        """
        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options={"n_rows": 100}
        )

        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(df.shape[1], 30)


class TestInputDataSetARename(unittest.TestCase):
    """
    A suite of unit tests for verifying input data handling the rename function in the InputDataSetA class.
    Ensures that specified columns are correctly renamed.
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
        Test renaming input data
        """
        df = self._get_input_data(
            self.config, file_type="parquet", read_file_options={}
        )

        self.assertFalse("filename" in df.columns)
        self.assertTrue("filename_new" in df.columns)

    def test_rename_with_incorrect_param(self):
        """
        Test renaming input data with empty parameter values
        """
        del self.config.get_step_params("input")["rename_dict"]

        with self.assertRaises(ValueError):
            _ = self._get_input_data(
                self.config, file_type="parquet", read_file_options={}
            )
