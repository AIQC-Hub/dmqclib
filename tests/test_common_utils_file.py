import os
import unittest
from pathlib import Path

import polars as pl

from dmqclib.common.utils.file import read_input_file


class TestReadInputFile(unittest.TestCase):
    """
    A suite of tests verifying various file reading scenarios
    (multiple file types, type inference, non-existent files, etc.)
    using 'read_input_file'.
    """

    def setUp(self):
        """
        Runs once before any tests. Sets the directory
        where test data files are located.
        """
        self.test_data_dir = Path(__file__).resolve().parent / "data" / "input"

    def test_read_input_file_explicit_type(self):
        """
        Verify that the function can read various file types
        when the file_type is explicitly specified.
        """
        test_cases = [
            ("nrt_cora_bo_test.parquet", 132342, "parquet"),
            ("nrt_cora_bo_test_2023_row1.csv", 1, "csv"),
            ("nrt_cora_bo_test_2023_row1.tsv", 1, "tsv"),
            ("nrt_cora_bo_test_2023_row1.csv.gz", 1, "csv.gz"),
            ("nrt_cora_bo_test_2023_row1.tsv.gz", 1, "tsv.gz"),
        ]
        for file_name, expected_rows, file_type in test_cases:
            with self.subTest(file_name=file_name, file_type=file_type):
                file_path = os.path.join(self.test_data_dir, file_name)
                df = read_input_file(file_path, file_type=file_type, options={})
                self.assertIsInstance(df, pl.DataFrame)
                self.assertEqual(df.shape[0], expected_rows)
                self.assertEqual(df.shape[1], 30)

    def test_read_input_file_infer_type(self):
        """
        Verify that the function can infer file types
        when file_type is not explicitly specified.
        """
        test_cases = [
            ("nrt_cora_bo_test.parquet", 132342),
            ("nrt_cora_bo_test_2023_row1.csv", 1),
            ("nrt_cora_bo_test_2023_row1.tsv", 1),
            ("nrt_cora_bo_test_2023_row1.csv.gz", 1),
            ("nrt_cora_bo_test_2023_row1.tsv.gz", 1),
        ]
        for file_name, expected_rows in test_cases:
            with self.subTest(file_name=file_name):
                file_path = os.path.join(self.test_data_dir, file_name)
                df = read_input_file(file_path, file_type=None, options={})
                self.assertIsInstance(df, pl.DataFrame)
                self.assertEqual(df.shape[0], expected_rows)
                self.assertEqual(df.shape[1], 30)

    def test_unsupported_file_type(self):
        """
        Verify that specifying an unsupported file type
        raises a ValueError.
        """
        file_path = os.path.join(self.test_data_dir, "nrt_cora_bo_test.parquet")
        with self.assertRaises(ValueError) as context:
            _ = read_input_file(file_path, file_type="foo", options={})
        self.assertIn("Unsupported file_type 'foo'", str(context.exception))

    def test_non_existent_file(self):
        """
        Verify that attempting to read a non-existent file
        raises FileNotFoundError.
        """
        with self.assertRaises(FileNotFoundError):
            _ = read_input_file("non_existent_file.csv", file_type="csv", options={})

    def test_pass_additional_options(self):
        """
        Demonstrate passing additional options to the reader,
        such as has_header=False for CSV.
        """
        file_name = "nrt_cora_bo_test_2023_row1.csv.gz"
        file_path = os.path.join(self.test_data_dir, file_name)
        df = read_input_file(
            file_path, file_type="csv.gz", options={"has_header": False}
        )
        self.assertIsInstance(df, pl.DataFrame)
