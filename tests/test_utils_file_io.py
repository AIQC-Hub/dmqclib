import os
from pathlib import Path
import unittest
import polars as pl
from dmqclib.utils.file_io import read_input_file


class TestReadInputFile(unittest.TestCase):
    def setUp(self):
        """
        Runs once before any tests. Sets the directory
        where your test data files are located.
        """
        self.test_data_dir = Path(__file__).resolve().parent / "data" / "input"

    def test_read_input_file_explicit_type(self):
        """
        Tests reading each supported file type with an explicitly specified file_type.
        """
        test_cases = [
            ("nrt_cora_bo_test.parquet", 132342, "parquet"),
            ("nrt_cora_bo_test_2023.csv.gz", 19480, "csv.gz"),
            ("nrt_cora_bo_test_2023.tsv.gz", 19480, "tsv.gz"),
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
        Tests reading each file type without specifying file_type
        (letting the function infer the correct format).
        """
        test_cases = [
            ("nrt_cora_bo_test.parquet", 132342),
            ("nrt_cora_bo_test_2023.csv.gz", 19480),
            ("nrt_cora_bo_test_2023.tsv.gz", 19480),
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
        Ensures that an unsupported file type raises a ValueError.
        """
        file_path = os.path.join(self.test_data_dir, "nrt_cora_bo_test.parquet")
        with self.assertRaises(ValueError) as context:
            _ = read_input_file(file_path, file_type="foo", options={})
        self.assertIn("Unsupported file_type 'foo'", str(context.exception))

    def test_non_existent_file(self):
        """
        Ensures that trying to read a non-existent file raises FileNotFoundError.
        """
        with self.assertRaises(FileNotFoundError):
            _ = read_input_file("non_existent_file.csv", file_type="csv", options={})

    def test_pass_additional_options(self):
        """
        Demonstrates passing extra options, e.g., has_header=False for CSV, if desired.
        Adjust logic/test according to how your data is formatted.
        """
        file_name = "nrt_cora_bo_test_2023.csv.gz"
        file_path = os.path.join(self.test_data_dir, file_name)
        # For demonstration, suppose we test with has_header=False (likely incorrect for real data).
        df = read_input_file(
            file_path, file_type="csv.gz", options={"has_header": False}
        )
        self.assertIsInstance(df, pl.DataFrame)
        # Insert additional assertions if your data truly lacks a header or you want to test specific behaviors.
        # self.assertEqual(df.shape[0], ...) etc.
