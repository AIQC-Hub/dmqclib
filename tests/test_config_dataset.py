import unittest
from pathlib import Path

from dmqclib.config.dataset_config import DataSetConfig


class TestDataSetConfig(unittest.TestCase):
    def setUp(self):
        """
        Called before each test method. We define the explicit path to
        the test data config file here for reuse.
        """
        self.config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_001.yaml"
        )

        self.template_file = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "prepare_config_template.yaml"
        )

    def test_valid_config(self):
        """
        Test valid config
        """
        ds = DataSetConfig(str(self.config_file_path))
        msg = ds.validate()
        self.assertIn("valid", msg)

    def test_invalid_config(self):
        """
        Test invalid config
        """
        config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_invalid.yaml"
        )
        ds = DataSetConfig(str(config_file_path))
        msg = ds.validate()
        self.assertIn("invalid", msg)

    def test_load_dataset_config(self):
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        self.assertEqual(len(ds.data["path_info"]), 6)
        self.assertEqual(len(ds.data["target_set"]), 2)
        self.assertEqual(len(ds.data["feature_set"]), 2)
        self.assertEqual(len(ds.data["feature_param_set"]), 2)
        self.assertEqual(len(ds.data["step_class_set"]), 2)
        self.assertEqual(len(ds.data["step_param_set"]), 2)

    def test_load_dataset_config_twise(self):
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")
        ds.select("NRT_BO_001")

    def test_invalid_dataset_name(self):
        ds = DataSetConfig(str(self.config_file_path))
        with self.assertRaises(ValueError):
            ds.select("INVALID_NAME")

    def test_input_folder(self):
        """
        Test input folder
        """
        ds = DataSetConfig(str(self.template_file))
        ds.select("NRT_BO_001")
        input_file_name = ds.get_full_file_name(
            "input",
            ds.data["input_file_name"],
            use_dataset_folder=False,
            folder_name_auto=False,
        )
        self.assertEqual(input_file_name, "/path/to/input/nrt_cora_bo_test.parquet")

    def test_summary_folder(self):
        """
        Test summary folder
        """
        ds = DataSetConfig(str(self.template_file))
        ds.select("NRT_BO_001")
        input_file_name = ds.get_full_file_name("summary", "test.txt")
        self.assertEqual(input_file_name, "/path/to/data/nrt_bo_001/summary/test.txt")

    def test_split_folder(self):
        """
        Test split folder
        """
        ds = DataSetConfig(str(self.template_file))
        ds.select("NRT_BO_001")
        input_file_name = ds.get_full_file_name("split", "test.txt")
        self.assertEqual(input_file_name, "/path/to/data/nrt_bo_001/training/test.txt")
