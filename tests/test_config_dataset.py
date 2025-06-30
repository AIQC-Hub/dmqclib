import unittest
from pathlib import Path

from dmqclib.config.dataset_config import DataSetConfig


class TestBaseConfigPathMethods(unittest.TestCase):
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

    def test_common_base_path(self):
        """
        Test file name with a correct entry.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.load_dataset_config("NRT_BO_001")

        base_path = ds.get_base_path("common")
        self.assertEqual("/path/to/data_1", base_path)

    def test_input_base_path(self):
        """
        Test file name without an entry.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.load_dataset_config("NRT_BO_001")

        base_path = ds.get_base_path("input")
        self.assertEqual("/path/to/input_1", base_path)

    def test_default_base_path(self):
        """
        Test file name without an entry.
        """
        config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        ds = DataSetConfig(str(config_file_path))
        ds.load_dataset_config("NRT_BO_001")

        base_path = ds.get_base_path("locate")
        self.assertEqual("/path/to/data_1", base_path)

    def test_input_step_folder_name(self):
        """
        Test file name without an entry.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.load_dataset_config("NRT_BO_001")

        folder_name = ds.get_step_folder_name("input")
        self.assertEqual("input_folder_1", folder_name)

    def test_auto_select_step_folder_name(self):
        """
        Test file name without an entry.
        """
        config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        ds = DataSetConfig(str(config_file_path))
        ds.load_dataset_config("NRT_BO_001")

        folder_name = ds.get_step_folder_name("select")
        self.assertEqual("select", folder_name)

    def test_no_auto_select_step_folder_name(self):
        """
        Test file name without an entry.
        """
        config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        ds = DataSetConfig(str(config_file_path))
        ds.load_dataset_config("NRT_BO_001")

        folder_name = ds.get_step_folder_name("select", folder_name_auto=False)
        self.assertEqual("", folder_name)

    def test_common_dataset_folder_name(self):
        """
        Test file name without an entry.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.load_dataset_config("NRT_BO_001")

        dataset_folder_name = ds.get_dataset_folder_name("input")
        self.assertEqual("nrt_bo_001", dataset_folder_name)

    def test_dataset_folder_name_in_step_params(self):
        """
        Test file name without an entry.
        """
        config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        ds = DataSetConfig(str(config_file_path))
        ds.load_dataset_config("NRT_BO_001")

        dataset_folder_name = ds.get_dataset_folder_name("summary")
        self.assertEqual("summary_dataset_folder", dataset_folder_name)

    def test_default_file_name(self):
        """
        Test file name with a correct entry.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.load_dataset_config("NRT_BO_001")

        file_name = ds.get_file_name("input", "default_file.txt")
        self.assertEqual("default_file.txt", file_name)

    def test_no_default_file_name(self):
        """
        Test file name with a correct entry.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.load_dataset_config("NRT_BO_001")

        with self.assertRaises(ValueError):
            _ = ds.get_file_name("input")

    def test_file_name_in_params(self):
        """
        Test file name with a correct entry.
        """
        config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        ds = DataSetConfig(str(config_file_path))
        ds.load_dataset_config("NRT_BO_001")

        file_name = ds.get_file_name("summary")
        self.assertEqual("summary_in_params.txt", file_name)

    def test_full_input_path(self):
        """
        Test with all normal, non-empty parameters.
        Expect paths to be joined with slashes correctly.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.load_dataset_config("NRT_BO_001")

        full_file_name = ds.get_full_file_name(
            "input", "test_input_file.txt", use_dataset_folder=False
        )

        self.assertEqual(
            full_file_name, "/path/to/input_1/input_folder_1/test_input_file.txt"
        )

    def test_full_input_path_with_dataset_folder(self):
        """
        Test with all normal, non-empty parameters.
        Expect paths to be joined with slashes correctly.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.load_dataset_config("NRT_BO_001")

        full_file_name = ds.get_full_file_name("input", "test_input_file.txt")

        self.assertEqual(
            full_file_name,
            "/path/to/input_1/nrt_bo_001/input_folder_1/test_input_file.txt",
        )

    def test_full_summary_path(self):
        """
        Test file name with a correct entry.
        """
        config_file_path = (
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_002.yaml"
        )
        ds = DataSetConfig(str(config_file_path))
        ds.load_dataset_config("NRT_BO_001")

        full_file_name = ds.get_full_file_name("summary", "test_input_file.txt")

        self.assertEqual(
            full_file_name,
            "/path/to/data_1/summary_dataset_folder/summary/summary_in_params.txt",
        )


class TesDataSetConfig(unittest.TestCase):
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
        ds.load_dataset_config("NRT_BO_001")

        self.assertEqual(len(ds.config["path_info"]), 6)
        self.assertEqual(len(ds.config["target_set"]), 2)
        self.assertEqual(len(ds.config["feature_set"]), 2)
        self.assertEqual(len(ds.config["feature_param_set"]), 2)
        self.assertEqual(len(ds.config["step_class_set"]), 2)
        self.assertEqual(len(ds.config["step_param_set"]), 2)

    def test_invalid_dataset_name(self):
        ds = DataSetConfig(str(self.config_file_path))
        with self.assertRaises(ValueError):
            ds.load_dataset_config("INVALID_NAME")
