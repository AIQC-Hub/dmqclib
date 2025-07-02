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
        ds.select("NRT_BO_001")

        base_path = ds.get_base_path("common")
        self.assertEqual("/path/to/data_1", base_path)

    def test_input_base_path(self):
        """
        Test file name without an entry.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

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
        ds.select("NRT_BO_001")

        base_path = ds.get_base_path("locate")
        self.assertEqual("/path/to/data_1", base_path)

    def test_input_step_folder_name(self):
        """
        Test file name without an entry.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

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
        ds.select("NRT_BO_001")

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
        ds.select("NRT_BO_001")

        folder_name = ds.get_step_folder_name("select", folder_name_auto=False)
        self.assertEqual("", folder_name)

    def test_common_dataset_folder_name(self):
        """
        Test file name without an entry.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

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
        ds.select("NRT_BO_001")

        dataset_folder_name = ds.get_dataset_folder_name("summary")
        self.assertEqual("summary_dataset_folder", dataset_folder_name)

    def test_default_file_name(self):
        """
        Test file name with a correct entry.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        file_name = ds.get_file_name("input", "default_file.txt")
        self.assertEqual("default_file.txt", file_name)

    def test_no_default_file_name(self):
        """
        Test file name with a correct entry.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

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
        ds.select("NRT_BO_001")

        file_name = ds.get_file_name("summary")
        self.assertEqual("summary_in_params.txt", file_name)

    def test_full_input_path(self):
        """
        Test with all normal, non-empty parameters.
        Expect paths to be joined with slashes correctly.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

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
        ds.select("NRT_BO_001")

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
        ds.select("NRT_BO_001")

        full_file_name = ds.get_full_file_name("summary", "test_input_file.txt")

        self.assertEqual(
            full_file_name,
            "/path/to/data_1/summary_dataset_folder/summary/summary_in_params.txt",
        )


class TestBaseConfigBaseClass(unittest.TestCase):
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

    def test_input_bass_class(self):
        """
        Test file name with a correct entry.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        base_class = ds.get_base_class("input")
        self.assertEqual("InputDataSetA", base_class)


class TestBaseConfigTargets(unittest.TestCase):
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

    def test_target_variables(self):
        """
        Test file name with a correct entry.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        target_variables = ds.get_target_variables()
        self.assertEqual(len(target_variables), 3)

    def test_target_names(self):
        """
        Test file name with a correct entry.
        """
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        target_names = ds.get_target_names()
        self.assertEqual(target_names, ["temp", "psal", "pres"])

    def test_target_dict(self):
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        target_dict = ds.get_target_dict()
        self.assertEqual(target_dict["temp"], {"name": "temp", "flag": "temp_qc"})
        self.assertEqual(target_dict["psal"], {"name": "psal", "flag": "psal_qc"})
        self.assertEqual(target_dict["pres"], {"name": "pres", "flag": "pres_qc"})

    def test_target_file_names(self):
        ds = DataSetConfig(str(self.config_file_path))
        ds.select("NRT_BO_001")

        target_file_names = ds.get_target_file_names(
            "select", "{target_name}_features.parquet"
        )

        self.assertEqual(
            target_file_names["temp"],
            "/path/to/select_1/nrt_bo_001/select_folder_1/temp_features.parquet",
        )
        self.assertEqual(
            target_file_names["psal"],
            "/path/to/select_1/nrt_bo_001/select_folder_1/psal_features.parquet",
        )
        self.assertEqual(
            target_file_names["pres"],
            "/path/to/select_1/nrt_bo_001/select_folder_1/pres_features.parquet",
        )
