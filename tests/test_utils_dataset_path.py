import unittest
from pathlib import Path
from dmqclib.utils.config import read_config
from dmqclib.utils.dataset_path import build_full_input_path
from dmqclib.utils.dataset_path import build_full_data_path


class TestBuildFullInputPath(unittest.TestCase):
    def setUp(self):
        """
        Called before each test method. We create a config file here for reuse.
        """
        self.explicit_config_file_path = (
            Path(__file__).resolve().parent / "data" / "config" / "datasets.yaml"
        )
        self.config = read_config(config_file=str(self.explicit_config_file_path))

    def test_normal_params(self):
        """
        Test with all normal, non-empty parameters.
        Expect paths to be joined with slashes correctly.
        """
        result = build_full_input_path(self.config, "my_subfolder", "datafile.csv")
        self.assertEqual(result, "/path/to/data/input/my_subfolder/datafile.csv")

    def test_empty_input_folder_arg(self):
        """
        Test when the input_folder argument is an empty string.
        Should skip including it in the path.
        """
        result = build_full_input_path(self.config, "", "something.txt")
        self.assertEqual(result, "/path/to/data/input/something.txt")

    def test_none_input_folder_arg(self):
        """
        Test when the input_folder argument is None.
        Should skip including it in the path.
        """
        result = build_full_input_path(self.config, None, "something_else.txt")
        self.assertEqual(result, "/path/to/data/input/something_else.txt")

    def test_empty_config_input_folder(self):
        """
        Test when config['path_info']['input_folder'] is an empty string.
        """
        self.config["path_info"]["input_folder"] = ""
        result = build_full_input_path(self.config, "some_folder", "file.csv")
        self.assertEqual(result, "/path/to/data/some_folder/file.csv")

    def test_none_config_input_folder(self):
        """
        Test when config['path_info']['input_folder'] is None.
        """
        self.config["path_info"]["input_folder"] = None
        result = build_full_input_path(self.config, "another_subfolder", "info.txt")
        self.assertEqual(result, "/path/to/data/another_subfolder/info.txt")

    def test_relative_paths(self):
        """
        Test when input_folder includes '..' or other relative segments.
        """
        result = build_full_input_path(self.config, "../relative", "file.csv")
        self.assertEqual(result, "/path/to/data/relative/file.csv")

    def test_only_file_name(self):
        """
        Test if the user sets both config['path_info']['input_folder'] and input_folder to empty.
        """
        self.config["path_info"]["input_folder"] = ""
        result = build_full_input_path(self.config, "", "filename.txt")
        self.assertEqual(result, "/path/to/data/filename.txt")


class TestBuildFullDataPath(unittest.TestCase):
    def setUp(self):
        """
        Called before each test method. We create a config file here for reuse.
        """
        self.explicit_config_file_path = (
            Path(__file__).resolve().parent / "data" / "config" / "datasets.yaml"
        )
        self.config = read_config(config_file=str(self.explicit_config_file_path))

    def test_normal_usage(self):
        """
        Test that a normal call (with non-empty data_folder, data_type, file_name)
        produces the expected path.
        """
        result = build_full_data_path(self.config, "my_data", "train", "datafile.csv")
        expected = "/path/to/data/my_data/train/datafile.csv"
        self.assertEqual(result, expected)

    def test_empty_data_folder(self):
        """
        Test when data_folder is an empty string. It should skip appending
        the data_folder level.
        """
        result = build_full_data_path(self.config, "", "validate", "val.csv")
        expected = "/path/to/data/validate/val.csv"
        self.assertEqual(result, expected)

    def test_none_data_folder(self):
        """
        Test when data_folder is None. It should skip appending the data_folder level.
        """
        result = build_full_data_path(self.config, None, "test", "testfile.csv")
        expected = "/path/to/data/test/testfile.csv"
        self.assertEqual(result, expected)

    def test_empty_data_type_folder(self):
        """
        Test when config["path_info"][data_type] is an empty string.
        """
        # Make "train" key empty to simulate having an empty folder name
        self.config["path_info"]["train_folder"] = ""
        result = build_full_data_path(self.config, "subfolder", "train", "somefile.txt")
        expected = "/path/to/data/subfolder/somefile.txt"
        self.assertEqual(result, expected)

    def test_none_data_type_folder(self):
        """
        Test when config["path_info"][data_type] is None.
        """
        self.config["path_info"]["train_folder"] = None
        result = build_full_data_path(
            self.config, "subfolder", "train", "anotherfile.txt"
        )
        expected = "/path/to/data/subfolder/anotherfile.txt"
        self.assertEqual(result, expected)

    def test_relative_paths(self):
        """
        Test when data_folder or config["path_info"][data_type] includes relative segments like '..'.
        """
        self.config["path_info"]["validate_folder"] = "../valid"
        result = build_full_data_path(
            self.config, "../archive", "validate", "mydata.csv"
        )
        expected = "/path/to/valid/mydata.csv"
        self.assertEqual(result, expected)
