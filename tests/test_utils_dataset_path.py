import unittest
from pathlib import Path
from dmqclib.utils.config import read_config
from dmqclib.utils.dataset_path import build_full_input_path
from dmqclib.utils.dataset_path import build_full_training_path


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
        result = build_full_input_path(
            self.config["path_info"], "my_subfolder", "datafile.csv"
        )
        self.assertEqual(result, "/path/to/data/input/my_subfolder/datafile.csv")

    def test_empty_input_folder_arg(self):
        """
        Test when the folder_name2 argument is an empty string.
        """
        result = build_full_input_path(self.config["path_info"], "", "something.txt")
        self.assertEqual(result, "/path/to/data/input/something.txt")

    def test_none_input_folder_arg(self):
        """
        Test when the folder_name2 argument is None.
        """
        result = build_full_input_path(
            self.config["path_info"], None, "something_else.txt"
        )
        self.assertEqual(result, "/path/to/data/input/something_else.txt")

    def test_empty_config_input_folder(self):
        """
        Test when config['path_info']['input_folder'] is an empty string.
        """
        self.config["path_info"]["input"]["folder_name"] = ""
        result = build_full_input_path(
            self.config["path_info"], "some_folder", "file.csv"
        )
        self.assertEqual(result, "/path/to/data/some_folder/file.csv")

    def test_none_config_input_folder(self):
        """
        Test when config['path_info']['input_folder'] is None.
        """
        self.config["path_info"]["input"]["folder_name"] = None
        result = build_full_input_path(
            self.config["path_info"], "another_subfolder", "info.txt"
        )
        self.assertEqual(result, "/path/to/data/another_subfolder/info.txt")

    def test_relative_paths(self):
        """
        Test when input_folder includes '..' or other relative segments.
        """
        result = build_full_input_path(
            self.config["path_info"], "../relative", "file.csv"
        )
        self.assertEqual(result, "/path/to/data/relative/file.csv")

    def test_only_file_name(self):
        """
        Test if the user sets both config['path_info']['input_folder'] and input_folder to empty.
        """
        self.config["path_info"]["input"]["folder_name"] = ""
        result = build_full_input_path(self.config["path_info"], "", "filename.txt")
        self.assertEqual(result, "/path/to/data/filename.txt")


class TestBuildFullTrainPath(unittest.TestCase):
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
        Test that a normal call produces the expected path.
        """
        result = build_full_training_path(
            self.config["path_info"], "my_data", "datafile.csv"
        )
        expected = "/path/to/data/my_data/train/datafile.csv"
        self.assertEqual(result, expected)

    def test_empty_data_folder(self):
        """
        Test when folder_name1 is an empty string.
        """
        result = build_full_training_path(self.config["path_info"], "", "val.csv")
        expected = "/path/to/data/train/val.csv"
        self.assertEqual(result, expected)

    def test_none_folder_name(self):
        """
        Test when folder_name1 is None.
        """
        result = build_full_training_path(
            self.config["path_info"], None, "testfile.csv"
        )
        expected = "/path/to/data/train/testfile.csv"
        self.assertEqual(result, expected)

    def test_empty_train_folder_name(self):
        """
        Test when config["path_info"]["train"]["folder_name"] is an empty string.
        """
        self.config["path_info"]["train"]["folder_name"] = ""
        result = build_full_training_path(
            self.config["path_info"], "subfolder", "somefile.txt"
        )
        expected = "/path/to/data/subfolder/somefile.txt"
        self.assertEqual(result, expected)

    def test_none_train_folder_name(self):
        """
        Test when config["path_info"]["train"]["folder_name"] is None.
        """
        self.config["path_info"]["train"]["folder_name"] = None
        result = build_full_training_path(
            self.config["path_info"], "subfolder", "anotherfile.txt"
        )
        expected = "/path/to/data/subfolder/anotherfile.txt"
        self.assertEqual(result, expected)

    def test_relative_paths(self):
        """
        Test when folder_name1 or config["path_info"]["train"]["folder_name"] includes relative segments like '..'.
        """
        self.config["path_info"]["train"]["folder_name"] = "../valid"
        result = build_full_training_path(
            self.config["path_info"], "../archive", "mydata.csv"
        )
        expected = "/path/to/valid/mydata.csv"
        self.assertEqual(result, expected)
