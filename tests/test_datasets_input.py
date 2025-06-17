import unittest
from pathlib import Path
from dmqclib.datasets.input.dataset_a import DataSetA


class TestDataSetA(unittest.TestCase):
    def setUp(self):
        """
        Called before each test method. We define the explicit path to
        the test data config file here for reuse.
        """
        self.explicit_config_file_path = (
            Path(__file__).resolve().parent / "data" / "config" / "datasets.yaml"
        )

    def test_init_valid_label(self):
        """Test that we can properly construct a DataSetA instance from the YAML."""
        ds = DataSetA("NRT_AL_001", str(self.explicit_config_file_path))
        self.assertEqual(ds.dataset_name, "NRT_AL_001")

    def test_init_invalid_label(self):
        """Test that constructing DataSetA with an invalid label raises ValueError."""
        with self.assertRaises(ValueError):
            DataSetA("NON_EXISTENT_LABEL", str(self.explicit_config_file_path))

    def test_config_file(self):
        """Test that config file is properly set in the corresponding member variable"""
        ds = DataSetA("NRT_AL_001", str(self.explicit_config_file_path))
        self.assertTrue("datasets.yaml" in ds.config_file)

    def test_input_file_name(self):
        """Test that config file is properly set in the corresponding member variable"""
        ds = DataSetA("NRT_AL_001", str(self.explicit_config_file_path))
        self.assertEqual("/path/to/data/input/nrt_al_001.parquet", str(ds.input_file_name))

    def test_no_input_file_name(self):
        """Test that config file is properly set in the corresponding member variable"""
        with self.assertRaises(ValueError):
            _ = DataSetA("NRT_AL_002", str(self.explicit_config_file_path))
