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
        self.assertEqual(ds.file, "nrt_al_001.parquet")
        self.assertTrue(ds.filter)
        self.assertEqual(ds.label, "NRT_AL_001")

    def test_init_invalid_label(self):
        """Test that constructing DataSetA with an invalid label raises ValueError."""
        with self.assertRaises(ValueError):
            DataSetA("NON_EXISTENT_LABEL", str(self.explicit_config_file_path))
