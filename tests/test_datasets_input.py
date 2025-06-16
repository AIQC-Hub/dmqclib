import unittest
from dmqclib.datasets.input.dataset_a import DataSetA

class TestDataSetA(unittest.TestCase):
    def test_init_valid_label(self):
        """Test that we can properly construct a DataSetA instance from the YAML."""
        ds = DataSetA("NRT_AL_001")
        self.assertEqual(ds.file, "nrt_al_001.parquet")
        self.assertTrue(ds.filter)
        self.assertEqual(ds.label, "NRT_AL_001")

    def test_init_invalid_label(self):
        """Test that constructing DataSetA with an invalid label raises ValueError."""
        with self.assertRaises(ValueError):
            DataSetA("NON_EXISTENT_LABEL")

    def test_init_class_mismatch(self):
        """
        If you had another label in datasets.yaml with 'class: SomethingElse',
        DataSetA should throw a ValueError for mismatch.
        """
        # Example if your YAML had:
        # MISMATCH_LABEL:
        #   class: SomeOtherClass
        #   file: mismatch.parquet
        #   filter: True
        #
        # with self.assertRaises(ValueError):
        #     DataSetA("MISMATCH_LABEL")
        pass
