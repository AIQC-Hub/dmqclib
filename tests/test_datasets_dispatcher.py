import unittest

from dmqclib.datasets.dispatcher import load_input_dataset
from dmqclib.datasets.input.dataset_a import DataSetA

class TestDispatcher(unittest.TestCase):
    def test_load_dataset_valid_label(self):
        """
        Test that load_dataset returns an instance of DataSetA for the known label.
        """
        ds = load_input_dataset("NRT_AL_001")
        self.assertIsInstance(ds, DataSetA)
        self.assertEqual(ds.file, "nrt_al_001.parquet")
        self.assertTrue(ds.filter)

    def test_load_dataset_invalid_label(self):
        """
        Test that calling load_dataset with an invalid label raises a ValueError.
        """
        with self.assertRaises(ValueError):
            load_input_dataset("NON_EXISTENT_LABEL")

    def test_load_dataset_unknown_class(self):
        """
        If you add a label with a class that isn't recognized,
        test that a ValueError is raised.
        """
        # Example if the config had:
        #   UNKNOWN_CLASS_LABEL:
        #     class: MysteryClass
        #     file: unknown.parquet
        #     filter: False
        #
        # with self.assertRaises(ValueError):
        #     load_dataset("UNKNOWN_CLASS_LABEL")
        pass
