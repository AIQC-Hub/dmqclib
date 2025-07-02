import os
import unittest
from pathlib import Path

import polars as pl

from dmqclib.common.loader.dataset_loader import load_step1_input_dataset
from dmqclib.common.loader.dataset_loader import load_step3_select_dataset
from dmqclib.config.dataset_config import DataSetConfig
from dmqclib.prepare.step4_locate.dataset_a import LocateDataSetA


class TestLocateDataSetA(unittest.TestCase):
    def setUp(self):
        """Set up test environment and load input and selected datasets."""
        self.config_file_path = str(
            Path(__file__).resolve().parent
            / "data"
            / "config"
            / "test_dataset_001.yaml"
        )
        self.config = DataSetConfig(str(self.config_file_path))
        self.config.select("NRT_BO_001")
        self.test_data_file = (
            Path(__file__).resolve().parent
            / "data"
            / "input"
            / "nrt_cora_bo_test.parquet"
        )
        self.ds_input = load_step1_input_dataset(self.config)
        self.ds_input.input_file_name = str(self.test_data_file)
        self.ds_input.read_input_data()

        self.ds_select = load_step3_select_dataset(
            self.config, input_data=self.ds_input.input_data
        )
        self.ds_select.label_profiles()

    def test_output_file_names(self):
        """Ensure output file names are set correctly."""
        ds = LocateDataSetA(self.config)
        self.assertEqual(
            "/path/to/locate_1/nrt_bo_001/locate_folder_1/temp_rows.parquet",
            str(ds.output_file_names["temp"]),
        )
        self.assertEqual(
            "/path/to/locate_1/nrt_bo_001/locate_folder_1/psal_rows.parquet",
            str(ds.output_file_names["psal"]),
        )

    def test_step_name(self):
        """Ensure the step name is set correctly."""
        ds = LocateDataSetA(self.config)
        self.assertEqual(ds.step_name, "locate")

    def test_input_data_and_selected_profiles(self):
        """Ensure input data and selected profiles are read correctly."""
        ds = LocateDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )

        self.assertIsInstance(ds.input_data, pl.DataFrame)
        self.assertEqual(ds.input_data.shape[0], 132342)
        self.assertEqual(ds.input_data.shape[1], 30)

        self.assertIsInstance(ds.selected_profiles, pl.DataFrame)
        self.assertEqual(ds.selected_profiles.shape[0], 44)
        self.assertEqual(ds.selected_profiles.shape[1], 8)

    def test_positive_rows(self):
        """Ensure positive row data is set correctly."""
        ds = LocateDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )
        ds.select_positive_rows("temp", {"flag": "temp_qc"})
        ds.select_positive_rows("psal", {"flag": "psal_qc"})

        self.assertIsInstance(ds.positive_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.positive_rows["temp"].shape[0], 64)
        self.assertEqual(ds.positive_rows["temp"].shape[1], 11)

        self.assertIsInstance(ds.positive_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.positive_rows["psal"].shape[0], 70)
        self.assertEqual(ds.positive_rows["psal"].shape[1], 11)

    def test_negative_rows(self):
        """Ensure negative row data is set correctly."""
        ds = LocateDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )
        ds.select_positive_rows("temp", {"flag": "temp_qc"})
        ds.select_positive_rows("psal", {"flag": "psal_qc"})
        ds.select_negative_rows("temp", {"flag": "temp_qc"})
        ds.select_negative_rows("psal", {"flag": "psal_qc"})

        self.assertIsInstance(ds.negative_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.negative_rows["temp"].shape[0], 64)
        self.assertEqual(ds.negative_rows["temp"].shape[1], 11)

        self.assertIsInstance(ds.negative_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.negative_rows["psal"].shape[0], 70)
        self.assertEqual(ds.negative_rows["psal"].shape[1], 11)

    def test_target_rows(self):
        """Ensure target rows are selected and set correctly."""
        ds = LocateDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )

        ds.process_targets()

        self.assertIsInstance(ds.target_rows["temp"], pl.DataFrame)
        self.assertEqual(ds.target_rows["temp"].shape[0], 128)
        self.assertEqual(ds.target_rows["temp"].shape[1], 9)

        self.assertIsInstance(ds.target_rows["psal"], pl.DataFrame)
        self.assertEqual(ds.target_rows["psal"].shape[0], 140)
        self.assertEqual(ds.target_rows["psal"].shape[1], 9)

    def test_write_target_rows(self):
        """Ensure target rows are written to parquet files correctly."""
        ds = LocateDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )
        data_path = Path(__file__).resolve().parent / "data" / "select"
        ds.output_file_names["temp"] = data_path / "temp_temp_rows.parquet"
        ds.output_file_names["psal"] = data_path / "temp_psal_rows.parquet"
        ds.output_file_names["pres"] = data_path / "temp_pres_rows.parquet"

        ds.process_targets()
        ds.write_target_rows()

        self.assertTrue(os.path.exists(ds.output_file_names["temp"]))
        self.assertTrue(os.path.exists(ds.output_file_names["psal"]))
        self.assertTrue(os.path.exists(ds.output_file_names["pres"]))
        os.remove(ds.output_file_names["temp"])
        os.remove(ds.output_file_names["psal"])
        os.remove(ds.output_file_names["pres"])

    def test_write_no_target_rows(self):
        """ "Ensure ValueError is raised for empty profiles."""
        ds = LocateDataSetA(
            self.config,
            input_data=self.ds_input.input_data,
            selected_profiles=self.ds_select.selected_profiles,
        )

        with self.assertRaises(ValueError):
            ds.write_target_rows()
