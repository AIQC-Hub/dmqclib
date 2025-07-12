"""
Module
"""

import io
import os
import pprint
from typing import List, Dict

import polars as pl

from dmqclib.common.config.dataset_config import DataSetConfig
from dmqclib.common.loader.dataset_loader import load_step1_input_dataset
from dmqclib.common.loader.dataset_loader import load_step2_summary_dataset


def get_summary_stats(input_file: str, summary_type: str) -> pl.DataFrame:
    """
    Show stats
    """
    config = DataSetConfig("template:data_sets")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File '{input_file}' does not exist.")
    config.select("NRT_BO_001")
    config.data["path_info"]["input"]["base_path"] = os.path.dirname(input_file)
    config.data["input_file_name"] = os.path.basename(input_file)

    ds_input = load_step1_input_dataset(config)
    ds_input.read_input_data()

    ds_summary = load_step2_summary_dataset(config, ds_input.input_data)
    ds_summary.calculate_stats()
    ds_summary.create_summary_stats_observation()
    ds_summary.create_summary_stats_profile()

    selectors = {
        "profiles": ds_summary.summary_stats_profile,
        "all": ds_summary.summary_stats_observation,
    }
    if summary_type not in selectors:
        raise ValueError(f"Summary type {summary_type} is not supported.")

    return selectors[summary_type]


def format_summary_stats(
    df: pl.DataFrame,
    variables: List[str] = [],
    summary_stats: List[str] = ["mean", "median", "sd", "pct25", "pct75"],
) -> str:
    """
    Format stats
    """
    if "stats" in df.columns:
        summary_type = "profiles"
    else:
        summary_type = "all"

    functions = {
        "profiles": _format_with_stats_column,
        "all": _format_without_stats_column,
    }

    result = functions[summary_type](df, variables, summary_stats)

    # Capture and print pretty output
    buf = io.StringIO()
    pprint.pprint(result, stream=buf, sort_dicts=False)
    pprint_output = buf.getvalue()
    buf.close()

    return pprint_output


def _format_with_stats_column(
    df: pl.DataFrame, variables: List[str], summary_stats: List[str]
) -> Dict:
    stats_dict = {}
    for row in df.iter_rows(named=True):
        if (row["stats"] not in summary_stats) or (
            len(variables) > 0 and row["variable"] not in variables
        ):
            continue
        stats_dict.setdefault(row["variable"], {})[row["stats"]] = {
            "min": round(row["min"], 2),
            "max": round(row["max"], 2),
        }

    return stats_dict


def _format_without_stats_column(
    df: pl.DataFrame, variables: List[str], _: List[str]
) -> Dict:
    stats_dict = {}
    for row in df.iter_rows(named=True):
        if len(variables) > 0 and row["variable"] not in variables:
            continue
        stats_dict[row["variable"]] = {
            "min": round(row["min"], 2),
            "max": round(row["max"], 2),
        }

    return stats_dict
