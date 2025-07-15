"""Utilities for generating and formatting summary statistics.

This module provides high-level functions to calculate and display summary
statistics for a given dataset file. It uses a predefined configuration
template to process the data, compute statistics at both global and
per-profile levels, and format the results for human-readable output.
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
    """Calculate and retrieve summary statistics from a dataset file.

    This function loads a dataset, computes global and per-profile summary
    statistics, and returns the requested type of summary as a Polars DataFrame.
    It uses a built-in configuration template and dynamically sets the input
    path based on the provided file.

    :param input_file: The path to the input dataset file (e.g., a TSV or Parquet file).
    :type input_file: str
    :param summary_type: The type of summary to return. Supported values are
                         "profiles" (for per-profile stats) and "all" (for global stats).
    :type summary_type: str
    :raises FileNotFoundError: If the ``input_file`` does not exist.
    :raises ValueError: If the ``summary_type`` is not a supported value.
    :return: A Polars DataFrame containing the requested summary statistics.
    :rtype: polars.DataFrame
    """
    config = DataSetConfig("template:data_sets")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File '{input_file}' does not exist.")
    config.select("dataset_0001")
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
    """Format a summary statistics DataFrame into a pretty-printed string.

    This function takes a DataFrame of statistics (as produced by
    :func:`get_summary_stats`) and converts it into a nested dictionary,
    which is then formatted into a string for display. The output can be
    filtered by variable and statistic type.

    :param df: The input DataFrame containing summary statistics. It is expected
               to have a "stats" column for profile-level summaries.
    :type df: polars.DataFrame
    :param variables: An optional list of variable names to include. If empty,
                      all variables are included.
    :type variables: list[str]
    :param summary_stats: An optional list of statistic names (e.g., "mean", "sd")
                          to include for profile-level summaries.
    :type summary_stats: list[str]
    :return: A string containing the pretty-printed, formatted statistics.
    :rtype: str
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

    buf = io.StringIO()
    pprint.pprint(result, stream=buf, sort_dicts=False)
    pprint_output = buf.getvalue()
    buf.close()

    return pprint_output


def _format_with_stats_column(
    df: pl.DataFrame, variables: List[str], summary_stats: List[str]
) -> Dict:
    """Format a DataFrame containing a 'stats' column into a nested dict."""
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
    """Format a DataFrame without a 'stats' column into a dict."""
    stats_dict = {}
    for row in df.iter_rows(named=True):
        if len(variables) > 0 and row["variable"] not in variables:
            continue
        stats_dict[row["variable"]] = {
            "min": round(row["min"], 2),
            "max": round(row["max"], 2),
        }

    return stats_dict
