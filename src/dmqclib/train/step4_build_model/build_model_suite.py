"""
This module defines the :class:`BuildModelSuite` class, a specialized component
for building and testing multiple machine learning models concurrently using
a model suite (e.g., ModelSuite).

It inherits from :class:`dmqclib.train.step4_build_model.build_model_base.BuildModelBase`
and aggregates the results across all methods into single output files per target.
"""

import copy
import os
from typing import Optional, Dict

import polars as pl

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.train.step4_build_model.build_model_base import BuildModelBase
from dmqclib.common.utils.metric_plots import create_multi_method_metric_plots
from dmqclib.common.loader.single_model_loader import load_single_model_class_with_class_name


class BuildModelSuite(BuildModelBase):
    """
    A subclass of :class:`BuildModelBase` designed to build and test models
    using a model suite (multi-model configuration).

    This class iterates through all ML methods defined in the provided base model.
    It saves individual models with composite keys, but aggregates test reports,
    predictions, and contingency tables into single datasets per target name
    by introducing a 'method' column.

    .. note::
       This class sets :attr:`expected_class_name` to ``"BuildModelSuite"``.
    """

    expected_class_name: str = "BuildModelSuite"

    def __init__(
        self,
        config: ConfigBase,
        training_sets: Optional[Dict[str, pl.DataFrame]] = None,
        test_sets: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> None:
        """
        Initializes the BuildModelSuite class with a training configuration,
        and training/test sets.
        """
        super().__init__(
            config=config, training_sets=training_sets, test_sets=test_sets
        )

        if not getattr(self.base_model, "multi", False):
            raise ValueError(
                "BuildModelSuite requires a base model with 'multi=True' "
                "(e.g., ModelSuite), but received a standard model class."
            )

        self.drop_cols = ["row_id", "platform_code", "profile_no", "observation_no"]
        self.test_cols =[
            "row_id",
            "platform_code",
            "profile_no",
            "observation_no",
            "label",
        ]

        # Consolidated files per target for data, but unique files per model/method
        self.default_file_names: Dict[str, str] = {
            "report": "test_report_{target_name}.tsv",
            "prediction": "test_prediction_{target_name}.parquet",
            "contingency_table": "test_contingency_tables_{target_name}.tsv",
            "metric_plot": "test_metric_plots_{target_name}.svg",
        }
        self.default_model_file_name: str = "model_{method}_{target_name}.joblib"

        # Populate base path mappings (using standard template for aggregated data)
        self.output_file_names: Dict[str, Dict[str, str]] = {
            k: self.config.get_target_file_names(step_name="build", default_file_name=v)
            for k, v in self.default_file_names.items()
        }

        # Populate paths specifically for individual model .joblib files
        base_models = self.config.get_target_file_names("model", self.default_model_file_name)
        self.model_file_names = {}
        for target_name in self.config.get_target_names():
            for method_name, method_obj in self.base_model.method_objs.items():
                method_lower = getattr(method_obj, "short_name", method_name).lower()
                comp_key = f"{method_lower}_{target_name}"
                self.model_file_names[comp_key] = base_models[target_name].replace("{method}", method_lower)

    def build_targets(self) -> None:
        """
        Iterate over all targets from the configuration, calling :meth:`build`
        for each, and then optionally calling :meth:`test` if test sets exist.
        """
        for target_name in self.config.get_target_names():
            self.build(target_name)
            if self.test_sets is not None and target_name in self.test_sets:
                self.test(target_name)

    def test_targets(self) -> None:
        """
        Iterate over all targets, ensuring that models have been built for all
        configured methods before calling :meth:`test`.

        :raises ValueError: If a target/method combination has no corresponding entry
                            in :attr:`models`.
        """
        for target_name in self.config.get_target_names():
            for method_name, method_obj in self.base_model.method_objs.items():
                method_lower = getattr(method_obj, "short_name", method_name).lower()
                comp_key = f"{method_lower}_{target_name}"

                # Check for the composite key instead of just target_name
                if comp_key not in self.models:
                    raise ValueError(
                        f"No valid model found for the variable '{target_name}' "
                        f"and method '{method_name}' (expected key '{comp_key}')."
                    )
            self.test(target_name)

    def build(self, target_name: str) -> None:
        """
        Build (train) models for the specified target across all configured methods,
        storing them in :attr:`models` with composite keys.
        """
        if not self.training_sets:
            raise ValueError("Member variable 'training_sets' must not be empty.")

        if not self.test_sets:
            raise ValueError("Member variable 'test_sets' must not be empty.")

        training_set = self.training_sets[target_name].drop(["k_fold"] + self.drop_cols)
        test_set = self.test_sets[target_name].drop(self.drop_cols)
        combined_set = training_set.vstack(test_set)

        for method_name, method_obj in self.base_model.method_objs.items():
            method_lower = getattr(method_obj, "short_name", method_name).lower()
            comp_key = f"{method_lower}_{target_name}"

            current_model = copy.deepcopy(method_obj)
            current_model.training_set = combined_set
            current_model.build()

            self.models[comp_key] = current_model

    def test(self, target_name: str) -> None:
        """
        Test the models for the given target across all methods, appending a
        'method' column and aggregating the results into single datasets.

        Data types for model outputs (class, score, etc.) are standardized
        to Int64 and Float64 to prevent Polars SchemaErrors when concatenating
        results from different ML libraries (e.g., XGBoost vs Scikit-Learn).
        """
        test_set = self.test_sets[target_name].drop(self.drop_cols)

        target_reports = []
        target_predictions =[]
        target_contingency =[]

        for method_name, method_obj in self.base_model.method_objs.items():
            method_lower = getattr(method_obj, "short_name", method_name).lower()
            comp_key = f"{method_lower}_{target_name}"

            current_model = self.models[comp_key]
            current_model.contingency_table = None  # Reset to prevent duplication
            current_model.test_set = test_set
            current_model.test()

            # Append method column to report and normalize potential mixed int/float types
            if current_model.report is not None:
                rep_df = current_model.report.with_columns([
                    pl.lit(method_name).alias("method")
                ])
                # Safely cast any integer column (like 'support') to Float64 to avoid concat errors
                if "support" in rep_df.columns:
                    rep_df = rep_df.with_columns(pl.col("support").cast(pl.Float64))
                target_reports.append(rep_df.select(["method", pl.exclude("method")]))

            # Append method column to predictions and standardize prediction types
            pred_df = pl.concat([
                    self.test_sets[target_name].select(self.test_cols),
                    current_model.predictions,
                ], how="horizontal")
            pred_df = pred_df.with_columns([
                pl.lit(method_name).alias("method"),
                pl.col("class").cast(pl.Int64),
                pl.col("score").cast(pl.Float64)
            ])
            target_predictions.append(pred_df.select(["method", pl.exclude("method")]))

            # Append method column to contingency table and standardize prediction types
            if current_model.contingency_table is not None:
                ct_df = current_model.contingency_table.with_columns([
                    pl.lit(method_name).alias("method"),
                    pl.col("k").cast(pl.Int64),
                    pl.col("label").cast(pl.Int64),
                    pl.col("score").cast(pl.Float64)
                ])
                target_contingency.append(ct_df.select(["method", pl.exclude("method")]))

        self.reports[target_name] = pl.concat(target_reports) if target_reports else None
        self.predictions[target_name] = pl.concat(target_predictions) if target_predictions else None
        self.contingency_tables[target_name] = pl.concat(target_contingency) if target_contingency else None

    def read_models(self) -> None:
        """
        Read and restore each target's models from disk for all methods in the suite,
        storing the loaded models in :attr:`models`.
        """
        for target_name in self.config.get_target_names():
            for method_name, method_obj in self.base_model.method_objs.items():
                method_lower = getattr(method_obj, "short_name", method_name).lower()
                comp_key = f"{method_lower}_{target_name}"

                path = self.model_file_names.get(comp_key)
                if not path or not os.path.exists(path):
                    raise FileNotFoundError(f"File '{path}' does not exist.")

                config_method = copy.deepcopy(self.config)
                config_method.set_base_class("model", method_name)
                new_model_instance = load_single_model_class_with_class_name(config_method, method_name)
                new_model_instance.load_model(path)
                new_model_instance = new_model_instance.update_nthreads(new_model_instance)

                self.models[comp_key] = new_model_instance

    def create_metric_plots(self) -> None:
        """
        Override parent method to call the multi-method metric plotter.
        """
        create_multi_method_metric_plots(self)
