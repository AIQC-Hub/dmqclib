import os
from typing import Dict, Optional

import polars as pl

from dmqclib.common.base.dataset_base import DataSetBase
from dmqclib.common.base.config_base import ConfigBase


class ConcatDatasetsBase(DataSetBase):
    """
    Abstract base class for concatenating predictions and the original dataset.

    Inherits from :class:`DataSetBase` to ensure configuration consistency.
    The concatenated dataset, once generated, can be written to Parquet files.
    """

    def __init__(
        self,
        config: ConfigBase,
        input_data: Optional[pl.DataFrame] = None,
        predictions: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> None:
        """
        Initialize the feature extraction base class.

        :param config: The configuration object, containing paths and target definitions.
        :type config: ConfigBase
        :param input_data: A Polars DataFrame providing the full dataset from which
                           features are extracted, defaults to None.
        :type input_data: pl.DataFrame, optional
        :param predictions: A dictionary mapping each target to its respective
                            subset of predictions,
                            defaults to None.
        :type predictions: dict of str to pl.DataFrame, optional
        :raises NotImplementedError: If the subclass does not define
                                     ``expected_class_name`` (when instantiating a real subclass).
        :raises ValueError: If the provided YAML config does not match this class's
                            ``expected_class_name``.
        """
        super().__init__("concat", config)

        #: The default pattern to use when writing feature files for each target.
        self.default_file_name: str = "predictions.parquet"

        #: Output file name to store the concatenated dataset
        self.output_file_name: str = self.config.get_full_file_name(
            "concat", self.default_file_name
        )

        self.input_data: Optional[pl.DataFrame] = input_data

        #: A dict of Polars DataFrames, one per target, containing classification results.
        self.predictions: Optional[Dict[str, pl.DataFrame]] = predictions

        self.merged_predictions: pl.DataFrame = None

    def merge_predictions(self) -> None:
        self.merged_predictions = (
            self.input_data.join(
                pl.concat([
                    df.rename({
                        'label': f'{key}_label',
                        'predicted': f'{key}_predicted'
                    }).select([
                        'platform_code',
                        'profile_no',
                        'observation_no',
                        f'{key}_label',
                        f'{key}_predicted'
                    ])
                    for key, df in self.predictions.items()
                ], how="align"),
                on = ["platform_code", "profile_no", "observation_no"]
            )
        )

    def write_merged_predictions(self) -> None:
        if self.merged_predictions is None:
            raise ValueError("Member variable 'merged_predictions' must not be empty.")

        os.makedirs(os.path.dirname(self.output_file_name), exist_ok=True)
        self.merged_predictions.write_parquet(self.output_file_name)
