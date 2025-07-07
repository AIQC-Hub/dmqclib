import numpy as np
import polars as pl
from typing import Optional, Dict

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.prepare.step6_split_dataset.split_base import SplitDataSetBase


class SplitDataSetA(SplitDataSetBase):
    """
    A subclass of :class:`SplitDataSetBase` that splits feature data into
    training and test sets for BO NRT + Cora test data.

    This class performs the following tasks:

      - Randomly samples a fraction of rows for the test set.
      - Ensures matching positive and negative rows are grouped by shared
        identifiers (e.g., ``pair_id``).
      - Splits out the remainder into a training set.
      - Assigns k-fold indices to the training set rows.
      - Optionally drops columns that are not required for subsequent analysis.

    .. note::

       The docstring states "SplitDataSetBase split feature data into training
       and test sets," but since this is :class:`SplitDataSetA`, the wording
       can be tailored further to mention "SplitDataSetA."
    """

    expected_class_name: str = "SplitDataSetA"

    def __init__(
        self,
        config: ConfigBase,
        target_features: Optional[Dict[str, pl.DataFrame]] = None,
    ) -> None:
        """
        Initialize the dataset splitting class with configuration
        and target features.

        :param config: A dataset configuration object that specifies
                       paths, test-set fraction, and k-fold details.
        :type config: ConfigBase
        :param target_features: A dictionary mapping target names to DataFrames
                                containing extracted features. Defaults to None.
        :type target_features: dict of str to pl.DataFrame, optional
        """
        super().__init__(config, target_features=target_features)

        #: Column names used for intermediate processing (e.g., to maintain
        #: matching references between positive and negative rows).
        self.work_col_names = [
            "profile_id",
            "pair_id",
            "platform_code",
            "profile_no",
            "observation_no",
        ]

    def split_test_set(self, target_name: str) -> None:
        """
        Split the specified target's DataFrame into training and test sets.

        1. A random fraction of rows labeled 1 (positive) is sampled to form
           the test set.
        2. Rows labeled 0 (negative) with matching ``pair_id`` are joined
           to that test set.
        3. The remaining rows form the training set.

        :param target_name: The target name identifying which DataFrame in
                            :attr:`target_features` to split.
        :type target_name: str
        """
        test_set_fraction = self.get_test_set_fraction()

        pos_test_set = (
            self.target_features[target_name]
            .filter(pl.col("label") == 1)
            .sample(fraction=test_set_fraction, shuffle=True)
        )

        neg_test_set = (
            self.target_features[target_name]
            .filter(pl.col("label") == 0)
            .join(pos_test_set.select([pl.col("pair_id")]), on="pair_id")
        )

        test_set = pos_test_set.vstack(neg_test_set)
        # Reassemble the final test set with "row_id" positioned as the first column.
        self.test_sets[target_name] = pl.concat(
            [
                test_set.select(["row_id"]),
                test_set,
            ],
            how="align_left",
        )

        self.training_sets[target_name] = self.target_features[target_name].join(
            self.test_sets[target_name].select([pl.col("row_id")]),
            on="row_id",
            how="anti",
        )

    def add_k_fold(self, target_name: str) -> None:
        """
        Assign a k-fold identifier to each row in the training set for cross-validation.

        1. Extracts rows labeled 1 (positive) and unevenly distributes them across
           the specified number of folds.
        2. Joins negative rows based on ``pair_id`` so they share the same fold
           assignment.

        :param target_name: The target name identifying the training set
                            within :attr:`training_sets`.
        :type target_name: str
        """
        k_fold = self.get_k_fold()
        pos_training_set = self.training_sets[target_name].filter(pl.col("label") == 1)
        df_size = pos_training_set.shape[0]

        n_per_value = df_size // k_fold
        k_values = np.array(
            [i for i in range(1, k_fold + 1) for _ in range(n_per_value)]
        )
        remaining = df_size % k_fold
        k_values = np.concatenate(
            [k_values, np.random.choice(range(1, k_fold + 1), remaining)]
        )
        np.random.shuffle(k_values)

        pos_training_set = pos_training_set.with_columns(pl.Series("k_fold", k_values))
        neg_training_set = (
            self.training_sets[target_name]
            .filter(pl.col("label") == 0)
            .join(
                pos_training_set.select([pl.col("pair_id"), pl.col("k_fold")]),
                on="pair_id",
            )
        )

        training_set = pos_training_set.vstack(neg_training_set)

        # Reassemble the final training set with "k_fold" positioned as the first column.
        self.training_sets[target_name] = pl.concat(
            [
                training_set.select(["k_fold", "row_id"]),
                training_set.drop(["k_fold"]),
            ],
            how="align_left",
        )

    def drop_columns(self, target_name: str) -> None:
        """
        Remove specified working columns from both the training and test sets,
        leaving only the essential columns for subsequent steps.

        :param target_name: The target name identifying which training and test sets
                            to modify.
        :type target_name: str
        """
        self.training_sets[target_name] = self.training_sets[target_name].drop(
            self.work_col_names
        )
        self.test_sets[target_name] = self.test_sets[target_name].drop(
            self.work_col_names
        )
