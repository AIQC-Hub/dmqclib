import polars as pl
import xgboost as xgb
from typing import Dict, Any

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.common.base.model_base import ModelBase


class XGBoost(ModelBase):
    """
    An XGBoost model wrapper class for training and testing using Polars data.

    Inherits from :class:`ModelBase` and implements the ``build`` and ``test`` methods
    specifically for an XGBoost classifier.

    Features include:

    - Conversion of Polars DataFrames to Pandas for compatibility with XGBoost.
    - Automatic application of ``model_params`` from the YAML config, if defined;
      otherwise, uses default hyperparameters.
    - Computation and storage of metrics (accuracy, balanced accuracy,
      classification report) in :attr:`result`.

    .. note::

       This class sets :attr:`expected_class_name` to ``"XGBoost"``, ensuring
       it can be matched in the YAML configuration if used within a
       loader or factory pattern.
    """

    expected_class_name: str = "XGBoost"

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the XGBoost model with default or user-specified parameters.

        :param config: A configuration object providing model parameters
                       (e.g., learning rate, max depth) and other metadata.
        :type config: ConfigBase
        :raises ValueError: If inherited requirements of :class:`ModelBase`
                            (like missing attributes) are not satisfied.
        """
        super().__init__(config)

        params: Dict[str, Any] = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "eval_metric": "logloss",
        }
        self.model_params = params if len(self.model_params) == 0 else self.model_params

    def build(self) -> None:
        """
        Train the XGBoost classifier using the assigned training set.

        Steps:

          1. Convert the Polars DataFrame (:attr:`training_set`) to Pandas.
          2. Separate features (X) and labels (y).
          3. Initialize and fit an XGBoost classifier with
             :attr:`model_params`.

        :raises ValueError: If :attr:`training_set` is None or empty.
        """
        if self.training_set is None:
            raise ValueError("Member variable 'training_set' must not be empty.")

        x_train = self.training_set.select(pl.exclude("label")).to_pandas()
        y_train = self.training_set["label"].to_pandas()

        self.model = xgb.XGBClassifier(**self.model_params)
        self.model.fit(x_train, y_train)

    def test(self) -> None:
        """
        Evaluate the trained XGBoost classifier on the assigned test set.

        Steps:

          1. Convert the Polars DataFrame (:attr:`test_set`) to Pandas.
          2. Generate predictions.
          3. Compute metrics such as accuracy, balanced accuracy, and
             precision/recall/f1 scores via :func:`sklearn.metrics.classification_report`.
          4. Store results in :attr:`result` as a Polars DataFrame.

        The :attr:`k` attribute (provided by parent class or
        cross-validation context) is used to identify the fold number:

          - If :attr:`k` is 0, the column is dropped from the final :attr:`result`.

        :raises ValueError: If :attr:`test_set` is None or empty.
        """
        if self.test_set is None:
            raise ValueError("Member variable 'test_set' must not be empty.")

        x_test = self.test_set.select(pl.exclude("label")).to_pandas()
        y_test = self.test_set["label"].to_pandas()

        y_pred = self.model.predict(x_test)

        # Build the base result DataFrame with placeholders for the "0" and "1" labels.
        self.result = pl.DataFrame(
            [
                {"k": self.k, "label": "0", "accuracy": None},
                {"k": self.k, "label": "1", "accuracy": None},
                {
                    "k": self.k,
                    "label": "macro avg",
                    "accuracy": accuracy_score(y_test, y_pred),
                },
                {
                    "k": self.k,
                    "label": "weighted avg",
                    "accuracy": balanced_accuracy_score(y_test, y_pred),
                },
            ]
        )

        # Join with the classification report for precision, recall, etc.
        classification_dict = classification_report(y_test, y_pred, output_dict=True)
        report_rows = []
        for label_key, metrics_dict in classification_dict.items():
            if isinstance(metrics_dict, dict):  # skip 'accuracy' row
                report_rows.append(
                    {
                        "k": self.k,
                        "label": label_key,
                        "precision": metrics_dict["precision"],
                        "recall": metrics_dict["recall"],
                        "f1-score": metrics_dict["f1-score"],
                        "support": metrics_dict["support"],
                    }
                )

        self.result = self.result.join(
            pl.DataFrame(report_rows),
            on=["k", "label"],
            how="left",
        )

        if self.k == 0:
            self.result = self.result.drop(["k"])
