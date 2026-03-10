"""
This module defines `SklearnModelBase`, an abstract base class for models
that adhere to the Scikit-Learn API (including XGBoost and native sklearn models).

It implements common workflows for data conversion, model building,
prediction, and reporting, reducing code duplication across specific
algorithm implementations.
"""

from abc import abstractmethod
from typing import Any, Self

import polars as pl
from sklearn.metrics import classification_report

from dmqclib.common.base.config_base import ConfigBase
from dmqclib.common.base.model_base import ModelBase


class SklearnModelBase(ModelBase):
    """
    Abstract base class for Scikit-Learn compatible models.

    This class implements the standard lifecycle methods (:meth:`build`,
    :meth:`test`, :meth:`predict`, :meth:`create_report`) assuming the
    underlying model object supports the standard ``fit``, ``predict``,
    and ``predict_proba`` methods.

    Subclasses must implement:
      - :meth:`_get_model_class`: To return the specific class type (e.g. XGBClassifier).
    """

    def __init__(self, config: ConfigBase) -> None:
        """
        Initialize the model base.

        :param config: A configuration object.
        :type config: ConfigBase
        """
        super().__init__(config=config)
        # self.model_params must be initialized by the child class before usage.

    @abstractmethod
    def _get_model_class(self) -> Any:
        """
        Return the class type of the underlying model to be instantiated.

        :return: The class object (e.g., xgboost.XGBClassifier, sklearn.linear_model.LogisticRegression).
        """
        pass

    def build(self) -> None:
        """
        Train the classifier using the assigned training set.

        Steps:
          1. Convert the Polars DataFrame (:attr:`training_set`) to Pandas.
          2. Separate features (X) and labels (y).
          3. Initialize the model class provided by :meth:`_get_model_class`
             with :attr:`model_params`.
          4. Fit the model.

        :raises ValueError: If :attr:`training_set` is ``None`` or empty.
        """
        if self.training_set is None:
            raise ValueError("Member variable 'training_set' must not be empty.")

        x_train = self.training_set.select(pl.exclude("label")).to_pandas()
        y_train = self.training_set["label"].to_pandas()

        model_class = self._get_model_class()
        self.model = model_class(**self.model_params)
        self.model.fit(x_train, y_train)

    def test(self) -> None:
        """
        Evaluate the trained classifier on the assigned test set.

        Steps:
          1. Call :meth:`predict` to generate predictions on the test set.
          2. Call :meth:`create_report` to compute metrics.
          3. Call :meth:`update_contingency_table` to store scores.

        :raises ValueError: If :attr:`test_set` is ``None``.
        """
        self.predict()
        self.create_report()
        self.update_contingency_table()

    def update_nthreads(self, model: Self) -> Self:
        """
        Update the number of threads set in the model.

        :param model: The model needs to be updated.
        :type model: Self
        :return: The updated model instance.
        """
        if "n_jobs" in self.model_params and hasattr(model.model, "n_jobs"):
            model.model.n_jobs = self.model_params["n_jobs"]

        return model

    def predict(self) -> None:
        """
        Generates predictions for the test set using the trained model.

        Converts the Polars test set to a Pandas DataFrame, makes predictions,
        and stores the results in :attr:`predictions`.

        :raises ValueError: If :attr:`test_set` is ``None``.
        """
        if self.test_set is None:
            raise ValueError("Member variable 'test_set' must not be empty.")

        x_test = self.test_set.select(pl.exclude("label")).to_pandas()

        self.predictions = pl.DataFrame(
            {
                "class": self.model.predict(x_test),
                "score": self.model.predict_proba(x_test)[:, 1],
            }
        )

    def create_report(self) -> None:
        """
        Computes and compiles a comprehensive classification report based on test results.

        Calculates precision, recall, f1-score, and support using
        :func:`sklearn.metrics.classification_report`. Stores the result
        in :attr:`report`.

        :raises ValueError: If :attr:`test_set` or :attr:`predictions` are ``None``.
        """
        if self.test_set is None:
            raise ValueError("Member variable 'test_set' must not be empty.")

        if self.predictions is None:
            raise ValueError("Member variable 'predictions' must not be empty.")

        y_test = self.test_set["label"].to_pandas()
        y_pred = self.predictions["class"].to_pandas()

        classification_dict = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        report_rows = []

        for label_key, metrics in classification_dict.items():
            if label_key == "accuracy":
                report_rows.append(
                    {"k": self.k, "metric_type": "overall_accuracy", "value": metrics}
                )
            elif label_key == "macro avg":
                balanced_accuracy = metrics.get("recall")
                report_rows.append(
                    {
                        "k": self.k,
                        "metric_type": "balanced_accuracy",
                        "value": balanced_accuracy,
                    }
                )
                report_rows.append(
                    {
                        "k": self.k,
                        "metric_type": "classification_report",
                        "label": label_key,
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1-score": metrics.get("f1-score"),
                        "support": metrics.get("support"),
                    }
                )
            else:
                report_rows.append(
                    {
                        "k": self.k,
                        "metric_type": "classification_report",
                        "label": label_key,
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1-score": metrics.get("f1-score"),
                        "support": metrics.get("support"),
                    }
                )

        self.report = pl.DataFrame(report_rows)

        if self.k == 0:
            self.report = self.report.drop("k")
