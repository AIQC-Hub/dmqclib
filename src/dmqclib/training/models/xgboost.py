import polars as pl
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)

from dmqclib.common.base.model_base import ModelBase


class XGBoost(ModelBase):
    expected_class_name = "XGBoost"

    def __init__(self, dataset_name: str, config_file: str = None):
        super().__init__(
            dataset_name,
            config_file=config_file,
        )

        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "eval_metric": "logloss",
        }
        self.model_params = params if len(self.model_params) == 0 else self.model_params

    def build(self):
        """
        Build model
        """
        if self.training_set is None:
            raise ValueError("Member variable 'training_set' must not be empty.")

        x_train = self.training_set.select(pl.exclude("label")).to_pandas()
        y_train = self.training_set["label"].to_pandas()

        self.built_model = xgb.XGBClassifier(**self.model_params)
        self.built_model.fit(x_train, y_train)

    def test(self):
        """
        Test model.
        """
        if self.built_model is None:
            raise ValueError("Member variable 'built_model' must not be empty.")

        if self.test_set is None:
            raise ValueError("Member variable 'test_set' must not be empty.")

        x_test = self.test_set.select(pl.exclude("label")).to_pandas()
        y_test = self.test_set["label"].to_pandas()

        y_pred = self.built_model.predict(x_test)

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
        ).join(
            pl.DataFrame(
                [
                    {
                        "k": self.k,
                        "label": k,
                        "precision": v["precision"],
                        "recall": v["recall"],
                        "f1-score": v["f1-score"],
                        "support": v["support"],
                    }
                    for k, v in classification_report(
                        y_test, y_pred, output_dict=True
                    ).items()
                    if isinstance(v, dict)
                ]
            ),
            on=["k", "label"],
            how="left",
        )
