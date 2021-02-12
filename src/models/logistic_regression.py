from typing import List

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


class LogisticRegressionModel:
    """Logistic regression model with same interface as DepthWiseXGBPipeline.
    Trains on concatenation of training and validation sets.
    """

    def __init__(self,
                 y_col: str,
                 include_cols: List[str],
                 random_state: int = None,
                 solver: str = 'lbfgs',
                 max_iter: int = 300,
                 class_weight: str = 'balanced'):
        self.y_col = y_col
        self.include_cols = include_cols
        self.model = LogisticRegression(
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight
        )

    def fit(self, train, valid):
        total = pd.concat((train, valid))
        self.model.fit(total[self.include_cols], total[self.y_col])

    def predict(self, data):
        return self.model.predict_proba(data[self.include_cols])[:, 1]


class LogisticRegressionOVRPredictor:

    def __init__(self,
                 features: List[str],
                 responses: List[str],
                 model_args: dict = {},
                 random_state: int = None):
        self.features = features
        self.responses = responses
        logistic_regression = LogisticRegression(
            random_state=random_state, **model_args
        )
        self.ovr_classifier = OneVsRestClassifier(logistic_regression)

    def fit(self,
            train: pd.DataFrame):
        self.ovr_classifier.fit(train[self.features], train[self.responses])

    def predict(self,
                data: pd.DataFrame):
        y_pred = self.ovr_classifier.predict_proba(data[self.features])
        return pd.DataFrame(y_pred, columns=self.responses, index=data.index)
