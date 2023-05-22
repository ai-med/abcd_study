import pandas as pd

from ..data.var_names import diagnoses
from .classifier_chain import ClassifierChainEnsemble
from .logistic_regression import (
    LogisticRegressionOVRPredictor, LogisticRegressionModel
)
from .xgboost_pipeline import DepthwiseXGBPipeline

logistic_regression_args = {
    'solver': 'lbfgs',
    'max_iter': 500,
    'class_weight': 'balanced'
}


class ModelIterator:
    def __init__(self, train_data, valid_data, features_selected, rnd):
        self.train_data = train_data
        self.valid_data = valid_data
        self.features_selected = features_selected
        self.rnd = rnd

    @property
    def data(self):
        return (self.train_data, self.valid_data,)

    def __iter__(self):
        ovr_predictor = LogisticRegressionOVRPredictor(
            features=self.features_selected,
            responses=diagnoses.features,
            model_args=logistic_regression_args,
            random_state=self.rnd.randint(0, 999999999)
        )
        data = pd.concat((self.train_data, self.valid_data))
        yield 'logistic_regression_ovr', ovr_predictor, (data,)
        del data, ovr_predictor

        lr_cce_predictor = ClassifierChainEnsemble(
            model=LogisticRegressionModel,
            features=self.features_selected,
            responses=diagnoses.features,
            num_chains=10,
            model_args=logistic_regression_args,
            random_state=self.rnd.randint(0, 999999999)
        )
        yield 'logistic_regression_cce', lr_cce_predictor, self.data
        del lr_cce_predictor

        xgboost_cce_predictor = ClassifierChainEnsemble(
            model=DepthwiseXGBPipeline,
            features=self.features_selected,
            responses=diagnoses.features,
            num_chains=10,
            model_args={
                'n_calls': 30
            },
            random_state=self.rnd.randint(0, 999999999)
        )
        yield 'xgboost_cce', xgboost_cce_predictor, self.data
