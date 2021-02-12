from abc import ABC, ABCMeta, abstractmethod
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skopt import gp_minimize, dump, load
from skopt.space import Dimension, Real
from skopt.utils import use_named_args
import xgboost as xgb

from sklearn.metrics import roc_curve, auc, log_loss


LOG = logging.getLogger(__name__)

ObjectiveFn = Callable[[Dict[str, Any]], float]
MakeObjectiveFn = Callable[[], ObjectiveFn]
OptStrList = Optional[List[str]]


class ErrorFunctions:
    """Contains static error function classes."""
    
    class BaseErrorFunction(ABC):
        """Parent class of error function classes.
        These are callable classes and contain the name of the error function.
        """
        
        name = None
        
        @abstractmethod
        def __call__(self,
                     y_pred: pd.Series,
                     y_true: pd.Series) -> float:
            pass
    
    class log_loss(BaseErrorFunction):
        """Aka. binary cross entropy"""
    
        name = "log_loss"

        def __call__(self,
                     y_pred: pd.Series,
                     y_true: pd.Series) -> float:
            if isinstance(y_pred, pd.Series):
                y_true, y_pred = y_true.align(y_pred, axis=0, join='inner')
            return log_loss(y_true, y_pred, eps=1e-5, labels=[0, 1])
        
    class negative_roc_auc(BaseErrorFunction):
        """Area under ROC curve, times (-1)"""
        
        name = "negative_roc_auc"
        
        def __call__(self,
                     y_pred: pd.Series,
                     y_true: pd.Series) -> float:
            if isinstance(y_pred, pd.Series):
                y_true, y_pred = y_true.align(y_pred, axis=0, join='inner')
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            return -auc(fpr, tpr)


class BasePipeline(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self,
                 y_col: str,
                 error_function: Callable,
                 include_cols: OptStrList = None,
                 n_calls: int = 100,
                 random_state: int = 0) -> None:
        self.include_cols = include_cols
        self.y_col = y_col
        self.n_calls = n_calls
        self.random_state = random_state
        self.error_function = error_function

        self.columns_ = None
        self.bayes_search_result_ = None
        self.model_ = None

    @property
    @abstractmethod
    def search_space(self) -> List[Dimension]:
        pass

    @property
    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        pass

    @property
    def initial_params(self):
        return None

    def _space_to_dict(self, space, filter_func=None):
        p = {}
        for val, s in zip(space, self.search_space):
            if filter_func is None or filter_func(s.name):
                p[s.name] = val
        return p

    def _fit_bayes_search(self,
                          make_objective_fn: MakeObjectiveFn) -> None:
        search_objective = make_objective_fn()

        LOG.info('Training model with Bayesian hyper-parameter search')
        
        res_gp = gp_minimize(
            func=search_objective,
            dimensions=self.search_space,
            x0=self.initial_params,
            n_calls=self.n_calls,
            random_state=self.random_state
        )

        LOG.info('Minimum is %.10f', res_gp.fun)

        self.bayes_search_result_ = res_gp

    @abstractmethod
    def fit(self, data_train: pd.DataFrame, data_valid: pd.DataFrame):
        """Fit the model."""

    def predict(self, data_test: pd.DataFrame) -> pd.Series:
        assert set(self.include_cols).issubset(data_test.columns), \
            ("data_test does not contain columns specified as 'include_cols' "
             "and 'y_col'")
        x_test = data_test[self.include_cols]
        y_pred = self._predict(x_test)
        return pd.Series(y_pred, index=data_test.index)

    def _predict(self, x_test):
        y_pred = self.model_.predict(x_test)
        return y_pred


class BaseXGBPipeline(BasePipeline):

    @abstractmethod
    def __init__(self,
                 y_col: str,
                 model_dir: str = None,
                 error_function: Callable = ErrorFunctions.log_loss(),
                 include_cols: OptStrList = None,
                 n_calls: int = 100,
                 random_state: int = 0) -> None:
        super(BaseXGBPipeline, self).__init__(
            y_col=y_col,
            error_function=error_function,
            include_cols=include_cols,
            n_calls=n_calls,
            random_state=random_state,
        )
        if model_dir is None:
            self.model_dir = None
        else:
            self.model_dir = Path(model_dir)
            self.gp_file = self.model_dir / 'result-skopt_minimize.gz'
            self.xgb_file = self.model_dir / 'model-xgb.dat'
            self.xgb_stats_file = self.model_dir / 'stats-xgb.txt'

    def fit(self, data_train: pd.DataFrame, data_valid: pd.DataFrame):
        """Fit pipeline.

        The model is fit on the training local_data, including tuning of hyper-parameters.
        Performance is evaluated on `data_valid` and the best hyper-parameters are
        retained.
        """

        assert set(self.include_cols).issubset(data_train.columns) and \
                self.y_col in data_train.columns,\
            ("data_train does not contain columns specified as 'include_cols' "
             "and 'y_col'")
        assert set(self.include_cols).issubset(data_valid.columns) and \
                self.y_col in data_valid.columns,\
            ("data_valid does not contain columns specified as 'include_cols' "
             "and 'y_col'")

        dtrain = xgb.DMatrix(
            data=data_train[self.include_cols],
            label=data_train[self.y_col]
        )

        if self.model_dir is not None:
            # model_dir has been specified. Load from and save to this directory
            gp_file = self.gp_file
            if gp_file.exists():
                LOG.info('Loading hyper-parameters from %s', gp_file)
                self.bayes_search_result_ = load(str(gp_file))

                LOG.info('Loading model from %s', self.xgb_file)
                est = xgb.Booster(self.hparams, model_file=str(self.xgb_file))
            else:
                if not self.model_dir.exists():
                    self.model_dir.mkdir(parents=True)

                dvalid = xgb.DMatrix(
                    data=data_valid[self.include_cols],
                    label=data_valid[self.y_col]
                )
                self._fit_bayes_search(
                    lambda: self._make_objective_fn(dtrain, dvalid)
                )
                LOG.info('Writing hyper-parameters to %s', gp_file)
                dump(self.bayes_search_result_, str(gp_file), store_objective=False)

                est = self._fit_xgboost(dtrain)

                LOG.info('Writing model to %s', self.xgb_file)
                est.save_model(str(self.xgb_file))
                est.dump_model(str(self.xgb_stats_file), with_stats=True)
        else:
            # model_dir has not been specified. Fit model without saving model
            # parameters.
            dvalid = xgb.DMatrix(
                data=data_valid[self.include_cols],
                label=data_valid[self.y_col]
            )
            self._fit_bayes_search(
                lambda: self._make_objective_fn(dtrain, dvalid)
            )
            est = self._fit_xgboost(dtrain)
        self.model_ = est

        return self

    def _make_objective_fn(self,
                           dtrain: xgb.DMatrix,
                           dvalid: xgb.DMatrix) -> ObjectiveFn:
        n_repeats = 100

        @use_named_args(self.search_space)
        def search_objective(**params):
            xparam = self.default_params
            xparam.update(params)
            num_boost_round = xparam.pop('num_boost_round')

            est = xgb.train(
                params=xparam, dtrain=dtrain, num_boost_round=num_boost_round
            )

            # Evaluate on 100 different 66% subsamples of validation set.
            valid_labels = dvalid.get_label().astype(int)
            valid_labels_indexed = np.vstack(
                (valid_labels, np.arange(len(valid_labels)))
            ).T

            errors = np.empty(n_repeats, dtype=float)
            for i in range(n_repeats):

                # Use train_test_split to stratify with respect to label
                sidx, _ = train_test_split(
                    valid_labels_indexed,
                    test_size=0.333,
                    stratify=valid_labels_indexed[:, 0],
                    random_state=self.random_state+i
                )
                sidx = sidx[:, 1].astype(int)

                d = dvalid.slice(sidx)
                pred = est.predict(d)
                errors[i] = self.error_function(y_pred=pred, y_true=d.get_label())

            return np.mean(errors) + np.std(errors, ddof=1)

        return search_objective

    @property
    def hparams(self):
        """Returns selected hyper-parameters in XGB format"""
        p = self._space_to_dict(
            self.bayes_search_result_.x,
            None
        )
        p.update(self.default_params)
        if 'num_boost_round' in p.keys():
            p.pop('num_boost_round')
        return p

    @property
    def num_boost_round(self):
        """Returns selected number of boosting rounds"""
        if 'num_boost_round' in self.default_params.keys():
            return self.default_params['num_boost_round']
        p = self._space_to_dict(
            self.bayes_search_result_.x,
            lambda x: x == 'num_boost_round'
        )
        return p['num_boost_round']

    def _fit_xgboost(self, dtrain: xgb.DMatrix) -> xgb.Booster:
        xparam = self.hparams
        LOG.info('Fitting model with final parameters: %s', xparam)
        est = xgb.train(
            params=xparam, dtrain=dtrain, num_boost_round=self.num_boost_round
        )
        return est

    def predict(self, data_test: pd.DataFrame) -> pd.Series:
        """Return predictions"""
        assert set(self.include_cols).issubset(data_test.columns), \
            ("data_test does not contain columns 'include_cols'")
        dtest = xgb.DMatrix(data=data_test[self.include_cols])
        y_pred = super(BaseXGBPipeline, self)._predict(dtest)
        return pd.Series(y_pred, index=data_test.index)


class DepthwiseXGBPipeline(BaseXGBPipeline):
    """Fit ensemble of gradient boosted regression trees.

    Args:
        y_col (str): Name of column denoting the prediction target.
        model_dir (str): Path to directory to write fitted model and
            hyper-parameters to.
        included_cols (list of str, optional): Names of columns in the
            local_data to consider during training or `None` to include all.
        n_calls (int, optional): Number of iterations for hyper-parameter search.
        random_state (int, optional): Random number seed.
    """

    def __init__(self,
                 y_col: str,
                 model_dir: str = None,
                 error_function: Callable = ErrorFunctions.log_loss(),
                 include_cols: OptStrList = None,
                 n_calls: int = 100,
                 random_state: int = 0) -> None:
        super(DepthwiseXGBPipeline, self).__init__(
            y_col=y_col,
            model_dir=model_dir,
            error_function=error_function,
            include_cols=include_cols,
            n_calls=n_calls,
            random_state=random_state,
        )

    @property
    def search_space(self):
        """Returns search space for hyper-parameter search"""
        space = [
            Real(10**-5, 1.25, "log-uniform", name='learning_rate'),
            Real(10**-6, 4096, "log-uniform", name='alpha'),  # L1 regularization term on weights
            Real(10**-6, 2048, "log-uniform", name='lambda')  # L2 regularization term on weights
        ]
        return space

    @property
    def default_params(self):
        """Returns parameters that are not optimized"""
        xparam = {
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'num_boost_round': 1000,
            'max_depth': 2,
            'subsample': 0.5,
            'grow_policy': 'depthwise',
            'verbosity': 1,
            'disable_default_eval_metric': 1
        }
        return xparam

    @property
    def initial_params(self):
        """Initial hyper-parameter configuration"""
        return [0.01, 1.0, 1.0]
