from typing import List
import pandas as pd

from sklearn.utils import check_random_state


class ValidationClassifierChain:
    """Classifier chain that takes training and validation set. Code based on
    sklearn.multioutput.ClassifierChain

    Args:
        model (DepthwiseXGBPipeline or LogisticRegressionModel): Class of
            base classifier model.
        features (list of str): List of features (predictors).
        responses (list of str): List of responses (dependent variables).
        model_args (dict, optional): Arguments passed to model constructor.
        random (bool, optional): Wether to randomly change prediction order
            of responses.
        random_state (int, optional): Random number seed.
    """

    def __init__(self,
                 model,
                 features: List[str],
                 responses: List[str],
                 model_args: dict = {},
                 random: bool = True,
                 random_state: int = None):
        self.model = model
        self.model_args = model_args
        self.features = features
        self.responses = responses
        self.random = random
        self.random_state = random_state

    def fit(self,
            train: pd.DataFrame,
            valid: pd.DataFrame):
        """Fit model using training and validation sets.

        Args:
            train (pd.DataFrame): Training set
            valid (pd.DataFrame): Validation set

        Returns:
            ValidationClassifierChain: self
        """

        self.responses_ = self.responses
        if self.random:
            random_state = check_random_state(self.random_state)
            self.responses_ = random_state.permutation(self.responses_)

        self.estimators_ = []
        for chain_idx, y_col in enumerate(self.responses_):
            estimator = self.model(
                y_col=y_col,
                include_cols=self.features + list(self.responses_[:chain_idx]),
                random_state=self.random_state,
                **self.model_args
            )
            estimator.fit(train, valid)
            self.estimators_.append(estimator)

        return self

    def predict_proba(self,
                      data: pd.DataFrame) -> pd.DataFrame:
        """Predict probability estimates.

        Args:
            data (pd.DataFrame): Input data
        """
        data_ = data[self.features]
        predictions_df = pd.DataFrame(columns=[], index=data.index)
        for chain_idx, (estimator, response) in \
                enumerate(zip(self.estimators_, self.responses_)):
            predictions_df[response] = estimator.predict(data_)
            # Predict new labels based on threshold predictions. Set threshold
            # to 0.5 arbitrarily.
            data_.loc[:, response] = predictions_df[response] > .5

        predictions_df = predictions_df[self.responses]
        return predictions_df


class ClassifierChainEnsemble:
    """Ensemble of ValidationClassifierChains.

    Args:
        model (DepthwiseXGBPipeline or LogisticRegressionModel): Class of
            base classifier model.
        features (list of str): List of features (predictors).
        responses (list of str): List of responses (dependent variables).
        num_chains (int): Number of individual chains.
        model_args (dict, optional): Arguments passed to model constructor.
        random_state (int, optional): Random number seed.
    """

    def __init__(self,
                 model,
                 features: List[str],
                 responses: List[str],
                 num_chains: int = 10,
                 model_args: dict = {},
                 random_state: int = None):
        self.chains = [
            ValidationClassifierChain(
                model=model,
                features=features,
                responses=responses,
                model_args=model_args,
                random=True,
                random_state=random_state + i
            ) for i in range(num_chains)
        ]

    def fit(self,
            train: pd.DataFrame,
            valid: pd.DataFrame):
        """Fit model using training and validation sets.

        Args:
            train (pd.DataFrame): Training set
            valid (pd.DataFrame): Validation set
        """
        for chain in self.chains:
            chain.fit(train, valid)

    def predict(self,
                data: pd.DataFrame) -> pd.DataFrame:
        """Predict probability estimates.

        Args:
            data (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Dataframe where columns are responses and index is
                same as data.index.
        """
        predictions = [chain.predict_proba(data) for chain in self.chains]
        y_pred_ensemble = sum(predictions) / len(predictions)
        return pd.DataFrame(
            y_pred_ensemble, columns=self.chains[0].responses, index=data.index
        )
