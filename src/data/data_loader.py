from typing import List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline


def encode_multilabel(multilabel_data: pd.DataFrame):
    """Encode multiple columns with binary target values as a single column with only
    one unique number."""
    encodings = []
    for row in multilabel_data.iterrows():
        binary = 0
        for col in multilabel_data.columns:
            binary = (binary << 1) + row[1][col].astype(int)
        encodings.append(binary)
    return pd.Series(data=encodings, index=multilabel_data.index)


def train_test_split_noproblem(df: pd.DataFrame,
                               test_size: float,
                               stratify,
                               random_state: int = None)\
        -> (pd.DataFrame, pd.DataFrame):
    # Some codes occur only once in the dataset. train_test_split does not work
    # then. For this reason we need to exclude them and split them randomly
    # afterwards.
    val_counts = stratify.value_counts()
    unique_codes = df.loc[stratify.isin(val_counts[val_counts == 1].index)]
    all_data_nonunique = df.drop(index=unique_codes.index)

    train_nonunique, test_nonunique = train_test_split(
        all_data_nonunique,
        test_size=test_size,
        stratify=stratify.drop(index=unique_codes.index),
        random_state=random_state
    )
    # Split rows with unique codes without stratification
    train_unique, test_unique = train_test_split(
        unique_codes, test_size=test_size, random_state=random_state
    )

    return (
        pd.concat((train_nonunique, train_unique)).sample(
            frac=1, random_state=random_state
        ),
        pd.concat((test_nonunique, test_unique)).sample(
            frac=1, random_state=random_state
        )
    )


class DropUseless(VarianceThreshold):

    def __init__(self, threshold: float = 0.0) -> None:
        super(DropUseless, self).__init__(threshold=threshold)

    def transform(self, X):
        Xt = super(DropUseless, self).transform(X)
        if isinstance(X, pd.DataFrame):
            Xt = pd.DataFrame(Xt, index=X.index, columns=X.columns[self.get_support()])
        return Xt


class RepeatedStratifiedKFoldDataloader:

    def __init__(self,
                 dataframe: pd.DataFrame,
                 features: List[str],
                 responses: List[str],
                 confounders: List[str],
                 n: int,
                 k: int,
                 val_ratio: float,
                 random_state: int = None,
                 ignore_adjustment: bool = False):
        self.df = dataframe
        self.features = features
        self.responses = responses
        self.confounders = confounders
        self.n = n
        self.k = k
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.ignore_adjustment = ignore_adjustment
        self.clean()
        # Encode multilabel subject assignment as one number. This will be
        # necessary for stratification with RepeatedStratifiedKFold.
        self.encoded_responses = encode_multilabel(self.df[self.responses])

    def clean(self):
        # Prepare dataset: Remove subjects with missing features, responses or confounders
        self.df = self.df.dropna(subset=self.features + self.confounders + self.responses)

    @property
    def df_size(self):
        return len(self.df)

    def __iter__(self):
        rskf = RepeatedStratifiedKFold(
            n_splits=self.k, n_repeats=self.n, random_state=self.random_state
        )
        self.split = rskf.split(self.df, self.encoded_responses)
        return self

    def __next__(self):
        trainval_indices, test_indices = next(self.split)
        trainval_indices = self.df.iloc[trainval_indices].index
        test_indices = self.df.iloc[test_indices].index
        _train, _valid = train_test_split_noproblem(
            df=self.encoded_responses.loc[trainval_indices],
            test_size=self.val_ratio,
            stratify=self.encoded_responses.loc[trainval_indices],
            random_state=self.random_state
        )
        train_indices, valid_indices = _train.index, _valid.index
        if not self.ignore_adjustment:
            output = self.residualize_features(self.df, train_indices)
        else:
            output = self.df.copy()
        output, features_selected = self.transform_features(output, train_indices)
        return (
            output.loc[train_indices],
            output.loc[valid_indices],
            output.loc[test_indices],
            features_selected
        )

    def residualize_features(self,
                             dataframe: pd.DataFrame,
                             train_indices: np.array) -> pd.DataFrame:
        """
        Regress out effect of confounding factors on features.
          1. Fit a linear regression model with self.confounders as independent variables
             to each brain structural feature in turn. Use only training set data for this
             to avoid data leakage from validation and test sets.
          2. Compute residuals of each brain structural feature by substracting from the
             true value the value computed by the related linear regression model. Do this
             for every subject in the training, validation, and test set (using the
             linear regression models fitted on training data)

        Parameters
        ==========

        dataframe : pd.DataFrame
            Dataframe containing features and confounders

        train_indices : np.array
            Array of indices of training set subjects

        Return value
        ============

        output : pd.DataFrame
            Complete dataset where each brain structural feature is replaced by its residualized
            value
        """

        output = dataframe.copy()

        # Do not residualize if there are no confounders
        if len(self.confounders) == 0:
            return output

        # Fit a linear regression model to each brain feature in turn and obtain residuals
        X_reg_train = output.loc[train_indices][self.confounders]
        X_reg_full = output[self.confounders]

        for feature in self.features:
            y_reg_train = output.loc[train_indices][feature]

            # Set up and fit linear regression model
            regression_model = LinearRegression()
            regression_model.fit(X_reg_train, y_reg_train)

            # Compute residuals and store
            output[feature] = output[feature] - regression_model.predict(X_reg_full)

        return output

    def transform_features(self,
                           dataframe: pd.DataFrame,
                           train_indices: np.array) -> (pd.DataFrame, List[str]):
        """
        In order, apply basic feature selection and linear transformation to
        features:
            1. Drop features with negligible variance
            2. Rescale features by applying a RobustScaler
        Do all fitting using training set data prior to transforming all data to
        prevent information leakage.
        """

        output = dataframe.copy()
        transformer = make_pipeline(
            DropUseless(0.001), RobustScaler(quantile_range=(5.0, 95.0))
        )
        transformer.fit(output.loc[train_indices][self.features])

        transformed = transformer.transform(output[self.features])
        columns_kept = transformer.steps[0][1].get_support()
        features_selected = output[self.features].columns[columns_kept]
        features_dropped = output[self.features].columns[~columns_kept]
        output.drop(columns=features_dropped)
        output[features_selected] = transformed
        return output, list(features_selected)
