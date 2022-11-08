import logging
import click
import random

import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger

from definitions import REPO_ROOT, PROCESSED_DATA_DIR
from src.data.data_loader import RepeatedStratifiedKFoldDataloader
from src.models.classifier_chain import ClassifierChainEnsemble
from src.models.logistic_regression import (
    LogisticRegressionOVRPredictor, LogisticRegressionModel
)
from src.models.xgboost_pipeline import DepthwiseXGBPipeline
import src.data.var_names as abcd_vars
from src.models.evaluation import ResultManager

DATA_DIR = PROCESSED_DATA_DIR / 'abcd_data.csv'

@click.command()
@click.option('--seed', default=0, help='Random number seed.', type=int)
@click.option('--k', default=5, help='Number of CV folds.', type=int)
@click.option('--n', help='Number of successive k-fold CV runs.', type=int)
@click.option('--unadjusted', help='Do not adjust features for confounders.',
              is_flag=True)
@click.option('--features',
              type=click.Choice(['all', 'freesurfer', 'sri']),
              default='freesurfer',
              help='Which subset of cortical and subcortical features to use.')
def main(seed: int, k: int, n: int, unadjusted: bool, features: str) -> None:

    logger = logging.getLogger(__name__)
    logger.info(
        'Running training and prediction on unpermuted dataset with '
        'seed=%(seed)s, k=%(k)s, n=%(n)s.',
        {'seed': seed, 'k': k, 'n': n},
    )

    logger.info('Load data')
    abcd_data = pd.read_csv(DATA_DIR, index_col='src_subject_id')

    if features == 'all':
        features_list = abcd_vars.all_brain_features.features
    elif features == 'freesurfer':
        features_list = abcd_vars.freesurfer.features
    elif features == 'sri':
        features_list = abcd_vars.sri24.features
    else:
        raise AssertionError()

    logger.info('Set up data structures')
    rnd = random.Random(x=seed)
    data_loader = RepeatedStratifiedKFoldDataloader(
        dataframe=abcd_data,
        features=features_list,
        responses=abcd_vars.diagnoses.features,
        confounders=abcd_vars.sociodem.features,
        n=n,
        k=k,
        val_ratio=0.2,
        random_state=rnd.randint(0, 999999999),
        ignore_adjustment=unadjusted
    )
    tensorboard_logger = TensorBoardLogger(REPO_ROOT / 'tensorboard')
    manager = ResultManager(
        tensorboard_logger=tensorboard_logger,
        save_root=REPO_ROOT / 'results',
        run_name=f"run_unpermuted_seed{seed}n{n}k{k}_{features}_"
                 f"{'unadjusted' if unadjusted else 'adjusted'}",
        save_params={
            'random_state': seed, 'n': n, 'k': k, 'unadjusted': unadjusted
        }
    )
    logistic_regression_args = {
        'solver': 'lbfgs',
        'max_iter': 500,
        'class_weight': 'balanced'
    }

    logger.info('Start training and prediction')
    for i, (train, valid, test, features_selected) in enumerate(data_loader):

        logger.info('Total fold %d: Fit OVR logistic regression classifier', i)
        ovr_predictor = LogisticRegressionOVRPredictor(
            features=features_selected,
            responses=abcd_vars.diagnoses.features,
            model_args=logistic_regression_args,
            random_state=rnd.randint(0, 999999999)
        )
        ovr_predictor.fit(pd.concat((train, valid)))
        logger.info('Total fold %d: Save OVR logistic regression predictions', i)
        for ds_str, ds in zip(['train', 'valid', 'test'], [train, valid, test]):
            manager.save_predictions(
                dataset_name='unpermuted',
                model_name='logistic_regression_ovr',
                fold=i,
                split_set=ds_str,
                y_true=ds[abcd_vars.diagnoses.features],
                y_pred=ovr_predictor.predict(ds[features_selected])
            )

        logger.info('Total fold %d: Fit CCE logistic regression classifier', i)
        lr_cce_predictor = ClassifierChainEnsemble(
            model=LogisticRegressionModel,
            features=features_selected,
            responses=abcd_vars.diagnoses.features,
            num_chains=10,
            model_args=logistic_regression_args,
            random_state=rnd.randint(0, 999999999)
        )
        lr_cce_predictor.fit(train, valid)
        logger.info('Total fold %d: Save CCE logistic regression predictions', i)
        for ds_str, ds in zip(['train', 'valid', 'test'], [train, valid, test]):
            manager.save_predictions(
                dataset_name='unpermuted',
                model_name='logistic_regression_cce',
                fold=i,
                split_set=ds_str,
                y_true=ds[abcd_vars.diagnoses.features],
                y_pred=lr_cce_predictor.predict(ds[features_selected])
            )

        logger.info('Total fold %d: Fit CCE XGBoost classifier', i)
        xgboost_cce_predictor = ClassifierChainEnsemble(
            model=DepthwiseXGBPipeline,
            features=features_selected,
            responses=abcd_vars.diagnoses.features,
            num_chains=10,
            model_args={
                'n_calls': 30
            },
            random_state=rnd.randint(0, 999999999)
        )
        xgboost_cce_predictor.fit(train, valid)
        logger.info('Total fold %d: Save CCE XGBoost predictions', i)
        for ds_str, ds in zip(['train', 'valid', 'test'], [train, valid, test]):
            manager.save_predictions(
                dataset_name='unpermuted',
                model_name='xgboost_cce',
                fold=i,
                split_set=ds_str,
                y_true=ds[abcd_vars.diagnoses.features],
                y_pred=xgboost_cce_predictor.predict(ds[features_selected])
            )

    logger.info('Save final ROC AUC values')
    manager.finish()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
