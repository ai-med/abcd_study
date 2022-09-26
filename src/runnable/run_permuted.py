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
@click.option('--num-permutations', help='Number of random permutations.', type=int)
@click.option('--features',
              type=click.Choice(['all', 'freesurfer', 'sri']),
              default='freesurfer',
              help='Which subset of cortical and subcortical features to use.')
def main(seed: int, k: int, n: int, num_permutations: int, features: str) -> None:

    logger = logging.getLogger(__name__)
    logger.info(f'Running training and prediction on {num_permutations} '
                f'permuted datasets with seed={seed}, k={k}, n={n}.')
    logger.info('Load data')
    abcd_data = pd.read_csv(DATA_DIR, index_col='src_subject_id')

    tensorboard_logger = TensorBoardLogger(
        REPO_ROOT / 'tensorboard' / f'seed={seed}k={k}'
    )
    manager = ResultManager(
        tensorboard_logger=tensorboard_logger,
        save_root=REPO_ROOT / 'results',
        run_name=f"run_permuted_seed{seed}n{n}k{k}num_permutations{num_permutations}_{features}",
        save_params={
            'random_state': seed, 'n': n, 'k': k,
            'num_permutations': num_permutations
        }
    )
    logistic_regression_args = {
        'solver': 'lbfgs',
        'max_iter': 500,
        'class_weight': 'balanced'
    }
    rnd = random.Random(x=seed)

    if features == 'all':
        features_list = abcd_vars.all_brain_features.features
    elif features == 'freesurfer':
        features_list = abcd_vars.freesurfer.features
    elif features == 'sri':
        features_list = abcd_vars.sri24.features
    else:
        raise AssertionError()

    for perm in range(num_permutations):

        logger.info(f'Create permutation no. {perm}')
        abcd_data_permuted = abcd_data.copy()
        abcd_data_permuted[abcd_vars.diagnoses.features] = \
            abcd_data_permuted[abcd_vars.diagnoses.features].sample(
                frac=1, random_state=rnd.randint(0, 999999999)
            ).to_numpy()

        logger.info('Set up data structures')
        data_loader = RepeatedStratifiedKFoldDataloader(
            dataframe=abcd_data_permuted,
            features=features_list,
            responses=abcd_vars.diagnoses.features,
            confounders=abcd_vars.sociodem.features,
            n=n,
            k=k,
            val_ratio=0.2,
            random_state=rnd.randint(0, 999999999)
        )

        logger.info('Start training and prediction')
        for i, (train, valid, test, features_selected) in enumerate(data_loader):

            logger.info(f'Permutation {perm}, total fold {i}: Fit OVR logistic regression '
                        f'classifier')
            ovr_predictor = LogisticRegressionOVRPredictor(
                features=features_selected,
                responses=abcd_vars.diagnoses.features,
                model_args=logistic_regression_args,
                random_state=rnd.randint(0, 999999999)
            )
            ovr_predictor.fit(pd.concat((train, valid)))
            logger.info(f'Permutation {perm}, total fold {i}: Save OVR logistic regression '
                        f'predictions')
            for ds_str, ds in zip(['train', 'valid', 'test'], [train, valid, test]):
                manager.save_predictions(
                    dataset_name=f'permuted_{perm}',
                    model_name='logistic_regression_ovr',
                    fold=i,
                    split_set=ds_str,
                    y_true=ds[abcd_vars.diagnoses.features],
                    y_pred=ovr_predictor.predict(ds[features_selected])
                )

            logger.info(f'Permutation {perm}, total fold {i}: Fit CCE logistic regression '
                        f'classifier')
            lr_cce_predictor = ClassifierChainEnsemble(
                model=LogisticRegressionModel,
                features=features_selected,
                responses=abcd_vars.diagnoses.features,
                num_chains=10,
                model_args=logistic_regression_args,
                random_state=rnd.randint(0, 999999999)
            )
            lr_cce_predictor.fit(train, valid)
            logger.info(f'Permutation {perm}, total fold {i}: Save CCE logistic regression '
                        f'predictions')
            for ds_str, ds in zip(['train', 'valid', 'test'], [train, valid, test]):
                manager.save_predictions(
                    dataset_name=f'permuted_{perm}',
                    model_name='logistic_regression_cce',
                    fold=i,
                    split_set=ds_str,
                    y_true=ds[abcd_vars.diagnoses.features],
                    y_pred=lr_cce_predictor.predict(ds[features_selected])
                )

            logger.info(f'Permutation {perm}, total fold {i}: Fit CCE XGBoost classifier')
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
            logger.info(f'Permutation {perm}, total fold {i}: Save CCE XGBoost predictions')
            for ds_str, ds in zip(['train', 'valid', 'test'], [train, valid, test]):
                manager.save_predictions(
                    dataset_name=f'permuted_{perm}',
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
