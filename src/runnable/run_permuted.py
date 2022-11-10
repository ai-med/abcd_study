import logging
import random
from typing import List

import click
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger

from definitions import REPO_ROOT, PROCESSED_DATA_DIR
from src.data.data_loader import RepeatedStratifiedKFoldDataloader
from src.models.base import ModelIterator
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
@click.option('--select-fold', multiple=True, type=int)
@click.option('--select-model', multiple=True)
@click.option('--select-permutation', multiple=True, type=int)
def main(
    seed: int,
    k: int,
    n: int,
    num_permutations: int,
    features: str,
    select_fold: List[int],
    select_model: List[str],
    select_permutation: List[int],
) -> None:
    def should_run(permutation: int, fold: int, model_name: str) -> bool:
        if len(select_permutation) > 0 and permutation not in select_permutation:
            return False
        if len(select_fold) > 0 and fold not in select_fold:
            return False
        if len(select_model) > 0 and model_name not in select_model:
            return False
        return True


    logger = logging.getLogger(__name__)
    logger.info(
        'Running training and prediction on %(num_permutations)s '
        'permuted datasets with seed=%(seed)s, k=%(k)s, n=%(n)s.',
        {'num_permutations': num_permutations, 'seed': seed, 'k': k, 'n': n},
    )
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

        logger.info('Create permutation no. %d', perm)
        abcd_data_permuted = abcd_data.copy()
        abcd_data_permuted.loc[:, abcd_vars.diagnoses.features] = \
            abcd_data_permuted.loc[:, abcd_vars.diagnoses.features].sample(
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
            model_iter = ModelIterator(train, valid, features_selected, rnd)
            for model_name, predictor, fit_data in model_iter:
                msg_args = {'perm': perm, 'i': i, 'model_name': model_name}
                if not should_run(perm, i, model_name):
                    logger.info(
                        'Skipping permutation %(perm)d, fold %(i)d, model %(model_name)s',
                        msg_args,
                    )
                    continue

                logger.info(
                    'Permutation %(perm)d, total fold %(i)d: Fit %(model_name)s classifier',
                    msg_args,
                )
                predictor.fit(*fit_data)

                logger.info(
                    'Permutation %(perm)d, total fold %(i)d: Save %(model_name)s predictions',
                    msg_args,
                )
                for ds_str, ds in zip(['train', 'valid', 'test'], [train, valid, test]):
                    manager.save_predictions(
                        dataset_name=f'permuted_{perm}',
                        model_name='logistic_regression_ovr',
                        fold=i,
                        split_set=ds_str,
                        y_true=ds.loc[:, abcd_vars.diagnoses.features],
                        y_pred=predictor.predict(ds.loc[:, features_selected])
                    )

        logger.info('Save final ROC AUC values')
        manager.finish()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
