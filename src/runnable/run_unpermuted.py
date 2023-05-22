import logging
import click
import random
from typing import List

import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.definitions import REPO_ROOT, PROCESSED_DATA_DIR
from src.data.data_loader import RepeatedStratifiedKFoldDataloader
from src.models.base import ModelIterator
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
@click.option('--select-fold', multiple=True, type=int)
@click.option('--select-model', multiple=True)
def main(
    seed: int,
    k: int,
    n: int,
    unadjusted: bool,
    features: str,
    select_fold: List[int],
    select_model: List[str],
) -> None:
    def should_run(fold: int, model_name: str) -> bool:
        if len(select_fold) > 0 and fold not in select_fold:
            return False
        if len(select_model) > 0 and model_name not in select_model:
            return False
        return True


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
    tensorboard_logger = SummaryWriter(
        log_dir=str(REPO_ROOT / 'tensorboard' /  f'seed{seed}_k{k}_n{n}' / 'unpermuted')
    )
    manager = ResultManager(
        tensorboard_logger=tensorboard_logger,
        save_root=REPO_ROOT / 'results',
        run_name=f"run_unpermuted_seed{seed}n{n}k{k}_{features}_"
                 f"{'unadjusted' if unadjusted else 'adjusted'}",
        save_params={
            'random_state': seed, 'n': n, 'k': k, 'unadjusted': unadjusted
        }
    )

    pbar = tqdm(total=n * k * 3)

    logger.info('Start training and prediction')
    for i, (train, valid, test, features_selected) in enumerate(data_loader):
        model_iter = ModelIterator(train, valid, features_selected, rnd)
        for model_name, predictor, fit_data in model_iter:
            if not should_run(i, model_name):
                logger.info('Skipping fold %d, model %s', i, model_name)
                continue

            logger.info('Total fold %d: Fit %s', i, model_name)
            predictor.fit(*fit_data)

            logger.info('Total fold %d: Save %s predictions', i, model_name)
            for ds_str, ds in zip(['train', 'valid', 'test'], [train, valid, test]):
                manager.save_predictions(
                    dataset_name='unpermuted',
                    model_name=model_name,
                    fold=i,
                    split_set=ds_str,
                    y_true=ds.loc[:, abcd_vars.diagnoses.features],
                    y_pred=predictor.predict(ds.loc[:, features_selected])
                )
            tensorboard_logger.flush()
            pbar.update()

    logger.info('Save final ROC AUC values')
    manager.finish()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
