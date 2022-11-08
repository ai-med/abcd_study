import logging
import click

from definitions import RAW_DATA_DIR, PROCESSED_DATA_DIR
import src.data.preprocess_data as prep

@click.command()
@click.option('--select-one-child-per-family',
              help='Whether to randomly select only one child per family.',
              is_flag=True)
@click.option('--seed',
              default=0,
              help='Random number seed for selecting one child per family.',
              type=int)
def main(select_one_child_per_family,
         seed):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    logger.info('Creating final data set from raw data.')

    logger.info('Load individual dataframes from %s', RAW_DATA_DIR)
    binary_diagnoses_df = prep.create_binary_diagnoses_df(RAW_DATA_DIR)
    sri24_df = prep.load_sri24_df(RAW_DATA_DIR)
    freesurfer_df = prep.load_freesurfer_df(RAW_DATA_DIR)
    sociodem_df = prep.load_sociodem_df(RAW_DATA_DIR)

    logger.info('Merge to final dataframe...')
    abcd_data_df = sociodem_df.merge(
        right=sri24_df, how='inner', left_index=True, right_index=True
    ).merge(
        right=freesurfer_df, how='inner', left_index=True, right_index=True
    ).merge(
        right=binary_diagnoses_df, how='inner', left_index=True, right_index=True
    )

    if select_one_child_per_family:
        logger.info('Randomly select only one child per family (seed=%d)', seed)
        abcd_data_df = prep.select_one_child_per_family(
            abcd_data_path=RAW_DATA_DIR,
            abcd_df=abcd_data_df,
            random_state=seed
        )

    logger.info('Save to %s', PROCESSED_DATA_DIR)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    abcd_data_df.to_csv(
        path_or_buf=PROCESSED_DATA_DIR / 'abcd_data.csv', index=True
    )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
