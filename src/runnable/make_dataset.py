import logging

from definitions import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data.preprocess_data import (
    create_binary_diagnoses_df,
    load_sri24_df,
    load_freesurfer_df,
    load_sociodem_df
)

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Creating final data set from raw data.')

    logger.info(f'Load individual dataframes from {input_filepath}')
    binary_diagnoses_df = create_binary_diagnoses_df(input_filepath)
    sri24_df = load_sri24_df(input_filepath)
    freesurfer_df = load_freesurfer_df(input_filepath)
    sociodem_df = load_sociodem_df(input_filepath)

    logger.info('Merge to final dataframe...')
    abcd_data_df = sociodem_df.merge(
        right=sri24_df, how='inner', left_index=True, right_index=True
    ).merge(
        right=freesurfer_df, how='inner', left_index=True, right_index=True
    ).merge(
        right=binary_diagnoses_df, how='inner', left_index=True, right_index=True
    )

    # Save
    logger.info(f'Save to {output_filepath}')
    abcd_data_df.to_csv(
        path_or_buf=output_filepath / 'abcd_data.csv', index=True
    )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main(RAW_DATA_DIR, PROCESSED_DATA_DIR)
