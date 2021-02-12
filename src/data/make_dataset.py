# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.data.preprocess_data import (
    create_binary_diagnoses_df,
    load_sri24_df,
    load_freesurfer_df,
    load_sociodem_df
)


@click.command()
@click.argument('input_filepath', type=Path)
@click.argument('output_filepath', type=Path)
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Load individual dataframes
    binary_diagnoses_df = create_binary_diagnoses_df(input_filepath)
    sri24_df = load_sri24_df(input_filepath)
    freesurfer_df = load_freesurfer_df(input_filepath)
    sociodem_df = load_sociodem_df(input_filepath)

    # Merge to full dataframe
    abcd_data_df = sociodem_df.merge(
        right=sri24_df, how='inner', left_index=True, right_index=True
    ).merge(
        right=freesurfer_df, how='inner', left_index=True, right_index=True
    ).merge(
        right=binary_diagnoses_df, how='inner', left_index=True, right_index=True
    )

    # Save
    abcd_data_df.to_csv(
        path_or_buf=output_filepath / 'abcd_data.csv', index=True
    )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
