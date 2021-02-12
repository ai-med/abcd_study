import os
from pathlib import Path

import pandas as pd
import numpy as np

def create_binary_diagnoses_df(abcd_data_path: Path):

    subindicators_table = pd.read_csv(abcd_data_path / 'subindicators_table.csv')

    # merge data tables with diagnosis variables into one
    labels_df = pd.DataFrame(columns=['src_subject_id'])  # initialize empty dataframe
    opened_files = {}  # dataframes from files already opened
    for row in subindicators_table.iterrows():
        filename = row[1]['File']
        if filename not in opened_files.keys():
            # file was not opened yet: open and save dataframe in opened_files
            newfile = pd.read_csv(abcd_data_path / f"{filename}.txt", sep='\t', skiprows=(1, 1))
            newfile = newfile.loc[newfile['eventname'] == "baseline_year_1_arm_1"]
            newfile['src_subject_id'] = newfile['src_subject_id'].str.upper()
            opened_files[filename] = newfile
        # extract column with score corresponding to 'row' and save in scores_data
        temp = opened_files[filename][['src_subject_id', row[1]['Feature']]]
        labels_df = labels_df.merge(right=temp, how='outer', on='src_subject_id')
    labels_df.index = labels_df['src_subject_id']
    labels_df = labels_df.drop(columns=['src_subject_id'])

    # Create binary_diagnosis_data using the and/or-rule:
    # Label a subject diagnosis-positive if at least one of the sub-indicators is positive
    diagnoses_list = subindicators_table['Diagnosis'].unique()
    diagnoses_list = [
        'Bipolar Disorder' if diagnosis == 'Bipolar I Disorder' else diagnosis for diagnosis in diagnoses_list
    ]
    diagnoses_list.remove('Bipolar II Disorder') # combine BD I and II to summary BD diagnosis
    cols = {}
    for diagnosis in diagnoses_list:
        # diag_vars: list of sub-indicators related to diagnosis (such as 'current', 'past', 'parent', 'youth')
        diag_vars = None
        if diagnosis == 'Bipolar Disorder':
            diag_vars = subindicators_table.loc[
                subindicators_table['Diagnosis'].isin(['Bipolar I Disorder', 'Bipolar II Disorder']), 'Feature'
            ]
        else:
            diag_vars = subindicators_table.loc[subindicators_table['Diagnosis'] == diagnosis, 'Feature']
        # Create binary diagnosis label from sub-indicators
        # Rules: if at least one of the sub-indicators is True, the binary diagnosis is True
        collapsed_diagnosis = labels_df[diag_vars].sum(axis=1) > 0
        # If no sub-indicator is True: If at least one variable is unknown (NaN), assume diagnosis also to be unknown
        # (NaN). Else, if all variables are known, the diagnosis is False.
        rowsNaN = (
                (labels_df[diag_vars].sum(axis=1) == 0) &
                (labels_df[diag_vars].isna().sum(axis=1) > 0)
        )
        collapsed_diagnosis.loc[rowsNaN] = np.nan
        cols[diagnosis] = collapsed_diagnosis
    binary_diagnoses_df = pd.DataFrame(data=cols)

    return binary_diagnoses_df


def load_sri24_df(abcd_data_path: Path) -> pd.DataFrame:
    path = abcd_data_path / 'btsv01.txt'
    sri24_df = pd.read_csv(path, sep='\t', skiprows=(1, 1))
    sri24_df['src_subject_id'] = sri24_df['src_subject_id'].str.upper()
    sri24_df.index = sri24_df['src_subject_id']
    sri24_df = sri24_df[sri24_df.columns[sri24_df.columns.str.startswith("sri24")].tolist()]
    return sri24_df


def load_freesurfer_df(abcd_data_path: Path) -> pd.DataFrame:
    path = abcd_data_path / 'abcd_freesurfer.csv'
    freesurfer_df = pd.read_csv(path)
    freesurfer_df['SRC_SUBJECT_ID'] = freesurfer_df['SRC_SUBJECT_ID'].str.upper()
    freesurfer_df = freesurfer_df.rename(columns={'SRC_SUBJECT_ID': 'src_subject_id'})
    freesurfer_df.index = freesurfer_df['src_subject_id']

    # FreeSurfer columns start with "FS_"
    relevant_cols = freesurfer_df.columns[freesurfer_df.columns.str.startswith("FS_")].tolist()
    freesurfer_df = freesurfer_df[relevant_cols]
    return freesurfer_df


def load_sociodem_df(abcd_data_path: Path) -> pd.DataFrame:
    path = abcd_data_path / 'sociodem_bl.csv'
    sociodem_df = pd.read_csv(path)
    sociodem_df = sociodem_df.drop(columns=['eventname', 'sex', 'anthro_bmi_calc'])
    sociodem_df['src_subject_id'] = sociodem_df['src_subject_id'].str.upper()
    sociodem_df.index = sociodem_df['src_subject_id']
    sociodem_df = sociodem_df.drop(columns=['src_subject_id'])

    # Drop household.income column since it contains too many NaNs. Then, drop all rows with NaNs.
    sociodem_df = sociodem_df.drop(columns=['household.income'])
    sociodem_df = sociodem_df.dropna()

    # Encode values numerically:
    # female to {0, 1}
    # married to {0, 1}
    # high.educ to one-hot-encoded dummy variables
    # race_ethnicity to one-hot-encoded dummy variables
    # abcd_site to one-hot-encoded dummy variables
    sociodem_df['female'] = (sociodem_df['female'] == "yes").astype(int)
    sociodem_df['married'] = (sociodem_df['married'] == "yes").astype(int)
    sociodem_df = pd.get_dummies(data=sociodem_df, columns=['race_ethnicity', 'high.educ', 'abcd_site'])

    return sociodem_df


def load_complete_df(abcd_data_path: Path) -> pd.DataFrame:
    binary_diagnoses_df = create_binary_diagnoses_df(abcd_data_path)
    sri24_df = load_sri24_df(abcd_data_path)
    freesurfer_df = load_freesurfer_df(abcd_data_path)
    sociodem_df = load_sociodem_df(abcd_data_path)

    complete_df = sociodem_df.merge(
        right=sri24_df, how='inner', left_index=True, right_index=True
    ).merge(
        right=freesurfer_df, how='inner', left_index=True, right_index=True
    ).merge(
        right=binary_diagnoses_df, how='inner', left_index=True, right_index=True
    )
    return complete_df


def select_one_child_per_family(abcd_data_path: Path,
                                abcd_df: pd.DataFrame,
                                random_state: int = None) -> pd.DataFrame:
    """Randomly select one subject per family"""

    path = abcd_data_path / 'acspsw03.txt'
    family_df = pd.read_csv(path, sep='\t', skiprows=(1, 1))
    family_df['src_subject_id'] = family_df['src_subject_id'].str.upper()
    family_df = family_df[['src_subject_id', 'rel_family_id']]

    # Group family_df by family ID and randomly select one subject per family ID
    subjects = family_df.groupby(by='rel_family_id').apply(
        lambda x: x.sample(1, random_state=random_state)
    ).reset_index(drop=True)['src_subject_id']

    new_df = abcd_df.merge(
        right=subjects, how='inner', left_index=True, right_on='src_subject_id'
    )
    new_df.index = new_df['src_subject_id']
    new_df = new_df.drop(columns=['src_subject_id'])

    return new_df
