import copy
from functools import partial
from pathlib import Path

import pandas as pd
import numpy as np

def create_binary_diagnoses_df(
    abcd_data_path: Path
):
    subindicators_table = pd.read_csv(abcd_data_path / 'subindicators_table.csv')

    # aggregate diagnoses via OR rule and remove '_or_rule' suffix
    binary_diagnoses_df = create_binary_diagnoses_df_detailed(
        abcd_data_path=abcd_data_path,
        subindicators_table=subindicators_table,
        or_rule=True,
        and_rule=False
    )
    binary_diagnoses_df = binary_diagnoses_df.rename(columns={
        col: col.replace('_or_rule', '') for col in binary_diagnoses_df.columns
    })

    return binary_diagnoses_df


def create_binary_diagnoses_df_detailed(
        abcd_data_path: Path,
        subindicators_table: pd.DataFrame,
        or_rule: bool,
        and_rule: bool
):
    # we want to summarize Bipolar I and II Disorder
    subindicators_table_ = copy.deepcopy(subindicators_table)
    subindicators_table_ = subindicators_table_.replace(to_replace='Bipolar I Disorder',
                                                        value='Bipolar Disorder')
    subindicators_table_ = subindicators_table_.replace(to_replace='Bipolar II Disorder',
                                                        value='Bipolar Disorder')

    # open files
    opened_dfs = []
    for filename in subindicators_table_['File'].unique():
        columns = list(
            subindicators_table_[subindicators_table_['File'] == filename]['Feature']) + [
                      'src_subject_id']
        new_file = pd.read_csv(abcd_data_path / f'{filename}.txt', sep='\t', skiprows=(1, 1))
        # we are only interested in the baseline assessment of the ABCD study
        new_file = new_file.loc[new_file['eventname'] == "baseline_year_1_arm_1"]
        # capitalize subject id to avoid false mismatches lateron
        new_file['src_subject_id'] = new_file['src_subject_id'].str.upper()
        # select only relevant columns and save to list
        new_file = new_file[columns]
        opened_dfs.append(new_file)

    # merge to one df
    raw_subindicators_df = opened_dfs[0]
    for df in opened_dfs[1:]:
        raw_subindicators_df = raw_subindicators_df.merge(right=df, how='outer',
                                                          on='src_subject_id')
    raw_subindicators_df.index = raw_subindicators_df['src_subject_id']
    raw_subindicators_df = raw_subindicators_df.drop(columns=['src_subject_id'])

    # first, apply or-rule for labels 'within interviewees'
    # e.g. if MDD has been diagnosed either at present or in the past in only the youth interview,
    # the youth label is positive
    dict_series = {}
    for diagnosis in subindicators_table_['Diagnosis'].unique():
        for interviewee in ['parent', 'youth']:
            cols = subindicators_table_[
                (subindicators_table_['Diagnosis'] == diagnosis) & \
                (subindicators_table_['Interview'] == f'{interviewee} interview')
                ]['Feature']
            if len(cols) == 0:
                break
            summarize_or = partial(summarize, rule='or')
            dict_series[f'{diagnosis}_{interviewee}'] = raw_subindicators_df[cols].apply(
                summarize_or, axis=1)
    interviewee_labels_df = pd.DataFrame(dict_series)

    # summarize within-interviewee labels via and/or-rule
    dict_series = {}
    for diagnosis in subindicators_table_['Diagnosis'].unique():
        cols = [f'{diagnosis}_{interviewee}' for interviewee in ['parent', 'youth'] \
                if f'{diagnosis}_{interviewee}' in interviewee_labels_df.columns]
        if or_rule:
            summarize_or = partial(summarize, rule='or')
            dict_series[f'{diagnosis}_or_rule'] = interviewee_labels_df[cols].apply(summarize_or,
                                                                                    axis=1)
        if and_rule:
            summarize_and = partial(summarize, rule='and')
            dict_series[f'{diagnosis}_and_rule'] = interviewee_labels_df[cols].apply(summarize_and,
                                                                                     axis=1)

    return pd.DataFrame(dict_series)


def summarize(x: pd.Series, rule: str):
    assert rule == 'or' or rule == 'and', "rule keyword can only be 'or' or 'and'"
    if rule == 'or':
        # if at leats one positive value -> overall positive
        if 1.0 in x.values:
            return 1.0
        # if no positive value and at least one NaN -> we cannot know -> overall NaN
        elif x.isnull().any():
            return np.nan
        # if only negative values -> safely overall negative
        else:
            return 0.0
    elif rule == 'and':
        # if all values positive -> overall positive
        if x.all() and not x.isnull().any():
            return 1.0
        # if some values negative -> overall negative
        elif 0.0 in x.values:
            return 0.0
        # if only positive and NaN values -> we cannot know -> overall NaN
        else:
            return np.nan


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
