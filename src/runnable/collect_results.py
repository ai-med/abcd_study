import click
from pathlib import Path
from typing import Iterable
import pandas as pd

classifiers = (
    'xgboost_cce',
    'logistic_regression_cce',
    'logistic_regression_ovr',
)

segmentations = (
    'freesurfer',
    'sri',
)


def load_results(filename: Path) -> pd.DataFrame:
    df = pd.read_csv(filename, index_col=0)
    df.index.name = 'fold'
    return df


def find_roc_auc(
    base_dir: Path,
    segmentation: str,
    kind: str,
    classifier: str,
    split: str,
) -> Iterable[Path]:
    for p0 in base_dir.glob(f"{segmentation}_*"):
        if not p0.is_dir():
            continue
        for p1 in (p0 / 'results').iterdir():
            if not p1.is_dir():
                continue

            f = p1 / kind / classifier / split / 'roc_auc.csv'
            if f.exists():
                yield f


def collect_results(
    base_dir: Path,
    segmentation: str,
    kind: str = 'unpermuted',
    split: str = 'test',
) -> pd.DataFrame:
    df = pd.concat([
        load_results(f).assign(Method=clsf, filename=str(f)).set_index('Method', append=True)
        for clsf in classifiers
        for f in find_roc_auc(base_dir, segmentation, kind, clsf, split)
    ], axis=0)
    df.sort_index(inplace=True)
    return df


@click.command()
@click.option('-d', '--directory',
              required=True,
              help='Path to directory containing roc_auc.csv files.')
@click.option('-k', '--kind',
              required=True,
              type=click.Choice(['permuted', 'unpermuted']),
              help='Which type of performance to aggregate.')
def main(directory, kind):
    for seg in segmentations:
        d = collect_results(Path(directory), seg, kind=kind)
        print(f"Found {d.shape[0]} results for {seg}:")
        print(d.groupby(level='Method').size())
        out_file = f'roc_auc_{seg}_{kind}.csv'
        print(f"Writing {out_file}")
        d.to_csv(out_file)
        print()


if __name__ == '__main__':
    main()
