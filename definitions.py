from pathlib import Path

REPO_ROOT = Path(__file__).resolve(strict=True).parent
RAW_DATA_DIR = REPO_ROOT / 'data' / 'raw'
PROCESSED_DATA_DIR = REPO_ROOT / 'data' / 'processed'
RESULTS_DIR = REPO_ROOT / 'results'

# Color blind friendly palette
okabe_ito_palette = {
    'black': '#000000',
    'orange': '#E69F00',
    'sky blue': '#56B4E9',
    'bluish green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'reddish purple': '#CC79A7'
}