from pathlib import Path

REPO_ROOT = Path(__file__).resolve(strict=True).parent
RAW_DATA_DIR = REPO_ROOT / 'data' / 'raw'
PROCESSED_DATA_DIR = REPO_ROOT / 'data' / 'processed'
