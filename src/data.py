from pathlib import Path


def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent


PROJECT_ROOT = get_project_root()
PATH_DATA = PROJECT_ROOT / "data"
PATH_DATA_RAW = PROJECT_ROOT / "data" / "raw"
PATH_DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
PATH_DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
