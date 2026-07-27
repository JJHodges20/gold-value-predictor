from data_sources.config import (
    ENV_FILE,
    PROJECT_ROOT,
    RAW_FRED_FOLDER,
    RAW_GOLD_FOLDER,
    PROCESSED_DATA_FOLDER,
    create_project_folders,
)


def test_project_root_exists() -> None:
    """The detected project root should exist."""

    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()


def test_environment_file_exists() -> None:
    """The root-level .env file should exist."""

    assert ENV_FILE.exists()
    assert ENV_FILE.is_file()


def test_project_folders_can_be_created() -> None:
    """The configured data folders should be created successfully."""

    create_project_folders()

    assert RAW_FRED_FOLDER.exists()
    assert RAW_FRED_FOLDER.is_dir()

    assert RAW_GOLD_FOLDER.exists()
    assert RAW_GOLD_FOLDER.is_dir()

    assert PROCESSED_DATA_FOLDER.exists()
    assert PROCESSED_DATA_FOLDER.is_dir()