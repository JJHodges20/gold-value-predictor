from pathlib import Path
import os

from dotenv import load_dotenv


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = PROJECT_ROOT / ".env"

DATA_FOLDER = PROJECT_ROOT / "data"
RAW_DATA_FOLDER = DATA_FOLDER / "raw"
RAW_FRED_FOLDER = RAW_DATA_FOLDER / "fred"
RAW_GOLD_FOLDER = RAW_DATA_FOLDER / "gold"

PROCESSED_DATA_FOLDER = DATA_FOLDER / "processed"
CHARTS_FOLDER = PROJECT_ROOT / "charts"


# --------------------------------------------------
# Environment variables
# --------------------------------------------------

load_dotenv(ENV_FILE)

FRED_API_KEY = os.getenv("FRED_API_KEY")


def create_project_folders() -> None:
    """Create all folders used by the application."""

    folders = [
        RAW_FRED_FOLDER,
        RAW_GOLD_FOLDER,
        PROCESSED_DATA_FOLDER,
        CHARTS_FOLDER,
    ]

    for folder in folders:
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )


def validate_configuration() -> None:
    """Confirm that required project settings exist."""

    if not ENV_FILE.exists():
        raise RuntimeError(
            "The root-level .env file was not found."
        )

    if not FRED_API_KEY:
        raise RuntimeError(
            "FRED_API_KEY was not found in the .env file."
        )


def initialize_project() -> None:
    """Validate configuration and create project folders."""

    validate_configuration()
    create_project_folders()