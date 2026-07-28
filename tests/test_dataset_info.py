import json
from pathlib import Path

import pandas as pd
import pytest

from data_sources.dataset_info import (
    build_dataset_health_report,
    calculate_column_health,
    format_generated_at,
    generate_dataset_health_report,
    load_master_dataset,
    load_master_metadata,
)


def write_master_csv(
    folder: Path,
    data: pd.DataFrame,
) -> Path:
    """
    Save a temporary master dataset.
    """

    output_path = (
        folder / "master_dataset.csv"
    )

    data.to_csv(
        output_path,
        index=False,
    )

    return output_path


def write_metadata_json(
    folder: Path,
    metadata: object,
) -> Path:
    """
    Save temporary master metadata.
    """

    output_path = (
        folder
        / "master_dataset_metadata.json"
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
        )

    return output_path


def sample_master_data() -> pd.DataFrame:
    """
    Return a small master dataset used by several tests.
    """

    return pd.DataFrame(
        {
            "Date": [
                "2024-01",
                "2024-02",
                "2024-03",
            ],
            "Gold Price": [
                2000.0,
                2050.0,
                2100.0,
            ],
            "CPI": [
                None,
                310.0,
                311.0,
            ],
        }
    )


def test_load_master_dataset(
    tmp_path: Path,
) -> None:
    """
    The master dataset should load with sorted datetime
    values.
    """

    data = sample_master_data().iloc[
        [2, 0, 1]
    ]

    write_master_csv(
        folder=tmp_path,
        data=data,
    )

    result = load_master_dataset(
        processed_folder=tmp_path,
    )

    assert pd.api.types.is_datetime64_any_dtype(
        result["Date"]
    )

    assert result["Date"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-03-01"),
    ]


def test_load_master_dataset_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """
    A missing master dataset should raise an error.
    """

    with pytest.raises(
        FileNotFoundError,
        match="Master dataset was not found",
    ):
        load_master_dataset(
            processed_folder=tmp_path,
        )


def test_load_master_dataset_rejects_empty_file(
    tmp_path: Path,
) -> None:
    """
    A completely empty file should raise an error.
    """

    file_path = (
        tmp_path / "master_dataset.csv"
    )

    file_path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="contains no readable data",
    ):
        load_master_dataset(
            processed_folder=tmp_path,
        )


def test_load_master_dataset_rejects_missing_date_column(
    tmp_path: Path,
) -> None:
    """
    The master dataset must contain a Date column.
    """

    write_master_csv(
        folder=tmp_path,
        data=pd.DataFrame(
            {
                "Month": [
                    "2024-01",
                ],
                "Gold Price": [
                    2000.0,
                ],
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="missing the Date column",
    ):
        load_master_dataset(
            processed_folder=tmp_path,
        )


def test_load_master_dataset_rejects_invalid_dates(
    tmp_path: Path,
) -> None:
    """
    Invalid date values should raise an error.
    """

    write_master_csv(
        folder=tmp_path,
        data=pd.DataFrame(
            {
                "Date": [
                    "not-a-date",
                ],
                "Gold Price": [
                    2000.0,
                ],
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="invalid date",
    ):
        load_master_dataset(
            processed_folder=tmp_path,
        )


def test_load_master_metadata(
    tmp_path: Path,
) -> None:
    """
    Valid metadata JSON should load as a dictionary.
    """

    metadata = {
        "generated_at": (
            "2026-07-27T18:42:11+00:00"
        ),
        "rows": 3,
    }

    write_metadata_json(
        folder=tmp_path,
        metadata=metadata,
    )

    result = load_master_metadata(
        processed_folder=tmp_path,
    )

    assert result == metadata


def test_load_master_metadata_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """
    Missing metadata should raise an error.
    """

    with pytest.raises(
        FileNotFoundError,
        match="Master metadata was not found",
    ):
        load_master_metadata(
            processed_folder=tmp_path,
        )


def test_load_master_metadata_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    """
    Invalid JSON should raise an error.
    """

    file_path = (
        tmp_path
        / "master_dataset_metadata.json"
    )

    file_path.write_text(
        "{invalid-json",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):
        load_master_metadata(
            processed_folder=tmp_path,
        )


def test_load_master_metadata_rejects_non_object(
    tmp_path: Path,
) -> None:
    """
    Metadata must contain a JSON object.
    """

    write_metadata_json(
        folder=tmp_path,
        metadata=[
            "Gold",
            "CPI",
        ],
    )

    with pytest.raises(
        ValueError,
        match="JSON object",
    ):
        load_master_metadata(
            processed_folder=tmp_path,
        )


def test_calculate_column_health() -> None:
    """
    Column health should report availability, missing
    values, coverage, and completeness.
    """

    data = sample_master_data()

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    result = calculate_column_health(
        data=data,
        column="CPI",
    )

    assert result == {
        "column": "CPI",
        "available_rows": 2,
        "missing_rows": 1,
        "completeness_percent": 66.67,
        "coverage_start": "2024-02",
        "coverage_end": "2024-03",
    }


def test_calculate_column_health_handles_all_missing() -> None:
    """
    A completely empty value column should have no
    coverage range.
    """

    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01",
                    "2024-02",
                ]
            ),
            "CPI": [
                None,
                None,
            ],
        }
    )

    result = calculate_column_health(
        data=data,
        column="CPI",
    )

    assert result[
        "available_rows"
    ] == 0

    assert result[
        "missing_rows"
    ] == 2

    assert result[
        "completeness_percent"
    ] == 0.0

    assert result[
        "coverage_start"
    ] is None

    assert result[
        "coverage_end"
    ] is None


def test_calculate_column_health_rejects_missing_column() -> None:
    """
    Unknown columns should raise an error.
    """

    data = sample_master_data()

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    with pytest.raises(
        ValueError,
        match="Column was not found",
    ):
        calculate_column_health(
            data=data,
            column="Oil",
        )


def test_build_dataset_health_report() -> None:
    """
    The complete health report should summarize the
    master dataset and every value column.
    """

    data = sample_master_data()

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    metadata = {
        "generated_at": (
            "2026-07-27T18:42:11+00:00"
        ),
        "datasets": [
            "Gold",
            "CPI",
        ],
        "source_files": {
            "Gold": "gold_price.csv",
            "CPI": "cpi.csv",
        },
    }

    result = build_dataset_health_report(
        data=data,
        metadata=metadata,
    )

    assert result["rows"] == 3
    assert result["columns"] == 3

    assert result[
        "overall_date_range"
    ] == {
        "start": "2024-01",
        "end": "2024-03",
    }

    assert result["datasets"] == [
        "Gold",
        "CPI",
    ]

    assert "Gold Price" in result[
        "column_health"
    ]

    assert "CPI" in result[
        "column_health"
    ]


def test_format_generated_at() -> None:
    """
    ISO timestamps should be formatted for display.
    """

    result = format_generated_at(
        "2026-07-27T18:42:11+00:00"
    )

    assert result == (
        "2026-07-27 18:42 UTC"
    )


def test_format_generated_at_handles_missing_value() -> None:
    """
    Missing timestamps should display as unknown.
    """

    assert format_generated_at(
        None
    ) == "Unknown"


def test_format_generated_at_preserves_invalid_value() -> None:
    """
    An unexpected timestamp format should be returned
    unchanged instead of crashing the report.
    """

    assert format_generated_at(
        "unexpected-format"
    ) == "unexpected-format"


def test_generate_dataset_health_report(
    tmp_path: Path,
) -> None:
    """
    The complete report pipeline should load both files
    and return the generated report.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_master_data(),
    )

    write_metadata_json(
        folder=tmp_path,
        metadata={
            "generated_at": (
                "2026-07-27T18:42:11+00:00"
            ),
            "datasets": [
                "Gold",
                "CPI",
            ],
            "source_files": {
                "Gold": "gold_price.csv",
                "CPI": "cpi.csv",
            },
        },
    )

    result = generate_dataset_health_report(
        processed_folder=tmp_path,
    )

    assert result["rows"] == 3
    assert result["columns"] == 3

    assert result[
        "column_health"
    ]["CPI"]["missing_rows"] == 1