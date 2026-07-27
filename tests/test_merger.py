import json
from pathlib import Path

import pandas as pd
import pytest

from data_sources.merger import (
    DATE_COLUMN,
    build_master_dataset,
    create_dataset_metadata,
    load_processed_dataset,
    merge_processed_datasets,
    save_dataset_metadata,
    save_master_dataset,
)


TEST_DATASETS: list[dict[str, str]] = [
    {
        "name": "Gold",
        "file_name": "gold.csv",
        "output_column": "Gold Price",
    },
    {
        "name": "CPI",
        "file_name": "cpi.csv",
        "output_column": "CPI",
    },
]


def write_csv(
    folder: Path,
    file_name: str,
    data: pd.DataFrame,
) -> Path:
    """
    Save test data to a temporary CSV file.

    Returns:
        Path to the saved CSV file.
    """

    output_path = folder / file_name

    data.to_csv(
        output_path,
        index=False,
    )

    return output_path


def test_load_processed_dataset(
    tmp_path: Path,
) -> None:
    """
    A valid processed dataset should load with parsed,
    sorted monthly dates.
    """

    write_csv(
        folder=tmp_path,
        file_name="gold.csv",
        data=pd.DataFrame(
            {
                "Date": [
                    "2024-02",
                    "2024-01",
                ],
                "Gold Price": [
                    2050.0,
                    2000.0,
                ],
            }
        ),
    )

    result = load_processed_dataset(
        dataset_config=TEST_DATASETS[0],
        processed_folder=tmp_path,
    )

    assert list(result.columns) == [
        "Date",
        "Gold Price",
    ]

    assert len(result) == 2

    assert result.iloc[0]["Date"] == pd.Timestamp(
        "2024-01-01"
    )

    assert result.iloc[1]["Date"] == pd.Timestamp(
        "2024-02-01"
    )

    assert result["Gold Price"].tolist() == [
        2000.0,
        2050.0,
    ]


def test_load_processed_dataset_standardizes_value_column(
    tmp_path: Path,
) -> None:
    """
    The source value column should be renamed to the
    configured master-dataset output column.
    """

    write_csv(
        folder=tmp_path,
        file_name="gold.csv",
        data=pd.DataFrame(
            {
                "Date": [
                    "2024-01",
                    "2024-02",
                ],
                "Unexpected Source Header": [
                    2000.0,
                    2050.0,
                ],
            }
        ),
    )

    result = load_processed_dataset(
        dataset_config=TEST_DATASETS[0],
        processed_folder=tmp_path,
    )

    assert list(result.columns) == [
        "Date",
        "Gold Price",
    ]

    assert result["Gold Price"].tolist() == [
        2000.0,
        2050.0,
    ]


def test_load_processed_dataset_converts_values_to_numeric(
    tmp_path: Path,
) -> None:
    """
    Numeric text should be converted into a numeric
    pandas data type.
    """

    write_csv(
        folder=tmp_path,
        file_name="gold.csv",
        data=pd.DataFrame(
            {
                "Date": [
                    "2024-01",
                    "2024-02",
                ],
                "Source Price": [
                    "2000.50",
                    "2050.75",
                ],
            }
        ),
    )

    result = load_processed_dataset(
        dataset_config=TEST_DATASETS[0],
        processed_folder=tmp_path,
    )

    assert pd.api.types.is_numeric_dtype(
        result["Gold Price"]
    )

    assert result["Gold Price"].tolist() == [
        2000.50,
        2050.75,
    ]


def test_load_processed_dataset_converts_invalid_values_to_nan(
    tmp_path: Path,
) -> None:
    """
    Invalid value entries should become missing values
    rather than causing the merge to fail.
    """

    write_csv(
        folder=tmp_path,
        file_name="gold.csv",
        data=pd.DataFrame(
            {
                "Date": [
                    "2024-01",
                    "2024-02",
                ],
                "Source Price": [
                    "2000.00",
                    "not-a-number",
                ],
            }
        ),
    )

    result = load_processed_dataset(
        dataset_config=TEST_DATASETS[0],
        processed_folder=tmp_path,
    )

    assert result.iloc[0]["Gold Price"] == 2000.0
    assert pd.isna(result.iloc[1]["Gold Price"])


def test_load_processed_dataset_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """
    A missing configured CSV should raise an error.
    """

    with pytest.raises(
        FileNotFoundError,
        match="dataset was not found",
    ):
        load_processed_dataset(
            dataset_config=TEST_DATASETS[0],
            processed_folder=tmp_path,
        )


def test_load_processed_dataset_rejects_empty_file(
    tmp_path: Path,
) -> None:
    """
    A completely empty CSV file should raise an error.
    """

    file_path = tmp_path / "gold.csv"

    file_path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="contains no readable data",
    ):
        load_processed_dataset(
            dataset_config=TEST_DATASETS[0],
            processed_folder=tmp_path,
        )


def test_load_processed_dataset_rejects_header_only_file(
    tmp_path: Path,
) -> None:
    """
    A CSV with columns but no observations should raise
    an error.
    """

    write_csv(
        folder=tmp_path,
        file_name="gold.csv",
        data=pd.DataFrame(
            columns=[
                "Date",
                "Gold Price",
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="dataset is empty",
    ):
        load_processed_dataset(
            dataset_config=TEST_DATASETS[0],
            processed_folder=tmp_path,
        )


def test_load_processed_dataset_rejects_missing_date_column(
    tmp_path: Path,
) -> None:
    """
    Every processed dataset must contain a Date column.
    """

    write_csv(
        folder=tmp_path,
        file_name="gold.csv",
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
        load_processed_dataset(
            dataset_config=TEST_DATASETS[0],
            processed_folder=tmp_path,
        )


def test_load_processed_dataset_rejects_no_value_column(
    tmp_path: Path,
) -> None:
    """
    A processed dataset must have one value column in
    addition to Date.
    """

    write_csv(
        folder=tmp_path,
        file_name="gold.csv",
        data=pd.DataFrame(
            {
                "Date": [
                    "2024-01",
                ],
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="exactly one value column",
    ):
        load_processed_dataset(
            dataset_config=TEST_DATASETS[0],
            processed_folder=tmp_path,
        )


def test_load_processed_dataset_rejects_multiple_value_columns(
    tmp_path: Path,
) -> None:
    """
    A processed source file should not contain multiple
    competing value columns.
    """

    write_csv(
        folder=tmp_path,
        file_name="gold.csv",
        data=pd.DataFrame(
            {
                "Date": [
                    "2024-01",
                ],
                "Open": [
                    1980.0,
                ],
                "Close": [
                    2000.0,
                ],
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="exactly one value column",
    ):
        load_processed_dataset(
            dataset_config=TEST_DATASETS[0],
            processed_folder=tmp_path,
        )


def test_load_processed_dataset_rejects_invalid_dates(
    tmp_path: Path,
) -> None:
    """
    Invalid dates should not be silently discarded.
    """

    write_csv(
        folder=tmp_path,
        file_name="gold.csv",
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
        load_processed_dataset(
            dataset_config=TEST_DATASETS[0],
            processed_folder=tmp_path,
        )


def test_load_processed_dataset_rejects_duplicate_months(
    tmp_path: Path,
) -> None:
    """
    Two observations that resolve to the same month
    should raise an error.
    """

    write_csv(
        folder=tmp_path,
        file_name="gold.csv",
        data=pd.DataFrame(
            {
                "Date": [
                    "2024-01-01",
                    "2024-01-20",
                ],
                "Gold Price": [
                    2000.0,
                    2020.0,
                ],
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="duplicate monthly date",
    ):
        load_processed_dataset(
            dataset_config=TEST_DATASETS[0],
            processed_folder=tmp_path,
        )


def test_merge_processed_datasets_uses_outer_join(
    tmp_path: Path,
) -> None:
    """
    The merge should preserve dates that appear in only
    one source dataset.
    """

    write_csv(
        folder=tmp_path,
        file_name="gold.csv",
        data=pd.DataFrame(
            {
                "Date": [
                    "2024-01",
                    "2024-02",
                    "2024-03",
                ],
                "Historical Gold Value": [
                    2000.0,
                    2050.0,
                    2100.0,
                ],
            }
        ),
    )

    write_csv(
        folder=tmp_path,
        file_name="cpi.csv",
        data=pd.DataFrame(
            {
                "Date": [
                    "2024-02",
                    "2024-03",
                    "2024-04",
                ],
                "CPI Observation": [
                    310.0,
                    311.0,
                    312.0,
                ],
            }
        ),
    )

    result = merge_processed_datasets(
        datasets=TEST_DATASETS,
        processed_folder=tmp_path,
    )

    assert len(result) == 4

    assert list(result.columns) == [
        DATE_COLUMN,
        "Gold Price",
        "CPI",
    ]

    assert result[DATE_COLUMN].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-03-01"),
        pd.Timestamp("2024-04-01"),
    ]

    january = result[
        result[DATE_COLUMN]
        == pd.Timestamp("2024-01-01")
    ].iloc[0]

    april = result[
        result[DATE_COLUMN]
        == pd.Timestamp("2024-04-01")
    ].iloc[0]

    assert pd.isna(january["CPI"])
    assert pd.isna(april["Gold Price"])


def test_merge_processed_datasets_rejects_empty_config(
    tmp_path: Path,
) -> None:
    """
    At least one dataset must be configured.
    """

    with pytest.raises(
        ValueError,
        match="No datasets are configured",
    ):
        merge_processed_datasets(
            datasets=[],
            processed_folder=tmp_path,
        )


def test_merge_processed_datasets_rejects_duplicate_output_columns(
    tmp_path: Path,
) -> None:
    """
    Two datasets cannot be configured to produce the
    same master-dataset column.
    """

    duplicate_config = [
        {
            "name": "Gold One",
            "file_name": "gold_one.csv",
            "output_column": "Gold Price",
        },
        {
            "name": "Gold Two",
            "file_name": "gold_two.csv",
            "output_column": "Gold Price",
        },
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate output columns",
    ):
        merge_processed_datasets(
            datasets=duplicate_config,
            processed_folder=tmp_path,
        )


def test_create_dataset_metadata() -> None:
    """
    Metadata should describe the master dataset, source
    files, date range, and missing values.
    """

    master_data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                ]
            ),
            "Gold Price": [
                2000.0,
                2050.0,
            ],
            "CPI": [
                None,
                310.0,
            ],
        }
    )

    metadata = create_dataset_metadata(
        master_data=master_data,
        datasets=TEST_DATASETS,
    )

    assert metadata["rows"] == 2
    assert metadata["columns"] == 3

    assert metadata["datasets"] == [
        "Gold",
        "CPI",
    ]

    assert metadata["source_files"] == {
        "Gold": "gold.csv",
        "CPI": "cpi.csv",
    }

    assert metadata["date_range"] == {
        "start": "2024-01",
        "end": "2024-02",
    }

    assert metadata["missing_values"] == {
        "Gold Price": 0,
        "CPI": 1,
    }

    assert isinstance(
        metadata["generated_at"],
        str,
    )

    assert metadata["generated_at"]


def test_create_dataset_metadata_rejects_empty_dataset() -> None:
    """
    Metadata cannot be created for an empty DataFrame.
    """

    empty_data = pd.DataFrame(
        columns=[
            "Date",
            "Gold Price",
        ]
    )

    with pytest.raises(
        ValueError,
        match="empty dataset",
    ):
        create_dataset_metadata(
            master_data=empty_data,
            datasets=TEST_DATASETS,
        )


def test_save_master_dataset(
    tmp_path: Path,
) -> None:
    """
    The saved CSV should use YYYY-MM dates without
    changing the in-memory DataFrame.
    """

    master_data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                ]
            ),
            "Gold Price": [
                2000.0,
                2050.0,
            ],
        }
    )

    output_path = save_master_dataset(
        master_data=master_data,
        processed_folder=tmp_path,
    )

    saved_data = pd.read_csv(output_path)

    assert output_path.exists()

    assert output_path.name == "master_dataset.csv"

    assert saved_data["Date"].tolist() == [
        "2024-01",
        "2024-02",
    ]

    assert saved_data["Gold Price"].tolist() == [
        2000.0,
        2050.0,
    ]

    assert pd.api.types.is_datetime64_any_dtype(
        master_data["Date"]
    )


def test_save_dataset_metadata(
    tmp_path: Path,
) -> None:
    """
    Metadata should be written as readable JSON.
    """

    metadata = {
        "rows": 2,
        "columns": 3,
        "datasets": [
            "Gold",
            "CPI",
        ],
    }

    output_path = save_dataset_metadata(
        metadata=metadata,
        processed_folder=tmp_path,
    )

    with output_path.open(
        mode="r",
        encoding="utf-8",
    ) as metadata_file:
        saved_metadata = json.load(
            metadata_file
        )

    assert output_path.exists()

    assert output_path.name == (
        "master_dataset_metadata.json"
    )

    assert saved_metadata == metadata


def test_build_master_dataset(
    tmp_path: Path,
) -> None:
    """
    The complete pipeline should merge the datasets and
    save both output files.
    """

    write_csv(
        folder=tmp_path,
        file_name="gold.csv",
        data=pd.DataFrame(
            {
                "Date": [
                    "2024-01",
                    "2024-02",
                ],
                "Original Gold Header": [
                    2000.0,
                    2050.0,
                ],
            }
        ),
    )

    write_csv(
        folder=tmp_path,
        file_name="cpi.csv",
        data=pd.DataFrame(
            {
                "Date": [
                    "2024-01",
                    "2024-02",
                ],
                "Original CPI Header": [
                    309.0,
                    310.0,
                ],
            }
        ),
    )

    result = build_master_dataset(
        datasets=TEST_DATASETS,
        processed_folder=tmp_path,
    )

    master_path = (
        tmp_path / "master_dataset.csv"
    )

    metadata_path = (
        tmp_path
        / "master_dataset_metadata.json"
    )

    assert len(result) == 2

    assert list(result.columns) == [
        "Date",
        "Gold Price",
        "CPI",
    ]

    assert master_path.exists()
    assert metadata_path.exists()

    saved_master = pd.read_csv(
        master_path
    )

    assert saved_master["Date"].tolist() == [
        "2024-01",
        "2024-02",
    ]

    with metadata_path.open(
        mode="r",
        encoding="utf-8",
    ) as metadata_file:
        saved_metadata = json.load(
            metadata_file
        )

    assert saved_metadata["rows"] == 2
    assert saved_metadata["columns"] == 3

    assert saved_metadata["datasets"] == [
        "Gold",
        "CPI",
    ]