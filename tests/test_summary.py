from pathlib import Path

import pandas as pd
import pytest

from analytics.summary import (
    build_dataset_summary,
    calculate_column_summary,
    format_number,
    generate_dataset_summary,
    print_dataset_summary,
)


def sample_data() -> pd.DataFrame:
    """
    Return a representative validated master dataset.
    """

    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-04-01",
                ]
            ),
            "Gold Price": [
                2000.0,
                2050.0,
                None,
                2200.0,
            ],
            "CPI": [
                308.0,
                309.0,
                310.0,
                311.0,
            ],
            "Market Label": [
                "A",
                "B",
                None,
                "D",
            ],
        }
    )


def write_master_csv(
    folder: Path,
    data: pd.DataFrame,
    file_name: str = "master_dataset.csv",
) -> Path:
    """
    Write a temporary master dataset.
    """

    output_path = folder / file_name

    data.to_csv(
        output_path,
        index=False,
    )

    return output_path


def test_calculate_numeric_column_summary() -> None:
    """
    Numeric columns should include coverage and
    descriptive statistics.
    """

    data = sample_data()

    result = calculate_column_summary(
        data=data,
        column="Gold Price",
    )

    expected_values = pd.Series(
        [
            2000.0,
            2050.0,
            2200.0,
        ]
    )

    assert result["column"] == "Gold Price"
    assert result["total_rows"] == 4
    assert result["available_rows"] == 3
    assert result["missing_rows"] == 1
    assert result["completeness_percent"] == 75.0
    assert result["coverage_start"] == "2024-01"
    assert result["coverage_end"] == "2024-04"
    assert result["minimum"] == 2000.0
    assert result["maximum"] == 2200.0

    assert result["mean"] == pytest.approx(
        expected_values.mean()
    )

    assert result["median"] == pytest.approx(
        expected_values.median()
    )

    assert result[
        "standard_deviation"
    ] == pytest.approx(
        expected_values.std()
    )


def test_calculate_non_numeric_column_summary() -> None:
    """
    Text columns should include coverage information but
    not numeric statistics.
    """

    result = calculate_column_summary(
        data=sample_data(),
        column="Market Label",
    )

    assert result["available_rows"] == 3
    assert result["missing_rows"] == 1
    assert result["completeness_percent"] == 75.0
    assert result["coverage_start"] == "2024-01"
    assert result["coverage_end"] == "2024-04"
    assert result["minimum"] is None
    assert result["maximum"] is None
    assert result["mean"] is None
    assert result["median"] is None
    assert result["standard_deviation"] is None


def test_calculate_column_summary_handles_all_missing() -> None:
    """
    A completely missing column should still return
    completeness information.
    """

    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                ]
            ),
            "Gold Price": pd.Series(
                [
                    float("nan"),
                    float("nan"),
                ],
                dtype="float64",
            ),
        }
    )

    result = calculate_column_summary(
        data=data,
        column="Gold Price",
    )

    assert result["available_rows"] == 0
    assert result["missing_rows"] == 2
    assert result["completeness_percent"] == 0.0
    assert result["coverage_start"] is None
    assert result["coverage_end"] is None
    assert result["minimum"] is None
    assert result["maximum"] is None


def test_calculate_column_summary_rejects_unknown_column() -> None:
    """
    Unknown columns should raise an error.
    """

    with pytest.raises(
        ValueError,
        match="Column was not found",
    ):
        calculate_column_summary(
            data=sample_data(),
            column="Oil Price",
        )


def test_calculate_column_summary_rejects_date_column() -> None:
    """
    Date should not be summarized as an indicator.
    """

    with pytest.raises(
        ValueError,
        match="Date column cannot be summarized",
    ):
        calculate_column_summary(
            data=sample_data(),
            column="Date",
        )


def test_build_dataset_summary() -> None:
    """
    The complete summary should contain dataset and
    column-level information.
    """

    result = build_dataset_summary(
        sample_data()
    )

    assert result["row_count"] == 4
    assert result["column_count"] == 4
    assert result["analytical_column_count"] == 3
    assert result["numeric_column_count"] == 2
    assert result["non_numeric_column_count"] == 1

    assert result["date_range"] == {
        "start": "2024-01",
        "end": "2024-04",
    }

    assert result["available_columns"] == [
        "Gold Price",
        "CPI",
        "Market Label",
    ]

    assert result["numeric_columns"] == [
        "Gold Price",
        "CPI",
    ]

    assert result["non_numeric_columns"] == [
        "Market Label",
    ]

    assert "Gold Price" in result["columns"]
    assert "CPI" in result["columns"]
    assert "Market Label" in result["columns"]


def test_build_dataset_summary_rejects_empty_data() -> None:
    """
    Empty datasets should not produce a summary.
    """

    with pytest.raises(
        ValueError,
        match="empty dataset",
    ):
        build_dataset_summary(
            pd.DataFrame()
        )


def test_build_dataset_summary_rejects_missing_date() -> None:
    """
    The dataset must contain a Date column.
    """

    data = pd.DataFrame(
        {
            "Gold Price": [
                2000.0,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="missing the Date column",
    ):
        build_dataset_summary(data)


def test_generate_dataset_summary(
    tmp_path: Path,
) -> None:
    """
    The public generator should load and summarize the
    master CSV.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_data(),
    )

    result = generate_dataset_summary(
        processed_folder=tmp_path,
    )

    assert result["row_count"] == 4
    assert result["date_range"]["start"] == "2024-01"
    assert result["columns"]["Gold Price"][
        "missing_rows"
    ] == 1


def test_generate_dataset_summary_supports_custom_file(
    tmp_path: Path,
) -> None:
    """
    A custom master filename should be supported.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_data(),
        file_name="custom_master.csv",
    )

    result = generate_dataset_summary(
        processed_folder=tmp_path,
        file_name="custom_master.csv",
    )

    assert result["row_count"] == 4


def test_format_number() -> None:
    """
    Numeric values should be formatted consistently.
    """

    assert format_number(None) == "N/A"
    assert format_number(1500) == "1,500"
    assert format_number(1500.5) == "1,500.50"

    assert (
        format_number(
            1500.5678,
            decimal_places=3,
        )
        == "1,500.568"
    )


def test_print_dataset_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    The console report should include its major sections.
    """

    summary = build_dataset_summary(
        sample_data()
    )

    print_dataset_summary(summary)

    output = capsys.readouterr().out

    assert (
        "GOLD VALUE PREDICTOR — DATASET SUMMARY"
        in output
    )
    assert "DATASET OVERVIEW" in output
    assert "AVAILABLE INDICATORS" in output
    assert "COLUMN DETAILS" in output
    assert "Gold Price" in output
    assert "CPI" in output
    assert "Market Label" in output
    assert "75.00%" in output


def test_single_observation_has_no_standard_deviation() -> None:
    """
    One numeric observation has no sample standard
    deviation.
    """

    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                ]
            ),
            "Gold Price": [
                2000.0,
            ],
        }
    )

    result = calculate_column_summary(
        data=data,
        column="Gold Price",
    )

    assert result["standard_deviation"] is None