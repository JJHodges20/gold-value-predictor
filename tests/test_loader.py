from pathlib import Path

import pandas as pd
import pytest

from analytics.loader import (
    available_columns,
    column_summary,
    dataset_date_range,
    load_date_range,
    load_master_data,
    numeric_columns,
    parse_month,
)


def write_master_csv(
    folder: Path,
    data: pd.DataFrame,
    file_name: str = "master_dataset.csv",
) -> Path:
    """
    Save a temporary master dataset for testing.
    """

    output_path = folder / file_name

    data.to_csv(
        output_path,
        index=False,
    )

    return output_path


def sample_master_data() -> pd.DataFrame:
    """
    Return a representative master dataset.
    """

    return pd.DataFrame(
        {
            "Date": [
                "2024-01",
                "2024-02",
                "2024-03",
                "2024-04",
            ],
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
                "C",
                "D",
            ],
        }
    )


def test_load_master_data(
    tmp_path: Path,
) -> None:
    """
    A valid master dataset should load successfully.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_master_data(),
    )

    result = load_master_data(
        processed_folder=tmp_path,
    )

    assert len(result) == 4

    assert list(result.columns) == [
        "Date",
        "Gold Price",
        "CPI",
        "Market Label",
    ]

    assert pd.api.types.is_datetime64_any_dtype(
        result["Date"]
    )


def test_load_master_data_sorts_dates(
    tmp_path: Path,
) -> None:
    """
    Loaded observations should be ordered by date.
    """

    data = sample_master_data().iloc[
        [2, 0, 3, 1]
    ]

    write_master_csv(
        folder=tmp_path,
        data=data,
    )

    result = load_master_data(
        processed_folder=tmp_path,
    )

    assert result["Date"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-03-01"),
        pd.Timestamp("2024-04-01"),
    ]


def test_load_master_data_returns_independent_copy(
    tmp_path: Path,
) -> None:
    """
    Separate loads should not share mutable DataFrames.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_master_data(),
    )

    first_result = load_master_data(
        processed_folder=tmp_path,
        copy=True,
    )

    first_result.loc[
        0,
        "Gold Price",
    ] = -1.0

    second_result = load_master_data(
        processed_folder=tmp_path,
        copy=True,
    )

    assert second_result.loc[
        0,
        "Gold Price",
    ] == 2000.0


def test_load_master_data_supports_custom_file_name(
    tmp_path: Path,
) -> None:
    """
    A caller should be able to load a custom CSV name.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_master_data(),
        file_name="custom_master.csv",
    )

    result = load_master_data(
        processed_folder=tmp_path,
        file_name="custom_master.csv",
    )

    assert len(result) == 4


def test_load_master_data_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """
    A missing master dataset should raise an error.
    """

    with pytest.raises(
        FileNotFoundError,
        match="Master dataset was not found",
    ):
        load_master_data(
            processed_folder=tmp_path,
        )


def test_load_master_data_rejects_empty_file(
    tmp_path: Path,
) -> None:
    """
    A completely empty CSV should raise an error.
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
        load_master_data(
            processed_folder=tmp_path,
        )


def test_load_master_data_rejects_header_only_file(
    tmp_path: Path,
) -> None:
    """
    A CSV with headers but no rows should be rejected.
    """

    write_master_csv(
        folder=tmp_path,
        data=pd.DataFrame(
            columns=[
                "Date",
                "Gold Price",
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="master dataset is empty",
    ):
        load_master_data(
            processed_folder=tmp_path,
        )


def test_load_master_data_rejects_missing_date_column(
    tmp_path: Path,
) -> None:
    """
    The master dataset must include a Date column.
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
        load_master_data(
            processed_folder=tmp_path,
        )


def test_load_master_data_rejects_invalid_dates(
    tmp_path: Path,
) -> None:
    """
    Invalid dates should not be silently removed.
    """

    write_master_csv(
        folder=tmp_path,
        data=pd.DataFrame(
            {
                "Date": [
                    "2024-01",
                    "not-a-date",
                ],
                "Gold Price": [
                    2000.0,
                    2050.0,
                ],
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="invalid date",
    ):
        load_master_data(
            processed_folder=tmp_path,
        )


def test_load_master_data_rejects_duplicate_months(
    tmp_path: Path,
) -> None:
    """
    Multiple observations in the same month should fail.
    """

    write_master_csv(
        folder=tmp_path,
        data=pd.DataFrame(
            {
                "Date": [
                    "2024-01-01",
                    "2024-01-20",
                ],
                "Gold Price": [
                    2000.0,
                    2050.0,
                ],
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="duplicate monthly dates",
    ):
        load_master_data(
            processed_folder=tmp_path,
        )


def test_parse_month() -> None:
    """
    A valid date should resolve to the first day of its
    month.
    """

    result = parse_month(
        value="2024-03-18",
        parameter_name="start",
    )

    assert result == pd.Timestamp(
        "2024-03-01"
    )


def test_parse_month_rejects_invalid_date() -> None:
    """
    Invalid range boundaries should raise an error.
    """

    with pytest.raises(
        ValueError,
        match="Invalid start date",
    ):
        parse_month(
            value="not-a-date",
            parameter_name="start",
        )


def test_load_date_range(
    tmp_path: Path,
) -> None:
    """
    Date-range filtering should include both boundaries.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_master_data(),
    )

    result = load_date_range(
        start="2024-02",
        end="2024-03",
        processed_folder=tmp_path,
    )

    assert result["Date"].tolist() == [
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-03-01"),
    ]


def test_load_date_range_allows_missing_start(
    tmp_path: Path,
) -> None:
    """
    An omitted start should use the earliest observation.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_master_data(),
    )

    result = load_date_range(
        end="2024-02",
        processed_folder=tmp_path,
    )

    assert result["Date"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
    ]


def test_load_date_range_allows_missing_end(
    tmp_path: Path,
) -> None:
    """
    An omitted end should use the latest observation.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_master_data(),
    )

    result = load_date_range(
        start="2024-03",
        processed_folder=tmp_path,
    )

    assert result["Date"].tolist() == [
        pd.Timestamp("2024-03-01"),
        pd.Timestamp("2024-04-01"),
    ]


def test_load_date_range_rejects_reversed_range(
    tmp_path: Path,
) -> None:
    """
    Start cannot occur after end.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_master_data(),
    )

    with pytest.raises(
        ValueError,
        match="start date cannot occur after",
    ):
        load_date_range(
            start="2024-04",
            end="2024-01",
            processed_folder=tmp_path,
        )


def test_load_date_range_rejects_empty_result(
    tmp_path: Path,
) -> None:
    """
    A valid range without observations should fail
    clearly.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_master_data(),
    )

    with pytest.raises(
        ValueError,
        match="No observations were found",
    ):
        load_date_range(
            start="2030-01",
            end="2030-12",
            processed_folder=tmp_path,
        )


def test_available_columns(
    tmp_path: Path,
) -> None:
    """
    Available columns should exclude Date.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_master_data(),
    )

    result = available_columns(
        processed_folder=tmp_path,
    )

    assert result == [
        "Gold Price",
        "CPI",
        "Market Label",
    ]


def test_numeric_columns(
    tmp_path: Path,
) -> None:
    """
    Numeric columns should exclude Date and text fields.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_master_data(),
    )

    result = numeric_columns(
        processed_folder=tmp_path,
    )

    assert result == [
        "Gold Price",
        "CPI",
    ]


def test_dataset_date_range(
    tmp_path: Path,
) -> None:
    """
    The dataset range should use YYYY-MM strings.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_master_data(),
    )

    result = dataset_date_range(
        processed_folder=tmp_path,
    )

    assert result == {
        "start": "2024-01",
        "end": "2024-04",
    }


def test_column_summary(
    tmp_path: Path,
) -> None:
    """
    Numeric summaries should exclude missing values from
    calculations and report them separately.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_master_data(),
    )

    result = column_summary(
        column="Gold Price",
        processed_folder=tmp_path,
    )

    expected_values = pd.Series(
        [
            2000.0,
            2050.0,
            2200.0,
        ]
    )

    assert result["column"] == "Gold Price"
    assert result["count"] == 3
    assert result["missing"] == 1

    assert result[
        "completeness_percent"
    ] == 75.0

    assert result[
        "coverage_start"
    ] == "2024-01"

    assert result[
        "coverage_end"
    ] == "2024-04"

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


def test_column_summary_rejects_unknown_column(
    tmp_path: Path,
) -> None:
    """
    Unknown columns should raise an error.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_master_data(),
    )

    with pytest.raises(
        ValueError,
        match="Column was not found",
    ):
        column_summary(
            column="Oil Price",
            processed_folder=tmp_path,
        )


def test_column_summary_rejects_date_column(
    tmp_path: Path,
) -> None:
    """
    Date should not be treated as an analytical value.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_master_data(),
    )

    with pytest.raises(
        ValueError,
        match="Date column cannot be summarized",
    ):
        column_summary(
            column="Date",
            processed_folder=tmp_path,
        )


def test_column_summary_rejects_non_numeric_column(
    tmp_path: Path,
) -> None:
    """
    Text columns should not receive numeric summaries.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_master_data(),
    )

    with pytest.raises(
        TypeError,
        match="Column is not numeric",
    ):
        column_summary(
            column="Market Label",
            processed_folder=tmp_path,
        )


def test_column_summary_rejects_all_missing_column(
    tmp_path: Path,
) -> None:
    """
    A numeric column with no usable observations should
    raise an error.
    """

    data = pd.DataFrame(
        {
            "Date": [
                "2024-01",
                "2024-02",
            ],
            "Gold Price": pd.Series(
                [
                    float("nan"),
                    float("nan"),
                ],
                dtype="float64",
            ),
        }
    )

    write_master_csv(
        folder=tmp_path,
        data=data,
    )

    with pytest.raises(
        ValueError,
        match="no usable observations",
    ):
        column_summary(
            column="Gold Price",
            processed_folder=tmp_path,
        )


def test_column_summary_handles_single_observation(
    tmp_path: Path,
) -> None:
    """
    A single observation has no sample standard
    deviation.
    """

    write_master_csv(
        folder=tmp_path,
        data=pd.DataFrame(
            {
                "Date": [
                    "2024-01",
                ],
                "Gold Price": [
                    2000.0,
                ],
            }
        ),
    )

    result = column_summary(
        column="Gold Price",
        processed_folder=tmp_path,
    )

    assert result[
        "standard_deviation"
    ] is None