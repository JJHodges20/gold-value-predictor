from pathlib import Path

import pandas as pd
import pytest

from analytics.returns import (
    build_return_summary,
    calculate_annual_returns,
    calculate_average_annual_return,
    calculate_average_monthly_return,
    calculate_best_month,
    calculate_cagr,
    calculate_cumulative_returns,
    calculate_monthly_returns,
    calculate_total_return,
    calculate_worst_month,
    calculate_year_to_date_return,
    format_percent,
    generate_return_summary,
    prepare_price_series,
    print_return_summary,
)


def sample_monthly_data() -> pd.DataFrame:
    """
    Return a simple monthly price dataset.
    """

    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2023-12-01",
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-04-01",
                    "2024-05-01",
                    "2024-06-01",
                    "2024-07-01",
                    "2024-08-01",
                    "2024-09-01",
                    "2024-10-01",
                    "2024-11-01",
                    "2024-12-01",
                ]
            ),
            "Gold Price": [
                100.0,
                105.0,
                110.0,
                99.0,
                108.0,
                115.0,
                120.0,
                126.0,
                130.0,
                125.0,
                135.0,
                140.0,
                150.0,
            ],
        }
    )


def sample_annual_data() -> pd.DataFrame:
    """
    Return year-end observations with predictable returns.
    """

    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2020-12-01",
                    "2021-12-01",
                    "2022-12-01",
                    "2023-12-01",
                ]
            ),
            "Gold Price": [
                100.0,
                110.0,
                121.0,
                133.1,
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

    file_path = folder / file_name

    data.to_csv(
        file_path,
        index=False,
    )

    return file_path


def test_prepare_price_series() -> None:
    """
    Price data should be sorted, indexed, and numeric.
    """

    data = pd.DataFrame(
        {
            "Date": [
                "2024-03",
                "2024-01",
                "2024-02",
            ],
            "Gold Price": [
                120.0,
                100.0,
                110.0,
            ],
        }
    )

    result = prepare_price_series(data)

    assert result.index.tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-03-01"),
    ]

    assert result.tolist() == [
        100.0,
        110.0,
        120.0,
    ]


def test_prepare_price_series_removes_missing_values() -> None:
    """
    Missing price observations should be removed.
    """

    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                ]
            ),
            "Gold Price": [
                100.0,
                None,
                120.0,
            ],
        }
    )

    result = prepare_price_series(data)

    assert len(result) == 2
    assert result.tolist() == [
        100.0,
        120.0,
    ]


def test_prepare_price_series_rejects_missing_column() -> None:
    """
    Unknown price columns should raise an error.
    """

    with pytest.raises(
        ValueError,
        match="Column was not found",
    ):
        prepare_price_series(
            data=sample_monthly_data(),
            column="Silver Price",
        )


def test_prepare_price_series_rejects_non_numeric_column() -> None:
    """
    Return calculations require numeric data.
    """

    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                ]
            ),
            "Market": [
                "Up",
                "Down",
            ],
        }
    )

    with pytest.raises(
        TypeError,
        match="Column is not numeric",
    ):
        prepare_price_series(
            data=data,
            column="Market",
        )


def test_prepare_price_series_rejects_duplicate_months() -> None:
    """
    Duplicate monthly observations should be rejected.
    """

    data = pd.DataFrame(
        {
            "Date": [
                "2024-01-01",
                "2024-01-20",
            ],
            "Gold Price": [
                100.0,
                105.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="duplicate monthly dates",
    ):
        prepare_price_series(data)


def test_calculate_monthly_returns() -> None:
    """
    Monthly returns should use percentage change.
    """

    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                ]
            ),
            "Gold Price": [
                100.0,
                110.0,
                99.0,
            ],
        }
    )

    result = calculate_monthly_returns(data)

    assert len(result) == 2
    assert result.iloc[0] == pytest.approx(0.10)
    assert result.iloc[1] == pytest.approx(-0.10)


def test_monthly_returns_can_keep_initial_missing_value() -> None:
    """
    The caller may retain the initial NaN return.
    """

    result = calculate_monthly_returns(
        data=sample_monthly_data(),
        drop_missing=False,
    )

    assert len(result) == 13
    assert pd.isna(result.iloc[0])


def test_calculate_cumulative_returns() -> None:
    """
    Cumulative return should measure growth from the
    first observation.
    """

    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                ]
            ),
            "Gold Price": [
                100.0,
                110.0,
                121.0,
            ],
        }
    )

    result = calculate_cumulative_returns(data)

    assert result.iloc[0] == pytest.approx(0.0)
    assert result.iloc[1] == pytest.approx(0.10)
    assert result.iloc[2] == pytest.approx(0.21)


def test_calculate_total_return() -> None:
    """
    Total return should compare first and last values.
    """

    result = calculate_total_return(
        sample_annual_data()
    )

    assert result == pytest.approx(0.331)


def test_total_return_requires_two_observations() -> None:
    """
    Total return requires a beginning and ending value.
    """

    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                ]
            ),
            "Gold Price": [
                100.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="At least two observations",
    ):
        calculate_total_return(data)


def test_total_return_rejects_zero_initial_value() -> None:
    """
    A zero initial value would cause division by zero.
    """

    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                ]
            ),
            "Gold Price": [
                0.0,
                100.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="initial value is zero",
    ):
        calculate_total_return(data)


def test_calculate_cagr() -> None:
    """
    A value doubling over two years should produce the
    expected CAGR.
    """

    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2022-01-01",
                    "2024-01-01",
                ]
            ),
            "Gold Price": [
                100.0,
                200.0,
            ],
        }
    )

    result = calculate_cagr(data)

    expected = (2.0 ** (1 / 2)) - 1

    assert result == pytest.approx(expected)


def test_cagr_rejects_non_positive_values() -> None:
    """
    CAGR requires positive boundary values.
    """

    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2023-01-01",
                    "2024-01-01",
                ]
            ),
            "Gold Price": [
                -100.0,
                110.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="positive initial and final",
    ):
        calculate_cagr(data)


def test_calculate_annual_returns() -> None:
    """
    Annual returns should compare consecutive year-end
    observations.
    """

    result = calculate_annual_returns(
        sample_annual_data()
    )

    assert result.index.tolist() == [
        2021,
        2022,
        2023,
    ]

    assert result.tolist() == pytest.approx(
        [
            0.10,
            0.10,
            0.10,
        ]
    )


def test_calculate_year_to_date_return() -> None:
    """
    YTD return should compare the first and last values
    within the requested year.
    """

    result = calculate_year_to_date_return(
        data=sample_monthly_data(),
        year=2024,
    )

    expected = (150.0 / 105.0) - 1

    assert result == pytest.approx(expected)


def test_year_to_date_uses_latest_year_by_default() -> None:
    """
    The latest available year should be selected when no
    year is supplied.
    """

    result = calculate_year_to_date_return(
        sample_monthly_data()
    )

    expected = (150.0 / 105.0) - 1

    assert result == pytest.approx(expected)


def test_year_to_date_requires_two_observations() -> None:
    """
    A year with fewer than two values cannot produce a
    return.
    """

    with pytest.raises(
        ValueError,
        match="At least two observations",
    ):
        calculate_year_to_date_return(
            data=sample_monthly_data(),
            year=2023,
        )


def test_average_monthly_return() -> None:
    """
    Average monthly return should equal the arithmetic
    mean of monthly percentage changes.
    """

    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                ]
            ),
            "Gold Price": [
                100.0,
                110.0,
                99.0,
            ],
        }
    )

    result = calculate_average_monthly_return(
        data
    )

    assert result == pytest.approx(0.0)


def test_average_annual_return() -> None:
    """
    Average annual return should average annual values.
    """

    result = calculate_average_annual_return(
        sample_annual_data()
    )

    assert result == pytest.approx(0.10)


def test_calculate_best_month() -> None:
    """
    The highest monthly return should be identified.
    """

    result = calculate_best_month(
        sample_monthly_data()
    )

    monthly_returns = calculate_monthly_returns(
        sample_monthly_data()
    )

    expected_date = monthly_returns.idxmax()
    expected_return = monthly_returns.max()

    assert result["date"] == expected_date.strftime(
        "%Y-%m"
    )

    assert result["return"] == pytest.approx(
        expected_return
    )


def test_calculate_worst_month() -> None:
    """
    The lowest monthly return should be identified.
    """

    result = calculate_worst_month(
        sample_monthly_data()
    )

    monthly_returns = calculate_monthly_returns(
        sample_monthly_data()
    )

    expected_date = monthly_returns.idxmin()
    expected_return = monthly_returns.min()

    assert result["date"] == expected_date.strftime(
        "%Y-%m"
    )

    assert result["return"] == pytest.approx(
        expected_return
    )


def test_build_return_summary() -> None:
    """
    A complete return summary should include all major
    metrics.
    """

    result = build_return_summary(
        sample_annual_data()
    )

    assert result["column"] == "Gold Price"
    assert result["observation_count"] == 4
    assert result["start_date"] == "2020-12"
    assert result["end_date"] == "2023-12"
    assert result["starting_value"] == 100.0
    assert result["ending_value"] == 133.1
    assert result["total_return"] == pytest.approx(
        0.331
    )
    assert result["cagr"] == pytest.approx(
        0.10
    )
    assert "best_month" in result
    assert "worst_month" in result


def test_generate_return_summary(
    tmp_path: Path,
) -> None:
    """
    The public generator should load and analyze the
    master dataset.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_annual_data(),
    )

    result = generate_return_summary(
        processed_folder=tmp_path,
    )

    assert result["column"] == "Gold Price"
    assert result["observation_count"] == 4
    assert result["total_return"] == pytest.approx(
        0.331
    )


def test_generate_return_summary_supports_custom_file(
    tmp_path: Path,
) -> None:
    """
    Custom master filenames should be supported.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_annual_data(),
        file_name="custom_master.csv",
    )

    result = generate_return_summary(
        processed_folder=tmp_path,
        file_name="custom_master.csv",
    )

    assert result["observation_count"] == 4


def test_format_percent() -> None:
    """
    Decimal returns should be formatted as percentages.
    """

    assert format_percent(0.10) == "10.00%"
    assert format_percent(-0.125) == "-12.50%"
    assert format_percent(
        0.12345,
        decimal_places=3,
    ) == "12.345%"


def test_print_return_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Console output should contain all major report
    sections.
    """

    summary = build_return_summary(
        sample_annual_data()
    )

    print_return_summary(summary)

    output = capsys.readouterr().out

    assert (
        "GOLD VALUE PREDICTOR — RETURN ANALYSIS"
        in output
    )
    assert "ANALYSIS PERIOD" in output
    assert "RETURN METRICS" in output
    assert "BEST AND WORST MONTHS" in output
    assert "Total return:" in output
    assert "CAGR:" in output