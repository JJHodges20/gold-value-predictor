from math import sqrt
from pathlib import Path

import pandas as pd
import pytest

from analytics.rolling import (
    MONTHS_PER_YEAR,
    build_rolling_summary,
    calculate_rolling_average,
    calculate_rolling_drawdown,
    calculate_rolling_high,
    calculate_rolling_low,
    calculate_rolling_range,
    calculate_rolling_return,
    calculate_rolling_volatility,
    format_percent,
    format_value,
    generate_rolling_summary,
    get_latest_valid_date,
    get_latest_valid_value,
    print_rolling_summary,
    validate_price_window,
    validate_return_window,
    validate_window,
)


def sample_rolling_data() -> pd.DataFrame:
    """
    Return a predictable monthly gold-price dataset.
    """

    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-04-01",
                    "2024-05-01",
                    "2024-06-01",
                ]
            ),
            "Gold Price": [
                100.0,
                110.0,
                90.0,
                120.0,
                105.0,
                126.0,
            ],
        }
    )


def steadily_rising_data() -> pd.DataFrame:
    """
    Return prices that increase by 10% every month.
    """

    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-04-01",
                    "2024-05-01",
                ]
            ),
            "Gold Price": [
                100.0,
                110.0,
                121.0,
                133.1,
                146.41,
            ],
        }
    )


def write_master_csv(
    folder: Path,
    data: pd.DataFrame,
    file_name: str = "master_dataset.csv",
) -> Path:
    """
    Write a temporary master dataset for generator tests.
    """

    file_path = folder / file_name

    data.to_csv(
        file_path,
        index=False,
    )

    return file_path


def test_validate_window_accepts_positive_integer() -> None:
    """
    Positive integer windows should be accepted.
    """

    validate_window(1)
    validate_window(12)


@pytest.mark.parametrize(
    "invalid_window",
    [
        1.5,
        "12",
        None,
        True,
        False,
    ],
)
def test_validate_window_rejects_non_integer_values(
    invalid_window: object,
) -> None:
    """
    Non-integer values, including booleans, should fail.
    """

    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        validate_window(invalid_window)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_window",
    [
        0,
        -1,
        -12,
    ],
)
def test_validate_window_rejects_values_below_one(
    invalid_window: int,
) -> None:
    """
    Windows must contain at least one observation.
    """

    with pytest.raises(
        ValueError,
        match="must be at least 1",
    ):
        validate_window(invalid_window)


def test_validate_price_window_accepts_enough_prices() -> None:
    """
    A price Series equal to the window size should pass.
    """

    prices = pd.Series(
        [100.0, 110.0, 120.0]
    )

    validate_price_window(
        prices=prices,
        window=3,
    )


def test_validate_price_window_rejects_short_series() -> None:
    """
    A price Series shorter than the window should fail.
    """

    prices = pd.Series(
        [100.0, 110.0]
    )

    with pytest.raises(
        ValueError,
        match="At least 3 price observations",
    ):
        validate_price_window(
            prices=prices,
            window=3,
        )


def test_validate_return_window_accepts_enough_returns() -> None:
    """
    A return Series equal to the requested window should pass.
    """

    returns = pd.Series(
        [0.10, -0.05, 0.08]
    )

    validate_return_window(
        monthly_returns=returns,
        window=3,
    )


def test_validate_return_window_rejects_short_series() -> None:
    """
    A return Series shorter than the window should fail.
    """

    returns = pd.Series(
        [0.10, -0.05]
    )

    with pytest.raises(
        ValueError,
        match="At least 3 monthly returns",
    ):
        validate_return_window(
            monthly_returns=returns,
            window=3,
        )


def test_calculate_rolling_average() -> None:
    """
    Rolling averages should use the requested number of prices.
    """

    result = calculate_rolling_average(
        data=sample_rolling_data(),
        window=3,
    )

    assert result.iloc[0:2].isna().all()

    assert result.iloc[2] == pytest.approx(
        (100.0 + 110.0 + 90.0) / 3
    )

    assert result.iloc[3] == pytest.approx(
        (110.0 + 90.0 + 120.0) / 3
    )

    assert result.iloc[5] == pytest.approx(
        (120.0 + 105.0 + 126.0) / 3
    )


def test_rolling_average_name() -> None:
    """
    The rolling-average Series should have a descriptive name.
    """

    result = calculate_rolling_average(
        data=sample_rolling_data(),
        window=3,
    )

    assert (
        result.name
        == "Gold Price 3-Month Rolling Average"
    )


def test_rolling_average_supports_custom_column() -> None:
    """
    Rolling calculations should support alternate price columns.
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
            "Silver Price": [
                20.0,
                22.0,
                24.0,
            ],
        }
    )

    result = calculate_rolling_average(
        data=data,
        column="Silver Price",
        window=2,
    )

    assert result.iloc[-1] == pytest.approx(23.0)

    assert (
        result.name
        == "Silver Price 2-Month Rolling Average"
    )


def test_rolling_average_rejects_short_dataset() -> None:
    """
    A dataset shorter than the requested window should fail.
    """

    with pytest.raises(
        ValueError,
        match="At least 7 price observations",
    ):
        calculate_rolling_average(
            data=sample_rolling_data(),
            window=7,
        )


def test_calculate_rolling_return() -> None:
    """
    Rolling returns should compare each price with the price
    the requested number of observations earlier.
    """

    result = calculate_rolling_return(
        data=sample_rolling_data(),
        window=2,
    )

    assert result.iloc[0:2].isna().all()

    assert result.iloc[2] == pytest.approx(
        (90.0 / 100.0) - 1
    )

    assert result.iloc[3] == pytest.approx(
        (120.0 / 110.0) - 1
    )

    assert result.iloc[5] == pytest.approx(
        (126.0 / 120.0) - 1
    )


def test_rolling_return_name() -> None:
    """
    The rolling-return Series should have a descriptive name.
    """

    result = calculate_rolling_return(
        data=sample_rolling_data(),
        window=2,
    )

    assert (
        result.name
        == "Gold Price 2-Month Rolling Return"
    )


def test_rolling_return_requires_window_plus_one_prices() -> None:
    """
    A period return requires both a starting and ending price.
    """

    with pytest.raises(
        ValueError,
        match="At least 7 price observations",
    ):
        calculate_rolling_return(
            data=sample_rolling_data(),
            window=6,
        )


def test_calculate_nonannualized_rolling_volatility() -> None:
    """
    Rolling volatility should equal the sample standard
    deviation of monthly returns.
    """

    data = sample_rolling_data()

    expected_returns = pd.Series(
        [
            0.10,
            (90.0 / 110.0) - 1,
            (120.0 / 90.0) - 1,
            (105.0 / 120.0) - 1,
            (126.0 / 105.0) - 1,
        ]
    )

    result = calculate_rolling_volatility(
        data=data,
        window=3,
        annualize=False,
    )

    # Six prices produce five monthly returns. A three-return
    # rolling window therefore has two initial missing values.
    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].notna().all()

    assert result.iloc[2] == pytest.approx(
        expected_returns.iloc[0:3].std(ddof=1)
    )

    assert result.iloc[3] == pytest.approx(
        expected_returns.iloc[1:4].std(ddof=1)
    )

    assert result.iloc[4] == pytest.approx(
        expected_returns.iloc[2:5].std(ddof=1)
    )


def test_calculate_annualized_rolling_volatility() -> None:
    """
    Annualized volatility should apply square-root-of-time
    scaling to rolling monthly volatility.
    """

    monthly_result = calculate_rolling_volatility(
        data=sample_rolling_data(),
        window=3,
        annualize=False,
    )

    annual_result = calculate_rolling_volatility(
        data=sample_rolling_data(),
        window=3,
        annualize=True,
    )

    assert annual_result.iloc[-1] == pytest.approx(
        monthly_result.iloc[-1]
        * sqrt(MONTHS_PER_YEAR)
    )


def test_rolling_volatility_names() -> None:
    """
    Annualized and monthly Series should have distinct names.
    """

    monthly_result = calculate_rolling_volatility(
        data=sample_rolling_data(),
        window=3,
        annualize=False,
    )

    annual_result = calculate_rolling_volatility(
        data=sample_rolling_data(),
        window=3,
        annualize=True,
    )

    assert (
        monthly_result.name
        == "Gold Price 3-Month Rolling Volatility"
    )

    assert (
        annual_result.name
        == (
            "Gold Price 3-Month "
            "Annualized Rolling Volatility"
        )
    )


def test_rolling_volatility_rejects_short_return_series() -> None:
    """
    Volatility should fail when too few returns are available.
    """

    with pytest.raises(
        ValueError,
        match="At least 6 monthly returns",
    ):
        calculate_rolling_volatility(
            data=sample_rolling_data(),
            window=6,
        )


def test_calculate_rolling_high() -> None:
    """
    Rolling highs should contain each window's maximum price.
    """

    result = calculate_rolling_high(
        data=sample_rolling_data(),
        window=3,
    )

    assert result.iloc[0:2].isna().all()
    assert result.iloc[2] == pytest.approx(110.0)
    assert result.iloc[3] == pytest.approx(120.0)
    assert result.iloc[5] == pytest.approx(126.0)


def test_rolling_high_name() -> None:
    """
    The rolling-high Series should have a descriptive name.
    """

    result = calculate_rolling_high(
        data=sample_rolling_data(),
        window=3,
    )

    assert (
        result.name
        == "Gold Price 3-Month Rolling High"
    )


def test_calculate_rolling_low() -> None:
    """
    Rolling lows should contain each window's minimum price.
    """

    result = calculate_rolling_low(
        data=sample_rolling_data(),
        window=3,
    )

    assert result.iloc[0:2].isna().all()
    assert result.iloc[2] == pytest.approx(90.0)
    assert result.iloc[3] == pytest.approx(90.0)
    assert result.iloc[5] == pytest.approx(105.0)


def test_rolling_low_name() -> None:
    """
    The rolling-low Series should have a descriptive name.
    """

    result = calculate_rolling_low(
        data=sample_rolling_data(),
        window=3,
    )

    assert (
        result.name
        == "Gold Price 3-Month Rolling Low"
    )


def test_calculate_rolling_drawdown() -> None:
    """
    Rolling drawdown should measure the current price's decline
    from the highest price in its window.
    """

    result = calculate_rolling_drawdown(
        data=sample_rolling_data(),
        window=3,
    )

    assert result.iloc[0:2].isna().all()

    assert result.iloc[2] == pytest.approx(
        (90.0 / 110.0) - 1
    )

    assert result.iloc[3] == pytest.approx(0.0)

    assert result.iloc[4] == pytest.approx(
        (105.0 / 120.0) - 1
    )

    assert result.iloc[5] == pytest.approx(0.0)


def test_rolling_drawdown_name() -> None:
    """
    The rolling-drawdown Series should have a descriptive name.
    """

    result = calculate_rolling_drawdown(
        data=sample_rolling_data(),
        window=3,
    )

    assert (
        result.name
        == "Gold Price 3-Month Rolling Drawdown"
    )


def test_rolling_drawdown_rejects_zero_high() -> None:
    """
    A zero rolling high would make drawdown division invalid.
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
                0.0,
                0.0,
                0.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="rolling high is zero",
    ):
        calculate_rolling_drawdown(
            data=data,
            window=2,
        )


def test_rising_prices_have_zero_rolling_drawdown() -> None:
    """
    New highs should produce zero rolling drawdown.
    """

    result = calculate_rolling_drawdown(
        data=steadily_rising_data(),
        window=3,
    )

    valid_results = result.dropna()

    assert valid_results.tolist() == pytest.approx(
        [0.0] * len(valid_results)
    )


def test_calculate_rolling_range() -> None:
    """
    Rolling range should equal rolling high minus rolling low.
    """

    result = calculate_rolling_range(
        data=sample_rolling_data(),
        window=3,
    )

    assert result.iloc[0:2].isna().all()
    assert result.iloc[2] == pytest.approx(20.0)
    assert result.iloc[3] == pytest.approx(30.0)
    assert result.iloc[5] == pytest.approx(21.0)


def test_rolling_range_name() -> None:
    """
    The rolling-range Series should have a descriptive name.
    """

    result = calculate_rolling_range(
        data=sample_rolling_data(),
        window=3,
    )

    assert (
        result.name
        == "Gold Price 3-Month Rolling Range"
    )


def test_get_latest_valid_value() -> None:
    """
    The helper should return the final non-missing value.
    """

    series = pd.Series(
        [
            float("nan"),
            10.0,
            float("nan"),
            20.0,
        ]
    )

    result = get_latest_valid_value(
        series=series,
        metric_name="test metric",
    )

    assert result == pytest.approx(20.0)


def test_get_latest_valid_value_rejects_all_missing() -> None:
    """
    A Series containing no valid values should fail clearly.
    """

    series = pd.Series(
        [
            float("nan"),
            float("nan"),
        ]
    )

    with pytest.raises(
        ValueError,
        match="No valid test metric values",
    ):
        get_latest_valid_value(
            series=series,
            metric_name="test metric",
        )


def test_get_latest_valid_date() -> None:
    """
    The helper should return the date of the final valid value.
    """

    series = pd.Series(
        [
            float("nan"),
            10.0,
            float("nan"),
            20.0,
        ],
        index=pd.to_datetime(
            [
                "2024-01-01",
                "2024-02-01",
                "2024-03-01",
                "2024-04-01",
            ]
        ),
    )

    result = get_latest_valid_date(
        series=series,
        metric_name="test metric",
    )

    assert result == "2024-04"


def test_get_latest_valid_date_rejects_all_missing() -> None:
    """
    A Series containing no dated valid value should fail.
    """

    series = pd.Series(
        [
            float("nan"),
            float("nan"),
        ],
        index=pd.to_datetime(
            [
                "2024-01-01",
                "2024-02-01",
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="No valid test metric values",
    ):
        get_latest_valid_date(
            series=series,
            metric_name="test metric",
        )


def test_build_rolling_summary() -> None:
    """
    The summary should contain the latest rolling metrics.
    """

    result = build_rolling_summary(
        data=sample_rolling_data(),
        window=3,
    )

    assert result["column"] == "Gold Price"
    assert result["window"] == 3
    assert result["observation_count"] == 6
    assert result["start_date"] == "2024-01"
    assert result["end_date"] == "2024-06"
    assert result["latest_metric_date"] == "2024-06"

    assert result["latest_price"] == pytest.approx(
        126.0
    )

    assert result["rolling_average"] == pytest.approx(
        (120.0 + 105.0 + 126.0) / 3
    )

    assert result["rolling_return"] == pytest.approx(
        (126.0 / 90.0) - 1
    )

    assert result["rolling_high"] == pytest.approx(
        126.0
    )

    assert result["rolling_low"] == pytest.approx(
        105.0
    )

    assert result["rolling_range"] == pytest.approx(
        21.0
    )

    assert result["rolling_drawdown"] == pytest.approx(
        0.0
    )

    assert (
        result["annualized_rolling_volatility"]
        > 0
    )


def test_build_rolling_summary_rejects_short_dataset() -> None:
    """
    A summary requires at least window plus one prices.
    """

    with pytest.raises(
        ValueError,
        match="At least 7 price observations",
    ):
        build_rolling_summary(
            data=sample_rolling_data(),
            window=6,
        )


def test_generate_rolling_summary(
    tmp_path: Path,
) -> None:
    """
    The public generator should load and analyze master data.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_rolling_data(),
    )

    result = generate_rolling_summary(
        processed_folder=tmp_path,
        window=3,
    )

    assert result["column"] == "Gold Price"
    assert result["window"] == 3
    assert result["observation_count"] == 6

    assert result["latest_price"] == pytest.approx(
        126.0
    )


def test_generate_rolling_summary_supports_custom_file(
    tmp_path: Path,
) -> None:
    """
    The generator should accept a custom master filename.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_rolling_data(),
        file_name="custom_master.csv",
    )

    result = generate_rolling_summary(
        processed_folder=tmp_path,
        file_name="custom_master.csv",
        window=3,
    )

    assert result["window"] == 3
    assert result["observation_count"] == 6


def test_generate_rolling_summary_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """
    A missing master file should produce a file error.
    """

    with pytest.raises(FileNotFoundError):
        generate_rolling_summary(
            processed_folder=tmp_path,
            file_name="missing.csv",
            window=3,
        )


def test_format_percent() -> None:
    """
    Decimal values should be formatted as percentages.
    """

    assert format_percent(0.10) == "10.00%"
    assert format_percent(-0.125) == "-12.50%"

    assert format_percent(
        0.12345,
        decimal_places=3,
    ) == "12.345%"


def test_format_value() -> None:
    """
    Numeric values should include grouping and decimal places.
    """

    assert format_value(1234.5) == "1,234.50"
    assert format_value(-50.25) == "-50.25"

    assert format_value(
        1234.5678,
        decimal_places=3,
    ) == "1,234.568"


def test_print_rolling_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Console output should contain the report's major sections.
    """

    summary = build_rolling_summary(
        data=sample_rolling_data(),
        window=3,
    )

    print_rolling_summary(summary)

    output = capsys.readouterr().out

    assert (
        "GOLD VALUE PREDICTOR — ROLLING ANALYSIS"
        in output
    )

    assert "ANALYSIS PERIOD" in output
    assert "LATEST ROLLING METRICS" in output
    assert "Rolling window:" in output
    assert "3 months" in output
    assert "3-month average:" in output
    assert "3-month return:" in output
    assert "Annualized volatility:" in output
    assert "Drawdown from rolling high:" in output