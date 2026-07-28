from math import sqrt
from pathlib import Path

import pandas as pd
import pytest

from analytics.risk import (
    MONTHS_PER_YEAR,
    annual_rate_to_monthly_rate,
    build_risk_summary,
    calculate_annualized_volatility,
    calculate_downside_deviation,
    calculate_drawdown_series,
    calculate_max_drawdown,
    calculate_monthly_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    format_percent,
    format_ratio,
    generate_risk_summary,
    print_risk_summary,
)


def sample_risk_data() -> pd.DataFrame:
    """
    Return price data containing positive and negative returns.
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
                99.0,
                108.9,
                87.12,
                104.544,
            ],
        }
    )


def drawdown_recovery_data() -> pd.DataFrame:
    """
    Return data with a clear peak, trough, and recovery.
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
                120.0,
                90.0,
                110.0,
                125.0,
            ],
        }
    )


def unrecovered_drawdown_data() -> pd.DataFrame:
    """
    Return data whose largest drawdown remains unrecovered.
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
                100.0,
                120.0,
                90.0,
                100.0,
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


def test_annual_rate_to_monthly_rate() -> None:
    """
    Compounding the monthly equivalent for 12 months should
    reproduce the annual rate.
    """

    annual_rate = 0.12

    monthly_rate = annual_rate_to_monthly_rate(
        annual_rate
    )

    compounded_rate = (
        (1 + monthly_rate) ** MONTHS_PER_YEAR
    ) - 1

    assert compounded_rate == pytest.approx(
        annual_rate
    )


def test_zero_annual_rate_converts_to_zero() -> None:
    """
    A zero annual rate should have a zero monthly equivalent.
    """

    assert annual_rate_to_monthly_rate(
        0.0
    ) == pytest.approx(0.0)


def test_annual_rate_rejects_negative_one() -> None:
    """
    Rates of -100% or less cannot be compounded.
    """

    with pytest.raises(
        ValueError,
        match="greater than -100%",
    ):
        annual_rate_to_monthly_rate(-1.0)


def test_calculate_monthly_volatility() -> None:
    """
    Monthly volatility should equal sample standard deviation.
    """

    monthly_returns = pd.Series(
        [
            0.10,
            -0.10,
            0.10,
            -0.20,
            0.20,
        ]
    )

    expected = monthly_returns.std(ddof=1)

    result = calculate_monthly_volatility(
        sample_risk_data()
    )

    assert result == pytest.approx(expected)


def test_monthly_volatility_requires_two_returns() -> None:
    """
    At least two returns are required for sample volatility.
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
                100.0,
                110.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="At least two monthly returns",
    ):
        calculate_monthly_volatility(data)


def test_calculate_annualized_volatility() -> None:
    """
    Annual volatility should equal monthly volatility times
    the square root of 12.
    """

    monthly_volatility = (
        calculate_monthly_volatility(
            sample_risk_data()
        )
    )

    result = calculate_annualized_volatility(
        sample_risk_data()
    )

    assert result == pytest.approx(
        monthly_volatility * sqrt(12)
    )


def test_calculate_drawdown_series() -> None:
    """
    Drawdowns should measure declines from the running peak.
    """

    result = calculate_drawdown_series(
        drawdown_recovery_data()
    )

    assert result.iloc[0] == pytest.approx(0.0)
    assert result.iloc[1] == pytest.approx(0.0)
    assert result.iloc[2] == pytest.approx(-0.25)

    assert result.iloc[3] == pytest.approx(
        (110.0 / 120.0) - 1
    )

    assert result.iloc[4] == pytest.approx(0.0)


def test_drawdown_series_name() -> None:
    """
    The drawdown Series should have a descriptive name.
    """

    result = calculate_drawdown_series(
        drawdown_recovery_data()
    )

    assert result.name == "Gold Price Drawdown"


def test_drawdown_rejects_zero_running_peak() -> None:
    """
    A zero running peak would cause division by zero.
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
                10.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="running peak is zero",
    ):
        calculate_drawdown_series(data)


def test_calculate_max_drawdown_with_recovery() -> None:
    """
    Maximum drawdown should identify its peak, trough,
    and recovery.
    """

    result = calculate_max_drawdown(
        drawdown_recovery_data()
    )

    assert result["drawdown"] == pytest.approx(
        -0.25
    )
    assert result["peak_date"] == "2024-02"
    assert result["trough_date"] == "2024-03"
    assert result["recovery_date"] == "2024-05"
    assert result["peak_value"] == 120.0
    assert result["trough_value"] == 90.0
    assert result["recovered"] is True


def test_calculate_max_drawdown_without_recovery() -> None:
    """
    An unrecovered decline should have no recovery date.
    """

    result = calculate_max_drawdown(
        unrecovered_drawdown_data()
    )

    assert result["drawdown"] == pytest.approx(
        -0.25
    )
    assert result["peak_date"] == "2024-02"
    assert result["trough_date"] == "2024-03"
    assert result["recovery_date"] is None
    assert result["recovered"] is False


def test_max_drawdown_for_only_new_highs() -> None:
    """
    A continuously rising series should have zero drawdown.
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
                120.0,
            ],
        }
    )

    result = calculate_max_drawdown(data)

    assert result["drawdown"] == pytest.approx(0.0)
    assert result["peak_date"] == "2024-01"
    assert result["trough_date"] == "2024-01"


def test_calculate_monthly_downside_deviation() -> None:
    """
    Non-annualized downside deviation should use negative
    return shortfalls and zeros for non-negative returns.
    """

    expected = sqrt(
        (
            0.0 ** 2
            + (-0.10) ** 2
            + 0.0 ** 2
            + (-0.20) ** 2
            + 0.0 ** 2
        )
        / 5
    )

    result = calculate_downside_deviation(
        data=sample_risk_data(),
        annualize=False,
    )

    assert result == pytest.approx(expected)


def test_calculate_annualized_downside_deviation() -> None:
    """
    Annualized downside deviation should apply square-root
    scaling.
    """

    monthly_result = calculate_downside_deviation(
        data=sample_risk_data(),
        annualize=False,
    )

    annual_result = calculate_downside_deviation(
        data=sample_risk_data(),
        annualize=True,
    )

    assert annual_result == pytest.approx(
        monthly_result * sqrt(12)
    )


def test_downside_deviation_with_positive_target() -> None:
    """
    A positive minimum acceptable return should increase
    shortfalls relative to a zero target.
    """

    zero_target = calculate_downside_deviation(
        data=sample_risk_data(),
        annual_minimum_acceptable_return=0.0,
        annualize=False,
    )

    positive_target = calculate_downside_deviation(
        data=sample_risk_data(),
        annual_minimum_acceptable_return=0.12,
        annualize=False,
    )

    assert positive_target > zero_target


def test_calculate_sharpe_ratio() -> None:
    """
    Sharpe ratio should annualize mean excess return divided
    by sample volatility.
    """

    returns = pd.Series(
        [
            0.10,
            -0.10,
            0.10,
            -0.20,
            0.20,
        ]
    )

    expected = (
        returns.mean()
        / returns.std(ddof=1)
    ) * sqrt(12)

    result = calculate_sharpe_ratio(
        sample_risk_data()
    )

    assert result == pytest.approx(expected)


def test_sharpe_ratio_uses_risk_free_rate() -> None:
    """
    Raising the risk-free rate should lower the Sharpe ratio
    for the sample data.
    """

    zero_rate_result = calculate_sharpe_ratio(
        data=sample_risk_data(),
        annual_risk_free_rate=0.0,
    )

    positive_rate_result = calculate_sharpe_ratio(
        data=sample_risk_data(),
        annual_risk_free_rate=0.05,
    )

    assert positive_rate_result < zero_rate_result


def test_sharpe_ratio_rejects_effectively_zero_volatility() -> None:
    """
    Mathematically constant returns may contain tiny
    floating-point differences and should still be treated
    as having zero volatility.
    """

    data = pd.DataFrame(
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
                100.0,
                110.0,
                121.0,
                133.1,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="volatility is zero",
    ):
        calculate_sharpe_ratio(data)


def test_calculate_sortino_ratio() -> None:
    """
    Sortino ratio should divide mean excess return by
    downside deviation and annualize it.
    """

    returns = pd.Series(
        [
            0.10,
            -0.10,
            0.10,
            -0.20,
            0.20,
        ]
    )

    shortfalls = returns.clip(upper=0)

    downside_deviation = sqrt(
        float((shortfalls ** 2).mean())
    )

    expected = (
        returns.mean()
        / downside_deviation
    ) * sqrt(12)

    result = calculate_sortino_ratio(
        sample_risk_data()
    )

    assert result == pytest.approx(expected)


def test_sortino_ratio_rejects_zero_downside() -> None:
    """
    A series with no downside returns has no usable Sortino
    denominator.
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

    with pytest.raises(
        ValueError,
        match="downside deviation is zero",
    ):
        calculate_sortino_ratio(data)


def test_build_risk_summary() -> None:
    """
    A complete risk summary should contain all major
    measurements.
    """

    result = build_risk_summary(
        data=sample_risk_data(),
        annual_risk_free_rate=0.02,
        annual_minimum_acceptable_return=0.01,
    )

    assert result["column"] == "Gold Price"
    assert result["observation_count"] == 6
    assert result["return_count"] == 5
    assert result["start_date"] == "2024-01"
    assert result["end_date"] == "2024-06"
    assert result["annual_risk_free_rate"] == 0.02

    assert (
        result["annual_minimum_acceptable_return"]
        == 0.01
    )

    assert result["monthly_volatility"] > 0
    assert result["annualized_volatility"] > 0

    assert (
        result["annualized_downside_deviation"]
        > 0
    )

    assert isinstance(
        result["sharpe_ratio"],
        float,
    )

    assert isinstance(
        result["sortino_ratio"],
        float,
    )

    assert "maximum_drawdown" in result


def test_generate_risk_summary(
    tmp_path: Path,
) -> None:
    """
    The public generator should load and analyze the master
    dataset.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_risk_data(),
    )

    result = generate_risk_summary(
        processed_folder=tmp_path,
    )

    assert result["column"] == "Gold Price"
    assert result["observation_count"] == 6
    assert result["return_count"] == 5


def test_generate_risk_summary_supports_custom_file(
    tmp_path: Path,
) -> None:
    """
    A custom master filename should be supported.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_risk_data(),
        file_name="custom_master.csv",
    )

    result = generate_risk_summary(
        processed_folder=tmp_path,
        file_name="custom_master.csv",
    )

    assert result["observation_count"] == 6


def test_format_percent() -> None:
    """
    Decimal values should be formatted as percentages.
    """

    assert format_percent(0.10) == "10.00%"
    assert format_percent(-0.25) == "-25.00%"

    assert format_percent(
        0.12345,
        decimal_places=3,
    ) == "12.345%"


def test_format_ratio() -> None:
    """
    Ratios should use three decimal places by default.
    """

    assert format_ratio(1.23456) == "1.235"
    assert format_ratio(-0.5) == "-0.500"


def test_print_risk_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Console output should contain the major report sections.
    """

    summary = build_risk_summary(
        sample_risk_data()
    )

    print_risk_summary(summary)

    output = capsys.readouterr().out

    assert (
        "GOLD VALUE PREDICTOR — RISK ANALYSIS"
        in output
    )
    assert "ANALYSIS PERIOD" in output
    assert "RISK ASSUMPTIONS" in output
    assert "VOLATILITY AND RISK RATIOS" in output
    assert "MAXIMUM DRAWDOWN" in output
    assert "Sharpe ratio:" in output
    assert "Sortino ratio:" in output
    assert "Recovered:" in output