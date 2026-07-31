import pandas as pd
import pytest

from analytics.forecasting import (
    calculate_historical_cagr,
    generate_growth_forecast,
    project_price_from_growth,
)


def create_test_data(
    months: int = 25,
) -> pd.DataFrame:
    dates = pd.date_range(
        start="2024-01-01",
        periods=months,
        freq="MS",
    )

    prices = [
        100 * (1.01 ** month)
        for month in range(months)
    ]

    return pd.DataFrame(
        {
            "Date": dates,
            "Gold Price": prices,
        }
    )


def test_project_price_from_growth():
    result = project_price_from_growth(
        current_price=1000,
        annual_growth_rate=0.10,
        forecast_years=2,
    )

    assert result == pytest.approx(1210.0)


def test_calculate_historical_cagr():
    data = create_test_data()

    result = calculate_historical_cagr(
        data=data,
        years=2,
    )

    expected = (
        data["Gold Price"].iloc[-1]
        / data["Gold Price"].iloc[0]
    ) ** (1 / 2) - 1

    assert result == pytest.approx(expected)


def test_generate_growth_forecast():
    data = create_test_data()

    result = generate_growth_forecast(
        data=data,
        historical_years=2,
        forecast_years=3,
    )

    assert result["method"] == "Historical CAGR"
    assert result["historical_years"] == 2
    assert result["forecast_years"] == 3
    assert result["projected_price"] > 0


def test_rejects_nonpositive_forecast_years():
    with pytest.raises(ValueError):
        project_price_from_growth(
            current_price=1000,
            annual_growth_rate=0.05,
            forecast_years=0,
        )