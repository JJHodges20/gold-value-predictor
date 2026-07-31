import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from visualizations.charts import (
    DEFAULT_CHART_TITLE,
    DEFAULT_FORECAST_BOUNDARY_LABEL,
    DEFAULT_FORECAST_HISTORY_LABEL,
    DEFAULT_HISTORICAL_PRICE_SUBTITLE,
    DEFAULT_PRICE_LABEL,
    plot_forecast,
    plot_forecast_scenarios,
    plot_historical_price,
    DEFAULT_ANNUAL_RETURNS_SUBTITLE,
    DEFAULT_CUMULATIVE_RETURNS_SUBTITLE,
    DEFAULT_MONTHLY_RETURNS_SUBTITLE,
    DEFAULT_RETURN_DISTRIBUTION_SUBTITLE,
    plot_annual_returns,
    plot_cumulative_returns,
    plot_monthly_returns,
    plot_return_distribution,
    DEFAULT_ROLLING_AVERAGE_SUBTITLE,
    DEFAULT_ROLLING_DRAWDOWN_SUBTITLE,
    DEFAULT_ROLLING_HIGH_LOW_SUBTITLE,
    DEFAULT_ROLLING_RETURN_SUBTITLE,
    DEFAULT_ROLLING_VOLATILITY_SUBTITLE,
    plot_rolling_average,
    plot_rolling_drawdown,
    plot_rolling_high_low,
    plot_rolling_return,
    plot_rolling_volatility,
    DEFAULT_FORECAST_DISCLAIMER,
    DEFAULT_FORECAST_SCENARIO_SUBTITLE,
    DEFAULT_FORECAST_SUBTITLE,
    DEFAULT_INFLATION_ADJUSTED_SUBTITLE,
    DEFAULT_NOMINAL_LABEL,
    DEFAULT_NOMINAL_VS_REAL_SUBTITLE,
    DEFAULT_REAL_LABEL,
    DEFAULT_RECENT_PRICE_SUBTITLE,
    plot_inflation_adjusted_price,
    plot_nominal_vs_real_price,
    plot_recent_price,
)


from visualizations.styling import (
    CUMULATIVE_RETURN_COLOR,
    DEFAULT_PERCENTAGE_FILL_ALPHA,
    HISTORICAL_PRICE_COLOR,
    NEGATIVE_RETURN_COLOR,
    NEUTRAL_RETURN_COLOR,
    NOMINAL_PRICE_COLOR,
    PLOT_BACKGROUND_COLOR,
    POSITIVE_RETURN_COLOR,
    REAL_PRICE_COLOR,
    RECENT_PRICE_COLOR,
    REFERENCE_LINE_COLOR,
    DRAWDOWN_COLOR,
    ROLLING_AVERAGE_COLOR,
    ROLLING_HIGH_COLOR,
    ROLLING_LOW_COLOR,
    ROLLING_RETURN_COLOR,
    VOLATILITY_COLOR,
    FORECAST_BASE_COLOR,
    FORECAST_BOUNDARY_COLOR,
    FORECAST_COLOR,
    FORECAST_CONSERVATIVE_COLOR,
    FORECAST_OPTIMISTIC_COLOR,
) 

def create_chart_data(
    months: int = 180,
) -> pd.DataFrame:
    """
    Create monthly gold-price data for chart tests.
    """

    dates = pd.date_range(
        start="2010-01-01",
        periods=months,
        freq="MS",
    )

    prices = [
        1000.0 + month * 5.0
        for month in range(months)
    ]

    return pd.DataFrame(
        {
            "Date": dates,
            "Gold Price": prices,
        }
    )

def create_inflation_chart_data(
    months: int = 180,
) -> pd.DataFrame:
    """
    Create monthly gold-price and CPI data for chart tests.
    """

    data = create_chart_data(
        months=months
    )

    data["CPI"] = [
        200.0 + month * 0.25
        for month in range(months)
    ]

    return data

@pytest.fixture(autouse=True)
def close_figures_after_test():
    """
    Prevent figures from accumulating during the test run.
    """

    yield

    plt.close("all")

# ------------------------------------------------------------------
# Historical-price pilot chart tests
# ------------------------------------------------------------------


def test_plot_historical_price_returns_figure_and_axes() -> None:
    data = create_chart_data()

    figure, axes = plot_historical_price(
        data=data,
    )

    assert isinstance(
        figure,
        Figure,
    )

    assert isinstance(
        axes,
        Axes,
    )


def test_plot_historical_price_contains_expected_line() -> None:
    data = create_chart_data()

    _, axes = plot_historical_price(
        data=data,
    )

    assert len(
        axes.lines
    ) == 1

    historical_line = axes.lines[0]

    assert (
        historical_line.get_label()
        == DEFAULT_PRICE_LABEL
    )


def test_plot_historical_price_uses_semantic_color() -> None:
    data = create_chart_data()

    _, axes = plot_historical_price(
        data=data,
    )

    historical_line = axes.lines[0]

    expected_color = (
        matplotlib.colors.to_rgba(
            HISTORICAL_PRICE_COLOR
        )
    )

    actual_color = (
        matplotlib.colors.to_rgba(
            historical_line.get_color()
        )
    )

    assert actual_color == expected_color


def test_plot_historical_price_uses_left_aligned_title() -> None:
    data = create_chart_data()

    _, axes = plot_historical_price(
        data=data,
    )

    assert (
        axes.get_title(
            loc="left"
        )
        == DEFAULT_CHART_TITLE
    )


def test_plot_historical_price_adds_subtitle() -> None:
    data = create_chart_data()

    _, axes = plot_historical_price(
        data=data,
    )

    subtitle_values = [
        text.get_text()
        for text in axes.texts
    ]

    assert (
        DEFAULT_HISTORICAL_PRICE_SUBTITLE
        in subtitle_values
    )


def test_plot_historical_price_allows_subtitle_to_be_omitted() -> None:
    data = create_chart_data()

    _, axes = plot_historical_price(
        data=data,
        subtitle=None,
    )

    subtitle_values = [
        text.get_text()
        for text in axes.texts
    ]

    assert (
        DEFAULT_HISTORICAL_PRICE_SUBTITLE
        not in subtitle_values
    )


def test_plot_historical_price_uses_shared_background() -> None:
    data = create_chart_data()

    _, axes = plot_historical_price(
        data=data,
    )

    expected_background = (
        matplotlib.colors.to_rgba(
            PLOT_BACKGROUND_COLOR
        )
    )

    assert (
        axes.get_facecolor()
        == expected_background
    )


def test_plot_historical_price_creates_legend() -> None:
    data = create_chart_data()

    _, axes = plot_historical_price(
        data=data,
    )

    legend = axes.get_legend()

    assert legend is not None

    legend_labels = [
        text.get_text()
        for text in legend.get_texts()
    ]

    assert (
        DEFAULT_PRICE_LABEL
        in legend_labels
    )


def test_plot_historical_price_formats_y_axis_as_currency() -> None:
    data = create_chart_data()

    _, axes = plot_historical_price(
        data=data,
    )

    assert isinstance(
        axes.yaxis.get_major_formatter(),
        FuncFormatter,
    )


def test_plot_historical_price_preserves_custom_title() -> None:
    data = create_chart_data()

    _, axes = plot_historical_price(
        data=data,
        title="Gold Price History",
    )

    assert (
        axes.get_title(
            loc="left"
        )
        == "Gold Price History"
    )


def test_plot_historical_price_preserves_custom_figure_size() -> None:
    data = create_chart_data()

    figure, _ = plot_historical_price(
        data=data,
        figure_size=(
            10.0,
            5.0,
        ),
    )

    width, height = (
        figure.get_size_inches()
    )

    assert width == pytest.approx(
        10.0
    )

    assert height == pytest.approx(
        5.0
    )

# ------------------------------------------------------------------
# Single forecast chart tests
# ------------------------------------------------------------------


def test_plot_forecast_returns_figure_and_axes() -> None:
    data = create_chart_data()

    figure, axes = plot_forecast(
        data=data,
        annual_growth_rate=0.05,
        forecast_years=5,
        history_years=10,
    )

    assert isinstance(
        figure,
        Figure,
    )

    assert isinstance(
        axes,
        Axes,
    )


def test_plot_forecast_contains_expected_series() -> None:
    data = create_chart_data()

    _, axes = plot_forecast(
        data=data,
        annual_growth_rate=0.05,
        forecast_years=5,
        history_years=10,
    )

    labels = [
        line.get_label()
        for line in axes.lines
    ]

    assert (
        DEFAULT_FORECAST_HISTORY_LABEL
        in labels
    )

    assert any(
        "Projected Price" in label
        for label in labels
    )


def test_plot_forecast_connects_to_last_historical_price() -> None:
    data = create_chart_data()

    _, axes = plot_forecast(
        data=data,
        annual_growth_rate=0.05,
        forecast_years=1,
        history_years=5,
    )

    historical_line = axes.lines[0]
    forecast_line = axes.lines[1]

    historical_y_values = (
        historical_line.get_ydata()
    )

    forecast_y_values = (
        forecast_line.get_ydata()
    )

    assert forecast_y_values[
        0
    ] == pytest.approx(
        historical_y_values[-1]
    )


def test_plot_forecast_uses_requested_forecast_length() -> None:
    data = create_chart_data()

    _, axes = plot_forecast(
        data=data,
        annual_growth_rate=0.05,
        forecast_years=2,
        history_years=5,
    )

    forecast_line = axes.lines[1]

    forecast_x_values = (
        forecast_line.get_xdata()
    )

    # One connecting historical value plus
    # 24 monthly forecast values.
    assert len(
        forecast_x_values
    ) == 25


def test_plot_forecast_formats_y_axis_as_currency() -> None:
    data = create_chart_data()

    _, axes = plot_forecast(
        data=data,
    )

    assert isinstance(
        axes.yaxis.get_major_formatter(),
        FuncFormatter,
    )


@pytest.mark.parametrize(
    "history_years",
    [
        0,
        -1,
    ],
)
def test_plot_forecast_rejects_invalid_history_years(
    history_years: int,
) -> None:
    data = create_chart_data()

    with pytest.raises(
        ValueError,
        match="history_years must be at least 1",
    ):
        plot_forecast(
            data=data,
            history_years=history_years,
        )


@pytest.mark.parametrize(
    "history_years",
    [
        5.5,
        "5",
        True,
    ],
)
def test_plot_forecast_rejects_noninteger_history_years(
    history_years: object,
) -> None:
    data = create_chart_data()

    with pytest.raises(
        TypeError,
        match=(
            "history_years must be "
            "an integer or None"
        ),
    ):
        plot_forecast(
            data=data,
            history_years=history_years,  # type: ignore[arg-type]
        )


def test_plot_forecast_allows_complete_history() -> None:
    data = create_chart_data(
        months=36
    )

    _, axes = plot_forecast(
        data=data,
        history_years=None,
        forecast_years=1,
    )

    historical_line = axes.lines[0]

    assert len(
        historical_line.get_xdata()
    ) == len(data)


# ------------------------------------------------------------------
# Scenario chart tests
# ------------------------------------------------------------------

def test_plot_forecast_scenarios_returns_figure_and_axes() -> None:
    data = create_chart_data()

    figure, axes = (
        plot_forecast_scenarios(
            data=data,
            forecast_years=5,
            history_years=10,
        )
    )

    assert isinstance(
        figure,
        Figure,
    )

    assert isinstance(
        axes,
        Axes,
    )


def test_plot_forecast_scenarios_draws_default_scenarios() -> None:
    data = create_chart_data()

    _, axes = plot_forecast_scenarios(
        data=data,
        forecast_years=1,
    )

    labels = [
        line.get_label()
        for line in axes.lines
    ]

    assert any(
        "Conservative" in label
        for label in labels
    )

    assert any(
        "Expected" in label
        for label in labels
    )

    assert any(
        "Optimistic" in label
        for label in labels
    )


def test_plot_forecast_scenarios_draws_history() -> None:
    data = create_chart_data()

    _, axes = plot_forecast_scenarios(
        data=data,
        forecast_years=1,
    )

    labels = [
        line.get_label()
        for line in axes.lines
    ]

    assert (
        DEFAULT_FORECAST_HISTORY_LABEL
        in labels
    )


def test_plot_forecast_scenarios_accepts_custom_rates() -> None:
    data = create_chart_data()

    growth_rates = {
        "Decline": -0.02,
        "Steady": 0.04,
    }

    _, axes = plot_forecast_scenarios(
        data=data,
        growth_rates=growth_rates,
        forecast_years=1,
    )

    labels = [
        line.get_label()
        for line in axes.lines
    ]

    assert any(
        "Decline" in label
        for label in labels
    )

    assert any(
        "Steady" in label
        for label in labels
    )

    assert not any(
        "Optimistic" in label
        for label in labels
    )


def test_plot_forecast_scenario_lines_start_at_same_price() -> None:
    data = create_chart_data()

    _, axes = plot_forecast_scenarios(
        data=data,
        forecast_years=1,
    )

    historical_line = axes.lines[0]

    last_historical_price = (
        historical_line.get_ydata()[-1]
    )

    scenario_lines = axes.lines[
        1:4
    ]

    for scenario_line in scenario_lines:
        assert (
            scenario_line.get_ydata()[0]
            == pytest.approx(
                last_historical_price
            )
        )


def test_plot_forecast_scenario_end_values_are_ordered() -> None:
    data = create_chart_data()

    _, axes = plot_forecast_scenarios(
        data=data,
        forecast_years=1,
    )

    scenario_lines = axes.lines[
        1:4
    ]

    ending_values = [
        line.get_ydata()[-1]
        for line in scenario_lines
    ]

    assert (
        ending_values[0]
        < ending_values[1]
        < ending_values[2]
    )


def test_plot_forecast_scenarios_rejects_empty_growth_rates() -> None:
    data = create_chart_data()

    with pytest.raises(
        ValueError,
        match="growth_rates cannot be empty",
    ):
        plot_forecast_scenarios(
            data=data,
            growth_rates={},
        )


# ------------------------------------------------------------------
# Recent-price chart tests
# ------------------------------------------------------------------


def test_plot_recent_price_uses_shared_style() -> None:
    data = create_chart_data()

    _, axes = plot_recent_price(
        data=data,
        years=5,
    )

    assert (
        axes.get_title(
            loc="left"
        )
        == "Recent Gold Prices: Last 5 Years"
    )

    assert any(
        text.get_text()
        == DEFAULT_RECENT_PRICE_SUBTITLE.format(
            years=5
        )
        for text in axes.texts
    )


def test_plot_recent_price_uses_semantic_color() -> None:
    data = create_chart_data()

    _, axes = plot_recent_price(
        data=data,
        years=5,
    )

    actual_color = matplotlib.colors.to_rgba(
        axes.lines[0].get_color()
    )

    expected_color = matplotlib.colors.to_rgba(
        RECENT_PRICE_COLOR
    )

    assert actual_color == expected_color


def test_plot_recent_price_creates_legend() -> None:
    data = create_chart_data()

    _, axes = plot_recent_price(
        data=data,
        years=5,
    )

    legend = axes.get_legend()

    assert legend is not None

    assert [
        text.get_text()
        for text in legend.get_texts()
    ] == [
        "Most Recent 5 Years"
    ]


def test_plot_recent_price_accepts_custom_subtitle() -> None:
    data = create_chart_data()

    _, axes = plot_recent_price(
        data=data,
        years=5,
        subtitle="Custom recent-period subtitle",
    )

    assert any(
        text.get_text()
        == "Custom recent-period subtitle"
        for text in axes.texts
    )


@pytest.mark.parametrize(
    "years",
    [
        True,
        False,
        5.5,
        "5",
    ],
)
def test_plot_recent_price_rejects_noninteger_years(
    years: object,
) -> None:
    data = create_chart_data()

    with pytest.raises(
        TypeError,
        match="years must be an integer",
    ):
        plot_recent_price(
            data=data,
            years=years,  # type: ignore[arg-type]
        )

# ------------------------------------------------------------------
# Inflation-adjusted chart tests
# ------------------------------------------------------------------


def test_plot_inflation_adjusted_price_uses_real_price_color() -> None:
    data = create_inflation_chart_data()

    _, axes = plot_inflation_adjusted_price(
        data=data,
    )

    actual_color = matplotlib.colors.to_rgba(
        axes.lines[0].get_color()
    )

    expected_color = matplotlib.colors.to_rgba(
        REAL_PRICE_COLOR
    )

    assert actual_color == expected_color


def test_plot_inflation_adjusted_price_uses_shared_title() -> None:
    data = create_inflation_chart_data()

    _, axes = plot_inflation_adjusted_price(
        data=data,
    )

    assert (
        axes.get_title(
            loc="left"
        )
        == "Inflation-Adjusted Gold Prices"
    )


def test_plot_inflation_adjusted_price_adds_subtitle() -> None:
    data = create_inflation_chart_data()

    _, axes = plot_inflation_adjusted_price(
        data=data,
    )

    assert any(
        text.get_text()
        == DEFAULT_INFLATION_ADJUSTED_SUBTITLE
        for text in axes.texts
    )


def test_plot_inflation_adjusted_price_creates_legend() -> None:
    data = create_inflation_chart_data()

    _, axes = plot_inflation_adjusted_price(
        data=data,
    )

    legend = axes.get_legend()

    assert legend is not None

    assert [
        text.get_text()
        for text in legend.get_texts()
    ] == [
        DEFAULT_REAL_LABEL
    ]

def create_mixed_return_chart_data() -> pd.DataFrame:
    """
    Create monthly prices containing positive and negative years.
    """

    dates = pd.date_range(
        start="2020-01-01",
        periods=48,
        freq="MS",
    )

    prices = (
        list(range(100, 112))
        + list(range(112, 100, -1))
        + list(range(101, 113))
        + list(range(113, 101, -1))
    )

    return pd.DataFrame(
        {
            "Date": dates,
            "Gold Price": prices,
        }
    )

# ------------------------------------------------------------------
# Nominal-versus-real chart tests
# ------------------------------------------------------------------


def test_plot_nominal_vs_real_price_draws_two_series() -> None:
    data = create_inflation_chart_data()

    _, axes = plot_nominal_vs_real_price(
        data=data,
    )

    assert len(
        axes.lines
    ) == 2

    labels = [
        line.get_label()
        for line in axes.lines
    ]

    assert labels == [
        DEFAULT_NOMINAL_LABEL,
        DEFAULT_REAL_LABEL,
    ]


def test_plot_nominal_vs_real_price_uses_semantic_colors() -> None:
    data = create_inflation_chart_data()

    _, axes = plot_nominal_vs_real_price(
        data=data,
    )

    actual_colors = [
        matplotlib.colors.to_rgba(
            line.get_color()
        )
        for line in axes.lines
    ]

    expected_colors = [
        matplotlib.colors.to_rgba(
            NOMINAL_PRICE_COLOR
        ),
        matplotlib.colors.to_rgba(
            REAL_PRICE_COLOR
        ),
    ]

    assert actual_colors == expected_colors


def test_plot_nominal_vs_real_price_adds_subtitle() -> None:
    data = create_inflation_chart_data()

    _, axes = plot_nominal_vs_real_price(
        data=data,
    )

    assert any(
        text.get_text()
        == DEFAULT_NOMINAL_VS_REAL_SUBTITLE
        for text in axes.texts
    )


def test_plot_nominal_vs_real_price_creates_legend() -> None:
    data = create_inflation_chart_data()

    _, axes = plot_nominal_vs_real_price(
        data=data,
    )

    legend = axes.get_legend()

    assert legend is not None

    labels = [
        text.get_text()
        for text in legend.get_texts()
    ]

    assert labels == [
        DEFAULT_NOMINAL_LABEL,
        DEFAULT_REAL_LABEL,
    ]


def test_price_and_inflation_charts_use_currency_formatters() -> None:
    price_data = create_chart_data()
    inflation_data = create_inflation_chart_data()

    charts = [
        plot_recent_price(
            data=price_data,
        ),
        plot_inflation_adjusted_price(
            data=inflation_data,
        ),
        plot_nominal_vs_real_price(
            data=inflation_data,
        ),
    ]

    for _, axes in charts:
        assert isinstance(
            axes.yaxis.get_major_formatter(),
            FuncFormatter,
        )

# ------------------------------------------------------------------
# Return-chart styling tests
# ------------------------------------------------------------------


def test_plot_monthly_returns_uses_positive_and_negative_colors() -> None:
    data = create_chart_data()

    _, axes = plot_monthly_returns(
        data=data,
    )

    line_colors = [
        matplotlib.colors.to_rgba(
            line.get_color()
        )
        for line in axes.lines[:2]
    ]

    expected_colors = [
        matplotlib.colors.to_rgba(
            POSITIVE_RETURN_COLOR
        ),
        matplotlib.colors.to_rgba(
            NEGATIVE_RETURN_COLOR
        ),
    ]

    assert line_colors == expected_colors


def test_plot_monthly_returns_adds_subtitle() -> None:
    data = create_chart_data()

    _, axes = plot_monthly_returns(
        data=data,
    )

    assert any(
        text.get_text()
        == DEFAULT_MONTHLY_RETURNS_SUBTITLE
        for text in axes.texts
    )


def test_plot_monthly_returns_has_zero_reference_line() -> None:
    data = create_chart_data()

    _, axes = plot_monthly_returns(
        data=data,
    )

    labels = [
        line.get_label()
        for line in axes.lines
    ]

    assert "Zero Return" in labels


def test_plot_cumulative_returns_uses_semantic_color() -> None:
    data = create_chart_data()

    _, axes = plot_cumulative_returns(
        data=data,
    )

    actual_color = matplotlib.colors.to_rgba(
        axes.lines[0].get_color()
    )

    expected_color = matplotlib.colors.to_rgba(
        CUMULATIVE_RETURN_COLOR
    )

    assert actual_color == expected_color


def test_plot_cumulative_returns_adds_subtitle() -> None:
    data = create_chart_data()

    _, axes = plot_cumulative_returns(
        data=data,
    )

    assert any(
        text.get_text()
        == DEFAULT_CUMULATIVE_RETURNS_SUBTITLE
        for text in axes.texts
    )


def test_plot_annual_returns_uses_positive_and_negative_bar_colors() -> None:
    data = create_chart_data()

    _, axes = plot_annual_returns(
        data=data,
    )

    bar_colors = {
        patch.get_facecolor()
        for patch in axes.patches
    }

    expected_positive = matplotlib.colors.to_rgba(
        POSITIVE_RETURN_COLOR
    )

    expected_negative = matplotlib.colors.to_rgba(
        NEGATIVE_RETURN_COLOR
    )

    assert expected_positive in bar_colors
    assert expected_negative in bar_colors


def test_plot_annual_returns_adds_subtitle() -> None:
    data = create_chart_data()

    _, axes = plot_annual_returns(
        data=data,
    )

    assert any(
        text.get_text()
        == DEFAULT_ANNUAL_RETURNS_SUBTITLE
        for text in axes.texts
    )


def test_plot_return_distribution_uses_neutral_color() -> None:
    data = create_chart_data()

    _, axes = plot_return_distribution(
        data=data,
    )

    assert axes.patches

    expected_color = matplotlib.colors.to_rgba(
        NEUTRAL_RETURN_COLOR,
        alpha=DEFAULT_PERCENTAGE_FILL_ALPHA,
    )

    actual_color = axes.patches[0].get_facecolor()

    assert actual_color == expected_color


def test_plot_return_distribution_adds_subtitle() -> None:
    data = create_chart_data()

    _, axes = plot_return_distribution(
        data=data,
    )

    assert any(
        text.get_text()
        == DEFAULT_RETURN_DISTRIBUTION_SUBTITLE
        for text in axes.texts
    )


def test_return_charts_use_percentage_formatters() -> None:
    data = create_chart_data()

    chart_results = [
        plot_monthly_returns(
            data=data,
        ),
        plot_cumulative_returns(
            data=data,
        ),
        plot_annual_returns(
            data=data,
        ),
    ]

    for _, axes in chart_results:
        assert isinstance(
            axes.yaxis.get_major_formatter(),
            FuncFormatter,
        )

    _, distribution_axes = (
        plot_return_distribution(
            data=data,
        )
    )

    assert isinstance(
        distribution_axes.xaxis.get_major_formatter(),
        FuncFormatter,
    )

def test_plot_annual_returns_uses_positive_and_negative_bar_colors() -> None:
    data = create_mixed_return_chart_data()

    _, axes = plot_annual_returns(
        data=data,
    )

    bar_colors = {
        patch.get_facecolor()
        for patch in axes.patches
    }

    expected_positive = matplotlib.colors.to_rgba(
        POSITIVE_RETURN_COLOR
    )

    expected_negative = matplotlib.colors.to_rgba(
        NEGATIVE_RETURN_COLOR
    )

    assert expected_positive in bar_colors
    assert expected_negative in bar_colors

def create_mixed_return_chart_data() -> pd.DataFrame:
    """
    Create monthly prices containing positive and negative years.
    """

    dates = pd.date_range(
        start="2020-01-01",
        periods=48,
        freq="MS",
    )

    prices = (
        list(range(100, 112))
        + list(range(112, 100, -1))
        + list(range(101, 113))
        + list(range(113, 101, -1))
    )

    return pd.DataFrame(
        {
            "Date": dates,
            "Gold Price": prices,
        }
    )

# ------------------------------------------------------------------
# Rolling-analysis chart styling tests
# ------------------------------------------------------------------


def test_plot_rolling_average_uses_semantic_colors() -> None:
    data = create_chart_data()

    _, axes = plot_rolling_average(
        data=data,
        window=12,
    )

    actual_colors = [
        matplotlib.colors.to_rgba(
            line.get_color()
        )
        for line in axes.lines[:2]
    ]

    expected_colors = [
        matplotlib.colors.to_rgba(
            HISTORICAL_PRICE_COLOR
        ),
        matplotlib.colors.to_rgba(
            ROLLING_AVERAGE_COLOR
        ),
    ]

    assert actual_colors == expected_colors


def test_plot_rolling_average_adds_subtitle() -> None:
    data = create_chart_data()

    _, axes = plot_rolling_average(
        data=data,
        window=12,
    )

    expected_subtitle = (
        DEFAULT_ROLLING_AVERAGE_SUBTITLE.format(
            window=12
        )
    )

    assert any(
        text.get_text() == expected_subtitle
        for text in axes.texts
    )


def test_plot_rolling_volatility_uses_semantic_color() -> None:
    data = create_chart_data()

    _, axes = plot_rolling_volatility(
        data=data,
        window=12,
    )

    actual_color = matplotlib.colors.to_rgba(
        axes.lines[0].get_color()
    )

    expected_color = matplotlib.colors.to_rgba(
        VOLATILITY_COLOR
    )

    assert actual_color == expected_color


def test_plot_rolling_volatility_updates_subtitle_when_not_annualized() -> None:
    data = create_chart_data()

    _, axes = plot_rolling_volatility(
        data=data,
        window=12,
        annualize=False,
    )

    expected_subtitle = (
        DEFAULT_ROLLING_VOLATILITY_SUBTITLE.format(
            window=12,
            volatility_type="monthly",
        )
    )

    assert any(
        text.get_text() == expected_subtitle
        for text in axes.texts
    )


def test_plot_rolling_return_uses_semantic_color() -> None:
    data = create_chart_data()

    _, axes = plot_rolling_return(
        data=data,
        window=12,
    )

    actual_color = matplotlib.colors.to_rgba(
        axes.lines[0].get_color()
    )

    expected_color = matplotlib.colors.to_rgba(
        ROLLING_RETURN_COLOR
    )

    assert actual_color == expected_color


def test_plot_rolling_return_includes_zero_line() -> None:
    data = create_chart_data()

    _, axes = plot_rolling_return(
        data=data,
        window=12,
    )

    labels = [
        line.get_label()
        for line in axes.lines
    ]

    assert "Zero Return" in labels


def test_plot_rolling_drawdown_uses_drawdown_color() -> None:
    data = create_chart_data()

    _, axes = plot_rolling_drawdown(
        data=data,
        window=12,
    )

    actual_color = matplotlib.colors.to_rgba(
        axes.lines[0].get_color()
    )

    expected_color = matplotlib.colors.to_rgba(
        DRAWDOWN_COLOR
    )

    assert actual_color == expected_color


def test_plot_rolling_drawdown_contains_filled_area() -> None:
    data = create_chart_data()

    _, axes = plot_rolling_drawdown(
        data=data,
        window=12,
    )

    assert axes.collections


def test_plot_rolling_high_low_uses_semantic_colors() -> None:
    data = create_chart_data()

    _, axes = plot_rolling_high_low(
        data=data,
        window=12,
    )

    actual_colors = [
        matplotlib.colors.to_rgba(
            line.get_color()
        )
        for line in axes.lines[:3]
    ]

    expected_colors = [
        matplotlib.colors.to_rgba(
            HISTORICAL_PRICE_COLOR
        ),
        matplotlib.colors.to_rgba(
            ROLLING_HIGH_COLOR
        ),
        matplotlib.colors.to_rgba(
            ROLLING_LOW_COLOR
        ),
    ]

    assert actual_colors == expected_colors


def test_rolling_charts_use_left_aligned_titles() -> None:
    data = create_chart_data()

    chart_results = [
        plot_rolling_average(
            data=data,
        ),
        plot_rolling_volatility(
            data=data,
        ),
        plot_rolling_return(
            data=data,
        ),
        plot_rolling_drawdown(
            data=data,
        ),
        plot_rolling_high_low(
            data=data,
        ),
    ]

    for _, axes in chart_results:
        assert axes.get_title(
            loc="left"
        )


def test_rolling_percentage_charts_use_percentage_formatters() -> None:
    data = create_chart_data()

    chart_results = [
        plot_rolling_volatility(
            data=data,
        ),
        plot_rolling_return(
            data=data,
        ),
        plot_rolling_drawdown(
            data=data,
        ),
    ]

    for _, axes in chart_results:
        assert isinstance(
            axes.yaxis.get_major_formatter(),
            FuncFormatter,
        )


def test_rolling_price_charts_use_currency_formatters() -> None:
    data = create_chart_data()

    chart_results = [
        plot_rolling_average(
            data=data,
        ),
        plot_rolling_high_low(
            data=data,
        ),
    ]

    for _, axes in chart_results:
        assert isinstance(
            axes.yaxis.get_major_formatter(),
            FuncFormatter,
        )

# ------------------------------------------------------------------
# Forecast-chart styling tests
# ------------------------------------------------------------------


def test_plot_forecast_uses_semantic_colors() -> None:
    data = create_chart_data()

    _, axes = plot_forecast(
        data=data,
        forecast_years=2,
    )

    actual_colors = [
        matplotlib.colors.to_rgba(
            line.get_color()
        )
        for line in axes.lines[:3]
    ]

    expected_colors = [
        matplotlib.colors.to_rgba(
            HISTORICAL_PRICE_COLOR
        ),
        matplotlib.colors.to_rgba(
            FORECAST_COLOR
        ),
        matplotlib.colors.to_rgba(
            FORECAST_BOUNDARY_COLOR
        ),
    ]

    assert actual_colors == expected_colors


def test_plot_forecast_adds_generated_subtitle() -> None:
    data = create_chart_data()

    _, axes = plot_forecast(
        data=data,
        annual_growth_rate=0.05,
        forecast_years=3,
    )

    expected_subtitle = (
        DEFAULT_FORECAST_SUBTITLE.format(
            forecast_years=3,
            growth_rate=0.05,
        )
    )

    assert any(
        text.get_text()
        == expected_subtitle
        for text in axes.texts
    )


def test_plot_forecast_adds_disclaimer() -> None:
    data = create_chart_data()

    figure, _ = plot_forecast(
        data=data,
    )

    assert any(
        text.get_text()
        == DEFAULT_FORECAST_DISCLAIMER
        for text in figure.texts
    )


def test_plot_forecast_can_hide_disclaimer() -> None:
    data = create_chart_data()

    figure, _ = plot_forecast(
        data=data,
        show_disclaimer=False,
    )

    assert not any(
        text.get_text()
        == DEFAULT_FORECAST_DISCLAIMER
        for text in figure.texts
    )


def test_plot_forecast_rejects_nonboolean_disclaimer() -> None:
    data = create_chart_data()

    with pytest.raises(
        TypeError,
        match=(
            "show_disclaimer must be "
            "a Boolean value"
        ),
    ):
        plot_forecast(
            data=data,
            show_disclaimer="yes",  # type: ignore[arg-type]
        )


def test_plot_forecast_scenarios_uses_default_semantic_colors() -> None:
    data = create_chart_data()

    _, axes = plot_forecast_scenarios(
        data=data,
        forecast_years=1,
    )

    actual_colors = [
        matplotlib.colors.to_rgba(
            line.get_color()
        )
        for line in axes.lines[:5]
    ]

    expected_colors = [
        matplotlib.colors.to_rgba(
            HISTORICAL_PRICE_COLOR
        ),
        matplotlib.colors.to_rgba(
            FORECAST_CONSERVATIVE_COLOR
        ),
        matplotlib.colors.to_rgba(
            FORECAST_BASE_COLOR
        ),
        matplotlib.colors.to_rgba(
            FORECAST_OPTIMISTIC_COLOR
        ),
        matplotlib.colors.to_rgba(
            FORECAST_BOUNDARY_COLOR
        ),
    ]

    assert actual_colors == expected_colors


def test_plot_forecast_scenarios_adds_generated_subtitle() -> None:
    data = create_chart_data()

    _, axes = plot_forecast_scenarios(
        data=data,
        forecast_years=4,
    )

    expected_subtitle = (
        DEFAULT_FORECAST_SCENARIO_SUBTITLE.format(
            forecast_years=4
        )
    )

    assert any(
        text.get_text()
        == expected_subtitle
        for text in axes.texts
    )


def test_plot_forecast_scenarios_adds_disclaimer() -> None:
    data = create_chart_data()

    figure, _ = plot_forecast_scenarios(
        data=data,
    )

    assert any(
        text.get_text()
        == DEFAULT_FORECAST_DISCLAIMER
        for text in figure.texts
    )


def test_plot_forecast_scenarios_can_hide_disclaimer() -> None:
    data = create_chart_data()

    figure, _ = plot_forecast_scenarios(
        data=data,
        show_disclaimer=False,
    )

    assert not any(
        text.get_text()
        == DEFAULT_FORECAST_DISCLAIMER
        for text in figure.texts
    )


def test_plot_forecast_scenarios_rejects_nonboolean_disclaimer() -> None:
    data = create_chart_data()

    with pytest.raises(
        TypeError,
        match=(
            "show_disclaimer must be "
            "a Boolean value"
        ),
    ):
        plot_forecast_scenarios(
            data=data,
            show_disclaimer="yes",  # type: ignore[arg-type]
        )


def test_plot_forecast_scenarios_supports_custom_colors() -> None:
    data = create_chart_data()

    custom_rates = {
        "Decline": -0.02,
        "Stable": 0.02,
        "Growth": 0.06,
        "Strong Growth": 0.10,
    }

    _, axes = plot_forecast_scenarios(
        data=data,
        growth_rates=custom_rates,
        forecast_years=1,
    )

    scenario_lines = axes.lines[
        1:1 + len(custom_rates)
    ]

    scenario_colors = [
        matplotlib.colors.to_rgba(
            line.get_color()
        )
        for line in scenario_lines
    ]

    assert len(
        set(scenario_colors)
    ) == len(custom_rates)


def test_forecast_charts_use_left_aligned_titles() -> None:
    data = create_chart_data()

    chart_results = [
        plot_forecast(
            data=data,
        ),
        plot_forecast_scenarios(
            data=data,
        ),
    ]

    for _, axes in chart_results:
        assert axes.get_title(
            loc="left"
        )


def test_forecast_charts_use_currency_formatters() -> None:
    data = create_chart_data()

    chart_results = [
        plot_forecast(
            data=data,
        ),
        plot_forecast_scenarios(
            data=data,
        ),
    ]

    for _, axes in chart_results:
        assert isinstance(
            axes.yaxis.get_major_formatter(),
            FuncFormatter,
        )

# ------------------------------------------------------------------
# Premium chart-annotation tests
# ------------------------------------------------------------------


def test_historical_chart_adds_latest_value_annotation() -> None:
    data = create_chart_data()

    _, axes = plot_historical_price(
        data=data,
    )

    annotation_texts = [
        text.get_text()
        for text in axes.texts
    ]

    assert any(
        text.startswith("Latest:")
        for text in annotation_texts
    )


def test_historical_chart_can_hide_latest_annotation() -> None:
    data = create_chart_data()

    _, axes = plot_historical_price(
        data=data,
        show_latest_value=False,
    )

    annotation_texts = [
        text.get_text()
        for text in axes.texts
    ]

    assert not any(
        text.startswith("Latest:")
        for text in annotation_texts
    )


def test_historical_chart_rejects_invalid_annotation_option() -> None:
    data = create_chart_data()

    with pytest.raises(
        TypeError,
        match=(
            "show_latest_value must be "
            "a Boolean value"
        ),
    ):
        plot_historical_price(
            data=data,
            show_latest_value="yes",  # type: ignore[arg-type]
        )


def test_recent_chart_adds_latest_annotation() -> None:
    data = create_mixed_return_chart_data()

    _, axes = plot_recent_price(
        data=data,
        years=3,
    )

    annotation_texts = [
        text.get_text()
        for text in axes.texts
    ]

    assert any(
        text.startswith("Latest:")
        for text in annotation_texts
    )


def test_recent_chart_adds_high_and_low_annotations() -> None:
    data = create_mixed_return_chart_data()

    _, axes = plot_recent_price(
        data=data,
        years=3,
        show_latest_value=False,
        show_high_low=True,
    )

    annotation_texts = [
        text.get_text()
        for text in axes.texts
    ]

    assert any(
        text.startswith("3-Year High:")
        for text in annotation_texts
    )

    assert any(
        text.startswith("3-Year Low:")
        for text in annotation_texts
    )


def test_recent_chart_can_hide_high_and_low_annotations() -> None:
    data = create_mixed_return_chart_data()

    _, axes = plot_recent_price(
        data=data,
        years=3,
        show_high_low=False,
    )

    annotation_texts = [
        text.get_text()
        for text in axes.texts
    ]

    assert not any(
        "Year High:" in text
        or "Year Low:" in text
        for text in annotation_texts
    )


@pytest.mark.parametrize(
    "parameter_name",
    [
        "show_latest_value",
        "show_high_low",
    ],
)
def test_recent_chart_rejects_invalid_annotation_options(
    parameter_name: str,
) -> None:
    data = create_chart_data()

    arguments = {
        parameter_name: "yes",
    }

    with pytest.raises(
        TypeError,
        match=(
            f"{parameter_name} must be "
            "a Boolean value"
        ),
    ):
        plot_recent_price(
            data=data,
            **arguments,  # type: ignore[arg-type]
        )


def test_drawdown_chart_adds_deepest_drawdown_annotation() -> None:
    data = create_mixed_return_chart_data()

    _, axes = plot_rolling_drawdown(
        data=data,
        window=6,
    )

    annotation_texts = [
        text.get_text()
        for text in axes.texts
    ]

    assert any(
        text.startswith(
            "Deepest Drawdown:"
        )
        for text in annotation_texts
    )


def test_drawdown_chart_can_hide_annotation() -> None:
    data = create_mixed_return_chart_data()

    _, axes = plot_rolling_drawdown(
        data=data,
        window=6,
        show_deepest_drawdown=False,
    )

    annotation_texts = [
        text.get_text()
        for text in axes.texts
    ]

    assert not any(
        text.startswith(
            "Deepest Drawdown:"
        )
        for text in annotation_texts
    )


def test_forecast_chart_adds_projected_value_annotation() -> None:
    data = create_chart_data()

    _, axes = plot_forecast(
        data=data,
        forecast_years=3,
    )

    annotation_texts = [
        text.get_text()
        for text in axes.texts
    ]

    assert any(
        text.startswith(
            "Projected 3-Year Value:"
        )
        for text in annotation_texts
    )


def test_forecast_chart_can_hide_projected_value_annotation() -> None:
    data = create_chart_data()

    _, axes = plot_forecast(
        data=data,
        show_projected_value=False,
    )

    annotation_texts = [
        text.get_text()
        for text in axes.texts
    ]

    assert not any(
        "Projected"
        in text
        and "Value:"
        in text
        for text in annotation_texts
    )


def test_forecast_chart_rejects_invalid_projected_value_option() -> None:
    data = create_chart_data()

    with pytest.raises(
        TypeError,
        match=(
            "show_projected_value must be "
            "a Boolean value"
        ),
    ):
        plot_forecast(
            data=data,
            show_projected_value="yes",  # type: ignore[arg-type]
        )

# ------------------------------------------------------------------
# Forecast projected-value annotation tests
# ------------------------------------------------------------------


def test_forecast_chart_adds_projected_value_annotation() -> None:
    data = create_chart_data()

    _, axes = plot_forecast(
        data=data,
        forecast_years=3,
    )

    annotation_texts = [
        text.get_text()
        for text in axes.texts
    ]

    assert any(
        text.startswith(
            "Projected 3-Year Value:"
        )
        for text in annotation_texts
    )


def test_forecast_chart_can_hide_projected_value_annotation() -> None:
    data = create_chart_data()

    _, axes = plot_forecast(
        data=data,
        show_projected_value=False,
    )

    annotation_texts = [
        text.get_text()
        for text in axes.texts
    ]

    assert not any(
        "Projected" in text
        and "Value:" in text
        for text in annotation_texts
    )


def test_forecast_chart_rejects_invalid_projected_value_option() -> None:
    data = create_chart_data()

    with pytest.raises(
        TypeError,
        match=(
            "show_projected_value must be "
            "a Boolean value"
        ),
    ):
        plot_forecast(
            data=data,
            show_projected_value="yes",  # type: ignore[arg-type]
        )


# ------------------------------------------------------------------
# Forecast boundary annotation tests
# ------------------------------------------------------------------


def test_plot_forecast_labels_forecast_boundary() -> None:
    data = create_chart_data()

    _, axes = plot_forecast(
        data=data,
        forecast_years=2,
    )

    text_values = [
        text.get_text()
        for text in axes.texts
    ]

    assert (
        DEFAULT_FORECAST_BOUNDARY_LABEL
        in text_values
    )


def test_plot_forecast_scenarios_labels_forecast_boundary() -> None:
    data = create_chart_data()

    _, axes = plot_forecast_scenarios(
        data=data,
        forecast_years=2,
    )

    text_values = [
        text.get_text()
        for text in axes.texts
    ]

    assert (
        DEFAULT_FORECAST_BOUNDARY_LABEL
        in text_values
    )


def test_plot_forecast_boundary_is_not_in_legend() -> None:
    data = create_chart_data()

    _, axes = plot_forecast(
        data=data,
    )

    legend = axes.get_legend()

    assert legend is not None

    legend_labels = [
        text.get_text()
        for text in legend.get_texts()
    ]

    assert (
        DEFAULT_FORECAST_BOUNDARY_LABEL
        not in legend_labels
    )


def test_plot_forecast_scenario_boundary_is_not_in_legend() -> None:
    data = create_chart_data()

    _, axes = plot_forecast_scenarios(
        data=data,
    )

    legend = axes.get_legend()

    assert legend is not None

    legend_labels = [
        text.get_text()
        for text in legend.get_texts()
    ]

    assert (
        DEFAULT_FORECAST_BOUNDARY_LABEL
        not in legend_labels
    )

# ------------------------------------------------------------------
# Source-note and watermark integration tests
# ------------------------------------------------------------------


def test_historical_chart_adds_source_note() -> None:
    data = create_chart_data()

    figure, _ = plot_historical_price(
        data=data,
        source_note="Source: Example dataset",
    )

    assert any(
        text.get_text()
        == "Source: Example dataset"
        for text in figure.texts
    )


def test_historical_chart_adds_watermark() -> None:
    data = create_chart_data()

    _, axes = plot_historical_price(
        data=data,
        watermark="Gold Value Predictor",
    )

    assert any(
        text.get_text()
        == "Gold Value Predictor"
        for text in axes.texts
    )


def test_historical_chart_omits_branding_by_default() -> None:
    data = create_chart_data()

    figure, axes = plot_historical_price(
        data=data,
    )

    assert not any(
        text.get_text().startswith("Source:")
        for text in figure.texts
    )

    assert not any(
        text.get_text()
        == "Gold Value Predictor"
        for text in axes.texts
    )


def test_recent_chart_adds_source_note_and_watermark() -> None:
    data = create_chart_data()

    figure, axes = plot_recent_price(
        data=data,
        source_note="Source: Example dataset",
        watermark="Gold Value Predictor",
    )

    assert any(
        text.get_text()
        == "Source: Example dataset"
        for text in figure.texts
    )

    assert any(
        text.get_text()
        == "Gold Value Predictor"
        for text in axes.texts
    )


def test_forecast_chart_adds_source_note_and_watermark() -> None:
    data = create_chart_data()

    figure, axes = plot_forecast(
        data=data,
        source_note="Source: Example dataset",
        watermark="Gold Value Predictor",
    )

    figure_texts = [
        text.get_text()
        for text in figure.texts
    ]

    assert (
        "Source: Example dataset"
        in figure_texts
    )

    assert (
        DEFAULT_FORECAST_DISCLAIMER
        in figure_texts
    )

    assert any(
        text.get_text()
        == "Gold Value Predictor"
        for text in axes.texts
    )


def test_forecast_scenario_chart_adds_source_note_and_watermark() -> None:
    data = create_chart_data()

    figure, axes = plot_forecast_scenarios(
        data=data,
        source_note="Source: Example dataset",
        watermark="Gold Value Predictor",
    )

    figure_texts = [
        text.get_text()
        for text in figure.texts
    ]

    assert (
        "Source: Example dataset"
        in figure_texts
    )

    assert (
        DEFAULT_FORECAST_DISCLAIMER
        in figure_texts
    )

    assert any(
        text.get_text()
        == "Gold Value Predictor"
        for text in axes.texts
    )


def test_forecast_source_note_works_without_disclaimer() -> None:
    data = create_chart_data()

    figure, _ = plot_forecast(
        data=data,
        show_disclaimer=False,
        source_note="Source: Example dataset",
    )

    figure_texts = [
        text.get_text()
        for text in figure.texts
    ]

    assert (
        "Source: Example dataset"
        in figure_texts
    )

    assert (
        DEFAULT_FORECAST_DISCLAIMER
        not in figure_texts
    )


@pytest.mark.parametrize(
    "chart_function",
    [
        plot_historical_price,
        plot_recent_price,
        plot_forecast,
        plot_forecast_scenarios,
    ],
)
def test_presentation_charts_reject_blank_source_notes(
    chart_function,
) -> None:
    data = create_chart_data()

    with pytest.raises(
        ValueError,
        match="source_note cannot be empty",
    ):
        chart_function(
            data=data,
            source_note="   ",
        )


@pytest.mark.parametrize(
    "chart_function",
    [
        plot_historical_price,
        plot_recent_price,
        plot_forecast,
        plot_forecast_scenarios,
    ],
)
def test_presentation_charts_reject_blank_watermarks(
    chart_function,
) -> None:
    data = create_chart_data()

    with pytest.raises(
        ValueError,
        match="watermark cannot be empty",
    ):
        chart_function(
            data=data,
            watermark="   ",
        )