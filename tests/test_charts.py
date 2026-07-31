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
)

from visualizations.charts import (
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
    NOMINAL_PRICE_COLOR,
    REAL_PRICE_COLOR,
    RECENT_PRICE_COLOR,
)

from visualizations.styling import (
    HISTORICAL_PRICE_COLOR,
    PLOT_BACKGROUND_COLOR,
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

    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)


def test_plot_forecast_contains_expected_lines() -> None:
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

    assert DEFAULT_FORECAST_HISTORY_LABEL in labels

    assert any(
        "Projected Price" in label
        for label in labels
    )

    assert (
        DEFAULT_FORECAST_BOUNDARY_LABEL
        in labels
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

    assert forecast_y_values[0] == pytest.approx(
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
    assert len(forecast_x_values) == 25


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

    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)


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


def test_plot_forecast_scenarios_draws_history_and_boundary() -> None:
    data = create_chart_data()

    _, axes = plot_forecast_scenarios(
        data=data,
        forecast_years=1,
    )

    labels = [
        line.get_label()
        for line in axes.lines
    ]

    assert DEFAULT_FORECAST_HISTORY_LABEL in labels

    assert (
        DEFAULT_FORECAST_BOUNDARY_LABEL
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

    scenario_lines = axes.lines[1:4]

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

    scenario_lines = axes.lines[1:4]

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