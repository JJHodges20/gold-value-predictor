"""
Visual review harness for Gold Value Predictor charts.

This script generates one representative chart from every major
visualization category. It can display the figures interactively,
export them for inspection, or do both.

Run from the project root:

    python visual_review.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from analytics.loader import load_master_data

from visualizations.charts import (
    plot_annual_returns,
    plot_cumulative_returns,
    plot_forecast,
    plot_forecast_scenarios,
    plot_historical_price,
    plot_inflation_adjusted_price,
    plot_monthly_returns,
    plot_nominal_vs_real_price,
    plot_recent_price,
    plot_return_distribution,
    plot_rolling_average,
    plot_rolling_drawdown,
    plot_rolling_high_low,
    plot_rolling_return,
    plot_rolling_volatility,
)

from visualizations.export import (
    export_figures,
)


# ------------------------------------------------------------------
# Review configuration
# ------------------------------------------------------------------


REVIEW_OUTPUT_DIRECTORY = (
    Path("reports")
    / "visual_review"
)

REVIEW_SOURCE_NOTE = (
    "Source: Gold Value Predictor master dataset"
)

REVIEW_WATERMARK = (
    "Gold Value Predictor"
)

REVIEW_ROLLING_WINDOW = 12

REVIEW_RECENT_YEARS = 5

REVIEW_FORECAST_YEARS = 5

REVIEW_FORECAST_HISTORY_YEARS = 10


# ------------------------------------------------------------------
# Figure generation
# ------------------------------------------------------------------


def generate_review_figures() -> dict[str, Figure]:
    """
    Generate one representative figure from every chart category.

    Returns:
        Dictionary mapping descriptive chart names to Matplotlib
        Figure objects.
    """

    data = load_master_data()

    figures: dict[str, Figure] = {}

    historical_figure, _ = (
        plot_historical_price(
            data=data,
            source_note=REVIEW_SOURCE_NOTE,
            watermark=REVIEW_WATERMARK,
        )
    )

    figures[
        "historical_price"
    ] = historical_figure

    recent_figure, _ = plot_recent_price(
        data=data,
        years=REVIEW_RECENT_YEARS,
        source_note=REVIEW_SOURCE_NOTE,
        watermark=REVIEW_WATERMARK,
    )

    figures[
        "recent_price"
    ] = recent_figure

    inflation_adjusted_figure, _ = (
        plot_inflation_adjusted_price(
            data=data,
        )
    )

    figures[
        "inflation_adjusted_price"
    ] = inflation_adjusted_figure

    nominal_real_figure, _ = (
        plot_nominal_vs_real_price(
            data=data,
        )
    )

    figures[
        "nominal_vs_real_price"
    ] = nominal_real_figure

    monthly_returns_figure, _ = (
        plot_monthly_returns(
            data=data,
        )
    )

    figures[
        "monthly_returns"
    ] = monthly_returns_figure

    cumulative_returns_figure, _ = (
        plot_cumulative_returns(
            data=data,
        )
    )

    figures[
        "cumulative_returns"
    ] = cumulative_returns_figure

    annual_returns_figure, _ = (
        plot_annual_returns(
            data=data,
        )
    )

    figures[
        "annual_returns"
    ] = annual_returns_figure

    return_distribution_figure, _ = (
        plot_return_distribution(
            data=data,
        )
    )

    figures[
        "return_distribution"
    ] = return_distribution_figure

    rolling_average_figure, _ = (
        plot_rolling_average(
            data=data,
            window=REVIEW_ROLLING_WINDOW,
        )
    )

    figures[
        "rolling_average"
    ] = rolling_average_figure

    rolling_volatility_figure, _ = (
        plot_rolling_volatility(
            data=data,
            window=REVIEW_ROLLING_WINDOW,
        )
    )

    figures[
        "rolling_volatility"
    ] = rolling_volatility_figure

    rolling_return_figure, _ = (
        plot_rolling_return(
            data=data,
            window=REVIEW_ROLLING_WINDOW,
        )
    )

    figures[
        "rolling_return"
    ] = rolling_return_figure

    rolling_drawdown_figure, _ = (
        plot_rolling_drawdown(
            data=data,
            window=REVIEW_ROLLING_WINDOW,
        )
    )

    figures[
        "rolling_drawdown"
    ] = rolling_drawdown_figure

    rolling_high_low_figure, _ = (
        plot_rolling_high_low(
            data=data,
            window=REVIEW_ROLLING_WINDOW,
        )
    )

    figures[
        "rolling_high_low"
    ] = rolling_high_low_figure

    forecast_figure, _ = plot_forecast(
        data=data,
        forecast_years=(
            REVIEW_FORECAST_YEARS
        ),
        history_years=(
            REVIEW_FORECAST_HISTORY_YEARS
        ),
        source_note=REVIEW_SOURCE_NOTE,
        watermark=REVIEW_WATERMARK,
    )

    figures[
        "forecast"
    ] = forecast_figure

    forecast_scenarios_figure, _ = (
        plot_forecast_scenarios(
            data=data,
            forecast_years=(
                REVIEW_FORECAST_YEARS
            ),
            history_years=(
                REVIEW_FORECAST_HISTORY_YEARS
            ),
            source_note=REVIEW_SOURCE_NOTE,
            watermark=REVIEW_WATERMARK,
        )
    )

    figures[
        "forecast_scenarios"
    ] = forecast_scenarios_figure

    return figures


# ------------------------------------------------------------------
# Export and display
# ------------------------------------------------------------------


def export_review_figures(
    figures: dict[str, Figure],
) -> dict[str, Path]:
    """
    Export all visual-review figures as high-resolution PNG files.

    Args:
        figures:
            Named figures to export.

    Returns:
        Mapping of figure names to exported file paths.
    """

    return export_figures(
        output_directory=(
            REVIEW_OUTPUT_DIRECTORY
            / "png"
        ),
        export_format="png",
        dpi=300,
        transparent=False,
        close_after_export=False,
        **figures,
    )


def export_review_figures_as_svg(
    figures: dict[str, Figure],
) -> dict[str, Path]:
    """
    Export all visual-review figures as SVG files.

    Args:
        figures:
            Named figures to export.

    Returns:
        Mapping of figure names to exported file paths.
    """

    return export_figures(
        output_directory=(
            REVIEW_OUTPUT_DIRECTORY
            / "svg"
        ),
        export_format="svg",
        dpi=300,
        transparent=False,
        close_after_export=False,
        **figures,
    )


def print_export_summary(
    exported_paths: dict[str, Path],
    *,
    heading: str,
) -> None:
    """
    Print exported chart paths in a readable format.
    """

    print()
    print(heading)
    print("-" * len(heading))

    for chart_name, path in (
        exported_paths.items()
    ):
        print(
            f"{chart_name}: {path}"
        )


def main() -> None:
    """
    Generate, export, and display the complete visual review.
    """

    print(
        "Generating visual review figures..."
    )

    figures = generate_review_figures()

    print(
        f"Generated {len(figures)} figures."
    )

    png_paths = export_review_figures(
        figures
    )

    svg_paths = export_review_figures_as_svg(
        figures
    )

    print_export_summary(
        png_paths,
        heading="PNG exports",
    )

    print_export_summary(
        svg_paths,
        heading="SVG exports",
    )

    print()
    print(
        "Opening figures for visual review..."
    )

    plt.show()


if __name__ == "__main__":
    main()