"""
Shared color definitions for Gold Value Predictor charts.

Colors are organized by semantic purpose rather than by chart.
This allows visualization functions to request colors based on
what a data series represents.
"""

# ------------------------------------------------------------------
# Neutral interface colors
# ------------------------------------------------------------------

BACKGROUND_COLOR = "#FFFFFF"

PLOT_BACKGROUND_COLOR = "#FAFAFA"

TEXT_COLOR = "#1F2933"

SECONDARY_TEXT_COLOR = "#52606D"

GRID_COLOR = "#CBD2D9"

SPINE_COLOR = "#9AA5B1"

REFERENCE_LINE_COLOR = "#7B8794"


# ------------------------------------------------------------------
# Gold and price-series colors
# ------------------------------------------------------------------

GOLD_COLOR = "#B8860B"

GOLD_HIGHLIGHT_COLOR = "#D4A017"

HISTORICAL_PRICE_COLOR = GOLD_COLOR

RECENT_PRICE_COLOR = GOLD_COLOR

NOMINAL_PRICE_COLOR = GOLD_COLOR

REAL_PRICE_COLOR = "#2F6B8A"


# ------------------------------------------------------------------
# Return and risk colors
# ------------------------------------------------------------------

POSITIVE_RETURN_COLOR = "#2E7D32"

NEGATIVE_RETURN_COLOR = "#C62828"

NEUTRAL_RETURN_COLOR = "#607D8B"

CUMULATIVE_RETURN_COLOR = "#1565C0"

ANNUAL_RETURN_COLOR = "#5E35B1"

VOLATILITY_COLOR = "#D97706"

DRAWDOWN_COLOR = "#B91C1C"


# ------------------------------------------------------------------
# Rolling-analysis colors
# ------------------------------------------------------------------

ROLLING_AVERAGE_COLOR = "#2563EB"

ROLLING_HIGH_COLOR = "#15803D"

ROLLING_LOW_COLOR = "#DC2626"

ROLLING_RANGE_COLOR = "#64748B"

ROLLING_RETURN_COLOR = "#7C3AED"


# ------------------------------------------------------------------
# Forecast colors
# ------------------------------------------------------------------

FORECAST_COLOR = "#7C3AED"

FORECAST_BOUNDARY_COLOR = "#6B7280"

FORECAST_CONSERVATIVE_COLOR = "#2563EB"

FORECAST_BASE_COLOR = "#7C3AED"

FORECAST_OPTIMISTIC_COLOR = "#15803D"


# ------------------------------------------------------------------
# Scenario palette
# ------------------------------------------------------------------

SCENARIO_COLORS = (
    FORECAST_CONSERVATIVE_COLOR,
    FORECAST_BASE_COLOR,
    FORECAST_OPTIMISTIC_COLOR,
    "#D97706",
    "#DC2626",
    "#0891B2",
    "#4F46E5",
    "#DB2777",
)