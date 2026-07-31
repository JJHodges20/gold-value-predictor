"""
Shared layout and formatting constants for visualizations.

These values control figure sizing, line styles, spacing, grid
appearance, legend behavior, and other presentation defaults.
"""

# ------------------------------------------------------------------
# Figure settings
# ------------------------------------------------------------------

DEFAULT_FIGURE_SIZE = (
    12.0,
    6.0,
)

WIDE_FIGURE_SIZE = (
    14.0,
    6.0,
)

TALL_FIGURE_SIZE = (
    10.0,
    8.0,
)

SQUARE_FIGURE_SIZE = (
    8.0,
    8.0,
)

DEFAULT_FIGURE_DPI = 100

DEFAULT_EXPORT_DPI = 300


# ------------------------------------------------------------------
# Line settings
# ------------------------------------------------------------------

DEFAULT_LINE_WIDTH = 2.0

SECONDARY_LINE_WIDTH = 1.5

REFERENCE_LINE_WIDTH = 1.25

THIN_LINE_WIDTH = 1.0

EMPHASIS_LINE_WIDTH = 2.5

DEFAULT_LINE_STYLE = "-"

DASHED_LINE_STYLE = "--"

DOTTED_LINE_STYLE = ":"

DASH_DOT_LINE_STYLE = "-."


# ------------------------------------------------------------------
# Marker settings
# ------------------------------------------------------------------

DEFAULT_MARKER_SIZE = 5.0

ANNOTATION_MARKER_SIZE = 6.0

DEFAULT_MARKER_EDGE_WIDTH = 0.75


# ------------------------------------------------------------------
# Grid settings
# ------------------------------------------------------------------

DEFAULT_GRID_ALPHA = 0.25

DEFAULT_GRID_LINE_WIDTH = 0.8

DEFAULT_GRID_LINE_STYLE = "-"

DEFAULT_GRID_AXIS = "both"

DEFAULT_GRID_WHICH = "major"


# ------------------------------------------------------------------
# Transparency settings
# ------------------------------------------------------------------

DEFAULT_FILL_ALPHA = 0.20

LIGHT_FILL_ALPHA = 0.10

EMPHASIS_FILL_ALPHA = 0.30

ANNOTATION_BOX_ALPHA = 0.90


# ------------------------------------------------------------------
# Axis and spine settings
# ------------------------------------------------------------------

DEFAULT_SPINE_LINE_WIDTH = 0.8

DEFAULT_TICK_LENGTH = 4.0

DEFAULT_TICK_WIDTH = 0.8

DEFAULT_TICK_PADDING = 6.0

DEFAULT_AXIS_MARGIN = 0.02

HORIZONTAL_REFERENCE_VALUE = 0.0


# ------------------------------------------------------------------
# Legend settings
# ------------------------------------------------------------------

DEFAULT_LEGEND_LOCATION = "best"

DEFAULT_LEGEND_FRAME_ON = False

DEFAULT_LEGEND_COLUMNS = 1

DEFAULT_LEGEND_HANDLE_LENGTH = 2.0

DEFAULT_LEGEND_BORDER_PADDING = 0.5

DEFAULT_LEGEND_LABEL_SPACING = 0.5


# ------------------------------------------------------------------
# Layout settings
# ------------------------------------------------------------------

DEFAULT_LAYOUT_PADDING = 1.5

DEFAULT_TITLE_PADDING = 18.0

DEFAULT_LABEL_PADDING = 10.0

DEFAULT_ANNOTATION_OFFSET = (
    12,
    12,
)

DEFAULT_X_MARGIN = 0.01

DEFAULT_Y_MARGIN = 0.05


# ------------------------------------------------------------------
# Date-axis settings
# ------------------------------------------------------------------

DEFAULT_MAJOR_DATE_INTERVAL = 1

DEFAULT_MINOR_DATE_INTERVAL = 3

DEFAULT_DATE_FORMAT = "%Y"

DEFAULT_MONTH_DATE_FORMAT = "%b %Y"


# ------------------------------------------------------------------
# Chart-specific presentation settings
# ------------------------------------------------------------------

DEFAULT_ZERO_LINE_WIDTH = 1.0

DEFAULT_FORECAST_BOUNDARY_WIDTH = 1.5

DEFAULT_FORECAST_LINE_STYLE = "--"

DEFAULT_FORECAST_BOUNDARY_STYLE = ":"

DEFAULT_RANGE_FILL_ALPHA = 0.10

DEFAULT_PERCENTAGE_FILL_ALPHA = 0.20