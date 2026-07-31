"""
Reusable annotation helpers for Matplotlib charts.

These functions add value labels, extrema markers, vertical
event labels, and informational notes to existing Axes objects.
They do not calculate financial metrics or create figures.
"""

from numbers import Real

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Annotation, Text

from visualizations.styling.colors import (
    BACKGROUND_COLOR,
    REFERENCE_LINE_COLOR,
    SECONDARY_TEXT_COLOR,
    TEXT_COLOR,
)

from visualizations.styling.constants import (
    ANNOTATION_BOX_ALPHA,
    ANNOTATION_MARKER_SIZE,
    DEFAULT_ANNOTATION_OFFSET,
    REFERENCE_LINE_WIDTH,
)

from visualizations.styling.formatting import (
    validate_axes,
    validate_figure,
    validate_optional_text,
)

from visualizations.styling.typography import (
    ANNOTATION_FONT_SIZE,
    ANNOTATION_FONT_WEIGHT,
    DEFAULT_FONT_FAMILY,
    SMALL_TEXT_FONT_SIZE,
)


# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------


def validate_numeric_series(
    values: pd.Series,
) -> pd.Series:
    """
    Validate and clean a numeric, date-indexed Series.

    Args:
        values:
            Series containing values to annotate.

    Returns:
        A sorted Series containing valid numeric observations.

    Raises:
        TypeError:
            If values is not a Pandas Series.

        ValueError:
            If no valid numeric observations remain.
    """

    if not isinstance(
        values,
        pd.Series,
    ):
        raise TypeError(
            "values must be a Pandas Series."
        )

    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    numeric_values = numeric_values.sort_index()

    if numeric_values.empty:
        raise ValueError(
            "values must contain at least one valid "
            "numeric observation."
        )

    return numeric_values


def validate_annotation_offset(
    offset: tuple[Real, Real],
) -> tuple[float, float]:
    """
    Validate an annotation offset.

    Args:
        offset:
            Two-value tuple representing horizontal and
            vertical point offsets.

    Returns:
        A normalized float tuple.

    Raises:
        TypeError:
            If offset is not a two-value numeric tuple.
    """

    if (
        not isinstance(offset, tuple)
        or len(offset) != 2
    ):
        raise TypeError(
            "offset must be a two-value tuple."
        )

    x_offset, y_offset = offset

    if (
        isinstance(x_offset, bool)
        or isinstance(y_offset, bool)
        or not isinstance(x_offset, Real)
        or not isinstance(y_offset, Real)
    ):
        raise TypeError(
            "offset values must be numeric."
        )

    return (
        float(x_offset),
        float(y_offset),
    )


# ------------------------------------------------------------------
# Value-label formatting
# ------------------------------------------------------------------


def format_annotation_value(
    value: Real,
    *,
    value_format: str = "currency",
) -> str:
    """
    Format a numeric value for an annotation.

    Args:
        value:
            Numeric value to format.

        value_format:
            One of: currency, percentage, or number.

    Returns:
        Formatted annotation text.

    Raises:
        TypeError:
            If value is not numeric or value_format is not
            a string.

        ValueError:
            If value_format is unsupported.
    """

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise TypeError(
            "value must be numeric."
        )

    if not isinstance(
        value_format,
        str,
    ):
        raise TypeError(
            "value_format must be a string."
        )

    normalized_format = (
        value_format.strip().lower()
    )

    if normalized_format == "currency":
        return f"${value:,.2f}"

    if normalized_format == "percentage":
        return f"{value:.2%}"

    if normalized_format == "number":
        return f"{value:,.2f}"

    raise ValueError(
        "value_format must be 'currency', "
        "'percentage', or 'number'."
    )


# ------------------------------------------------------------------
# Point annotations
# ------------------------------------------------------------------


def annotate_series_point(
    axes: Axes,
    *,
    x_value: object,
    y_value: Real,
    label: str,
    offset: tuple[Real, Real] = (
        DEFAULT_ANNOTATION_OFFSET
    ),
    marker_color: str = REFERENCE_LINE_COLOR,
) -> Annotation:
    """
    Annotate one point on a chart.

    Args:
        axes:
            Axes receiving the annotation.

        x_value:
            Horizontal-axis value.

        y_value:
            Vertical-axis value.

        label:
            Text displayed in the annotation.

        offset:
            Annotation offset in display points.

        marker_color:
            Marker and arrow color.

    Returns:
        The created Matplotlib Annotation.
    """

    validated_axes = validate_axes(
        axes
    )

    validated_label = validate_optional_text(
        label,
        "label",
    )

    if (
        isinstance(y_value, bool)
        or not isinstance(y_value, Real)
    ):
        raise TypeError(
            "y_value must be numeric."
        )

    normalized_offset = (
        validate_annotation_offset(
            offset
        )
    )

    validated_axes.scatter(
        [x_value],
        [y_value],
        color=marker_color,
        s=ANNOTATION_MARKER_SIZE ** 2,
        zorder=5,
    )

    annotation = validated_axes.annotate(
        validated_label,
        xy=(
            x_value,
            y_value,
        ),
        xytext=normalized_offset,
        textcoords="offset points",
        fontsize=ANNOTATION_FONT_SIZE,
        fontweight=ANNOTATION_FONT_WEIGHT,
        fontfamily=DEFAULT_FONT_FAMILY,
        color=TEXT_COLOR,
        arrowprops={
            "arrowstyle": "-",
            "color": marker_color,
            "linewidth": REFERENCE_LINE_WIDTH,
        },
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": BACKGROUND_COLOR,
            "edgecolor": marker_color,
            "alpha": ANNOTATION_BOX_ALPHA,
        },
        zorder=6,
    )

    return annotation


def annotate_latest_value(
    axes: Axes,
    values: pd.Series,
    *,
    prefix: str = "Latest",
    value_format: str = "currency",
    offset: tuple[Real, Real] = (
        DEFAULT_ANNOTATION_OFFSET
    ),
    marker_color: str = REFERENCE_LINE_COLOR,
) -> Annotation:
    """
    Annotate the latest valid observation in a Series.
    """

    cleaned_values = validate_numeric_series(
        values
    )

    latest_date = cleaned_values.index[-1]

    latest_value = float(
        cleaned_values.iloc[-1]
    )

    formatted_value = format_annotation_value(
        latest_value,
        value_format=value_format,
    )

    label = f"{prefix}: {formatted_value}"

    return annotate_series_point(
        axes,
        x_value=latest_date,
        y_value=latest_value,
        label=label,
        offset=offset,
        marker_color=marker_color,
    )


def annotate_high_value(
    axes: Axes,
    values: pd.Series,
    *,
    prefix: str = "High",
    value_format: str = "currency",
    offset: tuple[Real, Real] = (
        DEFAULT_ANNOTATION_OFFSET
    ),
    marker_color: str = REFERENCE_LINE_COLOR,
) -> Annotation:
    """
    Annotate the maximum valid observation in a Series.
    """

    cleaned_values = validate_numeric_series(
        values
    )

    high_date = cleaned_values.idxmax()

    high_value = float(
        cleaned_values.loc[high_date]
    )

    formatted_value = format_annotation_value(
        high_value,
        value_format=value_format,
    )

    label = f"{prefix}: {formatted_value}"

    return annotate_series_point(
        axes,
        x_value=high_date,
        y_value=high_value,
        label=label,
        offset=offset,
        marker_color=marker_color,
    )


def annotate_low_value(
    axes: Axes,
    values: pd.Series,
    *,
    prefix: str = "Low",
    value_format: str = "currency",
    offset: tuple[Real, Real] = (
        DEFAULT_ANNOTATION_OFFSET
    ),
    marker_color: str = REFERENCE_LINE_COLOR,
) -> Annotation:
    """
    Annotate the minimum valid observation in a Series.
    """

    cleaned_values = validate_numeric_series(
        values
    )

    low_date = cleaned_values.idxmin()

    low_value = float(
        cleaned_values.loc[low_date]
    )

    formatted_value = format_annotation_value(
        low_value,
        value_format=value_format,
    )

    label = f"{prefix}: {formatted_value}"

    return annotate_series_point(
        axes,
        x_value=low_date,
        y_value=low_value,
        label=label,
        offset=offset,
        marker_color=marker_color,
    )


# ------------------------------------------------------------------
# Event and note annotations
# ------------------------------------------------------------------


def annotate_vertical_event(
    axes: Axes,
    *,
    x_value: object,
    label: str,
    line_color: str = REFERENCE_LINE_COLOR,
    line_style: str = ":",
) -> Text:
    """
    Add a labeled vertical event line.

    Args:
        axes:
            Axes receiving the event marker.

        x_value:
            Horizontal-axis position.

        label:
            Event label.

        line_color:
            Vertical-line color.

        line_style:
            Matplotlib line style.

    Returns:
        The created text artist.
    """

    validated_axes = validate_axes(
        axes
    )

    validated_label = validate_optional_text(
        label,
        "label",
    )

    validated_axes.axvline(
        x=x_value,
        color=line_color,
        linewidth=REFERENCE_LINE_WIDTH,
        linestyle=line_style,
        zorder=2,
    )

    event_text = validated_axes.text(
        x_value,
        0.98,
        validated_label,
        transform=validated_axes.get_xaxis_transform(),
        rotation=90,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=SMALL_TEXT_FONT_SIZE,
        fontfamily=DEFAULT_FONT_FAMILY,
        color=SECONDARY_TEXT_COLOR,
    )

    return event_text


def add_axes_note(
    axes: Axes,
    note: str,
    *,
    x_position: float = 0.01,
    y_position: float = 0.01,
) -> Text:
    """
    Add a small informational note inside the Axes.

    Args:
        axes:
            Axes receiving the note.

        note:
            Note text.

        x_position:
            Horizontal location in Axes coordinates.

        y_position:
            Vertical location in Axes coordinates.

    Returns:
        The created text artist.
    """

    validated_axes = validate_axes(
        axes
    )

    validated_note = validate_optional_text(
        note,
        "note",
    )

    for value, parameter_name in (
        (
            x_position,
            "x_position",
        ),
        (
            y_position,
            "y_position",
        ),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
        ):
            raise TypeError(
                f"{parameter_name} must be numeric."
            )

    return validated_axes.text(
        float(x_position),
        float(y_position),
        validated_note,
        transform=validated_axes.transAxes,
        fontsize=SMALL_TEXT_FONT_SIZE,
        fontfamily=DEFAULT_FONT_FAMILY,
        color=SECONDARY_TEXT_COLOR,
        horizontalalignment="left",
        verticalalignment="bottom",
    )

def add_figure_source_note(
    figure: Figure,
    source_note: str,
    *,
    x_position: float = 0.01,
    y_position: float = 0.01,
) -> Text:
    """
    Add a small source note along the bottom of a Figure.

    Args:
        figure:
            Figure receiving the source note.

        source_note:
            Source or attribution text.

        x_position:
            Horizontal location in figure coordinates.

        y_position:
            Vertical location in figure coordinates.

    Returns:
        The created Matplotlib Text artist.

    Raises:
        TypeError:
            If figure is invalid or a position is not numeric.

        ValueError:
            If source_note is blank.
    """

    validated_figure = validate_figure(
        figure
    )

    validated_source_note = validate_optional_text(
        source_note,
        "source_note",
    )

    for value, parameter_name in (
        (
            x_position,
            "x_position",
        ),
        (
            y_position,
            "y_position",
        ),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
        ):
            raise TypeError(
                f"{parameter_name} must be numeric."
            )

    source_text = validated_figure.text(
        float(x_position),
        float(y_position),
        validated_source_note,
        fontsize=SMALL_TEXT_FONT_SIZE,
        fontfamily=DEFAULT_FONT_FAMILY,
        color=SECONDARY_TEXT_COLOR,
        horizontalalignment="left",
        verticalalignment="bottom",
    )

    return source_text

def add_watermark(
    axes: Axes,
    watermark: str,
    *,
    x_position: float = 0.99,
    y_position: float = 0.02,
    alpha: float = 0.16,
) -> Text:
    """
    Add a subtle text watermark inside an Axes.

    Args:
        axes:
            Axes receiving the watermark.

        watermark:
            Watermark text.

        x_position:
            Horizontal location in Axes coordinates.

        y_position:
            Vertical location in Axes coordinates.

        alpha:
            Watermark opacity between zero and one.

    Returns:
        The created Matplotlib Text artist.

    Raises:
        TypeError:
            If axes is invalid or a numeric option is invalid.

        ValueError:
            If watermark is blank or alpha is outside the
            inclusive range from zero to one.
    """

    validated_axes = validate_axes(
        axes
    )

    validated_watermark = validate_optional_text(
        watermark,
        "watermark",
    )

    for value, parameter_name in (
        (
            x_position,
            "x_position",
        ),
        (
            y_position,
            "y_position",
        ),
        (
            alpha,
            "alpha",
        ),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
        ):
            raise TypeError(
                f"{parameter_name} must be numeric."
            )

    normalized_alpha = float(
        alpha
    )

    if not 0.0 <= normalized_alpha <= 1.0:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    watermark_text = validated_axes.text(
        float(x_position),
        float(y_position),
        validated_watermark,
        transform=validated_axes.transAxes,
        fontsize=ANNOTATION_FONT_SIZE,
        fontweight=ANNOTATION_FONT_WEIGHT,
        fontfamily=DEFAULT_FONT_FAMILY,
        color=SECONDARY_TEXT_COLOR,
        alpha=normalized_alpha,
        horizontalalignment="right",
        verticalalignment="bottom",
        zorder=1,
    )

    return watermark_text