"""
Reusable formatting helpers for Matplotlib visualizations.

This module applies the shared colors, typography, spacing,
grid behavior, spine styling, legend settings, and layout
defaults defined by the visualization styling package.

The helpers operate only on existing Matplotlib Figure and
Axes objects. They do not calculate data or create charts.
"""

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from visualizations.styling.colors import (
    BACKGROUND_COLOR,
    GRID_COLOR,
    PLOT_BACKGROUND_COLOR,
    SPINE_COLOR,
    TEXT_COLOR,
)

from visualizations.styling.constants import (
    DEFAULT_GRID_ALPHA,
    DEFAULT_GRID_AXIS,
    DEFAULT_GRID_LINE_STYLE,
    DEFAULT_GRID_LINE_WIDTH,
    DEFAULT_GRID_WHICH,
    DEFAULT_LABEL_PADDING,
    DEFAULT_LAYOUT_PADDING,
    DEFAULT_LEGEND_BORDER_PADDING,
    DEFAULT_LEGEND_COLUMNS,
    DEFAULT_LEGEND_FRAME_ON,
    DEFAULT_LEGEND_HANDLE_LENGTH,
    DEFAULT_LEGEND_LABEL_SPACING,
    DEFAULT_LEGEND_LOCATION,
    DEFAULT_SPINE_LINE_WIDTH,
    DEFAULT_TICK_LENGTH,
    DEFAULT_TICK_PADDING,
    DEFAULT_TICK_WIDTH,
    DEFAULT_X_MARGIN,
    DEFAULT_Y_MARGIN,
)

from visualizations.styling.typography import (
    AXIS_LABEL_FONT_SIZE,
    AXIS_LABEL_FONT_WEIGHT,
    DEFAULT_FONT_FAMILY,
    LEGEND_FONT_SIZE,
    LEGEND_TITLE_FONT_WEIGHT,
    SUBTITLE_FONT_SIZE,
    SUBTITLE_FONT_WEIGHT,
    TICK_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    TITLE_FONT_WEIGHT,
    TITLE_PADDING,
)


# ------------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------------


def validate_figure(
    figure: Figure,
) -> Figure:
    """
    Validate a Matplotlib Figure.

    Args:
        figure:
            Figure object to validate.

    Returns:
        The validated Figure.

    Raises:
        TypeError:
            If figure is not a Matplotlib Figure.
    """

    if not isinstance(
        figure,
        Figure,
    ):
        raise TypeError(
            "figure must be a matplotlib Figure."
        )

    return figure


def validate_axes(
    axes: Axes,
) -> Axes:
    """
    Validate a Matplotlib Axes object.

    Args:
        axes:
            Axes object to validate.

    Returns:
        The validated Axes.

    Raises:
        TypeError:
            If axes is not a Matplotlib Axes.
    """

    if not isinstance(
        axes,
        Axes,
    ):
        raise TypeError(
            "axes must be a matplotlib Axes."
        )

    return axes


def validate_optional_text(
    value: str | None,
    parameter_name: str,
) -> str | None:
    """
    Validate an optional text value.

    Args:
        value:
            Optional text to validate.

        parameter_name:
            Name used in validation error messages.

    Returns:
        The original string or None.

    Raises:
        TypeError:
            If value is neither a string nor None.

        ValueError:
            If value is a blank string.
    """

    if value is None:
        return None

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{parameter_name} must be a string or None."
        )

    if not value.strip():
        raise ValueError(
            f"{parameter_name} cannot be empty."
        )

    return value


# ------------------------------------------------------------------
# Figure and axes backgrounds
# ------------------------------------------------------------------


def style_figure(
    figure: Figure,
) -> Figure:
    """
    Apply standard styling to a Matplotlib Figure.

    Args:
        figure:
            Figure to style.

    Returns:
        The styled Figure.
    """

    validated_figure = validate_figure(
        figure
    )

    validated_figure.set_facecolor(
        BACKGROUND_COLOR
    )

    return validated_figure


def style_axes_background(
    axes: Axes,
) -> Axes:
    """
    Apply the standard plotting-area background.

    Args:
        axes:
            Axes to style.

    Returns:
        The styled Axes.
    """

    validated_axes = validate_axes(
        axes
    )

    validated_axes.set_facecolor(
        PLOT_BACKGROUND_COLOR
    )

    return validated_axes


# ------------------------------------------------------------------
# Title and label styling
# ------------------------------------------------------------------


def style_title(
    axes: Axes,
    title: str,
) -> Axes:
    """
    Apply the standard chart title.

    Args:
        axes:
            Axes receiving the title.

        title:
            Chart title text.

    Returns:
        The styled Axes.
    """

    validated_axes = validate_axes(
        axes
    )

    validated_title = validate_optional_text(
        title,
        "title",
    )

    validated_axes.set_title(
        validated_title,
        fontsize=TITLE_FONT_SIZE,
        fontweight=TITLE_FONT_WEIGHT,
        fontfamily=DEFAULT_FONT_FAMILY,
        color=TEXT_COLOR,
        pad=TITLE_PADDING,
        loc="left",
    )

    return validated_axes


def style_axis_labels(
    axes: Axes,
    *,
    x_label: str | None = None,
    y_label: str | None = None,
) -> Axes:
    """
    Apply standard X-axis and Y-axis label styling.

    Labels are updated only when a value is supplied.

    Args:
        axes:
            Axes receiving the labels.

        x_label:
            Optional X-axis label.

        y_label:
            Optional Y-axis label.

    Returns:
        The styled Axes.
    """

    validated_axes = validate_axes(
        axes
    )

    validated_x_label = validate_optional_text(
        x_label,
        "x_label",
    )

    validated_y_label = validate_optional_text(
        y_label,
        "y_label",
    )

    if validated_x_label is not None:
        validated_axes.set_xlabel(
            validated_x_label,
            fontsize=AXIS_LABEL_FONT_SIZE,
            fontweight=AXIS_LABEL_FONT_WEIGHT,
            fontfamily=DEFAULT_FONT_FAMILY,
            color=TEXT_COLOR,
            labelpad=DEFAULT_LABEL_PADDING,
        )

    if validated_y_label is not None:
        validated_axes.set_ylabel(
            validated_y_label,
            fontsize=AXIS_LABEL_FONT_SIZE,
            fontweight=AXIS_LABEL_FONT_WEIGHT,
            fontfamily=DEFAULT_FONT_FAMILY,
            color=TEXT_COLOR,
            labelpad=DEFAULT_LABEL_PADDING,
        )

    return validated_axes


def style_subtitle(
    axes: Axes,
    subtitle: str,
) -> Axes:
    """
    Add a standard subtitle above the plotting area.

    The subtitle is positioned beneath the primary title and
    aligned with the left edge of the Axes.

    Args:
        axes:
            Axes receiving the subtitle.

        subtitle:
            Subtitle text.

    Returns:
        The styled Axes.
    """

    validated_axes = validate_axes(
        axes
    )

    validated_subtitle = validate_optional_text(
        subtitle,
        "subtitle",
    )

    validated_axes.text(
        0.0,
        1.02,
        validated_subtitle,
        transform=validated_axes.transAxes,
        fontsize=SUBTITLE_FONT_SIZE,
        fontweight=SUBTITLE_FONT_WEIGHT,
        fontfamily=DEFAULT_FONT_FAMILY,
        color=TEXT_COLOR,
        horizontalalignment="left",
        verticalalignment="bottom",
    )

    return validated_axes


# ------------------------------------------------------------------
# Grid, spine, and tick styling
# ------------------------------------------------------------------


def style_grid(
    axes: Axes,
    *,
    visible: bool = True,
) -> Axes:
    """
    Apply standard grid styling.

    Args:
        axes:
            Axes receiving the grid.

        visible:
            Whether the grid should be displayed.

    Returns:
        The styled Axes.

    Raises:
        TypeError:
            If visible is not a boolean.
    """

    validated_axes = validate_axes(
        axes
    )

    if not isinstance(
        visible,
        bool,
    ):
        raise TypeError(
            "visible must be a boolean."
        )

    validated_axes.grid(
        visible=visible,
        which=DEFAULT_GRID_WHICH,
        axis=DEFAULT_GRID_AXIS,
        color=GRID_COLOR,
        linestyle=DEFAULT_GRID_LINE_STYLE,
        linewidth=DEFAULT_GRID_LINE_WIDTH,
        alpha=DEFAULT_GRID_ALPHA,
    )

    validated_axes.set_axisbelow(
        True
    )

    return validated_axes


def style_spines(
    axes: Axes,
) -> Axes:
    """
    Apply standard chart-spine styling.

    The top and right spines are hidden. The bottom and left
    spines remain visible with the shared neutral style.

    Args:
        axes:
            Axes whose spines will be styled.

    Returns:
        The styled Axes.
    """

    validated_axes = validate_axes(
        axes
    )

    validated_axes.spines[
        "top"
    ].set_visible(False)

    validated_axes.spines[
        "right"
    ].set_visible(False)

    for spine_name in (
        "bottom",
        "left",
    ):
        spine = validated_axes.spines[
            spine_name
        ]

        spine.set_visible(
            True
        )

        spine.set_color(
            SPINE_COLOR
        )

        spine.set_linewidth(
            DEFAULT_SPINE_LINE_WIDTH
        )

    return validated_axes


def style_ticks(
    axes: Axes,
) -> Axes:
    """
    Apply standard tick-label and tick-mark styling.

    Args:
        axes:
            Axes whose ticks will be styled.

    Returns:
        The styled Axes.
    """

    validated_axes = validate_axes(
        axes
    )

    validated_axes.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_FONT_SIZE,
        colors=TEXT_COLOR,
        length=DEFAULT_TICK_LENGTH,
        width=DEFAULT_TICK_WIDTH,
        pad=DEFAULT_TICK_PADDING,
    )

    for tick_label in (
        validated_axes.get_xticklabels()
        + validated_axes.get_yticklabels()
    ):
        tick_label.set_fontfamily(
            DEFAULT_FONT_FAMILY
        )

    return validated_axes


def style_axis_margins(
    axes: Axes,
) -> Axes:
    """
    Apply shared horizontal and vertical axis margins.

    Args:
        axes:
            Axes whose margins will be updated.

    Returns:
        The styled Axes.
    """

    validated_axes = validate_axes(
        axes
    )

    validated_axes.margins(
        x=DEFAULT_X_MARGIN,
        y=DEFAULT_Y_MARGIN,
    )

    return validated_axes


# ------------------------------------------------------------------
# Legend styling
# ------------------------------------------------------------------


def style_legend(
    axes: Axes,
    *,
    title: str | None = None,
    location: str = DEFAULT_LEGEND_LOCATION,
) -> Axes:
    """
    Create and style a legend when labeled artists exist.

    If the Axes contains no labeled artists, no legend is
    created.

    Args:
        axes:
            Axes receiving the legend.

        title:
            Optional legend title.

        location:
            Matplotlib legend-location value.

    Returns:
        The styled Axes.

    Raises:
        TypeError:
            If location is not a string.

        ValueError:
            If location is blank.
    """

    validated_axes = validate_axes(
        axes
    )

    validated_title = validate_optional_text(
        title,
        "legend title",
    )

    if not isinstance(
        location,
        str,
    ):
        raise TypeError(
            "location must be a string."
        )

    if not location.strip():
        raise ValueError(
            "location cannot be empty."
        )

    handles, labels = (
        validated_axes.get_legend_handles_labels()
    )

    labeled_items = [
        (handle, label)
        for handle, label in zip(
            handles,
            labels,
            strict=True,
        )
        if label
        and not label.startswith("_")
    ]

    if not labeled_items:
        return validated_axes

    filtered_handles = [
        item[0]
        for item in labeled_items
    ]

    filtered_labels = [
        item[1]
        for item in labeled_items
    ]

    legend = validated_axes.legend(
        filtered_handles,
        filtered_labels,
        title=validated_title,
        loc=location,
        frameon=DEFAULT_LEGEND_FRAME_ON,
        ncols=DEFAULT_LEGEND_COLUMNS,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=DEFAULT_LEGEND_HANDLE_LENGTH,
        borderpad=DEFAULT_LEGEND_BORDER_PADDING,
        labelspacing=DEFAULT_LEGEND_LABEL_SPACING,
    )

    if legend is not None:
        legend_title = legend.get_title()

        legend_title.set_fontfamily(
            DEFAULT_FONT_FAMILY
        )

        legend_title.set_fontweight(
            LEGEND_TITLE_FONT_WEIGHT
        )

        for legend_text in (
            legend.get_texts()
        ):
            legend_text.set_fontfamily(
                DEFAULT_FONT_FAMILY
            )

            legend_text.set_color(
                TEXT_COLOR
            )

    return validated_axes


# ------------------------------------------------------------------
# Layout and combined formatting
# ------------------------------------------------------------------


def apply_layout(
    figure: Figure,
) -> Figure:
    """
    Apply the standard tight-layout configuration.

    Args:
        figure:
            Figure whose layout will be updated.

    Returns:
        The updated Figure.
    """

    validated_figure = validate_figure(
        figure
    )

    validated_figure.tight_layout(
        pad=DEFAULT_LAYOUT_PADDING
    )

    return validated_figure


def apply_standard_formatting(
    figure: Figure,
    axes: Axes,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    show_grid: bool = True,
    show_legend: bool = False,
    legend_title: str | None = None,
    legend_location: str = DEFAULT_LEGEND_LOCATION,
    apply_tight_layout: bool = True,
) -> tuple[Figure, Axes]:
    """
    Apply the complete standard chart presentation.

    This helper combines the shared figure, background, title,
    label, grid, spine, tick, margin, legend, and layout
    functions.

    Args:
        figure:
            Matplotlib Figure to format.

        axes:
            Matplotlib Axes to format.

        title:
            Optional chart title.

        subtitle:
            Optional chart subtitle.

        x_label:
            Optional X-axis label.

        y_label:
            Optional Y-axis label.

        show_grid:
            Whether to display the standard grid.

        show_legend:
            Whether to create and style a legend.

        legend_title:
            Optional legend title.

        legend_location:
            Matplotlib legend-location value.

        apply_tight_layout:
            Whether to apply the standard tight layout.

    Returns:
        A tuple containing the formatted Figure and Axes.

    Raises:
        TypeError:
            If boolean configuration values are not booleans.
    """

    validated_figure = validate_figure(
        figure
    )

    validated_axes = validate_axes(
        axes
    )

    if not isinstance(
        show_grid,
        bool,
    ):
        raise TypeError(
            "show_grid must be a boolean."
        )

    if not isinstance(
        show_legend,
        bool,
    ):
        raise TypeError(
            "show_legend must be a boolean."
        )

    if not isinstance(
        apply_tight_layout,
        bool,
    ):
        raise TypeError(
            "apply_tight_layout must be a boolean."
        )

    style_figure(
        validated_figure
    )

    style_axes_background(
        validated_axes
    )

    if title is not None:
        style_title(
            validated_axes,
            title,
        )

    if subtitle is not None:
        style_subtitle(
            validated_axes,
            subtitle,
        )

    style_axis_labels(
        validated_axes,
        x_label=x_label,
        y_label=y_label,
    )

    style_grid(
        validated_axes,
        visible=show_grid,
    )

    style_spines(
        validated_axes
    )

    style_ticks(
        validated_axes
    )

    style_axis_margins(
        validated_axes
    )

    if show_legend:
        style_legend(
            validated_axes,
            title=legend_title,
            location=legend_location,
        )

    if apply_tight_layout:
        apply_layout(
            validated_figure
        )

    return (
        validated_figure,
        validated_axes,
    )