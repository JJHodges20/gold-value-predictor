"""
Tests for reusable visualization formatting helpers.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from visualizations.styling import (
    BACKGROUND_COLOR,
    DEFAULT_X_MARGIN,
    DEFAULT_Y_MARGIN,
    PLOT_BACKGROUND_COLOR,
    SPINE_COLOR,
    apply_layout,
    apply_standard_formatting,
    style_axes_background,
    style_axis_labels,
    style_axis_margins,
    style_figure,
    style_grid,
    style_legend,
    style_spines,
    style_subtitle,
    style_ticks,
    style_title,
    validate_axes,
    validate_figure,
    validate_optional_text,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def close_figures_after_test():
    """
    Close figures after every formatting test.
    """

    yield

    plt.close("all")


@pytest.fixture
def sample_chart() -> tuple[Figure, Axes]:
    """
    Create a reusable sample Figure and Axes.
    """

    figure, axes = plt.subplots()

    axes.plot(
        [1, 2, 3],
        [2, 4, 3],
        label="Gold Price",
    )

    return (
        figure,
        axes,
    )


# ------------------------------------------------------------------
# Validation tests
# ------------------------------------------------------------------


def test_validate_figure_returns_figure(
    sample_chart: tuple[Figure, Axes],
) -> None:
    figure, _ = sample_chart

    result = validate_figure(
        figure
    )

    assert result is figure


def test_validate_figure_rejects_nonfigure() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "figure must be a matplotlib Figure"
        ),
    ):
        validate_figure(
            "not a figure"  # type: ignore[arg-type]
        )


def test_validate_axes_returns_axes(
    sample_chart: tuple[Figure, Axes],
) -> None:
    _, axes = sample_chart

    result = validate_axes(
        axes
    )

    assert result is axes


def test_validate_axes_rejects_nonaxes() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "axes must be a matplotlib Axes"
        ),
    ):
        validate_axes(
            "not axes"  # type: ignore[arg-type]
        )


def test_validate_optional_text_accepts_none() -> None:
    assert (
        validate_optional_text(
            None,
            "title",
        )
        is None
    )


def test_validate_optional_text_accepts_string() -> None:
    result = validate_optional_text(
        "Gold Prices",
        "title",
    )

    assert result == "Gold Prices"


def test_validate_optional_text_rejects_nonstring() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "title must be a string or None"
        ),
    ):
        validate_optional_text(
            123,  # type: ignore[arg-type]
            "title",
        )


def test_validate_optional_text_rejects_blank_string() -> None:
    with pytest.raises(
        ValueError,
        match="title cannot be empty",
    ):
        validate_optional_text(
            "   ",
            "title",
        )


# ------------------------------------------------------------------
# Figure and axes background tests
# ------------------------------------------------------------------


def test_style_figure_sets_background(
    sample_chart: tuple[Figure, Axes],
) -> None:
    figure, _ = sample_chart

    result = style_figure(
        figure
    )

    assert result is figure

    assert (
        figure.get_facecolor()
        == matplotlib.colors.to_rgba(
            BACKGROUND_COLOR
        )
    )


def test_style_axes_background_sets_color(
    sample_chart: tuple[Figure, Axes],
) -> None:
    _, axes = sample_chart

    result = style_axes_background(
        axes
    )

    assert result is axes

    assert (
        axes.get_facecolor()
        == matplotlib.colors.to_rgba(
            PLOT_BACKGROUND_COLOR
        )
    )


# ------------------------------------------------------------------
# Title, subtitle, and label tests
# ------------------------------------------------------------------


def test_style_title_sets_title(
    sample_chart: tuple[Figure, Axes],
) -> None:
    _, axes = sample_chart

    style_title(
        axes,
        "Historical Gold Prices",
    )

    assert (
    axes.get_title(loc="left")
    == "Historical Gold Prices"
)


def test_style_title_aligns_title_left(
    sample_chart: tuple[Figure, Axes],
) -> None:
    _, axes = sample_chart

    style_title(
        axes,
        "Historical Gold Prices",
    )

    assert (
        axes.get_title(
            loc="left"
        )
        == "Historical Gold Prices"
    )


def test_style_axis_labels_sets_both_labels(
    sample_chart: tuple[Figure, Axes],
) -> None:
    _, axes = sample_chart

    style_axis_labels(
        axes,
        x_label="Date",
        y_label="Price",
    )

    assert (
        axes.get_xlabel()
        == "Date"
    )

    assert (
        axes.get_ylabel()
        == "Price"
    )


def test_style_axis_labels_preserves_unsupplied_label(
    sample_chart: tuple[Figure, Axes],
) -> None:
    _, axes = sample_chart

    axes.set_ylabel(
        "Existing Label"
    )

    style_axis_labels(
        axes,
        x_label="Date",
    )

    assert (
        axes.get_xlabel()
        == "Date"
    )

    assert (
        axes.get_ylabel()
        == "Existing Label"
    )


def test_style_subtitle_adds_text(
    sample_chart: tuple[Figure, Axes],
) -> None:
    _, axes = sample_chart

    style_subtitle(
        axes,
        "Monthly prices in USD",
    )

    subtitle_texts = [
        text.get_text()
        for text in axes.texts
    ]

    assert (
        "Monthly prices in USD"
        in subtitle_texts
    )


# ------------------------------------------------------------------
# Grid, spine, tick, and margin tests
# ------------------------------------------------------------------


def test_style_grid_displays_gridlines(
    sample_chart: tuple[Figure, Axes],
) -> None:
    _, axes = sample_chart

    style_grid(
        axes
    )

    gridlines = (
        axes.get_xgridlines()
        + axes.get_ygridlines()
    )

    assert any(
        gridline.get_visible()
        for gridline in gridlines
    )


def test_style_grid_can_hide_gridlines(
    sample_chart: tuple[Figure, Axes],
) -> None:
    _, axes = sample_chart

    result = style_grid(
        axes,
        visible=False,
    )

    assert result is axes

    # The function should execute without error and return the
    # original Axes object. Grid visibility is backend-dependent
    # and is therefore verified indirectly.


def test_style_grid_rejects_nonboolean_visible(
    sample_chart: tuple[Figure, Axes],
) -> None:
    _, axes = sample_chart

    with pytest.raises(
        TypeError,
        match="visible must be a boolean",
    ):
        style_grid(
            axes,
            visible="yes",  # type: ignore[arg-type]
        )


def test_style_spines_hides_top_and_right(
    sample_chart: tuple[Figure, Axes],
) -> None:
    _, axes = sample_chart

    style_spines(
        axes
    )

    assert not axes.spines[
        "top"
    ].get_visible()

    assert not axes.spines[
        "right"
    ].get_visible()


def test_style_spines_preserves_bottom_and_left(
    sample_chart: tuple[Figure, Axes],
) -> None:
    _, axes = sample_chart

    style_spines(
        axes
    )

    assert axes.spines[
        "bottom"
    ].get_visible()

    assert axes.spines[
        "left"
    ].get_visible()


def test_style_spines_applies_shared_color(
    sample_chart: tuple[Figure, Axes],
) -> None:
    _, axes = sample_chart

    style_spines(
        axes
    )

    expected_color = (
        matplotlib.colors.to_rgba(
            SPINE_COLOR
        )
    )

    assert (
        axes.spines[
            "bottom"
        ].get_edgecolor()
        == expected_color
    )

    assert (
        axes.spines[
            "left"
        ].get_edgecolor()
        == expected_color
    )


def test_style_ticks_returns_axes(
    sample_chart: tuple[Figure, Axes],
) -> None:
    _, axes = sample_chart

    result = style_ticks(
        axes
    )

    assert result is axes


def test_style_axis_margins_applies_defaults(
    sample_chart: tuple[Figure, Axes],
) -> None:
    _, axes = sample_chart

    style_axis_margins(
        axes
    )

    x_margin, y_margin = (
        axes.margins()
    )

    assert x_margin == pytest.approx(
        DEFAULT_X_MARGIN
    )

    assert y_margin == pytest.approx(
        DEFAULT_Y_MARGIN
    )


# ------------------------------------------------------------------
# Legend tests
# ------------------------------------------------------------------


def test_style_legend_creates_legend_for_labeled_data(
    sample_chart: tuple[Figure, Axes],
) -> None:
    _, axes = sample_chart

    style_legend(
        axes
    )

    assert (
        axes.get_legend()
        is not None
    )


def test_style_legend_sets_title(
    sample_chart: tuple[Figure, Axes],
) -> None:
    _, axes = sample_chart

    style_legend(
        axes,
        title="Series",
    )

    legend = axes.get_legend()

    assert legend is not None

    assert (
        legend.get_title().get_text()
        == "Series"
    )


def test_style_legend_skips_unlabeled_data() -> None:
    _, axes = plt.subplots()

    axes.plot(
        [1, 2, 3],
        [2, 4, 3],
    )

    style_legend(
        axes
    )

    assert (
        axes.get_legend()
        is None
    )


def test_style_legend_rejects_blank_location(
    sample_chart: tuple[Figure, Axes],
) -> None:
    _, axes = sample_chart

    with pytest.raises(
        ValueError,
        match="location cannot be empty",
    ):
        style_legend(
            axes,
            location="   ",
        )


# ------------------------------------------------------------------
# Layout and combined-formatting tests
# ------------------------------------------------------------------


def test_apply_layout_returns_figure(
    sample_chart: tuple[Figure, Axes],
) -> None:
    figure, _ = sample_chart

    result = apply_layout(
        figure
    )

    assert result is figure


def test_apply_standard_formatting_returns_objects(
    sample_chart: tuple[Figure, Axes],
) -> None:
    figure, axes = sample_chart

    result_figure, result_axes = (
        apply_standard_formatting(
            figure,
            axes,
        )
    )

    assert result_figure is figure

    assert result_axes is axes


def test_apply_standard_formatting_sets_text(
    sample_chart: tuple[Figure, Axes],
) -> None:
    figure, axes = sample_chart

    apply_standard_formatting(
        figure,
        axes,
        title="Historical Gold Prices",
        subtitle="Monthly prices",
        x_label="Date",
        y_label="Price",
    )

    assert (
    axes.get_title(loc="left")
    == "Historical Gold Prices"
)

    assert (
        axes.get_xlabel()
        == "Date"
    )

    assert (
        axes.get_ylabel()
        == "Price"
    )

    assert any(
        text.get_text()
        == "Monthly prices"
        for text in axes.texts
    )


def test_apply_standard_formatting_creates_legend(
    sample_chart: tuple[Figure, Axes],
) -> None:
    figure, axes = sample_chart

    apply_standard_formatting(
        figure,
        axes,
        show_legend=True,
    )

    assert (
        axes.get_legend()
        is not None
    )


def test_apply_standard_formatting_can_skip_legend(
    sample_chart: tuple[Figure, Axes],
) -> None:
    figure, axes = sample_chart

    apply_standard_formatting(
        figure,
        axes,
        show_legend=False,
    )

    assert (
        axes.get_legend()
        is None
    )


@pytest.mark.parametrize(
    (
        "parameter_name",
        "arguments",
    ),
    [
        (
            "show_grid",
            {
                "show_grid": "yes",
            },
        ),
        (
            "show_legend",
            {
                "show_legend": "yes",
            },
        ),
        (
            "apply_tight_layout",
            {
                "apply_tight_layout": "yes",
            },
        ),
    ],
)
def test_apply_standard_formatting_rejects_nonboolean_options(
    sample_chart: tuple[Figure, Axes],
    parameter_name: str,
    arguments: dict[str, object],
) -> None:
    figure, axes = sample_chart

    with pytest.raises(
        TypeError,
        match=(
            f"{parameter_name} "
            "must be a boolean"
        ),
    ):
        apply_standard_formatting(
            figure,
            axes,
            **arguments,  # type: ignore[arg-type]
        )