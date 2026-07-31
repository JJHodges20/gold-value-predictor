"""
Tests for shared visualization styling constants.
"""

from visualizations.styling import (
    BACKGROUND_COLOR,
    DEFAULT_FIGURE_SIZE,
    DEFAULT_FONT_FAMILY,
    DEFAULT_GRID_ALPHA,
    DEFAULT_LINE_WIDTH,
    FORECAST_COLOR,
    GOLD_COLOR,
    SCENARIO_COLORS,
    TITLE_FONT_SIZE,
)


def test_default_figure_size_contains_two_positive_values() -> None:
    assert len(DEFAULT_FIGURE_SIZE) == 2

    assert all(
        isinstance(value, float)
        for value in DEFAULT_FIGURE_SIZE
    )

    assert all(
        value > 0
        for value in DEFAULT_FIGURE_SIZE
    )


def test_default_line_width_is_positive() -> None:
    assert DEFAULT_LINE_WIDTH > 0


def test_grid_alpha_is_valid() -> None:
    assert 0.0 <= DEFAULT_GRID_ALPHA <= 1.0


def test_title_font_size_is_positive() -> None:
    assert TITLE_FONT_SIZE > 0


def test_default_font_family_is_not_empty() -> None:
    assert DEFAULT_FONT_FAMILY.strip()


def test_primary_colors_are_hex_values() -> None:
    colors = (
        BACKGROUND_COLOR,
        GOLD_COLOR,
        FORECAST_COLOR,
    )

    for color in colors:
        assert color.startswith("#")
        assert len(color) == 7


def test_scenario_palette_contains_multiple_colors() -> None:
    assert len(SCENARIO_COLORS) >= 3


def test_scenario_palette_colors_are_unique() -> None:
    assert len(SCENARIO_COLORS) == len(
        set(SCENARIO_COLORS)
    )