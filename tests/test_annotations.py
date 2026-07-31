"""
Tests for reusable chart-annotation helpers.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.text import Annotation, Text

from visualizations.styling import (
    add_axes_note,
    add_figure_source_note,
    add_watermark,
    annotate_high_value,
    annotate_latest_value,
    annotate_low_value,
    annotate_series_point,
    annotate_vertical_event,
    format_annotation_value,
    validate_annotation_offset,
    validate_numeric_series,
)


@pytest.fixture(autouse=True)
def close_figures_after_test():
    """
    Close all figures after every annotation test.
    """

    yield

    plt.close("all")


@pytest.fixture
def sample_axes() -> Axes:
    """
    Create reusable Axes for annotation tests.
    """

    _, axes = plt.subplots()

    return axes


@pytest.fixture
def sample_series() -> pd.Series:
    """
    Create a predictable date-indexed Series.
    """

    return pd.Series(
        data=[
            100.0,
            125.0,
            90.0,
            140.0,
        ],
        index=pd.date_range(
            start="2026-01-01",
            periods=4,
            freq="MS",
        ),
        name="Gold Price",
    )


def test_validate_numeric_series_returns_clean_series(
    sample_series: pd.Series,
) -> None:
    result = validate_numeric_series(
        sample_series
    )

    pd.testing.assert_series_equal(
        result,
        sample_series,
    )


def test_validate_numeric_series_removes_invalid_values() -> None:
    values = pd.Series(
        [
            100,
            "invalid",
            125,
        ]
    )

    result = validate_numeric_series(
        values
    )

    assert list(result) == [
        100.0,
        125.0,
    ]


def test_validate_numeric_series_rejects_nonseries() -> None:
    with pytest.raises(
        TypeError,
        match="values must be a Pandas Series",
    ):
        validate_numeric_series(
            [100, 125]  # type: ignore[arg-type]
        )


def test_validate_numeric_series_rejects_empty_result() -> None:
    values = pd.Series(
        [
            None,
            "invalid",
        ]
    )

    with pytest.raises(
        ValueError,
        match="at least one valid numeric observation",
    ):
        validate_numeric_series(values)


def test_validate_annotation_offset_accepts_numeric_tuple() -> None:
    result = validate_annotation_offset(
        (
            10,
            -5,
        )
    )

    assert result == (
        10.0,
        -5.0,
    )


@pytest.mark.parametrize(
    "offset",
    [
        [10, 5],
        (10,),
        (10, 5, 2),
        ("10", 5),
        (True, 5),
    ],
)
def test_validate_annotation_offset_rejects_invalid_values(
    offset: object,
) -> None:
    with pytest.raises(TypeError):
        validate_annotation_offset(  # type: ignore[arg-type]
            offset
        )


@pytest.mark.parametrize(
    (
        "value",
        "value_format",
        "expected",
    ),
    [
        (
            1234.5,
            "currency",
            "$1,234.50",
        ),
        (
            0.0525,
            "percentage",
            "5.25%",
        ),
        (
            1234.5,
            "number",
            "1,234.50",
        ),
    ],
)
def test_format_annotation_value(
    value: float,
    value_format: str,
    expected: str,
) -> None:
    assert (
        format_annotation_value(
            value,
            value_format=value_format,
        )
        == expected
    )


def test_format_annotation_value_rejects_unknown_format() -> None:
    with pytest.raises(
        ValueError,
        match="value_format must be",
    ):
        format_annotation_value(
            100,
            value_format="unknown",
        )


def test_annotate_series_point_returns_annotation(
    sample_axes: Axes,
) -> None:
    annotation = annotate_series_point(
        sample_axes,
        x_value=1,
        y_value=100,
        label="Example",
    )

    assert isinstance(
        annotation,
        Annotation,
    )

    assert (
        annotation.get_text()
        == "Example"
    )


def test_annotate_latest_value_uses_last_observation(
    sample_axes: Axes,
    sample_series: pd.Series,
) -> None:
    annotation = annotate_latest_value(
        sample_axes,
        sample_series,
    )

    assert (
        annotation.get_text()
        == "Latest: $140.00"
    )


def test_annotate_high_value_uses_maximum(
    sample_axes: Axes,
    sample_series: pd.Series,
) -> None:
    annotation = annotate_high_value(
        sample_axes,
        sample_series,
    )

    assert (
        annotation.get_text()
        == "High: $140.00"
    )


def test_annotate_low_value_uses_minimum(
    sample_axes: Axes,
    sample_series: pd.Series,
) -> None:
    annotation = annotate_low_value(
        sample_axes,
        sample_series,
    )

    assert (
        annotation.get_text()
        == "Low: $90.00"
    )


def test_annotate_vertical_event_returns_text(
    sample_axes: Axes,
) -> None:
    text = annotate_vertical_event(
        sample_axes,
        x_value=pd.Timestamp(
            "2026-01-01"
        ),
        label="Forecast Begins",
    )

    assert isinstance(
        text,
        Text,
    )

    assert (
        text.get_text()
        == "Forecast Begins"
    )


def test_add_axes_note_returns_text(
    sample_axes: Axes,
) -> None:
    text = add_axes_note(
        sample_axes,
        "Source: Example data",
    )

    assert isinstance(
        text,
        Text,
    )

    assert (
        text.get_text()
        == "Source: Example data"
    )


def test_add_axes_note_rejects_nonnumeric_position(
    sample_axes: Axes,
) -> None:
    with pytest.raises(
        TypeError,
        match="x_position must be numeric",
    ):
        add_axes_note(
            sample_axes,
            "Example",
            x_position="left",  # type: ignore[arg-type]
        )

# ------------------------------------------------------------------
# Figure source-note and watermark tests
# ------------------------------------------------------------------


def test_add_figure_source_note_returns_text() -> None:
    figure, _ = plt.subplots()

    text = add_figure_source_note(
        figure,
        "Source: Gold Value Predictor dataset",
    )

    assert isinstance(
        text,
        Text,
    )

    assert (
        text.get_text()
        == "Source: Gold Value Predictor dataset"
    )


def test_add_figure_source_note_uses_figure_text() -> None:
    figure, _ = plt.subplots()

    text = add_figure_source_note(
        figure,
        "Source: Example",
    )

    assert text in figure.texts


def test_add_figure_source_note_rejects_blank_text() -> None:
    figure, _ = plt.subplots()

    with pytest.raises(
        ValueError,
        match="source_note cannot be empty",
    ):
        add_figure_source_note(
            figure,
            "   ",
        )


@pytest.mark.parametrize(
    (
        "parameter_name",
        "arguments",
    ),
    [
        (
            "x_position",
            {
                "x_position": "left",
            },
        ),
        (
            "y_position",
            {
                "y_position": "bottom",
            },
        ),
    ],
)
def test_add_figure_source_note_rejects_invalid_positions(
    parameter_name: str,
    arguments: dict[str, object],
) -> None:
    figure, _ = plt.subplots()

    with pytest.raises(
        TypeError,
        match=(
            f"{parameter_name} must be numeric"
        ),
    ):
        add_figure_source_note(
            figure,
            "Source: Example",
            **arguments,  # type: ignore[arg-type]
        )


def test_add_watermark_returns_text(
    sample_axes: Axes,
) -> None:
    text = add_watermark(
        sample_axes,
        "Gold Value Predictor",
    )

    assert isinstance(
        text,
        Text,
    )

    assert (
        text.get_text()
        == "Gold Value Predictor"
    )


def test_add_watermark_uses_axes_coordinates(
    sample_axes: Axes,
) -> None:
    text = add_watermark(
        sample_axes,
        "Gold Value Predictor",
    )

    assert (
        text.get_transform()
        == sample_axes.transAxes
    )


def test_add_watermark_applies_requested_alpha(
    sample_axes: Axes,
) -> None:
    text = add_watermark(
        sample_axes,
        "Gold Value Predictor",
        alpha=0.25,
    )

    assert text.get_alpha() == pytest.approx(
        0.25
    )


def test_add_watermark_rejects_blank_text(
    sample_axes: Axes,
) -> None:
    with pytest.raises(
        ValueError,
        match="watermark cannot be empty",
    ):
        add_watermark(
            sample_axes,
            "   ",
        )


@pytest.mark.parametrize(
    "alpha",
    [
        -0.01,
        1.01,
    ],
)
def test_add_watermark_rejects_invalid_alpha(
    sample_axes: Axes,
    alpha: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="alpha must be between 0 and 1",
    ):
        add_watermark(
            sample_axes,
            "Gold Value Predictor",
            alpha=alpha,
        )


@pytest.mark.parametrize(
    (
        "parameter_name",
        "arguments",
    ),
    [
        (
            "x_position",
            {
                "x_position": "right",
            },
        ),
        (
            "y_position",
            {
                "y_position": "bottom",
            },
        ),
        (
            "alpha",
            {
                "alpha": "faint",
            },
        ),
    ],
)
def test_add_watermark_rejects_nonnumeric_options(
    sample_axes: Axes,
    parameter_name: str,
    arguments: dict[str, object],
) -> None:
    with pytest.raises(
        TypeError,
        match=(
            f"{parameter_name} must be numeric"
        ),
    ):
        add_watermark(
            sample_axes,
            "Gold Value Predictor",
            **arguments,  # type: ignore[arg-type]
        )