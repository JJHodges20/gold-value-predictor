from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from visualizations.export import (
    DEFAULT_EXPORT_DIRECTORY,
    DEFAULT_EXPORT_DPI,
    DEFAULT_EXPORT_FORMAT,
    SUPPORTED_EXPORT_FORMATS,
    build_export_path,
    save_figure,
    validate_close_after_export,
    validate_export_dpi,
    validate_export_format,
    validate_filename,
    validate_output_directory,
    validate_transparent,
)

from visualizations.export import (
    DEFAULT_EXPORT_DIRECTORY,
    DEFAULT_EXPORT_DPI,
    DEFAULT_EXPORT_FORMAT,
    SUPPORTED_EXPORT_FORMATS,
    build_export_path,
    export_figures,
    save_figure,
    validate_close_after_export,
    validate_export_dpi,
    validate_export_format,
    validate_filename,
    validate_output_directory,
    validate_transparent,
)

@pytest.fixture(autouse=True)
def close_figures_after_test():
    """
    Close all Matplotlib figures after every test.
    """

    yield

    plt.close("all")


@pytest.fixture
def sample_figure() -> Figure:
    """
    Create a small Matplotlib figure for export tests.
    """

    figure, axes = plt.subplots()

    axes.plot(
        [1, 2, 3],
        [2, 4, 3],
    )

    return figure


# ------------------------------------------------------------------
# Export configuration tests
# ------------------------------------------------------------------

def test_default_export_directory_is_path() -> None:
    assert isinstance(
        DEFAULT_EXPORT_DIRECTORY,
        Path,
    )


def test_default_export_format_is_supported() -> None:
    assert (
        DEFAULT_EXPORT_FORMAT
        in SUPPORTED_EXPORT_FORMATS
    )


def test_default_export_dpi_is_positive() -> None:
    assert DEFAULT_EXPORT_DPI > 0


# ------------------------------------------------------------------
# Output-directory validation tests
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "output_directory",
    [
        "reports/charts",
        Path("reports/charts"),
        "custom_exports",
    ],
)
def test_validate_output_directory_returns_path(
    output_directory: str | Path,
) -> None:
    result = validate_output_directory(
        output_directory
    )

    assert isinstance(result, Path)

    assert result == Path(
        output_directory
    )


@pytest.mark.parametrize(
    "output_directory",
    [
        "",
        "   ",
    ],
)
def test_validate_output_directory_rejects_empty_strings(
    output_directory: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="output_directory cannot be empty",
    ):
        validate_output_directory(
            output_directory
        )


@pytest.mark.parametrize(
    "output_directory",
    [
        None,
        123,
        4.5,
        True,
    ],
)
def test_validate_output_directory_rejects_invalid_types(
    output_directory: object,
) -> None:
    with pytest.raises(
        TypeError,
        match=(
            "output_directory must be "
            "a string or Path"
        ),
    ):
        validate_output_directory(  # type: ignore[arg-type]
            output_directory
        )


# ------------------------------------------------------------------
# Export-format validation tests
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    (
        "export_format",
        "expected",
    ),
    [
        ("png", "png"),
        (".png", "png"),
        ("PNG", "png"),
        (" PDF ", "pdf"),
        ("jpeg", "jpeg"),
        ("SVG", "svg"),
    ],
)
def test_validate_export_format_normalizes_valid_formats(
    export_format: str,
    expected: str,
) -> None:
    result = validate_export_format(
        export_format
    )

    assert result == expected


@pytest.mark.parametrize(
    "export_format",
    [
        "",
        "   ",
        ".",
    ],
)
def test_validate_export_format_rejects_empty_values(
    export_format: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="export_format cannot be empty",
    ):
        validate_export_format(
            export_format
        )


@pytest.mark.parametrize(
    "export_format",
    [
        "gif",
        "bmp",
        "docx",
        "txt",
    ],
)
def test_validate_export_format_rejects_unsupported_formats(
    export_format: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported export format",
    ):
        validate_export_format(
            export_format
        )


@pytest.mark.parametrize(
    "export_format",
    [
        None,
        123,
        True,
    ],
)
def test_validate_export_format_rejects_invalid_types(
    export_format: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="export_format must be a string",
    ):
        validate_export_format(  # type: ignore[arg-type]
            export_format
        )


# ------------------------------------------------------------------
# DPI validation tests
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "dpi",
    [
        1,
        72,
        150,
        300,
        600,
    ],
)
def test_validate_export_dpi_accepts_positive_integers(
    dpi: int,
) -> None:
    validate_export_dpi(dpi)


@pytest.mark.parametrize(
    "dpi",
    [
        0,
        -1,
        -300,
    ],
)
def test_validate_export_dpi_rejects_nonpositive_values(
    dpi: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="dpi must be at least 1",
    ):
        validate_export_dpi(dpi)


@pytest.mark.parametrize(
    "dpi",
    [
        300.0,
        "300",
        None,
        True,
    ],
)
def test_validate_export_dpi_rejects_nonintegers(
    dpi: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="dpi must be an integer",
    ):
        validate_export_dpi(  # type: ignore[arg-type]
            dpi
        )


# ------------------------------------------------------------------
# Filename validation tests
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    (
        "filename",
        "expected",
    ),
    [
        ("forecast", "forecast"),
        ("forecast.png", "forecast.png"),
        (
            Path("forecast_scenarios"),
            "forecast_scenarios",
        ),
        (
            "historical-price.svg",
            "historical-price.svg",
        ),
    ],
)
def test_validate_filename_accepts_valid_names(
    filename: str | Path,
    expected: str,
) -> None:
    result = validate_filename(
        filename
    )

    assert result == expected


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "   ",
    ],
)
def test_validate_filename_rejects_empty_values(
    filename: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="filename cannot be empty",
    ):
        validate_filename(filename)


@pytest.mark.parametrize(
    "filename",
    [
        "reports/chart.png",
        "reports\\chart.png",
        "../chart.png",
        "./nested/chart.png",
    ],
)
def test_validate_filename_rejects_directories(
    filename: str,
) -> None:
    with pytest.raises(ValueError):
        validate_filename(filename)


@pytest.mark.parametrize(
    "filename",
    [
        "chart?.png",
        "chart*.png",
        "chart|final.png",
        "chart:final.png",
        'chart"final.png',
        "chart<final>.png",
    ],
)
def test_validate_filename_rejects_invalid_characters(
    filename: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="invalid characters",
    ):
        validate_filename(filename)


@pytest.mark.parametrize(
    "filename",
    [
        "chart.",
        "chart ",
    ],
)
def test_validate_filename_rejects_invalid_endings(
    filename: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "filename must not end with "
            "a period or space"
        ),
    ):
        validate_filename(filename)


@pytest.mark.parametrize(
    "filename",
    [
        None,
        123,
        True,
    ],
)
def test_validate_filename_rejects_invalid_types(
    filename: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="filename must be a string or Path",
    ):
        validate_filename(  # type: ignore[arg-type]
            filename
        )


# ------------------------------------------------------------------
# Boolean-option validation tests
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "transparent",
    [
        True,
        False,
    ],
)
def test_validate_transparent_accepts_booleans(
    transparent: bool,
) -> None:
    validate_transparent(transparent)


@pytest.mark.parametrize(
    "transparent",
    [
        1,
        0,
        "true",
        None,
    ],
)
def test_validate_transparent_rejects_nonbooleans(
    transparent: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="transparent must be a Boolean",
    ):
        validate_transparent(  # type: ignore[arg-type]
            transparent
        )


@pytest.mark.parametrize(
    "close_after_export",
    [
        True,
        False,
    ],
)
def test_validate_close_after_export_accepts_booleans(
    close_after_export: bool,
) -> None:
    validate_close_after_export(
        close_after_export
    )


@pytest.mark.parametrize(
    "close_after_export",
    [
        1,
        0,
        "false",
        None,
    ],
)
def test_validate_close_after_export_rejects_nonbooleans(
    close_after_export: object,
) -> None:
    with pytest.raises(
        TypeError,
        match=(
            "close_after_export must be "
            "a Boolean"
        ),
    ):
        validate_close_after_export(  # type: ignore[arg-type]
            close_after_export
        )


# ------------------------------------------------------------------
# Export-path tests
# ------------------------------------------------------------------

def test_build_export_path_adds_missing_extension() -> None:
    result = build_export_path(
        filename="forecast",
        output_directory="reports/charts",
        export_format="png",
    )

    assert result == Path(
        "reports/charts/forecast.png"
    )


def test_build_export_path_preserves_matching_extension() -> None:
    result = build_export_path(
        filename="forecast.svg",
        output_directory="exports",
        export_format="svg",
    )

    assert result == Path(
        "exports/forecast.svg"
    )


def test_build_export_path_normalizes_format() -> None:
    result = build_export_path(
        filename="forecast",
        output_directory="exports",
        export_format=".PNG",
    )

    assert result == Path(
        "exports/forecast.png"
    )


def test_build_export_path_rejects_mismatched_extension() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Filename extension does not "
            "match export_format"
        ),
    ):
        build_export_path(
            filename="forecast.pdf",
            export_format="png",
        )


def test_build_export_path_does_not_create_directory(
    tmp_path: Path,
) -> None:
    output_directory = (
        tmp_path
        / "not_created"
        / "charts"
    )

    result = build_export_path(
        filename="forecast",
        output_directory=output_directory,
    )

    assert result == (
        output_directory
        / "forecast.png"
    )

    assert not output_directory.exists()


# ------------------------------------------------------------------
# Figure-saving tests
# ------------------------------------------------------------------

def test_save_figure_creates_file(
    sample_figure: Figure,
    tmp_path: Path,
) -> None:
    result = save_figure(
        figure=sample_figure,
        filename="sample_chart",
        output_directory=tmp_path,
    )

    assert result.exists()
    assert result.is_file()
    assert result.suffix == ".png"


def test_save_figure_creates_nested_directories(
    sample_figure: Figure,
    tmp_path: Path,
) -> None:
    output_directory = (
        tmp_path
        / "reports"
        / "charts"
    )

    result = save_figure(
        figure=sample_figure,
        filename="nested_chart",
        output_directory=output_directory,
    )

    assert output_directory.exists()
    assert result.exists()


def test_save_figure_returns_expected_path(
    sample_figure: Figure,
    tmp_path: Path,
) -> None:
    result = save_figure(
        figure=sample_figure,
        filename="forecast.svg",
        output_directory=tmp_path,
        export_format="svg",
    )

    assert result == (
        tmp_path
        / "forecast.svg"
    )


def test_save_figure_preserves_open_figure_by_default(
    sample_figure: Figure,
    tmp_path: Path,
) -> None:
    figure_number = sample_figure.number

    save_figure(
        figure=sample_figure,
        filename="open_chart",
        output_directory=tmp_path,
    )

    assert plt.fignum_exists(
        figure_number
    )


def test_save_figure_closes_figure_when_requested(
    sample_figure: Figure,
    tmp_path: Path,
) -> None:
    figure_number = sample_figure.number

    save_figure(
        figure=sample_figure,
        filename="closed_chart",
        output_directory=tmp_path,
        close_after_export=True,
    )

    assert not plt.fignum_exists(
        figure_number
    )


def test_save_figure_accepts_transparent_export(
    sample_figure: Figure,
    tmp_path: Path,
) -> None:
    result = save_figure(
        figure=sample_figure,
        filename="transparent_chart",
        output_directory=tmp_path,
        transparent=True,
    )

    assert result.exists()


def test_save_figure_accepts_custom_dpi(
    sample_figure: Figure,
    tmp_path: Path,
) -> None:
    result = save_figure(
        figure=sample_figure,
        filename="custom_dpi_chart",
        output_directory=tmp_path,
        dpi=150,
    )

    assert result.exists()


def test_save_figure_rejects_nonfigure(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match=(
            "figure must be a "
            "matplotlib Figure"
        ),
    ):
        save_figure(
            figure="not a figure",  # type: ignore[arg-type]
            filename="chart",
            output_directory=tmp_path,
        )


def test_save_figure_does_not_create_file_when_validation_fails(
    sample_figure: Figure,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        save_figure(
            figure=sample_figure,
            filename="chart.pdf",
            output_directory=tmp_path,
            export_format="png",
        )

    assert not any(
        tmp_path.iterdir()
    )

# ------------------------------------------------------------------
# Batch figure-export tests
# ------------------------------------------------------------------

def test_export_figures_exports_multiple_files(
    tmp_path: Path,
) -> None:
    first_figure, first_axes = plt.subplots()
    second_figure, second_axes = plt.subplots()

    first_axes.plot(
        [1, 2, 3],
        [2, 4, 3],
    )

    second_axes.plot(
        [1, 2, 3],
        [5, 3, 6],
    )

    result = export_figures(
        output_directory=tmp_path,
        historical=first_figure,
        forecast=second_figure,
    )

    assert set(result) == {
        "historical",
        "forecast",
    }

    assert result[
        "historical"
    ] == tmp_path / "historical.png"

    assert result[
        "forecast"
    ] == tmp_path / "forecast.png"

    assert result[
        "historical"
    ].exists()

    assert result[
        "forecast"
    ].exists()


def test_export_figures_returns_path_for_each_figure(
    tmp_path: Path,
) -> None:
    first_figure, _ = plt.subplots()
    second_figure, _ = plt.subplots()
    third_figure, _ = plt.subplots()

    result = export_figures(
        output_directory=tmp_path,
        first=first_figure,
        second=second_figure,
        third=third_figure,
    )

    assert len(result) == 3

    assert all(
        isinstance(path, Path)
        for path in result.values()
    )


def test_export_figures_applies_custom_format(
    tmp_path: Path,
) -> None:
    first_figure, _ = plt.subplots()
    second_figure, _ = plt.subplots()

    result = export_figures(
        output_directory=tmp_path,
        export_format="svg",
        historical=first_figure,
        forecast=second_figure,
    )

    assert result[
        "historical"
    ].suffix == ".svg"

    assert result[
        "forecast"
    ].suffix == ".svg"

    assert result[
        "historical"
    ].exists()

    assert result[
        "forecast"
    ].exists()


def test_export_figures_creates_output_directory(
    tmp_path: Path,
) -> None:
    figure, _ = plt.subplots()

    output_directory = (
        tmp_path
        / "reports"
        / "charts"
    )

    result = export_figures(
        output_directory=output_directory,
        forecast=figure,
    )

    assert output_directory.exists()

    assert result[
        "forecast"
    ].exists()


def test_export_figures_preserves_figures_by_default(
    tmp_path: Path,
) -> None:
    first_figure, _ = plt.subplots()
    second_figure, _ = plt.subplots()

    first_number = first_figure.number
    second_number = second_figure.number

    export_figures(
        output_directory=tmp_path,
        first=first_figure,
        second=second_figure,
    )

    assert plt.fignum_exists(
        first_number
    )

    assert plt.fignum_exists(
        second_number
    )


def test_export_figures_closes_all_figures_when_requested(
    tmp_path: Path,
) -> None:
    first_figure, _ = plt.subplots()
    second_figure, _ = plt.subplots()

    first_number = first_figure.number
    second_number = second_figure.number

    export_figures(
        output_directory=tmp_path,
        close_after_export=True,
        first=first_figure,
        second=second_figure,
    )

    assert not plt.fignum_exists(
        first_number
    )

    assert not plt.fignum_exists(
        second_number
    )


def test_export_figures_accepts_transparent_setting(
    tmp_path: Path,
) -> None:
    figure, _ = plt.subplots()

    result = export_figures(
        output_directory=tmp_path,
        transparent=True,
        forecast=figure,
    )

    assert result[
        "forecast"
    ].exists()


def test_export_figures_accepts_custom_dpi(
    tmp_path: Path,
) -> None:
    figure, _ = plt.subplots()

    result = export_figures(
        output_directory=tmp_path,
        dpi=150,
        forecast=figure,
    )

    assert result[
        "forecast"
    ].exists()


def test_export_figures_rejects_empty_collection(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "At least one figure must "
            "be provided"
        ),
    ):
        export_figures(
            output_directory=tmp_path,
        )


def test_export_figures_rejects_nonfigure_value(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match=(
            "Figure 'forecast' must be "
            "a matplotlib Figure"
        ),
    ):
        export_figures(
            output_directory=tmp_path,
            forecast="not a figure",  # type: ignore[arg-type]
        )


def test_export_figures_rejects_empty_figure_name(
    tmp_path: Path,
) -> None:
    figure, _ = plt.subplots()

    with pytest.raises(
        ValueError,
        match="Figure names cannot be empty",
    ):
        export_figures(
            output_directory=tmp_path,
            **{
                "   ": figure,
            },
        )


def test_export_figures_stops_when_validation_fails(
    tmp_path: Path,
) -> None:
    valid_figure, _ = plt.subplots()

    with pytest.raises(TypeError):
        export_figures(
            output_directory=tmp_path,
            valid=valid_figure,
            invalid="not a figure",  # type: ignore[arg-type]
        )

def test_export_figures_does_not_write_partial_batch(
    tmp_path: Path,
) -> None:
    valid_figure, _ = plt.subplots()

    with pytest.raises(TypeError):
        export_figures(
            output_directory=tmp_path,
            valid=valid_figure,
            invalid="not a figure",  # type: ignore[arg-type]
        )

    assert not any(
        tmp_path.iterdir()
    )