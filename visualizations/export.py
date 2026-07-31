"""
Configuration and validation utilities for chart exports.

This module owns chart-export settings and file validation.
It does not generate analytics or render charts.
"""

from pathlib import Path
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Export configuration
# ------------------------------------------------------------------

DEFAULT_EXPORT_DIRECTORY = Path(
    "reports/charts"
)

DEFAULT_EXPORT_FORMAT = "png"

DEFAULT_EXPORT_DPI = 300

DEFAULT_EXPORT_BBOX = "tight"

DEFAULT_EXPORT_TRANSPARENT = False

DEFAULT_CLOSE_AFTER_EXPORT = False

SUPPORTED_EXPORT_FORMATS = frozenset(
    {
        "png",
        "jpg",
        "jpeg",
        "pdf",
        "svg",
    }
)

INVALID_FILENAME_CHARACTERS = frozenset(
    '<>:"/\\|?*'
)


# ------------------------------------------------------------------
# Export validation helpers
# ------------------------------------------------------------------

def validate_output_directory(
    output_directory: str | Path,
) -> Path:
    """
    Validate and normalize an export directory.

    This function does not create the directory. Directory
    creation is handled by the saving function in the next
    export section.

    Args:
        output_directory:
            Directory where exported files will be saved.

    Returns:
        The normalized directory as a Path object.

    Raises:
        TypeError:
            If output_directory is not a string or Path.

        ValueError:
            If output_directory is an empty string.
    """

    if not isinstance(
        output_directory,
        (str, Path),
    ):
        raise TypeError(
            "output_directory must be a string or Path."
        )

    if (
        isinstance(output_directory, str)
        and not output_directory.strip()
    ):
        raise ValueError(
            "output_directory cannot be empty."
        )

    return Path(output_directory)


def validate_export_format(
    export_format: str,
) -> str:
    """
    Validate and normalize a chart export format.

    A leading period is accepted, so both "png" and ".png"
    are valid inputs.

    Args:
        export_format:
            Requested image or document format.

    Returns:
        The normalized lowercase format without a leading
        period.

    Raises:
        TypeError:
            If export_format is not a string.

        ValueError:
            If export_format is empty or unsupported.
    """

    if not isinstance(export_format, str):
        raise TypeError(
            "export_format must be a string."
        )

    normalized_format = (
        export_format
        .strip()
        .lower()
        .lstrip(".")
    )

    if not normalized_format:
        raise ValueError(
            "export_format cannot be empty."
        )

    if (
        normalized_format
        not in SUPPORTED_EXPORT_FORMATS
    ):
        supported_text = ", ".join(
            sorted(SUPPORTED_EXPORT_FORMATS)
        )

        raise ValueError(
            f"Unsupported export format: "
            f"{export_format!r}. "
            f"Supported formats are: "
            f"{supported_text}."
        )

    return normalized_format


def validate_export_dpi(
    dpi: int,
) -> None:
    """
    Validate an export resolution.

    Args:
        dpi:
            Export resolution in dots per inch.

    Raises:
        TypeError:
            If dpi is not an integer.

        ValueError:
            If dpi is less than one.
    """

    if (
        isinstance(dpi, bool)
        or not isinstance(dpi, int)
    ):
        raise TypeError(
            "dpi must be an integer."
        )

    if dpi < 1:
        raise ValueError(
            "dpi must be at least 1."
        )


def validate_filename(
    filename: str | Path,
) -> str:
    """
    Validate an export filename.

    Filenames must contain only a filename, not an absolute
    path or nested directory path. The output directory is
    supplied separately.

    Args:
        filename:
            Desired export filename, with or without a file
            extension.

    Returns:
        The validated filename as a string.

    Raises:
        TypeError:
            If filename is not a string or Path.

        ValueError:
            If filename is empty, contains a directory path,
            uses invalid filename characters, or ends with a
            period or space.
    """

    if not isinstance(
        filename,
        (str, Path),
    ):
        raise TypeError(
            "filename must be a string or Path."
        )

    cleaned_filename = str(filename)

    if not cleaned_filename.strip():
        raise ValueError(
            "filename cannot be empty."
        )

    if cleaned_filename.endswith(
        (".", " ")
    ):
        raise ValueError(
            "filename must not end with a period or space."
        )

    filename_path = Path(
        cleaned_filename
    )

    if filename_path.is_absolute():
        raise ValueError(
            "filename must not be an absolute path."
        )

    if (
        "/" in cleaned_filename
        or "\\" in cleaned_filename
        or filename_path.parent != Path(".")
    ):
        raise ValueError(
            "filename must not contain directories."
        )

    invalid_characters_found = {
        character
        for character in cleaned_filename
        if character in INVALID_FILENAME_CHARACTERS
    }

    if invalid_characters_found:
        invalid_text = "".join(
            sorted(invalid_characters_found)
        )

        raise ValueError(
            "filename contains invalid characters: "
            f"{invalid_text!r}."
        )

    if cleaned_filename in {
        ".",
        "..",
    }:
        raise ValueError(
            "filename must contain a valid file name."
        )

    return cleaned_filename


def validate_transparent(
    transparent: bool,
) -> None:
    """
    Validate the transparent-background option.

    Args:
        transparent:
            Whether the exported figure should use a
            transparent background.

    Raises:
        TypeError:
            If transparent is not a Boolean.
    """

    if not isinstance(
        transparent,
        bool,
    ):
        raise TypeError(
            "transparent must be a Boolean."
        )


def validate_close_after_export(
    close_after_export: bool,
) -> None:
    """
    Validate the figure-closing option.

    Args:
        close_after_export:
            Whether the figure should be closed after it is
            saved.

    Raises:
        TypeError:
            If close_after_export is not a Boolean.
    """

    if not isinstance(
        close_after_export,
        bool,
    ):
        raise TypeError(
            "close_after_export must be a Boolean."
        )

def build_export_path(
    filename: str | Path,
    output_directory: str | Path = DEFAULT_EXPORT_DIRECTORY,
    export_format: str = DEFAULT_EXPORT_FORMAT,
) -> Path:
    """
    Build the complete destination path for an exported figure.

    If the filename does not include an extension, the requested
    export format is appended automatically.

    Args:
        filename:
            Desired filename.

        output_directory:
            Destination directory.

        export_format:
            Desired export format.

    Returns:
        Complete export path.

    Raises:
        ValueError:
            If the filename already contains a different extension.
    """

    directory = validate_output_directory(
        output_directory
    )

    cleaned_filename = validate_filename(
        filename
    )

    normalized_format = validate_export_format(
        export_format
    )

    path = Path(cleaned_filename)

    if path.suffix:
        existing_extension = (
            path.suffix.lower().lstrip(".")
        )

        if existing_extension != normalized_format:
            raise ValueError(
                "Filename extension does not match "
                "export_format."
            )
    else:
        path = path.with_suffix(
            f".{normalized_format}"
        )

    return directory / path

def save_figure(
    figure: Figure,
    filename: str | Path,
    output_directory: str | Path = DEFAULT_EXPORT_DIRECTORY,
    export_format: str = DEFAULT_EXPORT_FORMAT,
    dpi: int = DEFAULT_EXPORT_DPI,
    transparent: bool = DEFAULT_EXPORT_TRANSPARENT,
    close_after_export: bool = DEFAULT_CLOSE_AFTER_EXPORT,
) -> Path:
    """
    Save a Matplotlib figure to disk.

    Args:
        figure:
            Figure to export.

        filename:
            Desired filename.

        output_directory:
            Export directory.

        export_format:
            Desired file format.

        dpi:
            Export resolution.

        transparent:
            Export with transparent background.

        close_after_export:
            Close the figure after saving.

    Returns:
        Path to the exported file.
    """

    if not isinstance(
        figure,
        Figure,
    ):
        raise TypeError(
            "figure must be a matplotlib Figure."
        )

    validate_export_dpi(dpi)

    validate_transparent(
        transparent
    )

    validate_close_after_export(
        close_after_export
    )

    export_path = build_export_path(
        filename=filename,
        output_directory=output_directory,
        export_format=export_format,
    )

    export_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        export_path,
        dpi=dpi,
        bbox_inches=DEFAULT_EXPORT_BBOX,
        transparent=transparent,
    )

    if close_after_export:
        plt.close(figure)

    return export_path

def export_figures(
    *,
    output_directory: str | Path = DEFAULT_EXPORT_DIRECTORY,
    export_format: str = DEFAULT_EXPORT_FORMAT,
    dpi: int = DEFAULT_EXPORT_DPI,
    transparent: bool = DEFAULT_EXPORT_TRANSPARENT,
    close_after_export: bool = DEFAULT_CLOSE_AFTER_EXPORT,
    **figures: Figure,
) -> dict[str, Path]:
    """
    Export multiple Matplotlib figures using shared settings.

    Each keyword argument name becomes the exported filename.

    Args:
        output_directory:
            Directory where the figures will be saved.

        export_format:
            File format applied to every exported figure.

        dpi:
            Export resolution in dots per inch.

        transparent:
            Whether exported figures use transparent
            backgrounds.

        close_after_export:
            Whether each figure is closed after it is saved.

        **figures:
            Named Matplotlib figures to export.

    Returns:
        A dictionary mapping each figure name to its exported
        file path.

    Raises:
        ValueError:
            If no figures are provided or a figure name is
            invalid.

        TypeError:
            If a supplied value is not a Matplotlib Figure.
    """

    if not figures:
        raise ValueError(
            "At least one figure must be provided."
        )

    directory = validate_output_directory(
        output_directory
    )

    normalized_format = validate_export_format(
        export_format
    )

    validate_export_dpi(dpi)

    validate_transparent(
        transparent
    )

    validate_close_after_export(
        close_after_export
    )

    validated_figures: dict[str, Figure] = {}

    for figure_name, figure in figures.items():
        if not figure_name.strip():
            raise ValueError(
                "Figure names cannot be empty."
            )

        validate_filename(
            figure_name
        )

        if not isinstance(
            figure,
            Figure,
        ):
            raise TypeError(
                f"Figure {figure_name!r} must be "
                "a matplotlib Figure."
            )

        validated_figures[
            figure_name
        ] = figure

    exported_paths: dict[str, Path] = {}

    for figure_name, figure in (
        validated_figures.items()
    ):
        exported_path = save_figure(
            figure=figure,
            filename=figure_name,
            output_directory=directory,
            export_format=normalized_format,
            dpi=dpi,
            transparent=transparent,
            close_after_export=close_after_export,
        )

        exported_paths[
            figure_name
        ] = exported_path

    return exported_paths


def main() -> None:
    """
    Preview figure exporting.
    """

    figure = plt.figure()

    axes = figure.add_subplot(111)

    axes.plot(
        [1, 2, 3],
        [2, 4, 3],
    )

    export_path = save_figure(
        figure=figure,
        filename="preview_chart",
    )

    print(
        f"Exported successfully to:\n{export_path}"
    )

if __name__ == "__main__":
    main()    