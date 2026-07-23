from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# File locations
# --------------------------------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent
CSV_FILE = PROJECT_FOLDER / "gold_price.csv"
CHARTS_FOLDER = PROJECT_FOLDER / "charts"


# --------------------------------------------------
# Data loading and preparation
# --------------------------------------------------

def load_gold_data(file_path: Path) -> pd.DataFrame:
    """
    Load, validate, clean, and prepare monthly gold-price data.

    Expected CSV format:

    Date,Price
    1833-1,18.930
    1833-2,18.930
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path.name}. "
            "Make sure gold_price.csv is in the same folder "
            "as gold_analyzer.py."
        )

    data = pd.read_csv(file_path)

    required_columns = {"Date", "Price"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "The CSV must contain columns named Date and Price."
        )

    data["Date"] = pd.to_datetime(
        data["Date"],
        format="%Y-%m",
        errors="coerce",
    )

    data["Price"] = pd.to_numeric(
        data["Price"],
        errors="coerce",
    )

    data = data.dropna(subset=["Date", "Price"])

    data = data[data["Price"] > 0]

    data = (
        data
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    if data.empty:
        raise ValueError(
            "The CSV does not contain any usable records."
        )

    if len(data) < 2:
        raise ValueError(
            "The CSV must contain at least two usable records."
        )

    data["Monthly Change %"] = (
        data["Price"].pct_change() * 100
    )

    # A 12-month rolling average smooths short-term price movement.
    data["12-Month Rolling Average"] = (
        data["Price"].rolling(window=12).mean()
    )

    return data


# --------------------------------------------------
# Analysis calculations
# --------------------------------------------------

def calculate_summary(data: pd.DataFrame) -> dict:
    """
    Calculate summary measurements for any supplied date range.
    """

    if len(data) < 2:
        raise ValueError(
            "At least two records are needed for this analysis."
        )

    first_row = data.iloc[0]
    latest_row = data.iloc[-1]

    starting_date = first_row["Date"]
    latest_date = latest_row["Date"]

    starting_price = float(first_row["Price"])
    latest_price = float(latest_row["Price"])

    total_increase = latest_price - starting_price

    total_percentage_increase = (
        total_increase / starting_price
    ) * 100

    average_monthly_change = float(
        data["Monthly Change %"].mean()
    )

    number_of_years = (
        latest_date - starting_date
    ).days / 365.25

    if number_of_years > 0:
        cagr = (
            (latest_price / starting_price)
            ** (1 / number_of_years)
            - 1
        ) * 100
    else:
        cagr = 0.0

    highest_price_row = data.loc[
        data["Price"].idxmax()
    ]

    lowest_price_row = data.loc[
        data["Price"].idxmin()
    ]

    valid_changes = data.dropna(
        subset=["Monthly Change %"]
    )

    if valid_changes.empty:
        raise ValueError(
            "Monthly percentage changes could not be calculated."
        )

    best_month_row = valid_changes.loc[
        valid_changes["Monthly Change %"].idxmax()
    ]

    worst_month_row = valid_changes.loc[
        valid_changes["Monthly Change %"].idxmin()
    ]

    return {
        "record_count": len(data),
        "starting_date": starting_date,
        "starting_price": starting_price,
        "latest_date": latest_date,
        "latest_price": latest_price,
        "total_increase": total_increase,
        "total_percentage_increase":
            total_percentage_increase,
        "average_monthly_change":
            average_monthly_change,
        "number_of_years": number_of_years,
        "cagr": cagr,
        "highest_price_date":
            highest_price_row["Date"],
        "highest_price":
            float(highest_price_row["Price"]),
        "lowest_price_date":
            lowest_price_row["Date"],
        "lowest_price":
            float(lowest_price_row["Price"]),
        "best_month_date":
            best_month_row["Date"],
        "best_month_change":
            float(best_month_row["Monthly Change %"]),
        "worst_month_date":
            worst_month_row["Date"],
        "worst_month_change":
            float(worst_month_row["Monthly Change %"]),
    }


def calculate_full_summary(data: pd.DataFrame) -> dict:
    """
    Calculate the complete summary, including the recent
    five-year monthly average.
    """

    summary = calculate_summary(data)

    latest_date = data.iloc[-1]["Date"]
    five_year_cutoff = latest_date - pd.DateOffset(
        years=5
    )

    last_five_years = data[
        data["Date"] >= five_year_cutoff
    ]

    summary["average_five_year_monthly_change"] = float(
        last_five_years["Monthly Change %"].mean()
    )

    return summary


# --------------------------------------------------
# Display helpers
# --------------------------------------------------

def print_dashboard_row(
    label: str,
    value: str,
    label_width: int = 36,
) -> None:
    """
    Print one aligned dashboard row.
    """

    print(f"{label:<{label_width}}{value}")


def display_summary(
    summary: dict,
    title: str = "GOLD PRICE HISTORICAL ANALYSIS",
    show_five_year_average: bool = True,
) -> None:
    """
    Display analysis results as a terminal dashboard.
    """

    width = 74
    divider = "=" * width
    section_divider = "-" * width

    print("\n" + divider)
    print(title.center(width))
    print(divider)

    print("\nDATASET INFORMATION")
    print(section_divider)

    print_dashboard_row(
        "Starting record",
        f"{summary['starting_date']:%B %Y}",
    )

    print_dashboard_row(
        "Latest record",
        f"{summary['latest_date']:%B %Y}",
    )

    print_dashboard_row(
        "Years represented",
        f"{summary['number_of_years']:.1f}",
    )

    print_dashboard_row(
        "Monthly records",
        f"{summary['record_count']:,}",
    )

    print("\nGROWTH MEASUREMENTS")
    print(section_divider)

    print_dashboard_row(
        "Starting price",
        f"${summary['starting_price']:,.2f}",
    )

    print_dashboard_row(
        "Ending price",
        f"${summary['latest_price']:,.2f}",
    )

    print_dashboard_row(
        "Dollar change",
        f"${summary['total_increase']:,.2f}",
    )

    print_dashboard_row(
        "Total percentage change",
        f"{summary['total_percentage_increase']:+,.2f}%",
    )

    print_dashboard_row(
        "Average monthly percentage change",
        f"{summary['average_monthly_change']:+.4f}%",
    )

    if (
        show_five_year_average
        and "average_five_year_monthly_change" in summary
    ):
        print_dashboard_row(
            "Last 5-year monthly average",
            (
                f"{summary[
                    'average_five_year_monthly_change'
                ]:+.4f}%"
            ),
        )

    print_dashboard_row(
        "Compound annual growth (CAGR)",
        f"{summary['cagr']:+.4f}%",
    )

    print("\nHISTORICAL RECORDS")
    print(section_divider)

    print_dashboard_row(
        "Highest price",
        (
            f"${summary['highest_price']:,.2f} "
            f"({summary['highest_price_date']:%B %Y})"
        ),
    )

    print_dashboard_row(
        "Lowest price",
        (
            f"${summary['lowest_price']:,.2f} "
            f"({summary['lowest_price_date']:%B %Y})"
        ),
    )

    print("\nMONTHLY PERFORMANCE")
    print(section_divider)

    print_dashboard_row(
        "Best-performing month",
        (
            f"{summary['best_month_date']:%B %Y} "
            f"({summary['best_month_change']:+.2f}%)"
        ),
    )

    print_dashboard_row(
        "Worst-performing month",
        (
            f"{summary['worst_month_date']:%B %Y} "
            f"({summary['worst_month_change']:+.2f}%)"
        ),
    )

    print("\n" + divider)


def pause_program() -> None:
    """
    Pause before returning to a menu.
    """

    input("\nPress Enter to return to the menu...")


# --------------------------------------------------
# Date input and filtering
# --------------------------------------------------

def get_month_input(prompt: str) -> pd.Timestamp | None:
    """
    Ask the user for a month in YYYY-MM format.

    Entering Q cancels the operation.
    """

    while True:
        user_input = input(prompt).strip()

        if user_input.lower() == "q":
            return None

        try:
            date = pd.to_datetime(
                user_input,
                format="%Y-%m",
                errors="raise",
            )

            return date

        except ValueError:
            print(
                "Invalid date. Use YYYY-MM, such as 2020-06."
            )


def get_date_range(
    data: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Ask for a start and end month and return filtered data.
    """

    earliest_date = data.iloc[0]["Date"]
    latest_date = data.iloc[-1]["Date"]

    print(
        f"\nAvailable dates: "
        f"{earliest_date:%Y-%m} through "
        f"{latest_date:%Y-%m}"
    )

    print("Enter Q at any prompt to cancel.")

    start_date = get_month_input(
        "\nEnter the starting month (YYYY-MM): "
    )

    if start_date is None:
        return None

    end_date = get_month_input(
        "Enter the ending month (YYYY-MM): "
    )

    if end_date is None:
        return None

    if start_date > end_date:
        print(
            "\nThe starting date cannot be after "
            "the ending date."
        )
        return None

    filtered_data = data[
        (data["Date"] >= start_date)
        & (data["Date"] <= end_date)
    ].copy()

    if len(filtered_data) < 2:
        print(
            "\nThat range does not contain enough records. "
            "Choose a larger date range."
        )
        return None

    # Recalculate the first monthly change so it does not
    # depend on a record outside the selected range.
    filtered_data["Monthly Change %"] = (
        filtered_data["Price"].pct_change() * 100
    )

    filtered_data["12-Month Rolling Average"] = (
        filtered_data["Price"].rolling(window=12).mean()
    )

    return filtered_data


# --------------------------------------------------
# Menu option 1: Complete summary
# --------------------------------------------------

def view_complete_summary(
    data: pd.DataFrame,
) -> None:
    """
    Display the complete historical summary.
    """

    summary = calculate_full_summary(data)

    display_summary(summary)

    pause_program()


# --------------------------------------------------
# Menu option 2: Date-range analysis
# --------------------------------------------------

def analyze_date_range(
    data: pd.DataFrame,
) -> None:
    """
    Analyze a user-selected date range.
    """

    print("\n" + "=" * 74)
    print("ANALYZE A DATE RANGE".center(74))
    print("=" * 74)

    filtered_data = get_date_range(data)

    if filtered_data is None:
        return

    summary = calculate_summary(filtered_data)

    title = (
        f"GOLD ANALYSIS: "
        f"{summary['starting_date']:%B %Y} TO "
        f"{summary['latest_date']:%B %Y}"
    )

    display_summary(
        summary,
        title=title,
        show_five_year_average=False,
    )

    chart_answer = input(
        "\nWould you like to chart this date range? "
        "(yes/no): "
    ).strip().lower()

    if chart_answer in {"yes", "y"}:
        show_date_range_chart(filtered_data)

    pause_program()


# --------------------------------------------------
# Menu option 3: Compare two dates
# --------------------------------------------------

def find_exact_record(
    data: pd.DataFrame,
    selected_date: pd.Timestamp,
) -> pd.Series | None:
    """
    Find a record for an exact year and month.
    """

    matching_rows = data[
        data["Date"] == selected_date
    ]

    if matching_rows.empty:
        return None

    return matching_rows.iloc[0]


def compare_two_dates(
    data: pd.DataFrame,
) -> None:
    """
    Compare gold prices from two selected months.
    """

    print("\n" + "=" * 74)
    print("COMPARE TWO DATES".center(74))
    print("=" * 74)

    earliest_date = data.iloc[0]["Date"]
    latest_date = data.iloc[-1]["Date"]

    print(
        f"\nAvailable dates: "
        f"{earliest_date:%Y-%m} through "
        f"{latest_date:%Y-%m}"
    )

    print("Enter Q at any prompt to cancel.")

    first_date = get_month_input(
        "\nEnter the first month (YYYY-MM): "
    )

    if first_date is None:
        return

    second_date = get_month_input(
        "Enter the second month (YYYY-MM): "
    )

    if second_date is None:
        return

    first_record = find_exact_record(
        data,
        first_date,
    )

    second_record = find_exact_record(
        data,
        second_date,
    )

    if first_record is None:
        print(
            f"\nNo record was found for "
            f"{first_date:%Y-%m}."
        )
        pause_program()
        return

    if second_record is None:
        print(
            f"\nNo record was found for "
            f"{second_date:%Y-%m}."
        )
        pause_program()
        return

    first_price = float(first_record["Price"])
    second_price = float(second_record["Price"])

    dollar_change = second_price - first_price

    percentage_change = (
        dollar_change / first_price
    ) * 100

    elapsed_years = (
        second_date - first_date
    ).days / 365.25

    print("\n" + "-" * 74)

    print_dashboard_row(
        "First date",
        f"{first_date:%B %Y}",
    )

    print_dashboard_row(
        "First price",
        f"${first_price:,.2f}",
    )

    print_dashboard_row(
        "Second date",
        f"{second_date:%B %Y}",
    )

    print_dashboard_row(
        "Second price",
        f"${second_price:,.2f}",
    )

    print_dashboard_row(
        "Dollar change",
        f"${dollar_change:+,.2f}",
    )

    print_dashboard_row(
        "Percentage change",
        f"{percentage_change:+,.2f}%",
    )

    print_dashboard_row(
        "Time between dates",
        f"{abs(elapsed_years):.1f} years",
    )

    print("-" * 74)

    pause_program()


# --------------------------------------------------
# Menu option 4: Projection tool
# --------------------------------------------------

def estimate_future_price(
    current_price: float,
    monthly_growth_rate: float,
    number_of_months: int,
) -> float:
    """
    Estimate a future price using compound monthly growth.
    """

    decimal_rate = monthly_growth_rate / 100

    return current_price * (
        1 + decimal_rate
    ) ** number_of_months


def get_projection_rate(
    data: pd.DataFrame,
    choice: str,
) -> tuple[float, str]:
    """
    Return a selected historical monthly growth rate.
    """

    latest_date = data.iloc[-1]["Date"]

    if choice == "1":
        selected_data = data
        description = "Full historical monthly average"

    elif choice == "2":
        cutoff_date = latest_date - pd.DateOffset(
            years=5
        )

        selected_data = data[
            data["Date"] >= cutoff_date
        ]

        description = "Last five years' monthly average"

    elif choice == "3":
        cutoff_date = latest_date - pd.DateOffset(
            years=10
        )

        selected_data = data[
            data["Date"] >= cutoff_date
        ]

        description = "Last ten years' monthly average"

    else:
        raise ValueError("Invalid projection selection.")

    monthly_rate = float(
        selected_data["Monthly Change %"].mean()
    )

    return monthly_rate, description


def project_future_value(
    data: pd.DataFrame,
) -> None:
    """
    Run the simple future-value projection.
    """

    print("\n" + "=" * 74)
    print("SIMPLE GOLD PRICE PROJECTION".center(74))
    print("=" * 74)

    print(
        "\nThis is a mathematical projection based on "
        "historical averages."
    )

    print(
        "It is an educational estimate, not financial advice."
    )

    print("\nChoose a growth-rate method:")
    print("1. Full historical monthly average")
    print("2. Last five years' monthly average")
    print("3. Last ten years' monthly average")
    print("4. Return to main menu")

    choice = input("\nEnter your selection: ").strip()

    if choice == "4":
        return

    try:
        monthly_rate, description = get_projection_rate(
            data,
            choice,
        )

    except ValueError:
        print("\nInvalid selection.")
        pause_program()
        return

    try:
        years = float(
            input(
                "\nHow many years would you like "
                "to project? "
            )
        )

    except ValueError:
        print("\nPlease enter a valid number.")
        pause_program()
        return

    if years <= 0:
        print(
            "\nThe projection period must be "
            "greater than zero."
        )
        pause_program()
        return

    latest_date = data.iloc[-1]["Date"]
    latest_price = float(
        data.iloc[-1]["Price"]
    )

    number_of_months = round(years * 12)

    projected_price = estimate_future_price(
        current_price=latest_price,
        monthly_growth_rate=monthly_rate,
        number_of_months=number_of_months,
    )

    projected_date = latest_date + pd.DateOffset(
        months=number_of_months
    )

    print("\n" + "-" * 74)

    print_dashboard_row(
        "Latest recorded date",
        f"{latest_date:%B %Y}",
    )

    print_dashboard_row(
        "Latest recorded price",
        f"${latest_price:,.2f}",
    )

    print_dashboard_row(
        "Projection method",
        description,
    )

    print_dashboard_row(
        "Monthly growth rate",
        f"{monthly_rate:+.4f}%",
    )

    print_dashboard_row(
        "Projection period",
        f"{years:g} years",
    )

    print_dashboard_row(
        "Projected date",
        f"{projected_date:%B %Y}",
    )

    print_dashboard_row(
        "Estimated future price",
        f"${projected_price:,.2f}",
    )

    print("-" * 74)

    pause_program()


# --------------------------------------------------
# Chart helpers
# --------------------------------------------------

def prepare_chart(
    title: str,
    x_label: str,
    y_label: str,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create one consistently formatted chart.
    """

    figure, axes = plt.subplots(
        figsize=(12, 6)
    )

    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.grid(True, alpha=0.3)

    return figure, axes


def finish_chart(
    figure: plt.Figure,
    file_name: str,
) -> None:
    """
    Save a chart and display it.
    """

    CHARTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = CHARTS_FOLDER / file_name

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(
        f"\nChart saved to:\n{output_path}"
    )

    plt.show()

    plt.close(figure)


# --------------------------------------------------
# Individual charts
# --------------------------------------------------

def show_full_history_chart(
    data: pd.DataFrame,
) -> None:
    """
    Chart all available historical gold prices.
    """

    figure, axes = prepare_chart(
        title="Historical Gold Prices",
        x_label="Date",
        y_label="Gold Price",
    )

    axes.plot(
        data["Date"],
        data["Price"],
        label="Gold price",
    )

    axes.legend()

    finish_chart(
        figure,
        "full_historical_price_chart.png",
    )


def show_last_five_years_chart(
    data: pd.DataFrame,
) -> None:
    """
    Chart the latest five years of gold prices.
    """

    latest_date = data.iloc[-1]["Date"]

    cutoff_date = latest_date - pd.DateOffset(
        years=5
    )

    recent_data = data[
        data["Date"] >= cutoff_date
    ]

    figure, axes = prepare_chart(
        title="Gold Prices: Last Five Years",
        x_label="Date",
        y_label="Gold Price",
    )

    axes.plot(
        recent_data["Date"],
        recent_data["Price"],
        label="Gold price",
    )

    axes.legend()

    finish_chart(
        figure,
        "last_five_years_chart.png",
    )


def show_monthly_change_chart(
    data: pd.DataFrame,
) -> None:
    """
    Chart monthly percentage changes.
    """

    chart_data = data.dropna(
        subset=["Monthly Change %"]
    )

    figure, axes = prepare_chart(
        title="Monthly Gold Price Percentage Change",
        x_label="Date",
        y_label="Monthly Change (%)",
    )

    axes.plot(
        chart_data["Date"],
        chart_data["Monthly Change %"],
        label="Monthly percentage change",
    )

    axes.axhline(
        y=0,
        linewidth=1,
    )

    axes.legend()

    finish_chart(
        figure,
        "monthly_percentage_change_chart.png",
    )


def show_rolling_average_chart(
    data: pd.DataFrame,
) -> None:
    """
    Chart gold prices against their 12-month rolling average.
    """

    figure, axes = prepare_chart(
        title="Gold Price with 12-Month Rolling Average",
        x_label="Date",
        y_label="Gold Price",
    )

    axes.plot(
        data["Date"],
        data["Price"],
        label="Monthly gold price",
        alpha=0.6,
    )

    axes.plot(
        data["Date"],
        data["12-Month Rolling Average"],
        label="12-month rolling average",
        linewidth=2,
    )

    axes.legend()

    finish_chart(
        figure,
        "rolling_average_chart.png",
    )


def show_date_range_chart(
    data: pd.DataFrame,
) -> None:
    """
    Chart a selected date range.
    """

    starting_date = data.iloc[0]["Date"]
    ending_date = data.iloc[-1]["Date"]

    figure, axes = prepare_chart(
        title=(
            f"Gold Prices: "
            f"{starting_date:%B %Y} to "
            f"{ending_date:%B %Y}"
        ),
        x_label="Date",
        y_label="Gold Price",
    )

    axes.plot(
        data["Date"],
        data["Price"],
        label="Gold price",
    )

    axes.legend()

    finish_chart(
        figure,
        (
            f"date_range_"
            f"{starting_date:%Y-%m}_"
            f"to_{ending_date:%Y-%m}.png"
        ),
    )


# --------------------------------------------------
# Menu option 5: Charts menu
# --------------------------------------------------

def display_charts_menu() -> None:
    """
    Display chart options.
    """

    print("\n" + "=" * 74)
    print("GOLD PRICE CHARTS".center(74))
    print("=" * 74)

    print("\n1. Full historical price chart")
    print("2. Last five years chart")
    print("3. Monthly percentage-change chart")
    print("4. Twelve-month rolling average chart")
    print("5. Chart a particular date range")
    print("6. Return to main menu")


def run_charts_menu(
    data: pd.DataFrame,
) -> None:
    """
    Let the user select and display charts.
    """

    while True:
        display_charts_menu()

        choice = input(
            "\nEnter your selection: "
        ).strip()

        if choice == "1":
            show_full_history_chart(data)

        elif choice == "2":
            show_last_five_years_chart(data)

        elif choice == "3":
            show_monthly_change_chart(data)

        elif choice == "4":
            show_rolling_average_chart(data)

        elif choice == "5":
            selected_data = get_date_range(data)

            if selected_data is not None:
                show_date_range_chart(selected_data)

        elif choice == "6":
            return

        else:
            print(
                "\nInvalid selection. Enter a number "
                "from 1 through 6."
            )


# --------------------------------------------------
# Main interactive menu
# --------------------------------------------------

def display_main_menu() -> None:
    """
    Display the main program menu.
    """

    print("\n" + "=" * 74)
    print("GOLD VALUE PREDICTOR".center(74))
    print("=" * 74)

    print("\n1. View complete summary")
    print("2. Analyze a particular date range")
    print("3. Compare two dates")
    print("4. Project a future value")
    print("5. View charts")
    print("6. Exit")


def run_main_menu(
    data: pd.DataFrame,
) -> None:
    """
    Run the program until the user chooses to exit.
    """

    while True:
        display_main_menu()

        choice = input(
            "\nEnter your selection: "
        ).strip()

        if choice == "1":
            view_complete_summary(data)

        elif choice == "2":
            analyze_date_range(data)

        elif choice == "3":
            compare_two_dates(data)

        elif choice == "4":
            project_future_value(data)

        elif choice == "5":
            run_charts_menu(data)

        elif choice == "6":
            print(
                "\nThank you for using the "
                "Gold Value Predictor."
            )
            break

        else:
            print(
                "\nInvalid selection. Enter a number "
                "from 1 through 6."
            )


# --------------------------------------------------
# Main program
# --------------------------------------------------

def main() -> None:
    """
    Load the data and start the interactive program.
    """

    try:
        gold_data = load_gold_data(CSV_FILE)

        run_main_menu(gold_data)

    except FileNotFoundError as error:
        print(f"\nProgram error: {error}")

    except pd.errors.EmptyDataError:
        print(
            "\nProgram error: The CSV file is empty."
        )

    except pd.errors.ParserError:
        print(
            "\nProgram error: The CSV could not be read. "
            "Check its formatting."
        )

    except PermissionError:
        print(
            "\nProgram error: Permission to read or write "
            "a file was denied."
        )

    except ValueError as error:
        print(f"\nProgram error: {error}")

    except ModuleNotFoundError as error:
        print(
            f"\nMissing Python package: {error.name}"
        )

        print(
            "Install the required packages with:\n"
            "python -m pip install -r requirements.txt"
        )


if __name__ == "__main__":
    main()