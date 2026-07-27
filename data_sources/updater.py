import requests

from data_sources.fred import update_all_fred_series
from data_sources.gold import update_gold_data
from data_sources.merger import build_master_dataset


def print_section(title: str) -> None:
    """
    Print a consistently formatted pipeline section.
    """

    print("\n" + "-" * 72)
    print(title.center(72))
    print("-" * 72)


def update_market_data() -> bool:
    """
    Run the complete market-data pipeline.

    The pipeline performs these steps:

    1. Update every configured FRED dataset.
    2. Update the gold-price dataset.
    3. Build the merged master dataset.

    The master dataset is only rebuilt if both source
    update stages complete successfully.

    Returns:
        True when the entire pipeline succeeds.
        False when one or more stages fail.
    """

    failures: list[str] = []

    # --------------------------------------------------
    # Update FRED data
    # --------------------------------------------------

    print_section("UPDATING FRED DATA")

    try:
        update_all_fred_series()

    except (
        requests.RequestException,
        ValueError,
        OSError,
    ) as error:
        failures.append("FRED")

        print(
            f"\nFRED update failed: {error}"
        )

    # --------------------------------------------------
    # Update gold data
    # --------------------------------------------------

    print_section("UPDATING GOLD DATA")

    try:
        update_gold_data()

    except (
        requests.RequestException,
        ValueError,
        OSError,
    ) as error:
        failures.append("Gold")

        print(
            f"\nGold update failed: {error}"
        )

    # --------------------------------------------------
    # Stop before merging if source updates failed
    # --------------------------------------------------

    if failures:
        failed_updates = ", ".join(failures)

        print_section("PIPELINE STOPPED")

        print(
            "The master dataset was not rebuilt because "
            "one or more source updates failed."
        )

        print(
            f"Failed updates: {failed_updates}."
        )

        return False

    # --------------------------------------------------
    # Build master dataset
    # --------------------------------------------------

    print_section("BUILDING MASTER DATASET")

    try:
        build_master_dataset()

    except (
        FileNotFoundError,
        ValueError,
        OSError,
    ) as error:
        failures.append("Master Dataset")

        print(
            f"\nMaster dataset build failed: {error}"
        )

    # --------------------------------------------------
    # Final status
    # --------------------------------------------------

    if failures:
        failed_stages = ", ".join(failures)

        print_section("PIPELINE COMPLETED WITH ERRORS")

        print(
            f"The following stage failed: "
            f"{failed_stages}."
        )

        return False

    print_section("MARKET DATA PIPELINE COMPLETE")

    print(
        "All source datasets were updated successfully."
    )

    print(
        "The master dataset and metadata were rebuilt."
    )

    return True


def main() -> None:
    """
    Run the complete market-data pipeline.
    """

    print("\n" + "=" * 72)
    print("MARKET DATA PIPELINE".center(72))
    print("=" * 72)

    success = update_market_data()

    print("\n" + "=" * 72)

    if success:
        print(
            "PIPELINE FINISHED SUCCESSFULLY".center(72)
        )

    else:
        print(
            "PIPELINE FINISHED WITH ERRORS".center(72)
        )

    print("=" * 72)


if __name__ == "__main__":
    main()