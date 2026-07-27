import pytest
import requests

import data_sources.updater as updater


def test_update_market_data_runs_complete_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A successful pipeline should update FRED, update
    gold, and then build the master dataset.
    """

    calls: list[str] = []

    def fake_update_fred() -> None:
        calls.append("fred")

    def fake_update_gold() -> None:
        calls.append("gold")

    def fake_build_master_dataset() -> None:
        calls.append("merger")

    monkeypatch.setattr(
        updater,
        "update_all_fred_series",
        fake_update_fred,
    )

    monkeypatch.setattr(
        updater,
        "update_gold_data",
        fake_update_gold,
    )

    monkeypatch.setattr(
        updater,
        "build_master_dataset",
        fake_build_master_dataset,
    )

    result = updater.update_market_data()

    assert result is True

    assert calls == [
        "fred",
        "gold",
        "merger",
    ]


def test_update_market_data_skips_merger_after_fred_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A FRED failure should not prevent the gold update
    from being attempted, but the merger should not run.
    """

    calls: list[str] = []

    def fake_update_fred() -> None:
        calls.append("fred")

        raise requests.RequestException(
            "FRED unavailable"
        )

    def fake_update_gold() -> None:
        calls.append("gold")

    def fake_build_master_dataset() -> None:
        calls.append("merger")

    monkeypatch.setattr(
        updater,
        "update_all_fred_series",
        fake_update_fred,
    )

    monkeypatch.setattr(
        updater,
        "update_gold_data",
        fake_update_gold,
    )

    monkeypatch.setattr(
        updater,
        "build_master_dataset",
        fake_build_master_dataset,
    )

    result = updater.update_market_data()

    assert result is False

    assert calls == [
        "fred",
        "gold",
    ]


def test_update_market_data_skips_merger_after_gold_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A gold failure should occur after FRED succeeds,
    and the merger should not run.
    """

    calls: list[str] = []

    def fake_update_fred() -> None:
        calls.append("fred")

    def fake_update_gold() -> None:
        calls.append("gold")

        raise ValueError(
            "Invalid gold data"
        )

    def fake_build_master_dataset() -> None:
        calls.append("merger")

    monkeypatch.setattr(
        updater,
        "update_all_fred_series",
        fake_update_fred,
    )

    monkeypatch.setattr(
        updater,
        "update_gold_data",
        fake_update_gold,
    )

    monkeypatch.setattr(
        updater,
        "build_master_dataset",
        fake_build_master_dataset,
    )

    result = updater.update_market_data()

    assert result is False

    assert calls == [
        "fred",
        "gold",
    ]


def test_update_market_data_reports_multiple_source_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The updater should attempt both source updates and
    return False when both fail.
    """

    calls: list[str] = []

    def fake_update_fred() -> None:
        calls.append("fred")

        raise OSError(
            "Could not save FRED data"
        )

    def fake_update_gold() -> None:
        calls.append("gold")

        raise requests.RequestException(
            "Gold source unavailable"
        )

    def fake_build_master_dataset() -> None:
        calls.append("merger")

    monkeypatch.setattr(
        updater,
        "update_all_fred_series",
        fake_update_fred,
    )

    monkeypatch.setattr(
        updater,
        "update_gold_data",
        fake_update_gold,
    )

    monkeypatch.setattr(
        updater,
        "build_master_dataset",
        fake_build_master_dataset,
    )

    result = updater.update_market_data()

    assert result is False

    assert calls == [
        "fred",
        "gold",
    ]


def test_update_market_data_reports_merger_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A merger failure should return False after both
    source updates succeed.
    """

    calls: list[str] = []

    def fake_update_fred() -> None:
        calls.append("fred")

    def fake_update_gold() -> None:
        calls.append("gold")

    def fake_build_master_dataset() -> None:
        calls.append("merger")

        raise ValueError(
            "Invalid processed dataset"
        )

    monkeypatch.setattr(
        updater,
        "update_all_fred_series",
        fake_update_fred,
    )

    monkeypatch.setattr(
        updater,
        "update_gold_data",
        fake_update_gold,
    )

    monkeypatch.setattr(
        updater,
        "build_master_dataset",
        fake_build_master_dataset,
    )

    result = updater.update_market_data()

    assert result is False

    assert calls == [
        "fred",
        "gold",
        "merger",
    ]


def test_update_market_data_handles_missing_processed_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A missing processed file during the merge stage
    should cause the pipeline to fail cleanly.
    """

    def fake_update_fred() -> None:
        pass

    def fake_update_gold() -> None:
        pass

    def fake_build_master_dataset() -> None:
        raise FileNotFoundError(
            "CPI dataset was not found"
        )

    monkeypatch.setattr(
        updater,
        "update_all_fred_series",
        fake_update_fred,
    )

    monkeypatch.setattr(
        updater,
        "update_gold_data",
        fake_update_gold,
    )

    monkeypatch.setattr(
        updater,
        "build_master_dataset",
        fake_build_master_dataset,
    )

    result = updater.update_market_data()

    assert result is False