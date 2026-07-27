import pandas as pd
import pytest
import requests

from data_sources.gold import (
    GOLD_DATE_COLUMN,
    GOLD_SOURCE,
    GOLD_VALUE_COLUMN,
    download_gold_data,
    normalize_gold_data,
)


class FakeResponse:
    """
    Small fake HTTP response used by downloader tests.
    """

    def __init__(
        self,
        text: str,
        status_error: Exception | None = None,
    ) -> None:
        self.text = text
        self.status_error = status_error

    def raise_for_status(self) -> None:
        """
        Simulate requests.Response.raise_for_status().
        """

        if self.status_error is not None:
            raise self.status_error


def test_normalize_gold_data() -> None:
    """
    Valid daily observations should become monthly
    average gold prices.
    """

    raw_data = pd.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-15",
                "2024-02-01",
                "2024-02-20",
            ],
            "price": [
                "2000.00",
                "2020.00",
                "2040.00",
                "2060.00",
            ],
        }
    )

    result = normalize_gold_data(raw_data)

    assert list(result.columns) == [
        GOLD_DATE_COLUMN,
        GOLD_VALUE_COLUMN,
    ]

    assert len(result) == 2

    assert result.iloc[0][GOLD_DATE_COLUMN] == pd.Timestamp(
        "2024-01-01"
    )

    assert result.iloc[0][GOLD_VALUE_COLUMN] == 2010.00
    assert result.iloc[1][GOLD_VALUE_COLUMN] == 2050.00


def test_normalize_gold_data_removes_invalid_rows() -> None:
    """
    Invalid dates, missing prices, and nonnumeric prices
    should be removed.
    """

    raw_data = pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "invalid-date",
                "2024-02-01",
                "2024-03-01",
            ],
            "price": [
                "2000.00",
                "2010.00",
                "not-a-price",
                None,
            ],
        }
    )

    result = normalize_gold_data(raw_data)

    assert len(result) == 1
    assert result.iloc[0][GOLD_VALUE_COLUMN] == 2000.00


def test_normalize_gold_data_rejects_missing_columns() -> None:
    """
    Missing source columns should raise an error.
    """

    raw_data = pd.DataFrame(
        {
            "wrong_date": [
                "2024-01-01",
            ],
            "wrong_price": [
                2000.00,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        normalize_gold_data(raw_data)


def test_normalize_gold_data_rejects_empty_result() -> None:
    """
    Completely invalid source data should raise an error.
    """

    raw_data = pd.DataFrame(
        {
            "date": [
                "invalid-date",
            ],
            "price": [
                "not-a-price",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="No valid gold-price observations",
    ):
        normalize_gold_data(raw_data)


def test_normalize_gold_data_rejects_nonpositive_prices() -> None:
    """
    Zero and negative gold prices should not remain.
    """

    raw_data = pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-02-01",
            ],
            "price": [
                0,
                -100,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="No valid gold-price observations",
    ):
        normalize_gold_data(raw_data)


def test_normalize_gold_data_supports_custom_columns() -> None:
    """
    The normalizer should support different source
    column names.
    """

    raw_data = pd.DataFrame(
        {
            "Date": [
                "2024-01-01",
                "2024-02-01",
            ],
            "USD": [
                2000.00,
                2050.00,
            ],
        }
    )

    result = normalize_gold_data(
        data=raw_data,
        source_date_column="Date",
        source_value_column="USD",
    )

    assert len(result) == 2
    assert result.iloc[1][GOLD_VALUE_COLUMN] == 2050.00


def test_download_gold_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A valid CSV response should become a DataFrame.
    """

    csv_text = (
        "Date,Price\n"
        "2024-01,2000.00\n"
        "2024-02,2050.00\n"
    )

    def fake_get(
        url: str,
        timeout: int,
    ) -> FakeResponse:
        assert url == GOLD_SOURCE["url"]
        assert timeout == 30

        return FakeResponse(csv_text)

    monkeypatch.setattr(
        requests,
        "get",
        fake_get,
    )

    result = download_gold_data()

    assert list(result.columns) == [
        "Date",
        "Price",
    ]

    assert len(result) == 2
    assert result.iloc[0]["Price"] == 2000.00
    assert result.iloc[1]["Price"] == 2050.00


def test_download_gold_data_rejects_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An empty HTTP response should raise an error.
    """

    def fake_get(
        url: str,
        timeout: int,
    ) -> FakeResponse:
        return FakeResponse("")

    monkeypatch.setattr(
        requests,
        "get",
        fake_get,
    )

    with pytest.raises(
        ValueError,
        match="empty response",
    ):
        download_gold_data()


def test_download_gold_data_rejects_whitespace_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A whitespace-only response should raise an error.
    """

    def fake_get(
        url: str,
        timeout: int,
    ) -> FakeResponse:
        return FakeResponse("\n\n   ")

    monkeypatch.setattr(
        requests,
        "get",
        fake_get,
    )

    with pytest.raises(
        ValueError,
        match="empty response",
    ):
        download_gold_data()


def test_download_gold_data_rejects_header_only_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A CSV containing headers but no rows should be rejected.
    """

    def fake_get(
        url: str,
        timeout: int,
    ) -> FakeResponse:
        return FakeResponse("Date,Price\n")

    monkeypatch.setattr(
        requests,
        "get",
        fake_get,
    )

    with pytest.raises(
        ValueError,
        match="dataset is empty",
    ):
        download_gold_data()


def test_download_gold_data_propagates_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    HTTP errors should not be silently ignored.
    """

    http_error = requests.HTTPError(
        "404 Client Error"
    )

    def fake_get(
        url: str,
        timeout: int,
    ) -> FakeResponse:
        return FakeResponse(
            text="Not Found",
            status_error=http_error,
        )

    monkeypatch.setattr(
        requests,
        "get",
        fake_get,
    )

    with pytest.raises(requests.HTTPError):
        download_gold_data()


def test_download_gold_data_uses_custom_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The downloader should accept a custom source URL.
    """

    custom_url = (
        "https://example.com/gold.csv"
    )

    csv_text = (
        "Date,Price\n"
        "2024-01,2000.00\n"
    )

    def fake_get(
        url: str,
        timeout: int,
    ) -> FakeResponse:
        assert url == custom_url
        assert timeout == 30

        return FakeResponse(csv_text)

    monkeypatch.setattr(
        requests,
        "get",
        fake_get,
    )

    result = download_gold_data(
        url=custom_url,
    )

    assert len(result) == 1
    assert result.iloc[0]["Price"] == 2000.00