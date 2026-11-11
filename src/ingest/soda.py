from __future__ import annotations

import logging
from collections.abc import Iterator

import pandas as pd
import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    # ChunkedEncodingError / JSONDecodeError: the connection dropped mid-response
    if isinstance(exc, (requests.ConnectionError, requests.Timeout,
                        requests.exceptions.ChunkedEncodingError, requests.JSONDecodeError)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


class SodaClient:
    def __init__(
        self,
        domain: str,
        app_token: str | None = None,
        page_size: int = 50_000,
        timeout: int = 120,
        session: requests.Session | None = None,
    ):
        self.base = f"https://{domain}/resource"
        self.page_size = page_size
        self.timeout = timeout
        self.session = session or requests.Session()
        if app_token:
            self.session.headers["X-App-Token"] = app_token

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=2, max=60),
        reraise=True,
    )
    def _get(self, dataset_id: str, params: dict) -> list[dict]:
        resp = self.session.get(
            f"{self.base}/{dataset_id}.json", params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def iter_pages(
        self,
        dataset_id: str,
        columns: list[str],
        where: str | None = None,
        order: str | None = None,
    ) -> Iterator[pd.DataFrame]:

        offset = 0
        while True:
            params = {
                "$select": ",".join(columns),
                "$order": order or columns[0],
                "$limit": self.page_size,
                "$offset": offset,
            }
            if where:
                params["$where"] = where
            rows = self._get(dataset_id, params)
            if rows:
                # SODA omits null fields from JSON objects; reindex to pin the schema.
                yield pd.DataFrame.from_records(rows).reindex(columns=columns)
            if len(rows) < self.page_size:
                return
            offset += self.page_size

    def count(self, dataset_id: str, where: str | None = None) -> int:
        params = {"$select": "count(*) AS n"}
        if where:
            params["$where"] = where
        return int(self._get(dataset_id, params)[0]["n"])

    def fetch(self, dataset_id: str, columns: list[str], where=None, order=None) -> pd.DataFrame:
        pages = list(self.iter_pages(dataset_id, columns, where, order))
        if not pages:
            return pd.DataFrame(columns=columns)
        return pd.concat(pages, ignore_index=True)
