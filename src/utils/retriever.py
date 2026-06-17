"""External retrieval service client."""

import json
import time
import threading
import requests
from requests.adapters import HTTPAdapter


class Retriever:
    def __init__(self, url: str, max_retries: int = 3,
                 timeout: int = 30, max_concurrent: int = 32,
                 fail_on_empty: bool = False):
        self.url = url
        self.max_retries = max_retries
        self.timeout = timeout
        self.fail_on_empty = fail_on_empty
        # Limit concurrent requests so a single-process service isn't overwhelmed
        self._sem = threading.Semaphore(max_concurrent)
        self._local = threading.local()

    def _get_session(self) -> requests.Session:
        sess = getattr(self._local, "session", None)
        if sess is None:
            sess = requests.Session()
            adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32)
            sess.mount("http://", adapter)
            sess.mount("https://", adapter)
            self._local.session = sess
        return sess

    def retrieve(self, entity: str, data: list[str]) -> str:
        payload = json.dumps({"entity": entity, "data": data})
        headers = {"Content-Type": "application/json"}
        session = self._get_session()

        with self._sem:
            for attempt in range(self.max_retries):
                try:
                    resp = session.post(
                        self.url, headers=headers, data=payload,
                        timeout=self.timeout,
                    )
                    if resp.status_code == 200:
                        result = resp.json().get("response", "")
                        if self.fail_on_empty and not str(result).strip():
                            raise RuntimeError("Retriever returned an empty response")
                        return result
                    print(f"[Retriever] HTTP {resp.status_code}: {resp.text[:100]}")
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"[Retriever] attempt {attempt+1}/{self.max_retries} failed: {e}"
                          f"{f' — retrying in {wait}s' if attempt < self.max_retries-1 else ''}")
                    if attempt < self.max_retries - 1:
                        time.sleep(wait)
        if self.fail_on_empty:
            raise RuntimeError("Retriever failed after retries")
        return ""
