"""Disk cache keyed by SHA-256(model + messages + extra params)."""

import hashlib
import json
from pathlib import Path
from typing import Any


class DiskCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model: str, messages: list[dict], **extra) -> str:
        blob = json.dumps({"model": model, "messages": messages, **extra},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, model: str, messages: list[dict], **extra) -> Any | None:
        path = self.cache_dir / f"{self._key(model, messages, **extra)}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                try:
                    path.unlink()
                except OSError:
                    pass
                return None
        return None

    def put(self, model: str, messages: list[dict], value: Any, **extra) -> None:
        path = self.cache_dir / f"{self._key(model, messages, **extra)}.json"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
