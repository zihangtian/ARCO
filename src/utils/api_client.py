"""OpenAI SDK wrapper with exponential backoff retry and disk cache."""

import time
import openai
from .cache import DiskCache


class APIClient:
    def __init__(self, model: str, cache: DiskCache,
                 base_url: str | None = None,
                 api_key: str | None = None,
                 max_retries: int = 5, base_delay: float = 2.0, max_delay: float = 60.0):
        self.model = model
        self.cache = cache
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self.client = openai.OpenAI(**kwargs)

    def chat(self, messages: list[dict], temperature: float = 0.0,
             max_tokens: int = 512, use_cache: bool = True, **kwargs) -> str:
        extra = {"temperature": temperature, "max_tokens": max_tokens, **kwargs}
        if use_cache:
            cached = self.cache.get(self.model, messages, **extra)
            if cached is not None:
                return cached

        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content or ""
                if use_cache:
                    self.cache.put(self.model, messages, content, **extra)
                return content
            except (openai.RateLimitError, openai.APITimeoutError,
                    openai.APIConnectionError) as e:
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                print(f"[API] retry {attempt+1}/{self.max_retries} in {delay:.1f}s: {e}")
                time.sleep(delay)
            except openai.APIError as e:
                if attempt == self.max_retries - 1:
                    raise
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                print(f"[API] error retry {attempt+1}/{self.max_retries}: {e}")
                time.sleep(delay)
        raise RuntimeError(f"API call failed after {self.max_retries} retries")
