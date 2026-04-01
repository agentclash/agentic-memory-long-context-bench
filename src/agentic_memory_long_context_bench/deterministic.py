from __future__ import annotations

import hashlib
import math
import re


class HashingEmbedder:
    def __init__(self, dimensions: int = 64):
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    def embed_text(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self._dimensions
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
