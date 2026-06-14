"""NUCC taxonomy lookup and specialty normalization."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

from provider_intelligence.config import PROJECT_ROOT


class TaxonomyLookup:
    """Load and query NUCC taxonomy reference data."""

    def __init__(self, taxonomy_path: Path | None = None) -> None:
        path = taxonomy_path or PROJECT_ROOT / "data" / "reference" / "nucc_taxonomy_sample.csv"
        self._path = path
        self._df = self._load(path)
        self._code_to_desc = dict(zip(self._df["code"], self._df["description"]))
        self._descriptions = self._df["description"].tolist()

    @staticmethod
    def _load(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=["code", "description", "grouping", "classification"])
        return pd.read_csv(path, dtype=str).fillna("")

    def is_valid_code(self, code: str | None) -> bool:
        """Return True if taxonomy code exists in reference data."""
        return bool(code and code in self._code_to_desc)

    def get_description(self, code: str | None) -> str:
        """Return taxonomy description for a code."""
        if not code:
            return ""
        return self._code_to_desc.get(code, "")

    def fuzzy_match_description(self, specialty: str, threshold: int = 75) -> str:
        """Fuzzy match a specialty label to the closest taxonomy description."""
        if not specialty or not self._descriptions:
            return ""
        match = process.extractOne(specialty, self._descriptions, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= threshold:
            return match[0]
        return specialty

    def fuzzy_match_confidence(self, specialty: str) -> float:
        """Return 0–1 confidence for mapping a specialty label to NUCC taxonomy."""
        if not specialty or not self._descriptions:
            return 0.0
        match = process.extractOne(specialty, self._descriptions, scorer=fuzz.token_sort_ratio)
        if not match:
            return 0.0
        return round(match[1] / 100.0, 3)

    def fuzzy_match_code(self, specialty: str, threshold: int = 75) -> str | None:
        """Return best matching taxonomy code for a specialty label."""
        description = self.fuzzy_match_description(specialty, threshold=threshold)
        for code, desc in self._code_to_desc.items():
            if desc == description:
                return code
        return None


@lru_cache(maxsize=1)
def get_taxonomy_lookup() -> TaxonomyLookup:
    """Return cached taxonomy lookup instance."""
    return TaxonomyLookup()
