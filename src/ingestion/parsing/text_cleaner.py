"""Judgment text cleaning pipeline — fixes OCR artefacts, encoding issues, noise."""

from __future__ import annotations

import re
import unicodedata


class JudgmentTextCleaner:
    """
    Clean raw judgment text extracted from NigeriaLII HTML.

    Issues handled:
    - Encoding artefacts (mojibake from PDF→HTML pipeline)
    - OCR misreads (fi/fl ligature replacements)
    - Page headers / footers from PDF extraction
    - Inconsistent whitespace and paragraph spacing
    - Redundant legal-formatting patterns
    """

    # Common OCR ligature artefacts found in Nigerian judgments
    _OCR_REPLACEMENTS: dict[str, str] = {
        "?le": "file",
        "?led": "filed",
        "?rst": "first",
        "?nal": "final",
        "?nd": "find",
        "satis?ed": "satisfied",
        "justi?ed": "justified",
        "speci?c": "specific",
        "signi?cant": "significant",
        "ful?l": "fulfil",
        "quali?ed": "qualified",
        "certi?cate": "certificate",
        "bene?t": "benefit",
        "pro?t": "profit",
        "of?ce": "office",
        "of?cer": "officer",
        "suf?cient": "sufficient",
        "ef?cient": "efficient",
        "af?davit": "affidavit",
    }

    # Regex patterns for page noise removal
    _PAGE_NOISE_PATTERNS: list[str] = [
        r"Page\s+\d+\s+of\s+\d+",
        r"\n\s*-\s*\d+\s*-\s*\n",
        r"IN THE SUPREME COURT OF NIGERIA\s*\n(?=.*IN THE SUPREME COURT)",
        r"IN THE COURT OF APPEAL\s*\n(?=.*IN THE COURT OF APPEAL)",
        r"\f",  # form-feed characters from PDF splits
    ]

    def clean(self, text: str) -> str:
        """Full cleaning pipeline. Returns clean, normalised judgment text."""
        text = self._fix_encoding(text)
        text = self._fix_ocr_artifacts(text)
        text = self._remove_page_noise(text)
        text = self._normalise_whitespace(text)
        text = self._normalise_legal_formatting(text)
        return text.strip()

    # ── Individual pipeline stages ────────────────────────────────────────────

    def _fix_encoding(self, text: str) -> str:
        """Fix Unicode normalisation and common mojibake patterns."""
        text = unicodedata.normalize("NFKC", text)
        # Common mojibake from Windows-1252 → UTF-8 misread
        replacements = {
            "â€™": "'",
            "â€˜": "'",
            "â€œ": '"',
            "â€\x9d": '"',
            "â€”": "—",
            "â€“": "–",
            "â€¦": "…",
            "\u00a0": " ",  # non-breaking space → regular space
        }
        for wrong, right in replacements.items():
            text = text.replace(wrong, right)
        return text

    def _fix_ocr_artifacts(self, text: str) -> str:
        """Fix common OCR misreads (? substituted for fi/fl ligatures)."""
        for wrong, right in self._OCR_REPLACEMENTS.items():
            text = text.replace(wrong, right)
        return text

    def _remove_page_noise(self, text: str) -> str:
        """Remove page headers, footers, and PDF pagination artefacts."""
        for pattern in self._PAGE_NOISE_PATTERNS:
            text = re.sub(pattern, "\n", text, flags=re.IGNORECASE)
        return text

    def _normalise_whitespace(self, text: str) -> str:
        """Normalise whitespace while preserving paragraph structure."""
        # Collapse 3+ blank lines → double newline (paragraph boundary)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove trailing whitespace per line
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        # Collapse multiple spaces (but not leading indentation)
        text = re.sub(r"(?<=\S) {2,}", " ", text)
        return text

    def _normalise_legal_formatting(self, text: str) -> str:
        """Normalise legal-specific formatting and abbreviations."""
        # Standardise "versus" to v.  (matches "vs", "vs.", but not "v." already ok)
        text = re.sub(r"\bvs\.?\b", "v.", text, flags=re.IGNORECASE)
        # Normalise section abbreviation — careful not to clobber "S.C." etc.
        text = re.sub(r"\bS\.\s+(\d)", r"Section \1", text)
        return text
