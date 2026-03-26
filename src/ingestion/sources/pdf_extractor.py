"""Shared PDF-to-text extraction utility.

Extraction cascade:
  1. pdfplumber  — best for digital text (selectable text PDFs)
  2. pymupdf     — fallback for complex layouts or pdfplumber failures
  3. pytesseract — OCR fallback for scanned/image-only PDFs (optional dep)

Usage::

    extractor = PDFTextExtractor()
    text = extractor.extract(Path("judgment.pdf"))
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PDFTextExtractor:
    """Extract text from a PDF using a best-effort cascade of libraries."""

    MIN_CHARS = 200  # minimum non-whitespace characters to consider extraction successful

    @staticmethod
    def _has_content(text: str, min_chars: int) -> bool:
        return len(text.strip()) >= min_chars

    def extract(self, pdf_path: Path) -> str:
        """Return extracted text, trying pdfplumber → pymupdf → OCR in order."""
        text = self._try_pdfplumber(pdf_path)
        if self._has_content(text, self.MIN_CHARS):
            return text

        logger.debug(
            "pdfplumber yielded %d non-ws chars, trying pymupdf: %s",
            len(text.strip()),
            pdf_path.name,
        )
        text = self._try_pymupdf(pdf_path)
        if self._has_content(text, self.MIN_CHARS):
            return text

        logger.debug(
            "pymupdf yielded %d non-ws chars, trying OCR: %s", len(text.strip()), pdf_path.name
        )
        text = self._try_ocr(pdf_path)
        if not self._has_content(text, self.MIN_CHARS):
            logger.warning(
                "All extraction methods failed for %s (got %d non-ws chars)",
                pdf_path.name,
                len(text.strip()),
            )
        return text

    # ── primary: pdfplumber ────────────────────────────────────────────────────

    def _try_pdfplumber(self, pdf_path: Path) -> str:
        try:
            import pdfplumber  # type: ignore[import]

            pages: list[str] = []
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages.append(page_text)
            return "\n\n".join(pages)
        except Exception as exc:
            logger.debug("pdfplumber error on %s: %s", pdf_path.name, exc)
            return ""

    # ── fallback 1: pymupdf ────────────────────────────────────────────────────

    def _try_pymupdf(self, pdf_path: Path) -> str:
        try:
            import fitz  # type: ignore[import]  # pymupdf

            doc = fitz.open(str(pdf_path))
            pages: list[str] = []
            for page in doc:
                pages.append(str(page.get_text()))
            doc.close()
            return "\n\n".join(pages)
        except Exception as exc:
            logger.debug("pymupdf error on %s: %s", pdf_path.name, exc)
            return ""

    # ── fallback 2: OCR via pytesseract ──────────────────────────────────────

    def _try_ocr(self, pdf_path: Path) -> str:
        try:
            import io

            import fitz  # type: ignore[import]
            import pytesseract  # type: ignore[import]
            from PIL import Image  # type: ignore[import]

            doc = fitz.open(str(pdf_path))
            pages: list[str] = []
            for page in doc:
                # Render at 200 DPI for decent OCR quality
                mat = fitz.Matrix(200 / 72, 200 / 72)
                pix = page.get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img)
                if text.strip():
                    pages.append(text)
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            logger.debug("pytesseract or Pillow not installed — OCR unavailable")
            return ""
        except Exception as exc:
            logger.debug("OCR error on %s: %s", pdf_path.name, exc)
            return ""
