"""Pass 1: Structural rule-based judgment segmentation."""

from __future__ import annotations

import re

from .models import JudgmentSegment, SegmentType


class StructuralSegmenter:
    """
    First-pass segmenter using section-header patterns in Nigerian judgments.

    Nigerian SC/CA judgments follow a loose structure:

    1. Caption (court name, parties, suit number)
    2. Background / Procedural history
    3. Statement of facts
    4. Issues for determination
    5. Submissions by counsel
    6. Court's analysis (per issue)
    7. Holdings
    8. Orders / Disposition

    Segments are identified through:
    - Header / keyword patterns
    - Positional heuristics (early paragraphs → caption/background,
      late paragraphs → orders)
    - Default: ANALYSIS at confidence 0.3
    """

    # Section patterns ordered by specificity (most specific first).
    # Keys are the target SegmentType; values are regex strings to match
    # anywhere in the paragraph (case-insensitive).
    _SECTION_PATTERNS: dict[SegmentType, list[str]] = {
        SegmentType.BACKGROUND: [
            r"(?:BRIEF\s+)?(?:FACTS|BACKGROUND|HISTORY)\s*(?:OF\s+THE\s+CASE)?",
            r"PROCEDURAL\s+HISTORY",
            r"(?:The\s+)?(?:brief\s+)?facts\s+(?:of\s+)?(?:this|the)\s+case",
        ],
        SegmentType.ISSUES: [
            r"ISSUES?\s+FOR\s+DETERMINATION",
            r"The\s+(?:following\s+)?issues?\s+(?:were|are|is)\s+(?:raised|formulated|submitted|distilled)",
            r"(?:I|We)\s+(?:shall|will)\s+(?:consider|determine)\s+the\s+following\s+issues?",
        ],
        SegmentType.SUBMISSION: [
            r"(?:SUBMISSIONS?\s+OF|ARGUMENTS?\s+OF)\s+(?:LEARNED\s+)?(?:SENIOR\s+)?COUNSEL",
            r"(?:Mr\.|Chief|Dr\.)\s+\w+,?\s*(?:SAN|Esq\.?)?,?\s+(?:of\s+)?counsel\s+for\s+the\s+(?:appellant|respondent|applicant|plaintiff|defendant)",
            r"Learned\s+(?:Senior\s+)?Counsel\s+for\s+the\s+(?:Appellant|Respondent)",
        ],
        SegmentType.ANALYSIS: [
            r"(?:I|We)\s+have\s+(?:carefully\s+)?(?:considered|examined|perused)",
            r"(?:Having\s+)?(?:considered|examined)\s+the\s+(?:submissions|arguments|briefs)",
            r"(?:I|The\s+Court)\s+(?:shall|will)\s+now\s+(?:consider|examine|address)",
        ],
        SegmentType.HOLDING: [
            r"\bHELD\b",
            r"(?:I|We)\s+(?:therefore\s+)?hold\s+that",
            r"(?:In|For)\s+(?:the\s+)?(?:above|foregoing)\s+reasons?,?\s+"
            r"(?:I|we|this\s+court)\s+(?:hold|find|conclude)",
            r"(?:It|This)\s+is\s+(?:my|our)\s+(?:considered\s+)?(?:view|opinion|judgment)\s+that",
        ],
        SegmentType.ORDERS: [
            r"(?:FINAL\s+)?ORDER(?:S)?(?:\s+OF\s+(?:THE\s+)?COURT)?",
            r"(?:In|For)\s+the\s+(?:final\s+)?(?:result|conclusion)",
            r"The\s+appeal\s+(?:is|succeeds|fails|is\s+dismissed|is\s+allowed)",
            r"(?:Accordingly|Consequently),?\s+(?:I|we|this\s+court)\s+(?:order|direct|dismiss|allow)",
        ],
        SegmentType.DISSENT: [
            r"DISSENTING\s+(?:JUDGMENT|OPINION)",
            r"(?:I|My\s+Lord)\s+(?:respectfully\s+)?dissent",
        ],
        SegmentType.RATIO: [
            r"RATIO\s+DECIDENDI",
            r"The\s+(?:binding\s+)?(?:principle|rule)\s+(?:established|laid\s+down)\s+"
            r"(?:in\s+this\s+case\s+)?is",
        ],
    }

    # Minimum paragraph length (characters) to be treated as a real segment
    _MIN_PARAGRAPH_LENGTH: int = 30

    def segment(self, text: str) -> list[JudgmentSegment]:
        """
        Segment judgment text using structural patterns.

        Returns position-ordered, merged list of JudgmentSegments.
        """
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return []

        total = len(paragraphs)
        scored: list[JudgmentSegment] = [
            JudgmentSegment(
                segment_type=seg_type,
                content=para,
                position=i,
                confidence=confidence,
            )
            for i, para in enumerate(paragraphs)
            for seg_type, confidence in (self._classify_paragraph(para, i, total),)
        ]

        return self._merge_consecutive(scored)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split on blank lines; drop very short/empty paragraphs."""
        return [
            p.strip()
            for p in re.split(r"\n\s*\n", text)
            if len(p.strip()) >= self._MIN_PARAGRAPH_LENGTH
        ]

    def _classify_paragraph(
        self,
        text: str,
        position: int,
        total: int,
    ) -> tuple[SegmentType, float]:
        """
        Classify a paragraph by pattern match + positional heuristic.

        Pattern match → confidence 0.8 (wins over positional guess).
        Positional heuristic → used only when no pattern matched (< 0.5).
        Default → ANALYSIS at 0.3.
        """
        best_type = SegmentType.ANALYSIS
        best_conf = 0.3

        for seg_type, patterns in self._SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    if best_conf < 0.8:
                        best_type = seg_type
                        best_conf = 0.8
                    break  # one pattern match per type is sufficient

        if best_conf < 0.5:
            rel = position / max(total - 1, 1)
            if rel < 0.05:
                best_type = SegmentType.CAPTION
                best_conf = 0.5
            elif rel < 0.15:
                best_type = SegmentType.BACKGROUND
                best_conf = 0.4
            elif rel > 0.90:
                best_type = SegmentType.ORDERS
                best_conf = 0.4

        return best_type, best_conf

    @staticmethod
    def _merge_consecutive(
        segments: list[JudgmentSegment],
    ) -> list[JudgmentSegment]:
        """Merge consecutive segments of the same type into one."""
        if not segments:
            return []

        merged = [
            JudgmentSegment(
                segment_type=segments[0].segment_type,
                content=segments[0].content,
                position=segments[0].position,
                confidence=segments[0].confidence,
            )
        ]
        for seg in segments[1:]:
            prev = merged[-1]
            if seg.segment_type == prev.segment_type:
                prev.content += "\n\n" + seg.content
                prev.confidence = min(prev.confidence, seg.confidence)
            else:
                merged.append(
                    JudgmentSegment(
                        segment_type=seg.segment_type,
                        content=seg.content,
                        position=seg.position,
                        confidence=seg.confidence,
                    )
                )
        return merged
