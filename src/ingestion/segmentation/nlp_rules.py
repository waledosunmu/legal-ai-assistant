"""Pass 2: NLP-rule-based confidence rescoring for structural segments."""

from __future__ import annotations

import re

from .models import JudgmentSegment, SegmentType


# Weighted keyword patterns per segment type.
# Each entry is (regex_pattern, confidence_weight).
# Weights are additive; capped at _MAX_BOOST before applying.
_KEYWORD_WEIGHTS: dict[SegmentType, list[tuple[str, float]]] = {
    SegmentType.ISSUES: [
        (r"issues?\s+for\s+determination", 0.60),
        (r"issues?\s+(?:raised|formulated|distilled)", 0.50),
        (r"issue\s+(?:no\.?\s*)?\d+", 0.40),
        (r"\bwhether\b", 0.15),
    ],
    SegmentType.HOLDING: [
        (r"\bheld\b", 0.50),
        (r"in\s+the\s+result", 0.50),
        (r"for\s+the\s+(?:above|foregoing)\s+reasons?", 0.40),
        (r"(?:i|we)\s+hold\s+that", 0.50),
        (r"this\s+appeal\s+(?:is|succeeds|fails|is\s+dismissed|is\s+allowed)", 0.40),
    ],
    SegmentType.ORDERS: [
        (r"\border(?:s|ed)?\b", 0.30),
        (r"\baccordingly\b", 0.25),
        (r"\bis\s+hereby\b", 0.30),
        (r"costs?\s+(?:of|awarded|assessed)", 0.30),
        (r"appeal\s+(?:allowed|dismissed)", 0.50),
    ],
    SegmentType.RATIO: [
        (r"ratio\s+decidendi", 0.70),
        (r"\bper\s+incuriam\b", 0.40),
        (r"binding\s+(?:authority|precedent|principle)", 0.35),
        (r"the\s+law\s+is\s+settled", 0.35),
    ],
    SegmentType.DISSENT: [
        (r"\bi\s+dissent\b", 0.60),
        (r"respectfully\s+dissent", 0.60),
        (r"\bdissenting\b", 0.50),
    ],
    SegmentType.SUBMISSION: [
        (r"learned\s+(?:senior\s+)?counsel", 0.35),
        (r"(?:appellant|respondent|applicant)['']?s?\s+counsel", 0.30),
        (r"it\s+was\s+submitted", 0.30),
        (r"\b(?:argues?|contends?|submits?)\b", 0.20),
    ],
    SegmentType.ANALYSIS: [
        (r"(?:i|we)\s+have\s+(?:carefully\s+)?(?:considered|examined|perused)", 0.35),
        (r"having\s+(?:considered|examined)", 0.30),
        (r"\bthe\s+law\s+is\b", 0.20),
        (r"in\s+my\s+(?:view|opinion)", 0.20),
    ],
    SegmentType.BACKGROUND: [
        (r"brief\s+facts", 0.50),
        (r"procedural\s+history", 0.50),
        (r"the\s+trial\s+court", 0.25),
        (r"the\s+lower\s+court", 0.25),
        (r"at\s+the\s+(?:trial|lower|high)\s+court", 0.25),
    ],
}

# Segments with confidence at or above this threshold are left unchanged
_LOW_CONFIDENCE_THRESHOLD: float = 0.60

# Maximum additional confidence from keyword scoring
_MAX_BOOST: float = 0.30

# Minimum keyword score needed to override the existing segment type
_RECLASSIFY_MIN_SCORE: float = 0.15


class NLPSegmentClassifier:
    """
    Second-pass confidence rescorer for structural segmentation output.

    Takes the segment list produced by StructuralSegmenter and:

    1. Leaves high-confidence segments (≥ 0.60) unchanged.
    2. Detects all-caps section headings → reclassifies as CAPTION.
    3. Applies keyword-density scoring to low-confidence segments,
       boosting confidence and optionally re-assigning the type when
       keyword evidence is stronger than the original classification.

    No external APIs — purely deterministic.
    """

    def reclassify(
        self, segments: list[JudgmentSegment]
    ) -> list[JudgmentSegment]:
        """
        Return a new list with updated confidence scores (and types where
        keyword evidence is strong enough to override pass-1 guesses).
        """
        return [
            seg if seg.confidence >= _LOW_CONFIDENCE_THRESHOLD else self._rescore(seg)
            for seg in segments
        ]

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _rescore(self, seg: JudgmentSegment) -> JudgmentSegment:
        """Re-score a single low-confidence segment."""
        # All-caps headings (short paragraphs) are almost certainly CAPTION
        if self._is_heading(seg.content):
            return JudgmentSegment(
                segment_type=SegmentType.CAPTION,
                content=seg.content,
                position=seg.position,
                confidence=0.65,
                issue_number=seg.issue_number,
                metadata=seg.metadata,
            )

        # Keyword density scoring
        text_lower = seg.content.lower()
        scores: dict[SegmentType, float] = {}
        for seg_type, kw_list in _KEYWORD_WEIGHTS.items():
            total = sum(
                weight
                for pattern, weight in kw_list
                if re.search(pattern, text_lower, re.IGNORECASE)
            )
            if total > 0:
                scores[seg_type] = total

        if not scores:
            return seg  # No keyword signal — keep original unchanged

        best_type = max(scores, key=lambda t: scores[t])
        keyword_score = min(scores[best_type], _MAX_BOOST)
        new_conf = min(seg.confidence + keyword_score, 0.90)

        # Only switch type when keyword evidence is strong
        new_type = best_type if keyword_score >= _RECLASSIFY_MIN_SCORE else seg.segment_type

        return JudgmentSegment(
            segment_type=new_type,
            content=seg.content,
            position=seg.position,
            confidence=new_conf,
            issue_number=seg.issue_number,
            metadata=seg.metadata,
        )

    @staticmethod
    def _is_heading(text: str) -> bool:
        """
        Return True for all-caps section headings.

        Criteria: ≤ 10 words and ≥ 70 % of alpha characters are uppercase.
        """
        words = text.split()
        if len(words) > 10:
            return False
        alpha = [c for c in text if c.isalpha()]
        if not alpha:
            return False
        return sum(1 for c in alpha if c.isupper()) / len(alpha) >= 0.70
