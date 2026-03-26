"""Three-layer query parser for Nigerian legal queries.

Layer 1 — Deterministic (regex):
  Reuses NigerianCitationExtractor for case citations; regex for statutes + courts.

Layer 2 — Rule-based (lexicon):
  30+ legal concept detection, motion type classification, area of law inference.
  Sets ParsedQuery.confidence based on match strength.

Layer 3 — LLM-assisted (Claude Haiku, conditional):
  Triggered only when Layer 1+2 confidence < 0.5.
  Extracts motion type, step-back query, and key concepts via structured JSON prompt.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.citations.extractor import NigerianCitationExtractor
from retrieval.models import ParsedQuery

logger = logging.getLogger(__name__)

# ── Legal concept lexicon ──────────────────────────────────────────────────────

LEGAL_CONCEPTS: dict[str, list[str]] = {
    "jurisdiction": [
        "jurisdiction",
        "locus standi",
        "competence of court",
        "standing to sue",
        "subject matter jurisdiction",
        "territorial jurisdiction",
    ],
    "estoppel": [
        "estoppel",
        "res judicata",
        "issue estoppel",
        "cause of action estoppel",
        "collateral estoppel",
        "estoppel per rem judicatam",
    ],
    "contempt": [
        "contempt of court",
        "disobey order",
        "breach of court order",
        "committal for contempt",
    ],
    "limitation": [
        "limitation of action",
        "statute of limitations",
        "time-barred",
        "limitation period",
        "prescription",
    ],
    "natural_justice": [
        "natural justice",
        "fair hearing",
        "audi alteram partem",
        "nemo judex in causa sua",
        "bias",
        "opportunity to be heard",
    ],
    "contract": [
        "contract",
        "breach of contract",
        "offer and acceptance",
        "consideration",
        "misrepresentation",
        "frustration of contract",
        "privity",
    ],
    "evidence": [
        "admissibility",
        "hearsay",
        "documentary evidence",
        "burden of proof",
        "standard of proof",
        "confessional statement",
        "circumstantial evidence",
    ],
    "injunction": [
        "injunction",
        "interlocutory injunction",
        "mareva injunction",
        "mandatory injunction",
        "restraining order",
        "status quo",
    ],
    "appeal": [
        "ground of appeal",
        "leave to appeal",
        "notice of appeal",
        "competent appeal",
        "right of appeal",
        "appellate jurisdiction",
    ],
    "criminal": [
        "mens rea",
        "actus reus",
        "criminal intent",
        "reasonable doubt",
        "prosecution",
        "acquittal",
        "conviction",
        "sentence",
    ],
    "land": [
        "land law",
        "title to land",
        "trespass",
        "adverse possession",
        "customary land",
        "certificate of occupancy",
        "right of occupancy",
    ],
    "company": [
        "company law",
        "corporate veil",
        "shareholder",
        "director",
        "winding up",
        "insolvency",
        "memorandum of association",
    ],
    "constitutional": [
        "fundamental rights",
        "constitutional right",
        "section 33",
        "section 34",
        "section 35",
        "section 36",
        "section 41",
        "section 42",
        "constitutional validity",
        "unconstitutional",
    ],
    "tort": [
        "negligence",
        "duty of care",
        "tortious liability",
        "defamation",
        "libel",
        "slander",
        "nuisance",
        "trespass to person",
    ],
    "election": [
        "election petition",
        "electoral malpractice",
        "pre-election matter",
        "governorship election",
        "senatorial",
        "house of representatives",
    ],
    "chieftaincy": ["chieftaincy", "stool", "skin", "traditional ruler", "emirate"],
    "taxation": ["tax", "income tax", "value added tax", "stamp duty", "tax assessment"],
    "arbitration": ["arbitration", "arbitral award", "arbitral tribunal", "stay for arbitration"],
    "administrative": [
        "administrative action",
        "judicial review",
        "certiorari",
        "mandamus",
        "prohibition",
        "public body",
        "ultra vires",
    ],
    "enforcement": ["enforcement of judgment", "garnishee", "writ of execution", "attachment"],
    "damages": [
        "damages",
        "quantum of damages",
        "special damages",
        "general damages",
        "exemplary damages",
        "punitive damages",
        "loss of earnings",
    ],
    "service": [
        "service of process",
        "substituted service",
        "originating process",
        "writ of summons",
    ],
    "costs": ["costs", "security for costs", "award of costs", "taxation of costs"],
    "interlocutory": [
        "interlocutory",
        "interim",
        "pending trial",
        "balance of convenience",
        "irreparable damage",
        "status quo ante",
    ],
    "discovery": ["discovery", "interrogatories", "inspection of documents", "subpoena"],
    "amendment": ["amendment of pleadings", "amendment of claim", "leave to amend"],
    "joinder": ["joinder", "necessary party", "misjoinder", "non-joinder"],
    "striking_out": [
        "strike out",
        "struck out",
        "no reasonable cause of action",
        "abuse of process",
        "frivolous",
        "vexatious",
    ],
    "stay": [
        "stay of proceedings",
        "stay of execution",
        "pending appeal",
        "suspension of judgment",
    ],
    "summary_judgment": [
        "summary judgment",
        "undefended list",
        "judgment on the pleadings",
        "no defence",
        "admitted facts",
    ],
}

# ── Motion type keyword sets ───────────────────────────────────────────────────

MOTION_KEYWORDS: dict[str, list[str]] = {
    "motion_to_dismiss": [
        "dismiss",
        "struck out",
        "strike out",
        "no cause of action",
        "abuse of process",
        "frivolous",
        "vexatious",
        "no locus standi",
        "want of jurisdiction",
        "incompetent suit",
    ],
    "interlocutory_injunction": [
        "injunction",
        "restrain",
        "interim order",
        "interlocutory relief",
        "balance of convenience",
        "irreparable damage",
        "status quo",
        "mareva",
        "anton piller",
    ],
    "stay_of_proceedings": [
        "stay of proceedings",
        "stay of execution",
        "pending appeal",
        "suspend",
        "halt proceedings",
        "adjourn sine die",
    ],
    "summary_judgment": [
        "summary judgment",
        "undefended list",
        "judgment on the pleadings",
        "no triable issue",
        "no defence disclosed",
    ],
    "extension_of_time": [
        "extension of time",
        "time limit",
        "out of time",
        "enlarge time",
        "time within which",
        "appeal out of time",
        "regularise",
    ],
}

# ── Area of law keyword sets ───────────────────────────────────────────────────

AREA_KEYWORDS: dict[str, list[str]] = {
    "constitutional_law": ["fundamental rights", "constitution", "unconstitutional"],
    "criminal_law": ["criminal", "felony", "misdemeanour", "offence", "conviction"],
    "contract_law": ["contract", "breach", "offer", "acceptance", "consideration"],
    "land_law": ["land", "title", "trespass", "occupancy", "customary"],
    "company_law": ["company", "corporate", "shareholder", "director", "winding up"],
    "family_law": ["matrimonial", "divorce", "custody", "maintenance", "adoption"],
    "electoral_law": ["election", "electoral", "inec", "governorship", "senatorial"],
    "administrative_law": ["judicial review", "certiorari", "mandamus", "public body"],
    "tort_law": ["negligence", "defamation", "nuisance", "trespass"],
    "taxation_law": ["tax", "revenue", "customs", "excise"],
}

_CITATION_EXTRACTOR = NigerianCitationExtractor()


# ── Main parser class ──────────────────────────────────────────────────────────


class QueryParser:
    """Parse a free-text legal query into structured ParsedQuery.

    Args:
        anthropic_client: Optional pre-constructed Anthropic async client.
            If None, will be lazy-constructed from settings when Layer 3 fires.
    """

    def __init__(self, anthropic_client=None) -> None:
        self._anthropic = anthropic_client
        self._extraction_model: str | None = None

    async def parse(self, query: str) -> ParsedQuery:
        """Parse query through all three layers."""
        result = _layer1_regex(query)
        result = _layer2_rules(query, result)

        if result.confidence < _get_settings().llm_confidence_threshold:
            logger.debug(
                "query_parser.layer3_triggered confidence=%.2f query=%.60s",
                result.confidence,
                query,
            )
            result = await self._layer3_llm(query, result)

        return result

    async def _layer3_llm(self, query: str, partial: ParsedQuery) -> ParsedQuery:
        """Enrich low-confidence ParsedQuery via Claude Haiku structured output."""
        client = self._get_anthropic_client()
        settings = _get_settings()

        prompt = _build_layer3_prompt(query, partial)
        try:
            message = await client.messages.create(
                model=settings.extraction_model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
        except Exception as exc:
            logger.warning("query_parser.layer3_failed exc=%s", exc)
            return partial

        if data.get("motion_type") and not partial.motion_type:
            partial.motion_type = data["motion_type"]
        if data.get("concepts"):
            new_concepts = [c for c in data["concepts"] if c not in partial.detected_concepts]
            partial.detected_concepts.extend(new_concepts)
        if data.get("area_of_law") and not partial.area_of_law:
            partial.area_of_law = data["area_of_law"]
        if data.get("step_back_query"):
            partial.step_back_query = data["step_back_query"]
        partial.confidence = 0.6  # bumped after LLM enrichment
        return partial

    def _get_anthropic_client(self):
        if self._anthropic:
            return self._anthropic
        import anthropic

        settings = _get_settings()
        self._anthropic = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        return self._anthropic


# ── Layer implementations ──────────────────────────────────────────────────────


def _layer1_regex(query: str) -> ParsedQuery:
    """Extract deterministic signals: citations, statute refs, court mentions."""
    citations = _CITATION_EXTRACTOR.extract_all(query)
    case_references = [c.full_citation for c in citations]

    return ParsedQuery(
        original=query,
        case_references=case_references,
        confidence=0.2,  # base; boosted by Layer 2
    )


def _layer2_rules(query: str, partial: ParsedQuery) -> ParsedQuery:
    """Concept detection, motion type, area of law — all from lexicon matching."""
    q_lower = query.lower()
    confidence = partial.confidence

    # ── Concept detection ────────────────────────────────────────────────────
    matched_concepts: list[str] = []
    for concept, keywords in LEGAL_CONCEPTS.items():
        if any(kw in q_lower for kw in keywords):
            matched_concepts.append(concept)
    partial.detected_concepts = matched_concepts
    if matched_concepts:
        confidence += 0.15 * min(len(matched_concepts), 3)  # cap at 3 concepts' worth

    # ── Motion type detection ────────────────────────────────────────────────
    if not partial.motion_type:
        best_motion: str | None = None
        best_score = 0
        for motion, keywords in MOTION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in q_lower)
            if score > best_score:
                best_score = score
                best_motion = motion
        if best_motion and best_score >= 1:
            partial.motion_type = best_motion
            confidence += 0.20

    # ── Area of law inference ────────────────────────────────────────────────
    if not partial.area_of_law:
        for area, keywords in AREA_KEYWORDS.items():
            if any(kw in q_lower for kw in keywords):
                partial.area_of_law = area
                confidence += 0.10
                break

    # ── Bonus: explicit citations found ─────────────────────────────────────
    if partial.case_references:
        confidence += 0.10

    partial.confidence = min(confidence, 1.0)
    return partial


def _build_layer3_prompt(query: str, partial: ParsedQuery) -> str:
    known_motion = partial.motion_type or "unknown"
    known_concepts = ", ".join(partial.detected_concepts) if partial.detected_concepts else "none"
    return f"""You are a Nigerian legal query analyst. Analyse the following legal search query and return a JSON object with these fields:

{{
  "motion_type": one of ["motion_to_dismiss", "interlocutory_injunction", "stay_of_proceedings", "summary_judgment", "extension_of_time"] or null,
  "concepts": list of up to 5 relevant Nigerian legal concepts (strings),
  "area_of_law": one of ["constitutional_law", "criminal_law", "contract_law", "land_law", "company_law", "family_law", "electoral_law", "administrative_law", "tort_law", "taxation_law"] or null,
  "step_back_query": a broader restatement of the query focusing on the underlying legal principle (max 200 chars)
}}

Already detected: motion_type={known_motion}, concepts=[{known_concepts}]

Query: {query}

Return ONLY valid JSON with no markdown fences."""


def _get_settings():
    """Lazy-import settings to avoid circular imports in tests."""
    from config import settings

    return settings
