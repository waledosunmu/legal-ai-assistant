"""Unit tests for src/retrieval/query_parser.py"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from retrieval.query_parser import QueryParser, _layer1_regex, _layer2_rules


# ── Layer 1 — regex + citation extraction ─────────────────────────────────────


class TestLayer1:
    def test_no_citations(self) -> None:
        result = _layer1_regex("grounds for dismissal of a suit for want of jurisdiction")
        assert result.case_references == []
        assert result.confidence == pytest.approx(0.2)

    def test_citation_extracted(self) -> None:
        text = "see Bakare v. State (2000) 3 NWLR (Pt. 648) 1 for the principle"
        result = _layer1_regex(text)
        assert len(result.case_references) >= 1
        assert any("NWLR" in ref for ref in result.case_references)

    def test_original_preserved(self) -> None:
        q = "What are the requirements for an interlocutory injunction?"
        result = _layer1_regex(q)
        assert result.original == q


# ── Layer 2 — concept + motion + area detection ────────────────────────────────


class TestLayer2:
    def test_motion_to_dismiss_detected(self) -> None:
        q = "application to strike out the suit for want of jurisdiction"
        partial = _layer1_regex(q)
        result = _layer2_rules(q, partial)
        assert result.motion_type == "motion_to_dismiss"
        assert result.confidence > 0.2

    def test_injunction_motion_detected(self) -> None:
        q = "conditions for granting interlocutory injunction balance of convenience"
        partial = _layer1_regex(q)
        result = _layer2_rules(q, partial)
        assert result.motion_type == "interlocutory_injunction"

    def test_stay_motion_detected(self) -> None:
        q = "stay of execution pending appeal Supreme Court"
        partial = _layer1_regex(q)
        result = _layer2_rules(q, partial)
        assert result.motion_type == "stay_of_proceedings"

    def test_extension_of_time_detected(self) -> None:
        q = "extension of time to file appeal out of time application"
        partial = _layer1_regex(q)
        result = _layer2_rules(q, partial)
        assert result.motion_type == "extension_of_time"

    def test_concept_detected(self) -> None:
        q = "what is locus standi in Nigerian courts"
        partial = _layer1_regex(q)
        result = _layer2_rules(q, partial)
        assert "jurisdiction" in result.detected_concepts  # locus standi maps to jurisdiction

    def test_multiple_concepts(self) -> None:
        q = "jurisdiction and estoppel res judicata principles in contract law"
        partial = _layer1_regex(q)
        result = _layer2_rules(q, partial)
        assert "jurisdiction" in result.detected_concepts
        assert "estoppel" in result.detected_concepts
        assert "contract" in result.detected_concepts

    def test_area_of_law_criminal(self) -> None:
        q = "standard of proof in criminal proceedings beyond reasonable doubt"
        partial = _layer1_regex(q)
        result = _layer2_rules(q, partial)
        assert result.area_of_law == "criminal_law"

    def test_area_of_law_electoral(self) -> None:
        q = "time limit for filing election petition after governorship election"
        partial = _layer1_regex(q)
        result = _layer2_rules(q, partial)
        assert result.area_of_law == "electoral_law"

    def test_confidence_increases_with_matches(self) -> None:
        q_bare = "some query with no legal terms"
        q_rich = "motion to strike out suit for want of jurisdiction estoppel contract"
        partial_bare = _layer1_regex(q_bare)
        partial_rich = _layer1_regex(q_rich)
        result_bare = _layer2_rules(q_bare, partial_bare)
        result_rich = _layer2_rules(q_rich, partial_rich)
        assert result_rich.confidence > result_bare.confidence

    def test_confidence_capped_at_one(self) -> None:
        q = ("jurisdiction locus standi estoppel res judicata injunction "
             "natural justice fundamental rights contract breach tort negligence "
             "dismissal strike out stay of execution")
        partial = _layer1_regex(q)
        result = _layer2_rules(q, partial)
        assert result.confidence <= 1.0


# ── Layer 3 — LLM (mocked) ────────────────────────────────────────────────────


class TestLayer3:
    def _make_anthropic_client(self, response_json: dict) -> MagicMock:
        msg = MagicMock()
        msg.content = [MagicMock(text=json.dumps(response_json))]
        messages_mock = MagicMock()
        messages_mock.create = AsyncMock(return_value=msg)
        client = MagicMock()
        client.messages = messages_mock
        return client

    @pytest.mark.asyncio
    async def test_layer3_triggered_on_low_confidence(self) -> None:
        """QueryParser calls Haiku when L1+L2 confidence is low."""
        llm_response = {
            "motion_type": "motion_to_dismiss",
            "concepts": ["jurisdiction"],
            "area_of_law": "constitutional_law",
            "step_back_query": "What are the grounds to dismiss a suit in Nigeria?",
        }
        client = self._make_anthropic_client(llm_response)
        parser = QueryParser(anthropic_client=client)

        # Query with no keywords → low L2 confidence → triggers L3
        result = await parser.parse("some vague query without clear terms hmm")
        client.messages.create.assert_called_once()
        assert result.motion_type == "motion_to_dismiss"
        assert "jurisdiction" in result.detected_concepts
        assert result.step_back_query is not None

    @pytest.mark.asyncio
    async def test_layer3_not_triggered_on_high_confidence(self) -> None:
        """QueryParser skips Haiku when L1+L2 already confident."""
        client = MagicMock()
        client.messages = MagicMock()
        client.messages.create = AsyncMock()
        parser = QueryParser(anthropic_client=client)

        # Rich query → high L2 confidence → no LLM call
        result = await parser.parse(
            "application to dismiss suit for want of jurisdiction "
            "estoppel res judicata natural justice motion to strike out"
        )
        client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_layer3_llm_failure_returns_partial(self) -> None:
        """If Haiku fails, returns whatever Layer 2 produced (no crash)."""
        client = MagicMock()
        client.messages = MagicMock()
        client.messages.create = AsyncMock(side_effect=Exception("API down"))
        parser = QueryParser(anthropic_client=client)

        # Force low confidence with a vague query
        result = await parser.parse("something unclear")
        assert result.original == "something unclear"

    @pytest.mark.asyncio
    async def test_existing_motion_not_overwritten(self) -> None:
        """Layer 3 does not overwrite motion_type already detected by Layer 2."""
        llm_response = {
            "motion_type": "summary_judgment",  # different from L2 detection
            "concepts": ["contract"],
            "area_of_law": None,
            "step_back_query": "general principle",
        }
        client = self._make_anthropic_client(llm_response)
        parser = QueryParser(anthropic_client=client)

        result = await parser.parse(
            "application for stay of execution pending appeal vague text here"
        )
        # L2 detected stay_of_proceedings; L3 cannot overwrite it
        assert result.motion_type == "stay_of_proceedings"
