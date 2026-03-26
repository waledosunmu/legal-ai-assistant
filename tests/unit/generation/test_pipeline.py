"""Unit tests for generation.pipeline — MotionGenerationPipeline.

All LLM calls are mocked. Tests verify orchestration logic, JSON parsing,
citation extraction, and document assembly.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from generation.models import ArgumentSection, GenerationRequest
from generation.pipeline import MotionGenerationPipeline, _extract_principle_near_case
from generation.verification import CitationVerifier

# ── Fixtures ───────────────────────────────────────────────────────────────────


def _mock_anthropic_response(text: str):
    """Create a mock Anthropic API response."""
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def _mock_client(*responses: str):
    """Create a mock AsyncAnthropic that returns the given texts in order."""
    client = AsyncMock()
    side_effects = [_mock_anthropic_response(t) for t in responses]
    client.messages.create = AsyncMock(side_effect=side_effects)
    return client


def _gen_request(**kw) -> GenerationRequest:
    defaults = dict(
        case_facts="The defendant ABC Ltd breached a supply contract with XYZ Ltd by failing to deliver goods worth N50M.",
        motion_type="motion_to_dismiss",
        court_name="Federal High Court",
        division="Lagos",
        location="Lagos",
        suit_number="FHC/L/CS/1/2026",
        applicant_name="ABC Ltd",
        applicant_description="Applicant",
        respondent_name="XYZ Ltd",
        respondent_description="Respondent",
        position="applicant",
        relief_sought="An order dismissing this suit for want of jurisdiction",
        selected_cases=[
            {
                "case_id": "c1",
                "case_name": "Madukolu v. Nkemdilim",
                "case_name_short": "Madukolu v. Nkemdilim",
                "citation": "(1962) 2 SCNLR 341",
                "court": "NGSC",
                "year": 1962,
                "matched_segment": {"type": "RATIO", "content": "Jurisdiction is fundamental..."},
            },
        ],
        statutes=[
            {
                "section": "Section 251(1)(d)",
                "act": "CFRN 1999",
                "content": "Federal High Court jurisdiction",
            },
        ],
        counsel_name="A. Barrister Esq.",
        counsel_firm="Law Chambers",
        counsel_address="1 Law Lane, Lagos",
        date="19th March, 2026",
    )
    defaults.update(kw)
    return GenerationRequest(**defaults)


# ── Readiness assessment ───────────────────────────────────────────────────────


class TestReadinessAssessment:
    def test_ready_with_sc_cases(self):
        pipeline = MotionGenerationPipeline(
            anthropic_client=AsyncMock(),
            verifier=CitationVerifier(),
        )
        result = pipeline._assess_readiness(_gen_request())
        assert result["ready"] is True
        assert result["warnings"] == []

    def test_warns_no_cases(self):
        pipeline = MotionGenerationPipeline(
            anthropic_client=AsyncMock(),
            verifier=CitationVerifier(),
        )
        req = _gen_request(selected_cases=[])
        result = pipeline._assess_readiness(req)
        assert result["ready"] is False
        assert any("No cases" in w for w in result["warnings"])

    def test_warns_no_binding_precedent(self):
        pipeline = MotionGenerationPipeline(
            anthropic_client=AsyncMock(),
            verifier=CitationVerifier(),
        )
        req = _gen_request(
            selected_cases=[{"case_id": "c1", "case_name": "Lower v. Court", "court": "NGLAHC"}]
        )
        result = pipeline._assess_readiness(req)
        assert any("binding precedent" in w for w in result["warnings"])

    def test_warns_no_statutes(self):
        pipeline = MotionGenerationPipeline(
            anthropic_client=AsyncMock(),
            verifier=CitationVerifier(),
        )
        req = _gen_request(statutes=[])
        result = pipeline._assess_readiness(req)
        assert any("statutory" in w.lower() for w in result["warnings"])


# ── Issue formulation ──────────────────────────────────────────────────────────


class TestIssueFormulation:
    @pytest.mark.asyncio
    async def test_parses_json_issues(self):
        issues_json = json.dumps(
            [
                "Whether the court has jurisdiction?",
                "Whether the cause of action is disclosed?",
            ]
        )
        client = _mock_client(issues_json)
        pipeline = MotionGenerationPipeline(
            anthropic_client=client,
            verifier=CitationVerifier(),
        )
        issues = await pipeline._formulate_issues(_gen_request())
        assert len(issues) == 2
        assert "jurisdiction" in issues[0].lower()

    @pytest.mark.asyncio
    async def test_handles_markdown_wrapped_json(self):
        issues_json = '```json\n["Issue 1?", "Issue 2?"]\n```'
        client = _mock_client(issues_json)
        pipeline = MotionGenerationPipeline(
            anthropic_client=client,
            verifier=CitationVerifier(),
        )
        issues = await pipeline._formulate_issues(_gen_request())
        assert len(issues) == 2

    @pytest.mark.asyncio
    async def test_caps_at_four_issues(self):
        issues_json = json.dumps(["I1?", "I2?", "I3?", "I4?", "I5?"])
        client = _mock_client(issues_json)
        pipeline = MotionGenerationPipeline(
            anthropic_client=client,
            verifier=CitationVerifier(),
        )
        issues = await pipeline._formulate_issues(_gen_request())
        assert len(issues) == 4

    @pytest.mark.asyncio
    async def test_fallback_on_invalid_json(self):
        client = _mock_client("This is not JSON at all")
        pipeline = MotionGenerationPipeline(
            anthropic_client=client,
            verifier=CitationVerifier(),
        )
        issues = await pipeline._formulate_issues(_gen_request())
        # Should return fallback issues for motion_to_dismiss
        assert len(issues) >= 2
        assert any("jurisdiction" in i.lower() or "cause of action" in i.lower() for i in issues)


# ── Argument generation ────────────────────────────────────────────────────────


class TestArgumentGeneration:
    @pytest.mark.asyncio
    async def test_generates_argument_section(self):
        arg_text = (
            "It is trite law that jurisdiction is fundamental. "
            "In Madukolu v. Nkemdilim (1962) 2 SCNLR 341, the Supreme Court "
            "held that a court must be properly constituted."
        )
        client = _mock_client(arg_text)
        pipeline = MotionGenerationPipeline(
            anthropic_client=client,
            verifier=CitationVerifier(),
        )
        arg = await pipeline._generate_argument(
            issue_number=1,
            issue_text="Whether the court has jurisdiction?",
            req=_gen_request(),
        )
        assert isinstance(arg, ArgumentSection)
        assert arg.issue_number == 1
        assert "jurisdiction" in arg.argument_text.lower()

    @pytest.mark.asyncio
    async def test_extracts_citations_from_text(self):
        arg_text = (
            "As held in Madukolu v. Nkemdilim (1962) 2 SCNLR 341, "
            "the court must have jurisdiction."
        )
        client = _mock_client(arg_text)
        pipeline = MotionGenerationPipeline(
            anthropic_client=client,
            verifier=CitationVerifier(),
        )
        arg = await pipeline._generate_argument(
            issue_number=1,
            issue_text="Jurisdiction?",
            req=_gen_request(),
        )
        assert len(arg.cases_cited) >= 1
        assert arg.cases_cited[0]["name"] == "Madukolu v. Nkemdilim"

    @pytest.mark.asyncio
    async def test_extracts_statutes_from_text(self):
        arg_text = "By virtue of Section 251(1)(d) of the CFRN 1999, this court has jurisdiction."
        client = _mock_client(arg_text)
        pipeline = MotionGenerationPipeline(
            anthropic_client=client,
            verifier=CitationVerifier(),
        )
        arg = await pipeline._generate_argument(
            issue_number=1,
            issue_text="Jurisdiction?",
            req=_gen_request(),
        )
        assert len(arg.statutes_cited) >= 1


# ── Supporting sections ────────────────────────────────────────────────────────


class TestSupportingSections:
    @pytest.mark.asyncio
    async def test_parse_json_list_valid(self):
        pipeline = MotionGenerationPipeline(
            anthropic_client=AsyncMock(),
            verifier=CitationVerifier(),
        )
        result = pipeline._parse_json_list('["Prayer 1", "Prayer 2"]')
        assert result == ["Prayer 1", "Prayer 2"]

    @pytest.mark.asyncio
    async def test_parse_json_list_markdown_wrapped(self):
        pipeline = MotionGenerationPipeline(
            anthropic_client=AsyncMock(),
            verifier=CitationVerifier(),
        )
        result = pipeline._parse_json_list('```json\n["a", "b"]\n```')
        assert result == ["a", "b"]

    @pytest.mark.asyncio
    async def test_parse_json_list_fallback(self):
        pipeline = MotionGenerationPipeline(
            anthropic_client=AsyncMock(),
            verifier=CitationVerifier(),
        )
        result = pipeline._parse_json_list("1. First item\n2. Second item")
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_parse_json_list_empty_fallback(self):
        pipeline = MotionGenerationPipeline(
            anthropic_client=AsyncMock(),
            verifier=CitationVerifier(),
        )
        result = pipeline._parse_json_list("")
        assert len(result) >= 1  # fallback produces at least one item

    @pytest.mark.asyncio
    async def test_generate_prayers(self):
        prayers_json = json.dumps(["AN ORDER dismissing the suit", "Costs of this application"])
        client = _mock_client(prayers_json)
        pipeline = MotionGenerationPipeline(
            anthropic_client=client,
            verifier=CitationVerifier(),
        )
        prayers = await pipeline._generate_prayers(_gen_request(), ["Issue 1?"])
        assert len(prayers) == 2

    @pytest.mark.asyncio
    async def test_generate_grounds(self):
        grounds_json = json.dumps(["Ground 1", "Ground 2", "Ground 3"])
        client = _mock_client(grounds_json)
        pipeline = MotionGenerationPipeline(
            anthropic_client=client,
            verifier=CitationVerifier(),
        )
        grounds = await pipeline._generate_grounds(_gen_request(), ["Issue?"])
        assert len(grounds) == 3

    @pytest.mark.asyncio
    async def test_generate_affidavit(self):
        affi_json = json.dumps(["I am the Director", "The contract was signed"])
        client = _mock_client(affi_json)
        pipeline = MotionGenerationPipeline(
            anthropic_client=client,
            verifier=CitationVerifier(),
        )
        paragraphs = await pipeline._generate_affidavit(_gen_request())
        assert len(paragraphs) == 2

    @pytest.mark.asyncio
    async def test_generate_introduction(self):
        intro = "This Written Address is filed in support of the Applicant's Motion."
        client = _mock_client(intro)
        pipeline = MotionGenerationPipeline(
            anthropic_client=client,
            verifier=CitationVerifier(),
        )
        result = await pipeline._generate_introduction(_gen_request(), ["Issue?"])
        assert "Written Address" in result

    @pytest.mark.asyncio
    async def test_generate_conclusion(self):
        conclusion = "We urge this Honourable Court to grant the application."
        client = _mock_client(conclusion)
        pipeline = MotionGenerationPipeline(
            anthropic_client=client,
            verifier=CitationVerifier(),
        )
        result = await pipeline._generate_conclusion(_gen_request(), ["Issue?"], [])
        assert "urge" in result.lower()


# ── Strength assessment ────────────────────────────────────────────────────────


class TestStrengthAssessment:
    @pytest.mark.asyncio
    async def test_parses_strength_report(self):
        strength_json = json.dumps(
            {
                "legal_soundness": 8,
                "factual_applicability": 7,
                "authority_strength": 9,
                "vulnerability": 6,
                "weaknesses": [],
                "misattribution_risk": [],
            }
        )
        client = _mock_client(strength_json)
        pipeline = MotionGenerationPipeline(
            anthropic_client=client,
            verifier=CitationVerifier(),
        )
        arg = ArgumentSection(
            issue_number=1,
            issue_text="Test?",
            argument_text="Test argument",
            cases_cited=[{"name": "A v. B", "citation": "(2020) 1 NWLR 1"}],
        )
        report = await pipeline._assess_argument_strength([arg], _gen_request())
        assert len(report) == 1
        assert report[0]["overall_score"] == 7.5
        assert report[0]["issue_number"] == 1

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self):
        client = _mock_client("Not valid JSON")
        pipeline = MotionGenerationPipeline(
            anthropic_client=client,
            verifier=CitationVerifier(),
        )
        arg = ArgumentSection(issue_number=1, issue_text="T?", argument_text="T")
        report = await pipeline._assess_argument_strength([arg], _gen_request())
        assert report[0].get("_parse_error") is True


# ── Counter-arguments ──────────────────────────────────────────────────────────


class TestCounterArguments:
    @pytest.mark.asyncio
    async def test_parses_counter_arguments(self):
        ca_json = json.dumps(
            [
                {
                    "issue_number": 1,
                    "counter_argument": "The contract was validly formed",
                    "potential_authority": "A v. B (2020)",
                    "suggested_rebuttal": "However, the statute bars this",
                }
            ]
        )
        client = _mock_client(ca_json)
        pipeline = MotionGenerationPipeline(
            anthropic_client=client,
            verifier=CitationVerifier(),
        )
        arg = ArgumentSection(issue_number=1, issue_text="T?", argument_text="T")
        result = await pipeline._generate_counter_arguments([arg], _gen_request())
        assert len(result) == 1
        assert result[0]["issue_number"] == 1

    @pytest.mark.asyncio
    async def test_returns_empty_on_invalid_json(self):
        client = _mock_client("Invalid")
        pipeline = MotionGenerationPipeline(
            anthropic_client=client,
            verifier=CitationVerifier(),
        )
        arg = ArgumentSection(issue_number=1, issue_text="T?", argument_text="T")
        result = await pipeline._generate_counter_arguments([arg], _gen_request())
        assert result == []


# ── Document assembly ──────────────────────────────────────────────────────────


class TestDocumentAssembly:
    def test_assemble_motion_paper(self):
        pipeline = MotionGenerationPipeline(
            anthropic_client=AsyncMock(),
            verifier=CitationVerifier(),
        )
        mp = pipeline._assemble_motion_paper(
            _gen_request(),
            prayers=["Dismiss the suit"],
            grounds=["No jurisdiction"],
        )
        assert mp.court_name == "IN THE FEDERAL HIGH COURT"
        assert mp.applicant_name == "ABC Ltd"
        assert len(mp.prayers) == 1
        assert len(mp.grounds) == 1

    def test_assemble_affidavit(self):
        pipeline = MotionGenerationPipeline(
            anthropic_client=AsyncMock(),
            verifier=CitationVerifier(),
        )
        aff = pipeline._assemble_affidavit(
            _gen_request(),
            paragraphs=["Facts", "More facts"],
        )
        assert aff.deponent_name == "ABC Ltd"  # falls back to applicant_name
        assert len(aff.paragraphs) == 2

    def test_assemble_written_address(self):
        pipeline = MotionGenerationPipeline(
            anthropic_client=AsyncMock(),
            verifier=CitationVerifier(),
        )
        args = [ArgumentSection(issue_number=1, issue_text="Q?", argument_text="A")]
        wa = pipeline._assemble_written_address(
            _gen_request(),
            issues=["Question?"],
            arguments=args,
            introduction="Intro text",
            conclusion="Conclusion text",
        )
        assert wa.title == "Written Address in Support of Motion To Dismiss"
        assert len(wa.arguments) == 1
        assert wa.introduction == "Intro text"


# ── Helper functions ───────────────────────────────────────────────────────────


class TestHelpers:
    def test_prepare_case_context_empty(self):
        result = MotionGenerationPipeline._prepare_case_context([])
        assert result == "No cases available."

    def test_prepare_case_context_with_cases(self):
        cases = [
            {
                "case_name": "A v. B",
                "citation": "(2020) 1 NWLR",
                "court": "NGSC",
                "matched_segment": {"content": "holding text"},
            }
        ]
        result = MotionGenerationPipeline._prepare_case_context(cases)
        assert "A v. B" in result
        assert "NGSC" in result

    def test_format_statutes_empty(self):
        result = MotionGenerationPipeline._format_statutes([])
        assert "No statutory" in result

    def test_format_statutes_with_data(self):
        statutes = [{"section": "S.251", "act": "CFRN", "content": "jurisdiction"}]
        result = MotionGenerationPipeline._format_statutes(statutes)
        assert "S.251" in result

    def test_extract_pursuant_to_from_statutes(self):
        req = _gen_request()
        result = MotionGenerationPipeline._extract_pursuant_to(req)
        assert len(result) >= 1
        assert "Section 251" in result[0]

    def test_extract_pursuant_to_empty(self):
        req = _gen_request(statutes=[])
        result = MotionGenerationPipeline._extract_pursuant_to(req)
        assert "inherent jurisdiction" in result[0]

    def test_extract_principle_near_case(self):
        text = "In Madukolu v. Nkemdilim, the court held that jurisdiction is fundamental."
        principle = _extract_principle_near_case(text, "Madukolu v. Nkemdilim")
        assert "jurisdiction" in principle.lower()

    def test_extract_principle_not_found(self):
        text = "Some unrelated text without case names."
        principle = _extract_principle_near_case(text, "Nonexistent v. Case")
        assert principle == ""

    def test_update_citation_status(self):
        args = [
            ArgumentSection(
                issue_number=1,
                issue_text="Q?",
                argument_text="A",
                cases_cited=[
                    {"name": "A v. B", "citation": "(2020) 1 NWLR"},
                    {"name": "C v. D", "citation": "(2021) 2 NWLR"},
                ],
            )
        ]
        verified = [
            {
                "name": "A v. B",
                "citation": "(2020) 1 NWLR",
                "verified": True,
                "status": "fully_verified",
            },
            {
                "name": "C v. D",
                "citation": "(2021) 2 NWLR",
                "verified": False,
                "status": "not_in_corpus",
            },
        ]
        MotionGenerationPipeline._update_citation_status(args, verified)
        assert args[0].cases_cited[0]["verified"] is True
        assert args[0].cases_cited[1]["verification_status"] == "not_in_corpus"


# ── Full pipeline (mocked end-to-end) ──────────────────────────────────────────


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_end_to_end_generation(self):
        """Full pipeline with all LLM calls mocked."""
        # LLM responses in call order:
        responses = [
            # 1. Issue formulation (Sonnet)
            json.dumps(
                ["Whether the court has jurisdiction?", "Whether cause of action is disclosed?"]
            ),
            # 2-3. Argument generation per issue (Sonnet, parallel — 2 issues)
            "In Madukolu v. Nkemdilim the court held that jurisdiction is fundamental to adjudication.",
            "A cause of action requires facts disclosing a right recognised by law.",
            # 4-5. Strength assessment (Haiku, 2 issues)
            json.dumps(
                {
                    "legal_soundness": 8,
                    "factual_applicability": 7,
                    "authority_strength": 9,
                    "vulnerability": 6,
                    "weaknesses": [],
                    "misattribution_risk": [],
                }
            ),
            json.dumps(
                {
                    "legal_soundness": 7,
                    "factual_applicability": 6,
                    "authority_strength": 7,
                    "vulnerability": 5,
                    "weaknesses": [],
                    "misattribution_risk": [],
                }
            ),
            # 6. Counter-arguments (Haiku)
            json.dumps(
                [
                    {
                        "issue_number": 1,
                        "counter_argument": "CA1",
                        "potential_authority": "",
                        "suggested_rebuttal": "R1",
                    }
                ]
            ),
            # 7. Prayers (Haiku)
            json.dumps(["AN ORDER dismissing the suit"]),
            # 8. Grounds (Haiku)
            json.dumps(["The court lacks jurisdiction"]),
            # 9. Affidavit paragraphs (Haiku)
            json.dumps(["I am the Director of the Applicant"]),
            # 10. Introduction (Haiku)
            "This Written Address is filed in support of the Motion.",
            # 11. Conclusion (Haiku)
            "We urge this Honourable Court to grant the reliefs sought.",
        ]
        client = AsyncMock()
        client.messages.create = AsyncMock(
            side_effect=[_mock_anthropic_response(r) for r in responses]
        )

        verifier = CitationVerifier(db_pool=None)
        pipeline = MotionGenerationPipeline(
            anthropic_client=client,
            verifier=verifier,
        )

        result = await pipeline.generate(_gen_request())

        # Verify document structure
        assert result.motion_paper.applicant_name == "ABC Ltd"
        assert len(result.motion_paper.prayers) >= 1
        assert len(result.written_address.issues_for_determination) == 2
        assert len(result.written_address.arguments) == 2
        assert "jurisdiction" in result.written_address.arguments[0].argument_text.lower()

        # Verify audit trail
        event_types = [e["event_type"] for e in result.audit_log]
        assert "readiness_assessed" in event_types
        assert "issues_formulated" in event_types
        assert "arguments_generated" in event_types
        assert "citations_verified" in event_types
        assert "documents_assembled" in event_types

        # Verify readiness
        assert result.readiness_report["ready"] is True
