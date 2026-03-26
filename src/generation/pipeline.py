"""MotionGenerationPipeline — orchestrates all generation steps.

Pipeline:
  1. Readiness assessment (confidence gate)
  2. Issue formulation (Sonnet)
  3. Argument generation per issue (Sonnet, parallel)
  4. Supporting sections (Haiku, parallel) — prayers, grounds, affidavit, intro, conclusion
  5. Citation verification (5-level)
  6. Argument strength assessment (Haiku)
  7. Counter-argument analysis (Haiku)
  8. Document assembly via templates
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import cast

import anthropic
from anthropic.types import TextBlock

from generation.models import (
    ArgumentSection,
    AuditLog,
    GenerationRequest,
    GenerationResult,
    MotionPaper,
    SupportingAffidavit,
    WrittenAddress,
)
from generation.templates import FALLBACK_ISSUES
from generation.verification import CitationVerifier

logger = logging.getLogger(__name__)


class MotionGenerationPipeline:
    """Orchestrate generation of all three motion documents."""

    def __init__(
        self,
        anthropic_client: anthropic.AsyncAnthropic,
        verifier: CitationVerifier,
        generation_model: str = "claude-sonnet-4-6",
        extraction_model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.client = anthropic_client
        self.verifier = verifier
        self.generation_model = generation_model
        self.extraction_model = extraction_model

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate all three motion documents end-to-end."""
        audit = AuditLog(draft_id=request.draft_id or "new")

        # Step 0: Readiness assessment
        readiness = self._assess_readiness(request)
        audit.record("readiness_assessed", readiness)

        # Step 1: Formulate issues
        issues = await self._formulate_issues(request)
        audit.record("issues_formulated", {"issues": issues})

        # Step 2: Generate arguments in parallel
        arg_tasks = [
            self._generate_argument(i + 1, issue, request) for i, issue in enumerate(issues)
        ]
        arguments = await asyncio.gather(*arg_tasks)
        audit.record(
            "arguments_generated",
            {
                "count": len(arguments),
                "previews": [a.argument_text[:200] for a in arguments],
            },
        )

        # Step 3: Citation verification
        all_citations = []
        for arg in arguments:
            all_citations.extend(arg.cases_cited)
        verified_citations = await self.verifier.verify_batch(all_citations)
        self._update_citation_status(arguments, verified_citations)
        audit.record(
            "citations_verified",
            {
                "total": len(verified_citations),
                "results": [
                    {"name": c.get("name"), "status": c.get("status")} for c in verified_citations
                ],
            },
        )

        # Step 3b: Strength assessment + counter-arguments in parallel
        strength_task = self._assess_argument_strength(arguments, request)
        counter_task = self._generate_counter_arguments(arguments, request)

        # Step 4: Supporting sections in parallel
        prayers_task = self._generate_prayers(request, issues)
        grounds_task = self._generate_grounds(request, issues)
        affidavit_task = self._generate_affidavit(request)
        intro_task = self._generate_introduction(request, issues)
        conclusion_task = self._generate_conclusion(request, issues, arguments)

        (
            strength_report,
            counter_arguments,
            prayers,
            grounds,
            affidavit_paragraphs,
            introduction,
            conclusion,
        ) = cast(
            tuple[list[dict], list[dict], list[str], list[str], list[str], str, str],
            await asyncio.gather(
                strength_task,
                counter_task,
                prayers_task,
                grounds_task,
                affidavit_task,
                intro_task,
                conclusion_task,
            ),
        )

        audit.record("strength_assessed", {"report": strength_report})
        audit.record("counter_arguments_generated", {"count": len(counter_arguments)})

        # Step 5: Assemble documents
        motion_paper = self._assemble_motion_paper(request, prayers, grounds)
        affidavit = self._assemble_affidavit(request, affidavit_paragraphs)
        written_address = self._assemble_written_address(
            request,
            issues,
            list(arguments),
            introduction,
            conclusion,
        )
        audit.record(
            "documents_assembled",
            {
                "issues_count": len(issues),
                "arguments_count": len(arguments),
            },
        )

        return GenerationResult(
            motion_paper=motion_paper,
            affidavit=affidavit,
            written_address=written_address,
            citation_report=verified_citations,
            strength_report=strength_report,
            counter_arguments=counter_arguments,
            readiness_report=readiness,
            audit_log=audit.entries,
        )

    # ── Step 0: Readiness ──────────────────────────────────────────────────────

    def _assess_readiness(self, req: GenerationRequest) -> dict:
        """Pre-generation confidence gate."""
        warnings: list[str] = []
        if not req.selected_cases:
            warnings.append("No cases selected — arguments will lack authority.")
        else:
            courts = {c.get("court", "") for c in req.selected_cases}
            if not courts & {"NGSC", "NGCA"}:
                warnings.append(
                    "No Supreme Court or Court of Appeal authority selected. "
                    "Your motion will lack binding precedent."
                )
        if not req.statutes:
            warnings.append("No statutory provisions provided.")

        return {
            "ready": len(warnings) == 0,
            "warnings": warnings,
            "cases_count": len(req.selected_cases),
            "statutes_count": len(req.statutes),
        }

    # ── Step 1: Issue formulation ──────────────────────────────────────────────

    async def _formulate_issues(self, req: GenerationRequest) -> list[str]:
        """Generate 2-4 issues for determination using Sonnet."""
        case_summaries = self._prepare_case_context(req.selected_cases)
        statutes_text = self._format_statutes(req.statutes)

        prompt = f"""You are a senior Nigerian litigation lawyer formulating issues \
for determination in a {req.motion_type} application before the {req.court_name}.

CASE FACTS:
{req.case_facts[:3000]}

POSITION: We represent the {req.position}.
RELIEF SOUGHT: {req.relief_sought}

RELEVANT PRECEDENTS AVAILABLE:
{case_summaries}

RELEVANT STATUTES:
{statutes_text}

Formulate 2-4 issues for determination. Each issue should be:
1. Phrased as a question
2. Answerable from the available precedents and statutes
3. Leading towards the relief sought when answered favourably
4. Following Nigerian legal convention for issue formulation

Common patterns:
- "Whether this Honourable Court has the jurisdiction to..."
- "Whether the Applicant has established sufficient grounds for..."
- "Whether the Respondent is entitled to..."

Return ONLY a JSON array of issue strings. No markdown, no explanation.
["Issue 1?", "Issue 2?"]"""

        response = await self.client.messages.create(
            model=self.generation_model,
            max_tokens=1024,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )

        content = next((b.text for b in response.content if isinstance(b, TextBlock)), "").strip()
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            issues = json.loads(content)
            if isinstance(issues, list) and all(isinstance(i, str) for i in issues):
                return issues[:4]
        except (json.JSONDecodeError, TypeError):
            pass

        return FALLBACK_ISSUES.get(req.motion_type, FALLBACK_ISSUES["motion_to_dismiss"])

    # ── Step 2: Argument generation ────────────────────────────────────────────

    async def _generate_argument(
        self,
        issue_number: int,
        issue_text: str,
        req: GenerationRequest,
    ) -> ArgumentSection:
        """Generate IRAC-structured argument for a single issue using Sonnet."""
        relevant_cases = self._select_cases_for_issue(issue_text, req.selected_cases)
        case_context = self._prepare_detailed_case_context(relevant_cases)
        statutes_text = self._format_statutes(req.statutes)

        prompt = f"""You are a senior Nigerian litigation lawyer drafting the argument \
section of a Written Address for a {req.motion_type} application.

You are arguing ISSUE {issue_number}:
"{issue_text}"

CASE FACTS:
{req.case_facts[:2000]}

POSITION: We represent the {req.position}.

AVAILABLE PRECEDENTS (use ONLY these — do not invent cases):
{case_context}

AVAILABLE STATUTES:
{statutes_text}

Draft the argument for this issue following Nigerian legal convention:

1. STATE THE LEGAL PRINCIPLE: Begin by stating the settled legal position
   on this issue, citing the leading Supreme Court authority.

2. CITE SUPPORTING CASES: For each proposition, cite the specific case
   and quote or paraphrase the relevant holding. Use the EXACT case names
   and citations provided above. DO NOT invent or modify case names.

3. APPLY TO FACTS: Show how the legal principle applies to our client's
   specific facts.

4. ANTICIPATE COUNTER-ARGUMENTS: Briefly address the strongest argument
   the other side might make, and distinguish or refute it.

5. CONCLUDE: State the conclusion that supports our prayer.

CITATION FORMAT:
- When citing a case: "In [Case Name] [Citation], the Supreme Court
  held that [principle]."
- When citing a statute: "By virtue of Section [X] of the [Act],..."

CRITICAL RULES:
- ONLY cite cases from the AVAILABLE PRECEDENTS list above
- NEVER invent case names, citations, or holdings
- If you're not certain about a holding, describe it generally rather
  than fabricating a specific quote
- Every legal proposition must be supported by at least one authority
- Write in formal Nigerian legal English
- Target length: 400-800 words per issue

Return ONLY the argument text. No JSON, no markdown headers."""

        response = await self.client.messages.create(
            model=self.generation_model,
            max_tokens=2048,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )

        argument_text = next(
            (b.text for b in response.content if isinstance(b, TextBlock)), ""
        ).strip()
        extracted_citations = self._extract_citations_from_text(argument_text, relevant_cases)
        extracted_statutes = self._extract_statutes_from_text(argument_text, req.statutes)

        return ArgumentSection(
            issue_number=issue_number,
            issue_text=issue_text,
            argument_text=argument_text,
            cases_cited=extracted_citations,
            statutes_cited=extracted_statutes,
        )

    # ── Step 3b: Strength assessment ───────────────────────────────────────────

    async def _assess_argument_strength(
        self,
        arguments: list[ArgumentSection] | tuple[ArgumentSection, ...],
        req: GenerationRequest,
    ) -> list[dict]:
        """Evaluate each argument from a judicial perspective using Haiku."""
        assessments = []
        for arg in arguments:
            cited = "\n".join(
                f"- {c.get('name', 'Unknown')} ({c.get('citation', 'no citation')})"
                for c in arg.cases_cited
            )
            prompt = f"""You are a senior Nigerian judge evaluating a legal argument.

ISSUE: {arg.issue_text}

ARGUMENT:
{arg.argument_text[:2000]}

CASES CITED:
{cited}

Rate this argument on these dimensions (0-10 each):
1. legal_soundness: Is the legal principle correctly stated?
2. factual_applicability: Does the principle apply to these facts?
3. authority_strength: Are the cited cases binding, on point, and current?
4. vulnerability: How easily could opposing counsel rebut this? (10 = very hard to rebut)

For any score below 5, provide a specific explanation of the weakness.

Return JSON only:
{{"legal_soundness": 8, "factual_applicability": 7,
  "authority_strength": 9, "vulnerability": 6,
  "weaknesses": ["explanation if any score < 5"],
  "misattribution_risk": ["case name if any"]}}"""

            response = await self.client.messages.create(
                model=self.extraction_model,
                max_tokens=512,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

            try:
                text = next(
                    (b.text for b in response.content if isinstance(b, TextBlock)), ""
                ).strip()
                text = text.removeprefix("```json").removesuffix("```").strip()
                assessment = json.loads(text)
                assessment["issue_number"] = arg.issue_number
                assessment["issue_text"] = arg.issue_text
                assessment["overall_score"] = round(
                    (
                        assessment.get("legal_soundness", 5)
                        + assessment.get("factual_applicability", 5)
                        + assessment.get("authority_strength", 5)
                        + assessment.get("vulnerability", 5)
                    )
                    / 4,
                    1,
                )
            except (json.JSONDecodeError, KeyError):
                assessment = {
                    "issue_number": arg.issue_number,
                    "overall_score": None,
                    "_parse_error": True,
                }
            assessments.append(assessment)

        return assessments

    # ── Step 3c: Counter-arguments ─────────────────────────────────────────────

    async def _generate_counter_arguments(
        self,
        arguments: list[ArgumentSection] | tuple[ArgumentSection, ...],
        req: GenerationRequest,
    ) -> list[dict]:
        """Analyse each issue from the opposing party's perspective using Haiku."""
        args_summary = "\n\n".join(
            f"ISSUE {a.issue_number}: {a.issue_text}\nArgument: {a.argument_text[:500]}"
            for a in arguments
        )

        prompt = f"""You are senior counsel representing the RESPONDENT \
(opposing party) in this matter before a Nigerian court.

THE APPLICANT'S ARGUMENTS:
{args_summary}

THE FACTS:
{req.case_facts[:1500]}

For each issue, provide:
1. The single strongest counter-argument the respondent would make
2. Any Nigerian case law the respondent might cite (only cases you are \
confident actually exist)
3. A suggested preemptive rebuttal the applicant could include

Return JSON:
[{{"issue_number": 1, "counter_argument": "...",
   "potential_authority": "...", "suggested_rebuttal": "..."}}]"""

        response = await self.client.messages.create(
            model=self.extraction_model,
            max_tokens=1536,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            text = next((b.text for b in response.content if isinstance(b, TextBlock)), "").strip()
            text = text.removeprefix("```json").removesuffix("```").strip()
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    # ── Step 4: Supporting sections ────────────────────────────────────────────

    async def _generate_prayers(
        self,
        req: GenerationRequest,
        issues: list[str],
    ) -> list[str]:
        """Generate numbered prayers (reliefs sought) using Haiku."""
        prompt = f"""You are a Nigerian litigation lawyer drafting prayers \
(reliefs sought) for a {req.motion_type} motion.

CASE FACTS:
{req.case_facts[:1500]}

RELIEF SOUGHT: {req.relief_sought}

ISSUES FOR DETERMINATION:
{chr(10).join(f"{i + 1}. {iss}" for i, iss in enumerate(issues))}

Draft 2-5 specific prayers. Each prayer should be:
- A specific, actionable relief the court can grant
- Phrased in formal Nigerian legal convention
- Starting with "AN ORDER of this Honourable Court..." or \
"A DECLARATION that..."

Return ONLY a JSON array of prayer strings.
["Prayer 1", "Prayer 2"]"""

        response = await self.client.messages.create(
            model=self.extraction_model,
            max_tokens=1024,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._parse_json_list(
            next((b.text for b in response.content if isinstance(b, TextBlock)), "")
        )

    async def _generate_grounds(
        self,
        req: GenerationRequest,
        issues: list[str],
    ) -> list[str]:
        """Generate grounds of application using Haiku."""
        prompt = f"""You are a Nigerian litigation lawyer drafting grounds \
for a {req.motion_type} motion.

CASE FACTS:
{req.case_facts[:1500]}

ISSUES:
{chr(10).join(f"{i + 1}. {iss}" for i, iss in enumerate(issues))}

Draft 3-6 grounds of application. Each ground should:
- State a factual or legal basis for the motion
- Be numbered and concise
- Follow Nigerian legal convention

Return ONLY a JSON array of ground strings.
["Ground 1", "Ground 2"]"""

        response = await self.client.messages.create(
            model=self.extraction_model,
            max_tokens=1024,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._parse_json_list(
            next((b.text for b in response.content if isinstance(b, TextBlock)), "")
        )

    async def _generate_affidavit(self, req: GenerationRequest) -> list[str]:
        """Generate affidavit paragraphs (facts as THAT statements) using Haiku."""
        prompt = f"""You are a Nigerian litigation lawyer drafting an affidavit \
in support of a {req.motion_type} motion.

CASE FACTS:
{req.case_facts[:2000]}

Draft 5-15 affidavit paragraphs. Each paragraph must:
- Start with a statement of fact (the "THAT" prefix will be added automatically)
- Be a single factual statement (not legal argument)
- Be in first person ("I am", "I was", "The Respondent did")
- Follow Nigerian affidavit conventions

Return ONLY a JSON array of paragraph strings (without "THAT" prefix).
["I am the Managing Director of the Applicant company...",
 "The Respondent commenced this action by..."]"""

        response = await self.client.messages.create(
            model=self.extraction_model,
            max_tokens=2048,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._parse_json_list(
            next((b.text for b in response.content if isinstance(b, TextBlock)), "")
        )

    async def _generate_introduction(
        self,
        req: GenerationRequest,
        issues: list[str],
    ) -> str:
        """Generate the introduction section of the Written Address using Haiku."""
        prompt = f"""You are a Nigerian litigation lawyer writing the introduction \
paragraph of a Written Address in support of a {req.motion_type} motion \
before the {req.court_name}.

CASE FACTS (brief):
{req.case_facts[:1000]}

ISSUES:
{chr(10).join(f"{i + 1}. {iss}" for i, iss in enumerate(issues))}

Write a concise introduction (2-4 sentences) that:
- States what the Written Address is in support of
- Identifies the parties and the motion
- Briefly describes what is being sought
- Uses formal Nigerian legal English

Return ONLY the introduction text, no JSON."""

        response = await self.client.messages.create(
            model=self.extraction_model,
            max_tokens=512,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return next((b.text for b in response.content if isinstance(b, TextBlock)), "").strip()

    async def _generate_conclusion(
        self,
        req: GenerationRequest,
        issues: list[str],
        arguments: list[ArgumentSection] | tuple[ArgumentSection, ...],
    ) -> str:
        """Generate the conclusion section using Haiku."""
        issues_summary = "\n".join(f"Issue {i + 1}: {iss}" for i, iss in enumerate(issues))
        prompt = f"""You are a Nigerian litigation lawyer writing the conclusion \
of a Written Address for a {req.motion_type} motion.

ISSUES ARGUED:
{issues_summary}

RELIEF SOUGHT: {req.relief_sought}

Write a conclusion (2-4 sentences) that:
- Summarises the arguments made
- Urges the court to grant the reliefs sought
- Uses formal Nigerian legal English
- Ends with "We humbly urge this Honourable Court to..." or similar

Return ONLY the conclusion text, no JSON."""

        response = await self.client.messages.create(
            model=self.extraction_model,
            max_tokens=512,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return next((b.text for b in response.content if isinstance(b, TextBlock)), "").strip()

    # ── Step 5: Document assembly ──────────────────────────────────────────────

    def _assemble_motion_paper(
        self,
        req: GenerationRequest,
        prayers: list[str],
        grounds: list[str],
    ) -> MotionPaper:
        return MotionPaper(
            court_name=f"IN THE {req.court_name.upper()}",
            division=f"IN THE {req.division.upper()} JUDICIAL DIVISION",
            location=f"HOLDEN AT {req.location.upper()}",
            suit_number=req.suit_number,
            applicant_name=req.applicant_name,
            applicant_description=req.applicant_description,
            respondent_name=req.respondent_name,
            respondent_description=req.respondent_description,
            motion_type="MOTION ON NOTICE",
            brought_pursuant_to=self._extract_pursuant_to(req),
            prayers=prayers,
            grounds=grounds,
            date=req.date,
            counsel_name=req.counsel_name,
            counsel_firm=req.counsel_firm,
            counsel_address=req.counsel_address,
        )

    def _assemble_affidavit(
        self,
        req: GenerationRequest,
        paragraphs: list[str],
    ) -> SupportingAffidavit:
        court_header = (
            f"IN THE {req.court_name.upper()}\n"
            f"IN THE {req.division.upper()} JUDICIAL DIVISION\n"
            f"HOLDEN AT {req.location.upper()}"
        )
        parties = (
            f"{req.applicant_name.upper()} "
            f"……… {req.applicant_description.upper()}\n"
            f"AND\n"
            f"{req.respondent_name.upper()} "
            f"……… {req.respondent_description.upper()}"
        )
        return SupportingAffidavit(
            court_header=court_header,
            suit_number=req.suit_number,
            parties=parties,
            deponent_name=req.deponent_name or req.applicant_name,
            deponent_description=req.deponent_description or "Nigerian, of full age",
            deponent_capacity=req.deponent_capacity or "a party in this suit",
            paragraphs=paragraphs,
            jurat=f"Sworn to at {req.location}\nthis ______ day of _______ 20___",
        )

    def _assemble_written_address(
        self,
        req: GenerationRequest,
        issues: list[str],
        arguments: list[ArgumentSection],
        introduction: str,
        conclusion: str,
    ) -> WrittenAddress:
        court_header = (
            f"IN THE {req.court_name.upper()}\n"
            f"IN THE {req.division.upper()} JUDICIAL DIVISION\n"
            f"HOLDEN AT {req.location.upper()}"
        )
        parties = (
            f"{req.applicant_name.upper()} "
            f"……… {req.applicant_description.upper()}\n"
            f"AND\n"
            f"{req.respondent_name.upper()} "
            f"……… {req.respondent_description.upper()}"
        )
        counsel_sig = (
            f"___________________________\n"
            f"{req.counsel_name}\n"
            f"{req.counsel_firm}\n"
            f"{req.counsel_address}\n"
            f"Counsel to the {req.applicant_description}"
        )
        motion_label = req.motion_type.replace("_", " ").title()
        return WrittenAddress(
            court_header=court_header,
            suit_number=req.suit_number,
            parties=parties,
            title=f"Written Address in Support of {motion_label}",
            introduction=introduction,
            issues_for_determination=issues,
            arguments=arguments,
            conclusion=conclusion,
            counsel_signature=counsel_sig,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _prepare_case_context(cases: list[dict]) -> str:
        if not cases:
            return "No cases available."
        lines = []
        for c in cases[:10]:
            name = c.get("case_name", "Unknown")
            citation = c.get("citation", "")
            court = c.get("court", "")
            segment = c.get("matched_segment", {})
            content = segment.get("content", "")[:300] if isinstance(segment, dict) else ""
            lines.append(f"- {name} {citation} ({court})\n  Excerpt: {content}")
        return "\n".join(lines)

    @staticmethod
    def _prepare_detailed_case_context(cases: list[dict]) -> str:
        if not cases:
            return "No precedents available."
        lines = []
        for i, c in enumerate(cases[:8], 1):
            name = c.get("case_name", "Unknown")
            citation = c.get("citation", "")
            court = c.get("court", "")
            segment = c.get("matched_segment", {})
            content = segment.get("content", "") if isinstance(segment, dict) else ""
            lines.append(
                f"CASE {i}: {name} {citation} ({court})\nRelevant excerpt:\n{content[:500]}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _select_cases_for_issue(issue_text: str, cases: list[dict]) -> list[dict]:
        """Select the most relevant cases for a specific issue.

        Simple heuristic: all cases are passed through since the LLM
        prompt instructs it to select the ones most relevant. For a
        small number of selected cases (typically 3-8) this is fine.
        """
        return cases[:8]

    @staticmethod
    def _format_statutes(statutes: list[dict]) -> str:
        if not statutes:
            return "No statutory provisions provided."
        lines = []
        for s in statutes:
            section = s.get("section", "")
            act = s.get("act", s.get("title", ""))
            content = s.get("content", "")[:200]
            lines.append(f"- {section} of {act}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _extract_pursuant_to(req: GenerationRequest) -> list[str]:
        """Extract statutory bases from the request's statutes."""
        if not req.statutes:
            return ["the inherent jurisdiction of this Honourable Court"]
        return [
            f"{s.get('section', '')} of {s.get('act', s.get('title', ''))}"
            for s in req.statutes[:5]
        ]

    @staticmethod
    def _extract_citations_from_text(
        text: str,
        available_cases: list[dict],
    ) -> list[dict]:
        """Extract case citations from generated text and match to available cases."""
        citations = []
        for case in available_cases:
            name = case.get("case_name", "")
            short_name = case.get("case_name_short", name)
            # Check if the case name (or a variant) appears in the text
            for search_name in (name, short_name):
                if not search_name:
                    continue
                # Remove common suffixes for flexible matching
                base_name = search_name.split("(")[0].strip()
                if len(base_name) > 5 and base_name.lower() in text.lower():
                    # Find what principle is attributed
                    principle = _extract_principle_near_case(text, base_name)
                    citations.append(
                        {
                            "name": name,
                            "citation": case.get("citation", ""),
                            "case_id": case.get("case_id", ""),
                            "principle_cited": principle,
                        }
                    )
                    break
        return citations

    @staticmethod
    def _extract_statutes_from_text(
        text: str,
        statutes: list[dict],
    ) -> list[dict]:
        """Extract statute references from generated text."""
        found = []
        for s in statutes:
            section = s.get("section", "")
            if section and section.lower() in text.lower():
                found.append(
                    {
                        "section": section,
                        "act": s.get("act", s.get("title", "")),
                    }
                )
        return found

    @staticmethod
    def _update_citation_status(
        arguments: list[ArgumentSection] | tuple[ArgumentSection, ...],
        verified: list[dict],
    ) -> None:
        """Update citation dicts in arguments with verification results."""
        verified_map = {}
        for v in verified:
            key = (v.get("name", ""), v.get("citation", ""))
            verified_map[key] = v

        for arg in arguments:
            for cite in arg.cases_cited:
                key = (cite.get("name", ""), cite.get("citation", ""))
                if key in verified_map:
                    cite["verified"] = verified_map[key].get("verified", False)
                    cite["verification_status"] = verified_map[key].get("status", "unknown")

    @staticmethod
    def _parse_json_list(text: str) -> list[str]:
        """Parse LLM output as a JSON list of strings, with fallback."""
        text = text.strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [str(item) for item in result]
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback: split by newlines and clean up
        lines = [line.strip().lstrip("0123456789.-) ") for line in text.split("\n") if line.strip()]
        return lines if lines else ["As set out in the supporting affidavit."]


# ── Text extraction helpers ────────────────────────────────────────────────────


def _extract_principle_near_case(text: str, case_name: str) -> str:
    """Extract the legal principle attributed to a case in generated text.

    Looks for patterns like '... held that [principle]' or
    '... the court stated that [principle]' near the case name.
    """
    # Find the position of the case name
    lower_text = text.lower()
    pos = lower_text.find(case_name.lower())
    if pos == -1:
        return ""

    # Look in the ~500 chars after the case name for a holding
    region = text[pos : pos + 500]

    # Common patterns: "held that", "stated that", "decided that"
    patterns = [
        r"held\s+that\s+(.{20,200}?)(?:\.|$)",
        r"stated\s+that\s+(.{20,200}?)(?:\.|$)",
        r"decided\s+that\s+(.{20,200}?)(?:\.|$)",
        r"established\s+that\s+(.{20,200}?)(?:\.|$)",
        r"laid\s+down\s+(?:that\s+)?(.{20,200}?)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, region, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

    return ""
