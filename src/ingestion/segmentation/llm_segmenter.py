"""Pass 3: LLM-assisted segmentation for low-confidence cases (Claude Haiku)."""

from __future__ import annotations

import json

import structlog
from anthropic import Anthropic

logger = structlog.get_logger(__name__)

# Confidence threshold below which a segment triggers LLM re-evaluation
LOW_CONFIDENCE_THRESHOLD: float = 0.50

_SEGMENTATION_PROMPT = """\
You are a Nigerian legal text analyst. Given a Nigerian court judgment, \
extract the following structured information:

1. **Issues for Determination**: The specific legal questions the court addressed.
   Extract them verbatim if explicitly stated, or infer them if implicit.

2. **Holdings**: For each issue, what did the court decide? Include:
   - The issue number/description
   - The court's determination (brief)
   - The key reasoning (1-2 sentences)

3. **Ratio Decidendi**: The binding principle(s) of law established by this case.
   This is the legal rule that future courts must follow. Be precise and concise.

4. **Obiter Dicta**: Any significant non-binding observations or commentary
   by the court that may be persuasive in future cases.

5. **Orders**: The final orders or dispositions of the court.

6. **Cases Cited**: For each case cited in the judgment, provide:
   - Case name and citation
   - How it was used (followed, distinguished, overruled, mentioned)
   - Brief context of why it was cited

Respond in JSON format with these exact keys:
{{
  "issues": ["issue 1 text", "issue 2 text"],
  "holdings": [
    {{"issue": "...", "determination": "...", "reasoning": "..."}}
  ],
  "ratio_decidendi": "...",
  "obiter_dicta": ["observation 1", "observation 2"],
  "orders": ["order 1", "order 2"],
  "cases_cited": [
    {{
      "name": "...",
      "citation": "...",
      "treatment": "followed|distinguished|overruled|mentioned",
      "context": "..."
    }}
  ]
}}

JUDGMENT TEXT:
---
{judgment_text}
---

Return ONLY valid JSON. No markdown fences, no explanatory text.\
"""

# Haiku pricing (per million tokens, as of June 2025 — approximate)
_INPUT_COST_PER_M = 0.80
_OUTPUT_COST_PER_M = 4.00
_AVG_OUTPUT_TOKENS = 2_000


class LLMSegmenter:
    """
    Third-pass segmenter using Claude to extract structured segments
    from judgments where rule-based confidence is low.

    Uses claude-haiku-4-5 for cost efficiency.  Upgrade the model
    string to Sonnet/Opus for complex or ambiguous judgments.

    An optional ``client`` parameter allows injecting a mock Anthropic
    client in tests without patching.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        client: Anthropic | None = None,
    ) -> None:
        self.model = model
        self._client = client or Anthropic()

    def segment(self, text: str, max_input_chars: int = 80_000) -> dict:
        """
        Extract structured segments from judgment text via the Claude API.

        Very long judgments are truncated to ``max_input_chars``,
        keeping both the beginning (caption / facts) and the end
        (holdings / orders) which carry the most retrieval value.

        Returns a dict with keys: issues, holdings, ratio_decidendi,
        obiter_dicta, orders, cases_cited.  On JSON parse failure the
        dict includes ``_parse_error: True`` and ``_raw_response``.
        """
        if len(text) > max_input_chars:
            half = max_input_chars // 2
            text = (
                text[:half]
                + "\n\n[... middle portion omitted for processing ...]\n\n"
                + text[-half:]
            )

        logger.debug("llm_segmenter.calling_api", model=self.model, chars=len(text))

        response = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": _SEGMENTATION_PROMPT.format(judgment_text=text),
                }
            ],
        )

        content = response.content[0].text.strip()

        # Strip any accidental markdown code fences
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("llm_segmenter.json_parse_error", preview=content[:200])
            return {
                "issues": [],
                "holdings": [],
                "ratio_decidendi": None,
                "obiter_dicta": [],
                "orders": [],
                "cases_cited": [],
                "_parse_error": True,
                "_raw_response": content[:500],
            }

    def estimate_cost(
        self,
        num_cases: int,
        avg_tokens: int = 15_000,
    ) -> dict:
        """
        Estimate API cost for batch LLM segmentation.

        Returns a summary dict with token counts and USD cost.
        Example: 5,000 cases ≈ $68 with Haiku.
        """
        total_input = num_cases * avg_tokens
        total_output = num_cases * _AVG_OUTPUT_TOKENS
        cost = (
            total_input / 1_000_000 * _INPUT_COST_PER_M
            + total_output / 1_000_000 * _OUTPUT_COST_PER_M
        )
        return {
            "num_cases": num_cases,
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "estimated_cost_usd": round(cost, 4),
        }
