"""
Dimension Scoring Agent.

Scores a single dimension (Soundness, Presentation, or Contribution)
based on provided evidence and paper context.

Multiple instances of this agent can run in parallel for different dimensions.
"""

from typing import Any

from src.rubric.fixed_rubric import FixedRubric
from src.utils.llm_wrapper import LLMWrapper
from src.utils.logger import get_logger
from src.utils.parser import extract_number_from_text

logger = get_logger(__name__)


class DimensionScoringAgent:
    """Agent that scores a specific rubric dimension."""

    def __init__(
        self,
        dimension: str,
        llm: LLMWrapper | None = None,
    ):
        self.dimension = dimension.lower()
        self.llm = llm or LLMWrapper()
        self.rubric = FixedRubric()

    def run(
        self,
        paper_context: str,
        evidence: list[dict[str, Any]],
        title: str = "",
    ) -> dict[str, Any]:
        """Score the assigned dimension.

        Returns:
            Dict with 'dimension', 'score', 'confidence', 'justification'.
        """
        logger.info(f"ScoringAgent [{self.dimension}]: scoring '{title[:50]}...'")

        # Filter relevant evidence
        relevant = self._filter_evidence(evidence)

        # Build prompt
        prompt = self._build_prompt(paper_context, title, relevant)

        # Generate score
        system_prompt = (
            f"You are an expert academic reviewer scoring the '{self.dimension}' dimension. "
            f"Be STRICT and CRITICAL. Most papers have significant room for improvement. "
            f"Do not inflate scores. Use the full scale. "
            f"A middle-range score means 'fair but flawed', not 'acceptable'. "
            f"Reserve high scores only for truly exceptional work. "
            f"Provide a numeric score and brief justification. Cite evidence."
        )

        raw_output = self.llm.generate(prompt, system_prompt=system_prompt, max_tokens=1500)

        # Parse score
        score = self._parse_score(raw_output)
        confidence = self._parse_confidence(raw_output)

        return {
            "dimension": self.dimension,
            "score": score,
            "confidence": confidence,
            "justification": raw_output.strip(),
            "evidence_used": len(relevant),
        }

    def _filter_evidence(self, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter evidence relevant to this dimension."""
        dim_name = self.dimension.capitalize()
        relevant = []
        for ev in evidence:
            rel_dim = ev.get("related_dimension", "")
            if rel_dim and dim_name.lower() in rel_dim.lower():
                relevant.append(ev)
        return relevant[:5]  # Top 5 most relevant

    def _build_prompt(
        self,
        paper_context: str,
        title: str,
        evidence: list[dict[str, Any]],
    ) -> str:
        lines = [f"Paper: {title}", ""]

        # Add rubric description
        dim_info = self.rubric.get_dimension(self.dimension)
        if dim_info:
            lines.append(f"Dimension: {dim_info['name']}")
            lines.append(f"Description: {dim_info['description']}")
            lines.append(f"Scale: {dim_info['scale'][0]}-{dim_info['scale'][1]}")

        # Add evidence
        if evidence:
            lines.append("\nRelevant Evidence:")
            for i, ev in enumerate(evidence, 1):
                lines.append(f"{i}. [{ev.get('source_type', '')}] {ev.get('evidence_text', '')[:200]}")

        # Add paper context (truncated)
        lines.append(f"\nPaper Context:\n{paper_context[:4000]}")

        lines.append(
            f"\nBased on the evidence and paper, provide a {self.dimension} score "
            f"({dim_info['scale'][0]}-{dim_info['scale'][1]}) "
            f"with precision of 0.01 (e.g., 3.25, 7.50). "
            f"Format: Score: X. Justification: ..."
        )

        # Few-shot examples to anchor score standards
        scale_min, scale_max = dim_info["scale"]
        mid = (scale_min + scale_max) / 2
        low = scale_min + (scale_max - scale_min) * 0.25
        high = scale_min + (scale_max - scale_min) * 0.75
        lines.append(f"\nScore calibration examples:")
        lines.append(f"- Score {low:.2f}: Major flaws or insufficient evidence.")
        lines.append(f"- Score {mid:.2f}: Fair but noticeable weaknesses.")
        lines.append(f"- Score {high:.2f}: Good with minor issues.")
        lines.append(f"- Score {scale_max:.2f}: Truly exceptional, no significant weaknesses.")

        return "\n".join(lines)

    def _parse_score(self, text: str) -> float | None:
        """Extract numeric score from response."""
        # Look for "Score: X" pattern
        match = __import__('re').search(r'[Ss]core[:\s]+(\d+(?:\.\d+)?)', text)
        if match:
            return round(float(match.group(1)), 2)
        # Fallback to first number
        val = extract_number_from_text(text)
        return round(val, 2) if val is not None else None

    def _parse_confidence(self, text: str) -> float | None:
        """Extract confidence from response."""
        match = __import__('re').search(r'[Cc]onfidence[:\s]+(\d+(?:\.\d+)?)', text)
        if match:
            return round(float(match.group(1)), 2)
        return None
