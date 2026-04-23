"""
Explanation Generation Agent.

Produces structured, evidence-driven explanations using templates.
Must cite real evidence and rubric dimensions.
Does not fabricate evidence.
"""

from typing import Any

from src.rubric.fixed_rubric import FixedRubric
from src.utils.llm_wrapper import LLMWrapper
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExplanationAgent:
    """Agent that generates template-based explanations."""

    def __init__(self, llm: LLMWrapper | None = None):
        self.llm = llm or LLMWrapper()
        self.rubric = FixedRubric()

    def run(
        self,
        title: str,
        final_scores: dict[str, float | None],
        evidence: list[dict[str, Any]],
        arbitration_notes: list[str],
    ) -> dict[str, Any]:
        """Generate explanation for the final scores.

        Returns:
            Dict with 'explanation_text' and structured fields.
        """
        logger.info(f"ExplanationAgent: generating explanation for '{title[:50]}...'")

        explanations = []

        for dim in ["soundness", "presentation", "contribution", "rating"]:
            score = final_scores.get(dim)
            if score is None:
                continue

            dim_info = self.rubric.get_dimension(dim)
            dim_name = dim_info["name"] if dim_info else dim.capitalize()
            scale = dim_info["scale"] if dim_info else [1, 5]

            # Find relevant evidence
            relevant = self._filter_evidence(evidence, dim)

            # Build template-based explanation
            exp_text = self._build_explanation(
                title=title,
                dimension=dim_name,
                score=score,
                scale=scale,
                evidence=relevant,
                notes=arbitration_notes if dim == "rating" else [],
            )
            explanations.append({
                "dimension": dim,
                "score": score,
                "explanation": exp_text,
                "evidence_cited": [e.get("evidence_id", "") for e in relevant],
            })

        return {
            "explanations": explanations,
            "full_text": "\n\n".join(e["explanation"] for e in explanations),
        }

    def _filter_evidence(self, evidence: list[dict[str, Any]], dimension: str) -> list[dict[str, Any]]:
        """Filter evidence relevant to dimension."""
        dim_name = dimension.capitalize()
        return [
            ev for ev in evidence
            if dim_name.lower() in ev.get("related_dimension", "").lower()
        ][:3]

    def _build_explanation(
        self,
        title: str,
        dimension: str,
        score: float,
        scale: list[int],
        evidence: list[dict[str, Any]],
        notes: list[str],
    ) -> str:
        """Build template-based explanation."""
        lines = [
            f"## {dimension} Assessment",
            "",
            f"**Paper**: {title}",
            f"**Score**: {score:.1f} / {scale[1]}",
            "",
        ]

        # Evidence section
        if evidence:
            lines.append("**Evidence:**")
            for ev in evidence:
                text = ev.get("evidence_text", "")[:150]
                section = ev.get("section", "unknown")
                lines.append(f"- [{ev.get('source_type', '')}] {text} (Source: {section})")
            lines.append("")
        else:
            lines.append("**Evidence**: No direct evidence found for this dimension.")
            lines.append("")

        # Assessment
        if score >= scale[1] * 0.8:
            assessment = "Strong performance"
        elif score >= scale[1] * 0.6:
            assessment = "Adequate performance"
        elif score >= scale[1] * 0.4:
            assessment = "Below expectations"
        else:
            assessment = "Significant issues identified"

        lines.append(f"**Assessment**: {assessment}")
        lines.append("")

        # Arbitration notes (for rating only)
        if notes:
            lines.append("**Arbitration Notes**:")
            for note in notes:
                lines.append(f"- {note}")
            lines.append("")

        # Suggestions
        if score < scale[1] * 0.7:
            lines.append("**Suggestions for Improvement**:")
            if dimension == "Soundness":
                lines.append("- Strengthen methodology description and validation.")
            elif dimension == "Presentation":
                lines.append("- Improve clarity, organization, and visual aids.")
            elif dimension == "Contribution":
                lines.append("- Clarify novelty and significance compared to prior work.")
            lines.append("")

        return "\n".join(lines)
