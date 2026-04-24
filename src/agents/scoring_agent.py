"""
Dimension Scoring Agent.

Scores a single dimension (Soundness, Presentation, Contribution, or Rating)
based on provided evidence and paper context.

KEY DESIGN CHOICE: Direct scoring with distribution anchoring.
We do NOT use deduction-based scoring because LLMs often misidentify
"flaws" relative to human reviewers, which can invert rankings (negative Spearman).
Instead, we anchor scores to the known dataset distribution and ask the LLM
to compare the paper against typical NeurIPS/ICML submissions.

Multiple instances of this agent can run in parallel for different dimensions.
"""

from typing import Any

from src.rubric.fixed_rubric import FixedRubric
from src.utils.llm_wrapper import LLMWrapper
from src.utils.logger import get_logger
from src.utils.parser import extract_number_from_text

logger = get_logger(__name__)


# Dataset distribution anchors (DeepReview human review statistics)
# These anchor the LLM to realistic score ranges.
DATASET_ANCHORS = {
    "rating": {
        "mean": 5.5,
        "std": 1.4,
        "typical_range": "5.0-6.5",
        "scale": [1, 10],
        "anchors": [
            (9.0, "Truly exceptional, landmark contribution, flawless execution"),
            (7.5, "Strong paper, minor issues only, clear accept"),
            (6.0, "Above average, some weaknesses but acceptable"),
            (5.0, "Average/marginal, noticeable flaws"),
            (3.5, "Below average, significant methodological or presentation problems"),
            (2.0, "Poor, major errors or insufficient evidence"),
        ],
    },
    "soundness": {
        "mean": 2.8,
        "std": 0.7,
        "typical_range": "2.5-3.5",
        "scale": [1, 4],
        "anchors": [
            (4.0, "Methodology is rigorous, correct, and well-validated"),
            (3.25, "Sound methodology with minor gaps in validation or proof"),
            (2.75, "Acceptable methodology but noticeable weaknesses in validation or correctness"),
            (2.0, "Significant methodological concerns, limited validation"),
            (1.25, "Major methodological flaws or incorrect claims"),
        ],
    },
    "presentation": {
        "mean": 2.8,
        "std": 0.6,
        "typical_range": "2.5-3.5",
        "scale": [1, 4],
        "anchors": [
            (4.0, "Crystal clear, well-organized, excellent figures and writing"),
            (3.25, "Good presentation, minor clarity or organization issues"),
            (2.75, "Adequate but noticeable clarity or structure problems"),
            (2.0, "Poor organization or confusing exposition"),
            (1.25, "Very difficult to follow, major communication barriers"),
        ],
    },
    "contribution": {
        "mean": 2.7,
        "std": 0.7,
        "typical_range": "2.0-3.0",
        "scale": [1, 4],
        "anchors": [
            (4.0, "Major novel contribution, opens new direction"),
            (3.25, "Solid incremental contribution with clear value"),
            (2.75, "Moderate contribution, incremental but useful"),
            (2.0, "Limited novelty, minor incremental improvement"),
            (1.25, "Trivial or well-known contribution"),
        ],
    },
}


class DimensionScoringAgent:
    """Agent that scores a specific rubric dimension with distribution anchoring."""

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

        # Build system prompt with distribution anchoring
        system_prompt = self._build_system_prompt()

        raw_output = self.llm.generate(prompt, system_prompt=system_prompt, max_tokens=1500)

        # Parse score
        score = self._parse_score(raw_output)
        confidence = self._parse_confidence(raw_output)

        # Clamp to valid range
        dim_info = self.rubric.get_dimension(self.dimension)
        if dim_info and score is not None:
            scale_min, scale_max = dim_info["scale"]
            score = max(scale_min, min(scale_max, score))
            score = round(score, 2)

        return {
            "dimension": self.dimension,
            "score": score,
            "confidence": confidence,
            "justification": raw_output.strip(),
            "evidence_used": len(relevant),
        }

    def _build_system_prompt(self) -> str:
        """Build system prompt anchored to dataset distribution."""
        anchor = DATASET_ANCHORS.get(self.dimension)
        if not anchor:
            # Fallback generic prompt
            return (
                f"You are an expert academic reviewer scoring the '{self.dimension}' dimension. "
                f"Be objective and calibrated. Most papers are average; only exceptional work deserves top scores."
            )

        scale_min, scale_max = anchor["scale"]
        anchor_lines = "\n".join(
            f"  {score:.2f}: {desc}" for score, desc in anchor["anchors"]
        )

        return (
            f"You are an expert academic reviewer evaluating the '{self.dimension.upper()}' dimension.\n\n"
            f"DATASET CALIBRATION (human reviewer statistics):\n"
            f"- Scale: {scale_min}-{scale_max}\n"
            f"- Typical papers score around {anchor['mean']:.1f} (range: {anchor['typical_range']})\n"
            f"- Most submissions are average; reserve top scores for truly outstanding work\n\n"
            f"SCORE ANCHORS (compare the paper against these standards):\n"
            f"{anchor_lines}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Compare this paper against the anchors above.\n"
            f"2. Do NOT start from maximum and deduct. Instead, judge the paper's absolute quality.\n"
            f"3. Most papers should cluster near the dataset mean ({anchor['mean']:.1f}).\n"
            f"4. Provide a specific numeric score with 0.01 precision.\n"
            f"5. Briefly justify your score by referencing specific evidence.\n\n"
            f"Output format:\n"
            f"Score: [numeric value]\n"
            f"Justification: [your reasoning]"
        )

    def _filter_evidence(self, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter evidence relevant to this dimension."""
        dim_name = self.dimension.capitalize()
        relevant = []
        for ev in evidence:
            rel_dim = ev.get("related_dimension", "")
            if not rel_dim:
                continue
            # Handle both string and list types for related_dimension
            if isinstance(rel_dim, list):
                rel_dim_str = " ".join(str(d) for d in rel_dim)
            else:
                rel_dim_str = str(rel_dim)
            if dim_name.lower() in rel_dim_str.lower():
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

        anchor = DATASET_ANCHORS.get(self.dimension)
        if anchor:
            lines.append(
                f"\nReminder: typical papers score {anchor['mean']:.1f} on this dimension. "
                f"Score this paper relative to typical NeurIPS/ICML submissions."
            )

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
