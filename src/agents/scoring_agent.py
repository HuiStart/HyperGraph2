"""
Dimension Scoring Agent.

Scores a single dimension (Soundness, Presentation, Contribution, or Rating)
based on provided evidence and paper context.

Aligned with DeepReview official scoring agent:
- Enforces chain-of-thought: Strengths -> Weaknesses -> Score
- Does NOT use deduction-based scoring or distribution anchoring
- Lets the LLM judge holistically after analyzing evidence
"""

from typing import Any

from src.rubric.fixed_rubric import FixedRubric
from src.utils.llm_wrapper import LLMWrapper
from src.utils.logger import get_logger
from src.utils.parser import extract_number_from_text, round_to_step

logger = get_logger(__name__)


class DimensionScoringAgent:
    """Agent that scores a specific rubric dimension with chain-of-thought reasoning."""

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

        # System prompt aligned with DeepReview official style
        system_prompt = self._build_system_prompt()

        raw_output = self.llm.generate(prompt, system_prompt=system_prompt, max_tokens=1500)

        # Parse score
        score = self._parse_score(raw_output)
        confidence = self._parse_confidence(raw_output)

        # Clamp to valid range and round to 0.05 step
        dim_info = self.rubric.get_dimension(self.dimension)
        if dim_info and score is not None:
            scale_min, scale_max = dim_info["scale"]
            score = max(scale_min, min(scale_max, score))
            score = round_to_step(score)

        return {
            "dimension": self.dimension,
            "score": score,
            "confidence": confidence,
            "justification": raw_output.strip(),
            "evidence_used": len(relevant),
        }

    def _build_system_prompt(self) -> str:
        """Build system prompt aligned with DeepReview official style."""
        dim_name = self.dimension.capitalize()

        if self.dimension == "rating":
            return (
                "You are an expert reviewer acting as part of the DeepReview system.\n"
                "Evaluate the paper comprehensively. You MUST follow the reasoning structure "
                "before assigning any numerical scores. Do not skip steps.\n\n"
                "After completing the analysis, provide the final score with 0.05 precision."
            )
        else:
            return (
                f"You are an expert reviewer evaluating the '{dim_name}' dimension "
                f"as part of the DeepReview system.\n"
                f"You MUST follow the reasoning structure before assigning a score. "
                f"Do not skip steps. After completing the analysis, provide the final score "
                f"with 0.05 precision."
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
        dim_info = self.rubric.get_dimension(self.dimension)
        scale_text = ""
        if dim_info:
            scale_text = f"Scale: {dim_info['scale'][0]}-{dim_info['scale'][1]}. "

        # Evidence section
        evidence_section = ""
        if evidence:
            evidence_lines = ["Relevant Evidence:"]
            for i, ev in enumerate(evidence, 1):
                evidence_lines.append(
                    f"{i}. [{ev.get('source_type', '')}] {ev.get('evidence_text', '')[:200]}"
                )
            evidence_section = "\n".join(evidence_lines)

        # Chain-of-thought prompt aligned with DeepReview official
        if self.dimension == "rating":
            reasoning_steps = (
                "1. Summary: Provide a concise overview of the core ideas and findings.\n"
                "2. Strengths: Detail the key advantages, novelty, and positive attributes.\n"
                "3. Weaknesses: Explicitly identify methodological flaws, missing baselines, or clarity issues.\n"
                "4. Suggestions: Provide actionable feedback for the authors.\n"
                "5. Overall Assessment: Synthesize the above into a holistic judgment."
            )
        elif self.dimension == "soundness":
            reasoning_steps = (
                "=== ANTI-GULLIBILITY RULES (Critical) ===\n"
                "1. DO NOT be fooled by bold claims like 'we rigorously prove'. Look for concrete empirical baselines and ablation studies.\n"
                "2. DO NOT penalize honest limitation discussions. Acknowledging limitations is a sign of HIGH soundness, not weakness.\n"
                "3. Complex math without ablation studies is NOT rigorous. Simple, clear, validated methods score higher.\n\n"
                "=== STRUCTURED CHECKS (Answer before scoring) ===\n"
                "A. Baseline Check: Are compared baselines strong and recent? (Yes/No/Unclear)\n"
                "B. Ablation Check: Did they isolate the source of improvement via ablation? (Yes/No/Unclear)\n"
                "C. Limitation Check: Did they honestly discuss limitations? (Yes = good, No = concern)\n\n"
                "=== REASONING ===\n"
                "1. Strengths: Detail methodological rigor, solid experimental design, strong baselines.\n"
                "2. Weaknesses: Identify real flaws, missing validations, unsupported claims.\n"
                "3. Overall Assessment: Judge whether the methodology is sound and well-supported."
            )
        elif self.dimension == "presentation":
            reasoning_steps = (
                "1. Strengths: Identify clear writing, good structure, helpful figures/tables.\n"
                "2. Weaknesses: Point out confusing sections, poor organization, or unclear notation.\n"
                "3. Overall Assessment: Judge how well the paper communicates its ideas."
            )
        elif self.dimension == "contribution":
            reasoning_steps = (
                "1. Strengths: Highlight novelty, significant advances, or valuable insights.\n"
                "2. Weaknesses: Note incremental or trivial contributions, lack of novelty.\n"
                "3. Overall Assessment: Judge the significance and novelty relative to the field."
            )
        else:
            reasoning_steps = (
                "1. Strengths: Detail the positive attributes of this dimension.\n"
                "2. Weaknesses: Explicitly identify flaws or shortcomings.\n"
                "3. Overall Assessment: Synthesize into a holistic judgment."
            )

        return (
            f"Paper: {title}\n\n"
            f"Dimension: {self.dimension.capitalize()}\n"
            f"{scale_text}\n"
            f"{evidence_section}\n\n"
            f"Paper Content:\n{paper_context[:8000]}\n\n"
            f"INSTRUCTIONS:\n"
            f"You MUST follow this exact reasoning structure before assigning a score. Do not skip steps.\n\n"
            f"{reasoning_steps}\n\n"
            f"Only after completing the above, provide your final score strictly in this format:\n"
            f"Score: [numeric value with 0.05 precision]\n"
            f"Justification: [brief reasoning]\n"
            f"Confidence: [1-5]"
        )

    def _parse_score(self, text: str) -> float | None:
        """Extract numeric score from response."""
        # Look for "Score: X" pattern
        match = __import__('re').search(r'[Ss]core[:\s]+(\d+(?:\.\d+)?)', text)
        if match:
            return round_to_step(float(match.group(1)))
        # Fallback to first number
        val = extract_number_from_text(text)
        return round_to_step(val) if val is not None else None

    def _parse_confidence(self, text: str) -> float | None:
        """Extract confidence from response."""
        match = __import__('re').search(r'[Cc]onfidence[:\s]+(\d+(?:\.\d+)?)', text)
        if match:
            return round_to_step(float(match.group(1)))
        return None
