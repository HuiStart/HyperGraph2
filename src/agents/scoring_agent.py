"""
Dimension Scoring Agent.

Scores a single dimension (Soundness, Presentation, Contribution, or Rating)
based on provided evidence and paper context.

<<<<<<< HEAD
Aligned with DeepReview official scoring agent:
- Enforces chain-of-thought: Strengths -> Weaknesses -> Score
- Does NOT use deduction-based scoring or distribution anchoring
- Lets the LLM judge holistically after analyzing evidence
=======
Uses DEDUCTION-BASED scoring to avoid score inflation:
- Forces the LLM to identify specific flaws first
- Calculates score by subtracting deductions from maximum
- This produces discriminative scores across the full scale

Multiple instances of this agent can run in parallel for different dimensions.
>>>>>>> newb
"""

from typing import Any

from src.rubric.fixed_rubric import FixedRubric
from src.utils.llm_wrapper import LLMWrapper
from src.utils.logger import get_logger
from src.utils.parser import extract_number_from_text, round_to_step

logger = get_logger(__name__)


class DimensionScoringAgent:
<<<<<<< HEAD
    """Agent that scores a specific rubric dimension with chain-of-thought reasoning."""
=======
    """Agent that scores a specific rubric dimension using deduction-based scoring."""
>>>>>>> newb

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
        """Score the assigned dimension using deduction-based scoring.

        Returns:
            Dict with 'dimension', 'score', 'confidence', 'justification'.
        """
        logger.info(f"ScoringAgent [{self.dimension}]: scoring '{title[:50]}...'")

        # Filter relevant evidence
        relevant = self._filter_evidence(evidence)

        # Build prompt
        prompt = self._build_prompt(paper_context, title, relevant)

<<<<<<< HEAD
        # System prompt aligned with DeepReview official style
        system_prompt = self._build_system_prompt()
=======
        # Deduction-based scoring system prompt
        dim_info = self.rubric.get_dimension(self.dimension)
        scale_min, scale_max = dim_info["scale"] if dim_info else (1, 10)

        system_prompt = self._build_system_prompt(scale_min, scale_max)
>>>>>>> newb

        raw_output = self.llm.generate(prompt, system_prompt=system_prompt, max_tokens=1500)

        # Parse score
        score = self._parse_score(raw_output, scale_min, scale_max)
        confidence = self._parse_confidence(raw_output)

<<<<<<< HEAD
        # Clamp to valid range and round to 0.05 step
        dim_info = self.rubric.get_dimension(self.dimension)
        if dim_info and score is not None:
            scale_min, scale_max = dim_info["scale"]
            score = max(scale_min, min(scale_max, score))
            score = round_to_step(score)
=======
        # If score is still inflated (e.g., > 80% of scale for most papers),
        # apply post-hoc calibration based on justification length (#flaws mentioned)
        if score is not None:
            score = self._calibrate_by_flaws(score, raw_output, scale_min, scale_max)
>>>>>>> newb

        return {
            "dimension": self.dimension,
            "score": score,
            "confidence": confidence,
            "justification": raw_output.strip(),
            "evidence_used": len(relevant),
        }

<<<<<<< HEAD
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
=======
    def _build_system_prompt(self, scale_min: int, scale_max: int) -> str:
        """Build strict system prompt enforcing deduction-based scoring."""

        if self.dimension == "rating":
            return (
                f"You are a harsh academic reviewer evaluating the OVERALL quality of a research paper. "
                f"You MUST use deduction-based scoring. Do NOT give a 'gut feeling' score.\n\n"
                f"SCORING RULES (Rating scale {scale_min}-{scale_max}):\n"
                f"1. Start from the MAXIMUM score ({scale_max}).\n"
                f"2. Identify EVERY flaw, weakness, or limitation you can find.\n"
                f"3. For each flaw, assign severity:\n"
                f"   - minor flaw (clarity issue, minor omission): deduct 0.5\n"
                f"   - moderate flaw (methodological concern, missing comparison): deduct 1.0\n"
                f"   - major flaw (incorrect claim, missing critical experiment, fatal flaw): deduct 2.0\n"
                f"4. Calculate: Score = {scale_max} - sum_of_deductions.\n"
                f"5. Minimum score is {scale_min}.\n\n"
                f"MOST papers have significant flaws. A typical NeurIPS/ICML submission deserves 5-6. "
                f"Only truly exceptional, landmark papers deserve 8+. "
                f"Papers with major methodological errors can go below 4.\n\n"
                f"You MUST list at least 3 flaws. If you cannot find 3 flaws, you are not being critical enough.\n\n"
                f"Output format:\n"
                f"Flaws:\n"
                f"1. [description] (severity: minor/moderate/major, deduction: X)\n"
                f"2. ...\n"
                f"Score: [calculated score with 0.01 precision]\n"
                f"Justification: [brief reasoning]"
            )
        else:
            return (
                f"You are a harsh academic reviewer evaluating the '{self.dimension.upper()}' dimension. "
                f"You MUST use deduction-based scoring. Do NOT give a 'gut feeling' score.\n\n"
                f"SCORING RULES ({self.dimension.upper()} scale {scale_min}-{scale_max}):\n"
                f"1. Start from the MAXIMUM score ({scale_max}).\n"
                f"2. Identify EVERY flaw related to {self.dimension.upper()} you can find.\n"
                f"3. For each flaw, assign severity:\n"
                f"   - minor flaw: deduct 0.25\n"
                f"   - moderate flaw: deduct 0.50\n"
                f"   - major flaw: deduct 1.00\n"
                f"4. Calculate: Score = {scale_max} - sum_of_deductions.\n"
                f"5. Minimum score is {scale_min}.\n\n"
                f"MOST papers have room for improvement in {self.dimension.upper()}. "
                f"A typical paper gets 2.0-2.5. Good but flawed papers get 2.5-3.0. "
                f"Only truly outstanding work deserves 3.5+. "
                f"Papers with serious problems in this dimension can go below 2.0.\n\n"
                f"You MUST list at least 2 flaws. If you cannot find 2 flaws, you are not being critical enough.\n\n"
                f"Output format:\n"
                f"Flaws:\n"
                f"1. [description] (severity: minor/moderate/major, deduction: X)\n"
                f"2. ...\n"
                f"Score: [calculated score with 0.01 precision]\n"
                f"Justification: [brief reasoning]"
>>>>>>> newb
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
        semantic_rubric = ""
        if dim_info:
            scale_min, scale_max = dim_info['scale']
            scale_text = f"Scale: {scale_min}-{scale_max}.\n"

            if self.dimension == "rating":
                semantic_rubric = (
                    "SCORING SEMANTICS (CRITICAL):\n"
                    " - 9-10: Award quality. Groundbreaking work.\n"
                    " - 7-8: Clear Accept. Strong methodology and novelty.\n"
                    " - 5-6: Borderline/Average. Incremental work or noticeable gaps. (Most papers fall here).\n"
                    " - 3-4: Clear Reject. Major flaws, missing baselines, or lacks novelty.\n"
                    " - 1-2: Fundamentally flawed or unreadable.\n\n"
                    "BASELINE RULE: Assume the paper is a 5.0 (Borderline). "
                    "You MUST find explicit, strong evidence in 'Strengths' to score above 6.0.\n"
                )
            else:
                semantic_rubric = (
                    "SCORING SEMANTICS (CRITICAL):\n"
                    " - 4: Excellent. Flawless execution in this dimension.\n"
                    " - 3: Good. Solid, but with minor easily fixable issues.\n"
                    " - 2: Fair. Noticeable weaknesses that undermine confidence.\n"
                    " - 1: Poor. Severe flaws.\n\n"
                    "BASELINE RULE: The default score is 2.0 or 2.5. "
                    "Do not give a 3 or 4 unless you have verified strong evidence.\n"
                )

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
                "=== FOCUS RULE ===\n"
                "In this dimension, tolerate experimental weaknesses. "
                "Focus ONLY on core Idea novelty and potential impact on the field. "
                "Do NOT let Soundness flaws (missing baselines, no ablation) lower this score.\n\n"
                "=== REASONING ===\n"
                "1. Strengths: Highlight novelty, significant advances, or valuable insights.\n"
                "2. Weaknesses: Note trivial or well-known contributions, lack of novelty.\n"
                "3. Overall Assessment: Judge the significance and novelty relative to the field."
            )
        else:
            reasoning_steps = (
                "1. Strengths: Detail the positive attributes of this dimension.\n"
                "2. Weaknesses: Explicitly identify flaws or shortcomings.\n"
                "3. Overall Assessment: Synthesize into a holistic judgment."
            )

<<<<<<< HEAD
        return (
            f"Paper: {title}\n\n"
            f"Dimension: {self.dimension.capitalize()}\n"
            f"{scale_text}"
            f"{semantic_rubric}\n"
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
=======
        lines.append(
            f"\nBased on the evidence and paper, evaluate the {self.dimension.upper()} dimension. "
            f"Follow the deduction-based scoring process in your system instructions. "
            f"Be CRITICAL. Most papers have significant weaknesses."
        )

        return "\n".join(lines)

    def _parse_score(self, text: str, scale_min: int, scale_max: int) -> float | None:
>>>>>>> newb
        """Extract numeric score from response."""
        # Look for "Score: X" pattern
        match = __import__('re').search(r'[Ss]core[:\s]+(\d+(?:\.\d+)?)', text)
        if match:
<<<<<<< HEAD
            return round_to_step(float(match.group(1)))
        # Fallback to first number
        val = extract_number_from_text(text)
        return round_to_step(val) if val is not None else None
=======
            score = round(float(match.group(1)), 2)
            return max(scale_min, min(scale_max, score))
        # Fallback to first number
        val = extract_number_from_text(text)
        if val is not None:
            score = round(val, 2)
            return max(scale_min, min(scale_max, score))
        return None
>>>>>>> newb

    def _parse_confidence(self, text: str) -> float | None:
        """Extract confidence from response."""
        match = __import__('re').search(r'[Cc]onfidence[:\s]+(\d+(?:\.\d+)?)', text)
        if match:
            return round_to_step(float(match.group(1)))
        return None

    def _calibrate_by_flaws(self, score: float, text: str, scale_min: int, scale_max: int) -> float:
        """Post-hoc calibration: if few flaws mentioned, push score down."""
        import re

        # Count how many flaws were explicitly listed
        flaw_patterns = [
            r'^\s*\d+\.\s+.+\(severity:\s*\w+',
            r'^\s*[-*]\s+.+flaw',
            r'\bflaw\b|\bweakness\b|\blimitation\b|\berror\b|\bissue\b',
        ]

        flaw_count = 0
        lines = text.split('\n')
        for line in lines:
            for pattern in flaw_patterns[:2]:
                if re.search(pattern, line, re.IGNORECASE):
                    flaw_count += 1
                    break

        # If very few flaws mentioned but score is high, penalize
        if self.dimension == "rating":
            # Rating 1-10
            if flaw_count < 2 and score > 6.0:
                # Not critical enough, push down
                score = max(scale_min, score - 2.0)
            elif flaw_count < 3 and score > 7.0:
                score = max(scale_min, score - 1.5)
            elif flaw_count < 4 and score > 8.0:
                score = max(scale_min, score - 1.0)
        else:
            # Sub-dims 1-4
            if flaw_count < 1 and score > 2.5:
                score = max(scale_min, score - 1.0)
            elif flaw_count < 2 and score > 3.0:
                score = max(scale_min, score - 0.5)

        return round(score, 2)
