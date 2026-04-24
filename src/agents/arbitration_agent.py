"""
Arbitration Agent.

Aggregates multiple dimension scores and resolves disagreements.
Produces final consensus scores.
"""

from typing import Any

from src.scoring.aggregation import aggregate_scores
from src.utils.llm_wrapper import LLMWrapper
from src.utils.logger import get_logger
from src.utils.parser import round_to_step

logger = get_logger(__name__)


class ArbitrationAgent:
    """Agent that arbitrates between multiple dimension scores."""

    def __init__(self, llm: LLMWrapper | None = None):
        self.llm = llm or LLMWrapper()

    def run(self, dimension_scores: list[dict[str, Any]]) -> dict[str, Any]:
        """Arbitrate and produce final scores.

        Args:
            dimension_scores: List of scoring agent outputs.

        Returns:
            Dict with final scores and arbitration notes.
        """
        logger.info(f"ArbitrationAgent: arbitrating {len(dimension_scores)} dimension scores")

        # Group by dimension
        by_dim = {}
        for ds in dimension_scores:
            dim = ds.get("dimension", "")
            if dim:
                by_dim.setdefault(dim, []).append(ds)

        final_scores = {}
        arbitration_notes = []

        for dim, scores_list in by_dim.items():
            score_values = [s["score"] for s in scores_list if s.get("score") is not None]
            confidences = [s.get("confidence", 1.0) for s in scores_list]

            if not score_values:
                continue

            # If only one score, use it
            if len(score_values) == 1:
                final_scores[dim] = score_values[0]
                continue

            # Check for high disagreement
            import numpy as np
            std = np.std(score_values)

            if std > 1.5:
                # High disagreement, use LLM to arbitrate
                arbitrated = self._llm_arbitrate(dim, scores_list)
                final_scores[dim] = arbitrated.get("score", np.median(score_values))
                arbitration_notes.append(
                    f"{dim}: high disagreement (std={std:.2f}), arbitrated to {final_scores[dim]}"
                )
            else:
                # Low disagreement, use median to avoid inflation from weighted mean
                final_scores[dim] = round_to_step(float(np.median(score_values)))

        # Round all dimension scores to 0.05 step
        for k in list(final_scores.keys()):
            if isinstance(final_scores[k], float):
                final_scores[k] = round_to_step(final_scores[k])

        # Compute decision from rating (threshold: >= 6.5 is accept, aligned with config)
        rating = final_scores.get("rating", 0)
        final_scores["decision"] = "accept" if rating >= 6.5 else "reject"

        return {
            "final_scores": final_scores,
            "arbitration_notes": arbitration_notes,
            "disagreement_detected": len(arbitration_notes) > 0,
        }

    def _llm_arbitrate(self, dimension: str, scores_list: list[dict[str, Any]]) -> dict[str, Any]:
        """Use LLM to arbitrate high-disagreement cases."""
        prompt_lines = [
            f"Multiple agents scored the '{dimension}' dimension differently:",
            "",
        ]
        for s in scores_list:
            prompt_lines.append(
                f"- Score: {s.get('score')}, Confidence: {s.get('confidence', 'N/A')}"
            )
            prompt_lines.append(f"  Justification: {s.get('justification', '')[:200]}")
            prompt_lines.append("")

        prompt_lines.append(
            "As an expert arbitrator, provide a final score and brief reasoning. "
            "Format: Final Score: X. Reasoning: ..."
        )

        prompt = "\n".join(prompt_lines)
        system_prompt = "You are an expert arbitration agent resolving scoring disagreements."

        try:
            raw = self.llm.generate(prompt, system_prompt=system_prompt, max_tokens=1000)
            match = __import__('re').search(r'[Ff]inal [Ss]core[:\s]+(\d+(?:\.\d+)?)', raw)
            if match:
                return {"score": float(match.group(1)), "reasoning": raw.strip()}
        except Exception as e:
            logger.warning(f"LLM arbitration failed: {e}")

        # Fallback to median
        import numpy as np
        scores = [s["score"] for s in scores_list if s.get("score") is not None]
        return {"score": float(np.median(scores)) if scores else None, "reasoning": "Fallback to median"}
