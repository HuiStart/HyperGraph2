"""
Arbitration Agent.

Aggregates multiple dimension scores and resolves disagreements.
Produces final consensus scores.
"""

from typing import Any

from src.scoring.aggregation import aggregate_scores
from src.utils.llm_wrapper import LLMWrapper
from src.utils.logger import get_logger

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
                # Low disagreement, use weighted mean by confidence
                valid = [(s, c) for s, c in zip(score_values, confidences) if s is not None]
                if valid:
                    scores, weights = zip(*valid)
                    final_scores[dim] = aggregate_scores(list(scores), method="weighted", weights=list(weights))
                else:
                    final_scores[dim] = score_values[0]

        # Compute overall rating from dimension scores (heuristic)
        dim_avg = []
        for dim in ["soundness", "presentation", "contribution"]:
            if dim in final_scores and final_scores[dim] is not None:
                dim_avg.append(final_scores[dim])

        # Round all dimension scores to 2 decimals
        for k in list(final_scores.keys()):
            if isinstance(final_scores[k], float):
                final_scores[k] = round(final_scores[k], 2)

        if dim_avg:
            # Scale to 1-10: average of 1-4 dimensions, doubled
            raw_avg = sum(dim_avg) / len(dim_avg)
            final_scores["rating"] = round(min(10.0, raw_avg * 2), 2)

        # Compute decision from rating (threshold: >= 5 is accept)
        rating = final_scores.get("rating", 0)
        final_scores["decision"] = "accept" if rating >= 5 else "reject"

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
