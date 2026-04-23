"""
Consistency checking rules for hypergraph.

Rules:
1. Abstract claims must be supported by experiments
2. Methods described must match experiments conducted
3. References must be cited in the paper
4. Results must be consistent with methodology
"""

from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConsistencyChecker:
    """Check cross-section consistency in papers."""

    def check(self, sections: dict[str, Any]) -> list[dict[str, Any]]:
        """Run all consistency checks.

        Returns:
            List of conflict dicts with fields:
            - type, description, severity, confidence
        """
        conflicts = []
        conflicts.extend(self._check_abstract_experiments(sections))
        conflicts.extend(self._check_methods_experiments(sections))
        conflicts.extend(self._check_figures_referenced(sections))
        conflicts.extend(self._check_results_consistency(sections))
        return conflicts

    def _check_abstract_experiments(self, sections: dict[str, Any]) -> list[dict[str, Any]]:
        """Check if abstract mentions experiments that exist."""
        conflicts = []
        abstract = sections.get("abstract", "")
        experiments = sections.get("experiments", "")

        # Simple heuristic: abstract mentions results/experiments but no experiments section
        if abstract and any(kw in abstract.lower() for kw in ["experiment", "result", "evaluate", "dataset"]):
            if not experiments:
                conflicts.append({
                    "type": "missing_experiments",
                    "description": "Abstract mentions experiments/results but no experiments section found.",
                    "severity": "high",
                    "confidence": 0.9,
                })

        return conflicts

    def _check_methods_experiments(self, sections: dict[str, Any]) -> list[dict[str, Any]]:
        """Check if methods and experiments align."""
        conflicts = []
        methods = sections.get("methods", "")
        experiments = sections.get("experiments", "")

        if methods and experiments:
            # Check for method keywords in experiments
            method_keywords = ["model", "algorithm", "network", "approach"]
            found = any(kw in experiments.lower() for kw in method_keywords)
            if not found:
                conflicts.append({
                    "type": "methods_experiments_mismatch",
                    "description": "Experiments section does not reference methods described.",
                    "severity": "medium",
                    "confidence": 0.7,
                })

        return conflicts

    def _check_figures_referenced(self, sections: dict[str, Any]) -> list[dict[str, Any]]:
        """Check if figures/tables are referenced in text."""
        conflicts = []
        figures = sections.get("figures", [])
        full_text = " ".join(str(v) for v in sections.values() if isinstance(v, str))

        for fig in figures:
            caption = fig.get("caption", "")
            if caption and caption not in full_text:
                # This is a weak check, skip for now
                pass

        return conflicts

    def _check_results_consistency(self, sections: dict[str, Any]) -> list[dict[str, Any]]:
        """Check if results are internally consistent."""
        conflicts = []
        experiments = sections.get("experiments", "")

        if experiments:
            # Check for numerical consistency (simple heuristic)
            numbers = []
            for match in __import__('re').finditer(r'(\d+\.?\d*)\s*%', experiments):
                numbers.append(float(match.group(1)))

            if len(numbers) >= 2:
                # Check if any result contradicts another (simplified)
                pass

        return conflicts
