"""
Risk Assessment Agent.

Determines if a sample should be escalated to human review.
Conditions (any one triggers escalation):
1. Multi-agent score disagreement (std > threshold)
2. Insufficient evidence (count < threshold)
3. Low confidence (any agent confidence < threshold)
4. Output format errors
5. Hypergraph consistency conflicts
"""

from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RiskAgent:
    """Agent that assesses risk and decides on human escalation."""

    def __init__(
        self,
        score_std_threshold: float = 1.5,
        confidence_threshold: float = 2.0,
        min_evidence_count: int = 3,
        max_conflict_count: int = 1,
    ):
        self.score_std_threshold = score_std_threshold
        self.confidence_threshold = confidence_threshold
        self.min_evidence_count = min_evidence_count
        self.max_conflict_count = max_conflict_count

    def run(
        self,
        dimension_scores: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        conflicts: list[dict[str, Any]],
        parse_success: bool = True,
    ) -> dict[str, Any]:
        """Assess risk and return escalation decision.

        Returns:
            Dict with 'escalate', 'reasons', 'risk_level'.
        """
        reasons = []
        risk_level = "low"

        # Check 1: Multi-agent disagreement
        by_dim = {}
        for ds in dimension_scores:
            dim = ds.get("dimension", "")
            if dim:
                by_dim.setdefault(dim, []).append(ds.get("score"))

        for dim, scores in by_dim.items():
            valid = [s for s in scores if s is not None]
            if len(valid) >= 2:
                std = np.std(valid)
                if std > self.score_std_threshold:
                    reasons.append(f"High disagreement in {dim} (std={std:.2f})")
                    risk_level = "high"

        # Check 2: Insufficient evidence
        if len(evidence) < self.min_evidence_count:
            reasons.append(f"Insufficient evidence ({len(evidence)} < {self.min_evidence_count})")
            if risk_level != "high":
                risk_level = "medium"

        # Check 3: Low confidence
        for ds in dimension_scores:
            conf = ds.get("confidence")
            if conf is not None and conf < self.confidence_threshold:
                reasons.append(f"Low confidence in {ds.get('dimension', '')} ({conf:.1f})")
                if risk_level != "high":
                    risk_level = "medium"

        # Check 4: Parse errors
        if not parse_success:
            reasons.append("Output format error (parse failed)")
            risk_level = "high"

        # Check 5: Consistency conflicts
        if len(conflicts) > self.max_conflict_count:
            reasons.append(f"Too many consistency conflicts ({len(conflicts)})")
            risk_level = "high"

        escalate = len(reasons) > 0

        return {
            "escalate": escalate,
            "reasons": reasons,
            "risk_level": risk_level,
            "evidence_count": len(evidence),
            "conflict_count": len(conflicts),
        }
