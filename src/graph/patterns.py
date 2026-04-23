"""
Pattern definitions for hypergraph hyperedges.

Each pattern defines a type of relationship between evidence, paper sections,
rubric dimensions, and risks.
"""

from typing import Any


class HyperEdgePattern:
    """Base class for hyperedge patterns."""

    def __init__(self, name: str, description: str, weight: float = 0.5):
        self.name = name
        self.description = description
        self.weight = weight

    def match(self, nodes: list[dict[str, Any]]) -> bool:
        """Check if a set of nodes matches this pattern."""
        raise NotImplementedError


class SoundnessPattern(HyperEdgePattern):
    """Connects methodology evidence and experimental evidence to Soundness dimension."""

    def __init__(self):
        super().__init__(
            "SoundnessPattern",
            "Methodology supported by experiments",
            weight=0.8,
        )

    def match(self, nodes: list[dict[str, Any]]) -> bool:
        types = {n.get("node_type", "") for n in nodes}
        return "Evidence" in types and "RubricDimension" in types


class ContributionPattern(HyperEdgePattern):
    """Connects claim evidence and references to Contribution dimension."""

    def __init__(self):
        super().__init__(
            "ContributionPattern",
            "Claims and novelty assessment",
            weight=0.75,
        )


class PresentationPattern(HyperEdgePattern):
    """Connects section structure and figures/tables to Presentation dimension."""

    def __init__(self):
        super().__init__(
            "PresentationPattern",
            "Clarity and organization assessment",
            weight=0.6,
        )


class ConsistencyPattern(HyperEdgePattern):
    """Flags inconsistencies between paper sections."""

    def __init__(self):
        super().__init__(
            "ConsistencyPattern",
            "Cross-section consistency check",
            weight=0.7,
        )


class RiskPattern(HyperEdgePattern):
    """Connects risks to affected dimensions."""

    def __init__(self):
        super().__init__(
            "RiskPattern",
            "Identified risks and their impact",
            weight=0.9,
        )
