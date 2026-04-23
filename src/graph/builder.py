"""
Hypergraph builder for evidence-rubric-code-text relationships.

In DeepReview context, nodes are:
- PaperSection: abstract, intro, methods, experiments, conclusion
- Claim: paper claims/contributions
- Evidence: extracted evidence items
- RubricDimension: Rating, Soundness, Presentation, Contribution
- Risk: identified risks

Hyperedges connect multiple nodes via pattern types.
"""

from typing import Any

import networkx as nx

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HyperGraphBuilder:
    """Build hypergraph from paper sections and evidence."""

    def __init__(self):
        self.graph = nx.Graph()
        self.hyperedges = []  # List of {type, nodes, weight, metadata}

    def add_paper_sections(self, sections: dict[str, str]) -> None:
        """Add paper section nodes."""
        for section_name, text in sections.items():
            if isinstance(text, str):
                self.graph.add_node(
                    f"section:{section_name}",
                    node_type="PaperSection",
                    name=section_name,
                    text=text[:1000],
                )

    def add_evidence(self, evidence_list: list[dict[str, Any]]) -> None:
        """Add evidence nodes and connect to sections."""
        for ev in evidence_list:
            ev_id = ev.get("evidence_id", "")
            if not ev_id:
                continue

            self.graph.add_node(
                f"evidence:{ev_id}",
                node_type="Evidence",
                source_type=ev.get("source_type", ""),
                section=ev.get("section", ""),
                text=ev.get("evidence_text", "")[:500],
                related_dimension=ev.get("related_dimension", ""),
                confidence=ev.get("confidence", 0.5),
            )

            # Connect to source section
            section = ev.get("section", "")
            if section and f"section:{section}" in self.graph:
                self.graph.add_edge(
                    f"section:{section}",
                    f"evidence:{ev_id}",
                    edge_type="source",
                    weight=ev.get("confidence", 0.5),
                )

    def add_rubric_dimensions(self, dimensions: list[dict[str, Any]]) -> None:
        """Add rubric dimension nodes."""
        for dim in dimensions:
            dim_key = dim.get("key", dim.get("name", "").lower())
            self.graph.add_node(
                f"dimension:{dim_key}",
                node_type="RubricDimension",
                name=dim.get("name", ""),
                description=dim.get("description", ""),
                scale=dim.get("scale", [1, 5]),
            )

    def add_risks(self, risks: list[dict[str, Any]]) -> None:
        """Add risk nodes."""
        for i, risk in enumerate(risks):
            risk_id = f"risk:{i + 1}"
            self.graph.add_node(
                risk_id,
                node_type="Risk",
                description=risk.get("description", ""),
                severity=risk.get("severity", "medium"),
            )

    def build_hyperedges(self, evidence_list: list[dict[str, Any]]) -> None:
        """Build pattern hyperedges from evidence and dimensions.

        Pattern types:
        - SoundnessPattern: method + experiment evidence -> Soundness
        - ContributionPattern: claim + reference evidence -> Contribution
        - PresentationPattern: section structure + figures/tables -> Presentation
        - ConsistencyPattern: abstract vs experiments
        - RiskPattern: missing evidence -> Risk
        """
        # Group evidence by type
        by_type = {}
        for ev in evidence_list:
            et = ev.get("source_type", "")
            by_type.setdefault(et, []).append(ev)

        # SoundnessPattern
        method_evs = by_type.get("method_evidence", [])
        exp_evs = by_type.get("experiment_evidence", [])
        if method_evs and exp_evs:
            self.hyperedges.append({
                "type": "SoundnessPattern",
                "nodes": ["dimension:soundness"] +
                         [f"evidence:{e['evidence_id']}" for e in method_evs[:2]] +
                         [f"evidence:{e['evidence_id']}" for e in exp_evs[:2]],
                "weight": 0.8,
                "metadata": {"description": "Methodology supported by experiments"},
            })

        # ContributionPattern
        claim_evs = by_type.get("claim_evidence", [])
        ref_evs = by_type.get("reference_evidence", [])
        if claim_evs:
            nodes = ["dimension:contribution"] + [f"evidence:{e['evidence_id']}" for e in claim_evs[:3]]
            if ref_evs:
                nodes.append(f"evidence:{ref_evs[0]['evidence_id']}")
            self.hyperedges.append({
                "type": "ContributionPattern",
                "nodes": nodes,
                "weight": 0.75,
                "metadata": {"description": "Claims and novelty assessment"},
            })

        # ConsistencyPattern
        consistency_evs = by_type.get("consistency_evidence", [])
        for ev in consistency_evs:
            self.hyperedges.append({
                "type": "ConsistencyPattern",
                "nodes": ["dimension:soundness", f"evidence:{ev['evidence_id']}"],
                "weight": ev.get("confidence", 0.7),
                "metadata": {"description": ev.get("evidence_text", "")},
            })

        # RiskPattern: missing evidence
        if not method_evs:
            self.hyperedges.append({
                "type": "RiskPattern",
                "nodes": ["dimension:soundness", "risk:1"],
                "weight": 0.9,
                "metadata": {"description": "No methodology evidence found"},
            })

    def get_related_evidence(self, dimension: str) -> list[dict[str, Any]]:
        """Get evidence related to a dimension via hyperedges."""
        related = []
        dim_node = f"dimension:{dimension.lower()}"

        for he in self.hyperedges:
            if dim_node in he["nodes"]:
                for node in he["nodes"]:
                    if node.startswith("evidence:") and node in self.graph:
                        data = dict(self.graph.nodes[node])
                        data["hyperedge_type"] = he["type"]
                        data["hyperedge_weight"] = he["weight"]
                        related.append(data)

        return related

    def get_consistency_conflicts(self) -> list[dict[str, Any]]:
        """Get consistency conflicts from hyperedges."""
        conflicts = []
        for he in self.hyperedges:
            if he["type"] == "ConsistencyPattern":
                conflicts.append({
                    "type": he["type"],
                    "description": he["metadata"].get("description", ""),
                    "weight": he["weight"],
                })
        return conflicts

    def to_context_summary(self, dimension: str) -> str:
        """Convert hypergraph substructure to text summary for LLM context."""
        lines = [f"Rubric: {dimension.capitalize()}", ""]

        # Evidence
        evidence = self.get_related_evidence(dimension)
        if evidence:
            lines.append("Evidence:")
            for ev in evidence:
                text = ev.get("text", "")[:200]
                lines.append(f"- {text}")

        # Consistency conflicts
        conflicts = self.get_consistency_conflicts()
        if conflicts:
            lines.append("\nConsistency Checks:")
            for c in conflicts:
                lines.append(f"- {c['description']}")

        return "\n".join(lines)


def build_hypergraph(
    sections: dict[str, str],
    evidence: list[dict[str, Any]],
    dimensions: list[dict[str, Any]],
    risks: list[dict[str, Any]] | None = None,
) -> HyperGraphBuilder:
    """Convenience function to build a complete hypergraph."""
    builder = HyperGraphBuilder()
    builder.add_paper_sections(sections)
    builder.add_evidence(evidence)
    builder.add_rubric_dimensions(dimensions)
    if risks:
        builder.add_risks(risks)
    builder.build_hyperedges(evidence)
    return builder
