"""
Multi-agent workflow orchestration using LangGraph.

Workflow:
    Start -> EvidenceAgent -> [Parallel] ScoringAgents -> ArbitrationAgent -> RiskAgent
                                                     \
                                               ExplanationAgent (if not escalated)

This is a simplified implementation that can run sequentially or with basic parallelism.
For true parallel scoring, use threading or LangGraph's built-in parallel nodes.
"""

from typing import Any

from src.agents.arbitration_agent import ArbitrationAgent
from src.agents.evidence_agent import EvidenceExtractionAgent
from src.agents.explanation_agent import ExplanationAgent
from src.agents.risk_agent import RiskAgent
from src.agents.scoring_agent import DimensionScoringAgent
from src.graph.builder import build_hypergraph
from src.graph.consistency import ConsistencyChecker
from src.rubric.fixed_rubric import FixedRubric
from src.utils.llm_wrapper import LLMWrapper
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReviewWorkflow:
    """Orchestrated multi-agent review workflow."""

    def __init__(
        self,
        llm: LLMWrapper | None = None,
        use_llm_evidence: bool = False,
    ):
        self.llm = llm or LLMWrapper()
        self.rubric = FixedRubric()

        # Initialize agents
        self.evidence_agent = EvidenceExtractionAgent(self.llm, use_llm=use_llm_evidence)
        self.arbitration_agent = ArbitrationAgent(self.llm)
        self.explanation_agent = ExplanationAgent(self.llm)
        self.risk_agent = RiskAgent()
        self.consistency_checker = ConsistencyChecker()

        # Scoring agents per dimension (including rating for direct scoring)
        self.scoring_agents = {
            dim["key"]: DimensionScoringAgent(dim["key"], self.llm)
            for dim in self.rubric.get_all_dimensions()
        }

    def run(self, paper_context: str, title: str = "", sample_id: str = "") -> dict[str, Any]:
        """Run the full multi-agent workflow on a single paper.

        Returns:
            Complete review result with scores, explanation, and risk assessment.
        """
        logger.info(f"Workflow: processing sample {sample_id}")

        # Step 1: Evidence Extraction
        evidence_result = self.evidence_agent.run(paper_context, title)
        evidence = evidence_result["evidence"]
        sections = evidence_result["sections"]

        # Build hypergraph
        dimensions = self.rubric.get_all_dimensions()
        conflicts = self.consistency_checker.check(sections)
        hypergraph = build_hypergraph(sections, evidence, dimensions, risks=conflicts)

        # Step 2: Parallel Dimension Scoring
        dimension_scores = []
        for dim_key, agent in self.scoring_agents.items():
            # Enhance context with hypergraph evidence
            related = hypergraph.get_related_evidence(dim_key)
            all_evidence = evidence + [
                {
                    "evidence_id": f"HG_{i}",
                    "source_type": "hypergraph",
                    "section": "hypergraph",
                    "evidence_text": r.get("text", ""),
                    "related_dimension": dim_key.capitalize(),
                    "confidence": r.get("confidence", 0.5),
                }
                for i, r in enumerate(related)
            ]

            score_result = agent.run(paper_context, all_evidence, title)
            dimension_scores.append(score_result)

        # Step 3: Arbitration
        arbitration_result = self.arbitration_agent.run(dimension_scores)
        final_scores = arbitration_result["final_scores"]

        # Step 4: Risk Assessment
        risk_result = self.risk_agent.run(
            dimension_scores=dimension_scores,
            evidence=evidence,
            conflicts=conflicts,
            parse_success=True,
        )

        # Step 5: Explanation (only if not escalated)
        explanation_result = None
        if not risk_result["escalate"]:
            explanation_result = self.explanation_agent.run(
                title=title,
                final_scores=final_scores,
                evidence=evidence,
                arbitration_notes=arbitration_result.get("arbitration_notes", []),
            )

        # Build final output
        output = {
            "sample_id": sample_id,
            "title": title,
            "mode": "ours",
            "scores": final_scores,
            "evidence": {
                "count": len(evidence),
                "items": evidence,
            },
            "dimension_scores": dimension_scores,
            "arbitration": arbitration_result,
            "risk": risk_result,
            "explanation": explanation_result,
            "hypergraph": {
                "num_nodes": hypergraph.graph.number_of_nodes(),
                "num_edges": hypergraph.graph.number_of_edges(),
                "num_hyperedges": len(hypergraph.hyperedges),
                "conflicts": conflicts,
            },
        }

        return output

    def run_batch(
        self,
        samples: list[dict[str, Any]],
        output_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run workflow on a batch of samples."""
        results = []
        for i, sample in enumerate(samples):
            logger.info(f"Batch {i + 1}/{len(samples)}: {sample.get('id', '')}")
            try:
                result = self.run(
                    paper_context=sample.get("paper_context", ""),
                    title=sample.get("title", ""),
                    sample_id=sample.get("id", ""),
                )
                results.append(result)

                # Incremental save
                if output_path and (i + 1) % 5 == 0:
                    self._save_results(results, output_path)

            except Exception as e:
                logger.error(f"Workflow failed for sample {sample.get('id')}: {e}")
                results.append({
                    "sample_id": sample.get("id", ""),
                    "error": str(e),
                    "scores": {},
                })

        if output_path:
            self._save_results(results, output_path)

        return results

    @staticmethod
    def _save_results(results: list[dict[str, Any]], path: str) -> None:
        import json
        from pathlib import Path

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
