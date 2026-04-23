"""
Evidence Extraction Agent.

Only extracts evidence, does not score.
Outputs structured evidence list for downstream agents.
"""

from typing import Any

from src.evidence.extractor import EvidenceExtractor
from src.preprocess.latex_parser import LaTeXParser
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvidenceExtractionAgent:
    """Agent responsible for extracting evidence from papers."""

    def __init__(self, llm_wrapper=None, use_llm: bool = False):
        self.extractor = EvidenceExtractor(llm=llm_wrapper, use_llm=use_llm)
        self.parser = LaTeXParser()

    def run(self, paper_context: str, title: str = "") -> dict[str, Any]:
        """Extract evidence from a paper.

        Returns:
            Dict with 'evidence' list and 'sections' dict.
        """
        logger.info(f"EvidenceAgent: extracting evidence for '{title[:50]}...'")

        # Parse sections
        sections = self.parser.parse(paper_context)

        # Extract evidence
        evidence = self.extractor.extract(paper_context, title)

        return {
            "sections": sections,
            "evidence": evidence,
            "num_evidence": len(evidence),
        }
