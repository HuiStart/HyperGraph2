"""
Evidence extraction module.

Extracts structured evidence from academic papers for review scoring.
Evidence types (aligned with DeepReview academic context):
- claim_evidence: core claims from abstract/intro/conclusion
- method_evidence: methodology descriptions
- experiment_evidence: experimental setup and results
- reference_evidence: key citations
- consistency_evidence: cross-section consistency checks

Extraction methods:
1. Rule-based: regex on LaTeX structure
2. LLM-based: dedicated prompt for evidence extraction
"""

import re
from typing import Any

from src.preprocess.latex_parser import LaTeXParser
from src.utils.llm_wrapper import LLMWrapper
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvidenceExtractor:
    """Extract structured evidence from paper text."""

    EVIDENCE_TYPES = [
        "claim_evidence",
        "method_evidence",
        "experiment_evidence",
        "reference_evidence",
        "consistency_evidence",
    ]

    def __init__(self, llm: LLMWrapper | None = None, use_llm: bool = True):
        self.llm = llm
        self.use_llm = use_llm
        self.latex_parser = LaTeXParser()

    def extract(self, paper_context: str, title: str = "") -> list[dict[str, Any]]:
        """Extract all evidence types from a paper.

        Returns:
            List of evidence dicts with fields:
            - evidence_id, source_type, section, evidence_text,
              related_dimension, confidence
        """
        evidence = []

        # Parse LaTeX structure
        sections = self.latex_parser.parse(paper_context)

        # Rule-based extraction
        evidence.extend(self._extract_claims(sections, title))
        evidence.extend(self._extract_methods(sections))
        evidence.extend(self._extract_experiments(sections))
        evidence.extend(self._extract_references(sections))
        evidence.extend(self._extract_consistency(sections))

        # LLM-based extraction (if enabled)
        if self.use_llm and self.llm:
            evidence.extend(self._extract_with_llm(paper_context, title))

        # Deduplicate by evidence_text
        seen = set()
        unique_evidence = []
        for e in evidence:
            text_hash = hash(e["evidence_text"][:200])
            if text_hash not in seen:
                seen.add(text_hash)
                unique_evidence.append(e)

        # Assign IDs
        for i, e in enumerate(unique_evidence):
            e["evidence_id"] = f"E{i + 1:03d}"

        return unique_evidence

    def _extract_claims(self, sections: dict[str, Any], title: str) -> list[dict[str, Any]]:
        """Extract core claims from abstract and introduction."""
        evidence = []

        for section_name in ["abstract", "introduction"]:
            text = sections.get(section_name, "")
            if not text:
                continue

            # Split into sentences (simple heuristic)
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for sent in sentences[:5]:  # First few sentences usually contain claims
                sent = sent.strip()
                if len(sent) > 30:
                    evidence.append({
                        "source_type": "claim_evidence",
                        "section": section_name,
                        "evidence_text": sent,
                        "related_dimension": "Contribution",
                        "confidence": 0.85,
                    })

        return evidence

    def _extract_methods(self, sections: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract methodology evidence."""
        evidence = []

        method_text = sections.get("methods", "")
        if not method_text:
            # Try to find methods in other sections
            for key in sections:
                if any(x in key.lower() for x in ["method", "approach", "model", "algorithm"]):
                    method_text = sections[key]
                    break

        if method_text:
            # Extract first paragraph (usually describes the core method)
            paragraphs = method_text.split('\n\n')
            for para in paragraphs[:2]:
                para = para.strip()
                if len(para) > 50:
                    evidence.append({
                        "source_type": "method_evidence",
                        "section": "methods",
                        "evidence_text": para[:500],
                        "related_dimension": "Soundness",
                        "confidence": 0.8,
                    })

        return evidence

    def _extract_experiments(self, sections: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract experimental evidence."""
        evidence = []

        for section_name in ["experiments", "results", "evaluation"]:
            text = sections.get(section_name, "")
            if text:
                paragraphs = text.split('\n\n')
                for para in paragraphs[:3]:
                    para = para.strip()
                    if len(para) > 50:
                        evidence.append({
                            "source_type": "experiment_evidence",
                            "section": section_name,
                            "evidence_text": para[:500],
                            "related_dimension": "Soundness",
                            "confidence": 0.75,
                        })

        return evidence

    def _extract_references(self, sections: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract reference/citation evidence."""
        evidence = []

        refs = sections.get("references", [])
        if isinstance(refs, list) and refs:
            evidence.append({
                "source_type": "reference_evidence",
                "section": "related_work",
                "evidence_text": f"Key citations: {', '.join(refs[:10])}",
                "related_dimension": "Contribution",
                "confidence": 0.7,
            })

        return evidence

    def _extract_consistency(self, sections: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract consistency evidence (cross-section checks)."""
        evidence = []

        abstract = sections.get("abstract", "")
        experiments = sections.get("experiments", "")

        if abstract and not experiments:
            evidence.append({
                "source_type": "consistency_evidence",
                "section": "global",
                "evidence_text": "Paper claims experiments in abstract but no experiments section found.",
                "related_dimension": "Soundness",
                "confidence": 0.9,
            })

        # Check for method-experiment alignment
        methods = sections.get("methods", "")
        if methods and experiments:
            # Simple keyword check for alignment
            method_keywords = set(re.findall(r'\b\w+\b', methods.lower()))
            exp_keywords = set(re.findall(r'\b\w+\b', experiments.lower()))
            overlap = len(method_keywords & exp_keywords)
            if overlap < 5:
                evidence.append({
                    "source_type": "consistency_evidence",
                    "section": "global",
                    "evidence_text": "Low overlap between methods and experiments sections. Potential inconsistency.",
                    "related_dimension": "Soundness",
                    "confidence": 0.6,
                })

        return evidence

    def _extract_with_llm(self, paper_context: str, title: str) -> list[dict[str, Any]]:
        """Use LLM to extract evidence."""
        if not self.llm:
            return []

        prompt = self._build_extraction_prompt(paper_context, title)

        try:
            response = self.llm.generate_json(prompt, max_tokens=4000)
            if isinstance(response, dict) and "evidence" in response:
                return response["evidence"]
        except Exception as e:
            logger.warning(f"LLM evidence extraction failed: {e}")

        return []

    def _build_extraction_prompt(self, paper_context: str, title: str) -> str:
        return (
            f"Extract structured evidence from the following research paper.\n\n"
            f"Title: {title}\n\n"
            f"Paper:\n{paper_context[:8000]}\n\n"
            f"For each piece of evidence, provide:\n"
            f"1. source_type: one of [claim_evidence, method_evidence, experiment_evidence, reference_evidence, consistency_evidence]\n"
            f"2. section: which section it came from\n"
            f"3. evidence_text: the actual text (max 300 chars)\n"
            f"4. related_dimension: which scoring dimension it relates to [Rating, Soundness, Presentation, Contribution]\n"
            f"5. confidence: 0-1 score\n\n"
            f"Output as JSON: {{'evidence': [...]}}"
        )


class EvidenceMerger:
    """Merge and deduplicate evidence from multiple sources."""

    @staticmethod
    def merge(evidence_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Merge multiple evidence lists, deduplicating by text similarity."""
        all_evidence = []
        for ev_list in evidence_lists:
            all_evidence.extend(ev_list)

        # Simple dedup: truncate and compare
        seen = set()
        merged = []
        for e in all_evidence:
            key = e["evidence_text"][:100].lower().strip()
            if key not in seen:
                seen.add(key)
                merged.append(e)

        return merged
