"""
Prompt-Only Scorer (Ablation).

This is the MOST MINIMAL version of our scoring system.
NO evidence extraction, NO hypergraph, NO multi-agent, NO arbitration.
Only a carefully crafted prompt sent to the LLM once.

Purpose: Isolate the effect of prompt engineering alone.
"""

import re
from pathlib import Path
from typing import Any

from src.scoring.baseline import BaseScorer
from src.utils.llm_wrapper import LLMWrapper
from src.utils.logger import get_logger
from src.utils.parser import round_to_step

logger = get_logger(__name__)


class PromptOnlyScorer(BaseScorer):
    """Pure prompt-based scorer. One LLM call, all dimensions at once."""

    SYSTEM_PROMPT = (
        "You are a STRICT and CRITICAL expert reviewer for NeurIPS/ICML-level conferences.\n"
        "Your job is to identify flaws and give accurate, discriminating scores.\n"
        "Do NOT be polite. Do NOT inflate scores. Most papers are average or below.\n\n"

        "=== DATASET CALIBRATION (Human Reviewer Statistics) ===\n"
        "- Rating (1-10): typical papers score 5.5, most fall in 4-7 range\n"
        "  9.0+: Paradigm-shifting (Top 1%, extremely rare)\n"
        "  7.5-8.5: Strong accept (flawless methodology + significant empirical superiority)\n"
        "  6.0-7.0: Above average / borderline\n"
        "  5.0-5.5: Average (incremental work with noticeable gaps)\n"
        "  3.5-4.5: Below average (missing baselines, weak evaluation)\n"
        "  1.0-3.0: Poor (fundamentally flawed)\n"
        "- Soundness / Presentation / Contribution (1-4): typical papers score 2.5-3.0\n"
        "  4.0: Outstanding and flawless (rare)\n"
        "  3.0: Good, solid with minor issues\n"
        "  2.0: Fair, noticeable weaknesses\n"
        "  1.0: Poor, severe flaws\n\n"

        "=== UNIVERSAL ANTI-GULLIBILITY RULES ===\n"
        "1. DO NOT treat author claims in abstract/introduction as facts.\n"
        "   'We propose a novel method' is NOT a strength. Only empirical results and proofs count.\n"
        "2. DO NOT penalize honest limitation discussions. Acknowledging limitations is a sign of HIGH quality.\n"
        "3. Complex math without ablation studies is NOT rigorous. Simple, validated methods score higher.\n"
        "4. If the paper only combines existing techniques, it is INCREMENTAL, not groundbreaking.\n\n"

        "=== STRICT SCORING CAPS (MUST OBEY) ===\n"
        "Rating (1-10):\n"
        "- CAP AT 7.0: If you identify ANY notable issue (missing baseline, lack of clarity, weak evaluation),\n"
        "  the rating MUST NOT exceed 7.0, regardless of strengths.\n"
        "- CAP AT 5.5: If the contribution is merely combining existing techniques,\n"
        "  or lacks rigorous ablation studies, the rating MUST NOT exceed 5.5.\n"
        "- DEFAULT TO 5.0: When in doubt, score between 4.5 and 5.5.\n\n"

        "Soundness (1-4):\n"
        "- CAP AT 3.0: If you write ANYTHING in Weaknesses, score CANNOT be 4.0. Must be 3.0 or lower.\n"
        "- CAP AT 2.5: If ablation studies are missing or baselines are weak, MUST NOT exceed 2.5.\n"
        "- CAP AT 2.0: If methodology is unclear or unsupported, MUST NOT exceed 2.0.\n\n"

        "Presentation (1-4):\n"
        "- CAP AT 3.0: If you write ANYTHING in Weaknesses, score CANNOT be 4.0. Must be 3.0 or lower.\n"
        "- CAP AT 2.5: If figures/tables are unclear or notation is confusing, MUST NOT exceed 2.5.\n\n"

        "Contribution (1-4):\n"
        "- CAP AT 3.0: Unless the paper solves a long-standing open problem, maximum is 3.0.\n"
        "- CAP AT 2.5: If the method just applies an existing algorithm to a slightly new dataset/task,\n"
        "  it is MINOR incremental work. MUST NOT exceed 2.5.\n"
        "- CAP AT 2.0: If the core idea is well-known or trivial, MUST NOT exceed 2.0.\n\n"

        "=== REASONING ORDER (DO NOT SKIP) ===\n"
        "You MUST think in this order:\n"
        "1. Summary: Concise overview of core ideas.\n"
        "2. Weaknesses FIRST: Identify ALL flaws before assigning scores. Be specific and concrete.\n"
        "3. Strengths: Only after weaknesses. Be sparing with praise.\n"
        "4. Soundness Assessment: Judge methodology rigor.\n"
        "5. Presentation Assessment: Judge clarity and organization.\n"
        "6. Contribution Assessment: Judge novelty and significance ONLY. Ignore experimental weaknesses here.\n"
        "7. Overall Rating: Synthesize holistically AFTER all above.\n\n"

        "=== OUTPUT FORMAT (STRICT) ===\n"
        "Provide your review in this exact format. Use 0.05 precision for all scores.\n\n"
        "Soundness: [1.00-4.00]\n"
        "Presentation: [1.00-4.00]\n"
        "Contribution: [1.00-4.00]\n"
        "Rating: [1.00-10.00]\n"
        "Confidence: [1-5]\n"
        "Decision: [Accept or Reject]  (Accept ONLY if Rating >= 6.5)\n\n"
        "Weaknesses:\n"
        "- [weakness 1]\n"
        "- [weakness 2]\n"
        "...\n\n"
        "Strengths:\n"
        "- [strength 1]\n"
        "...\n\n"
        "Summary: [brief summary]\n"
    )

    def score(self, paper_context: str, title: str = "") -> dict[str, Any]:
        user_prompt = self._build_user_prompt(paper_context, title)

        raw_output = self.llm.generate(
            prompt=user_prompt,
            system_prompt=self.SYSTEM_PROMPT,
            max_tokens=4000,
        )

        scores = self._parse_scores(raw_output)

        return {
            "mode": "prompt_only",
            "raw_output": raw_output,
            "scores": scores,
        }

    @staticmethod
    def _build_user_prompt(paper_context: str, title: str) -> str:
        header = f"Title: {title}\n\n" if title else ""
        return (
            f"{header}Review the following research paper.\n\n"
            f"=== PAPER CONTENT ===\n"
            f"{paper_context[:8000]}\n\n"
            f"=== INSTRUCTIONS ===\n"
            f"Follow the reasoning order in the system prompt.\n"
            f"First identify ALL weaknesses, then assign scores.\n"
            f"Remember the scoring caps - they are hard limits.\n"
            f"Output ONLY the structured format specified."
        )

    @staticmethod
    def _parse_scores(text: str) -> dict[str, Any]:
        """Extract scores from structured output with robust pattern matching."""
        scores = {
            "soundness": None,
            "presentation": None,
            "contribution": None,
            "rating": None,
            "confidence": None,
            "decision": None,
        }

        if not text:
            return scores

        # Extract each dimension with multiple fallback patterns
        for dim in ["soundness", "presentation", "contribution", "rating", "confidence"]:
            val = None
            # Pattern 1: "Rating: 8.0" or "rating = 8" or "Rating: [8.0]"
            match = re.search(rf'{dim}[\s]*[:=][\s]*[\[\(]?(\d+(?:\.\d+)?)[\]\)]?', text, re.IGNORECASE)
            if match:
                val = float(match.group(1))
            # Pattern 2: markdown bold "**Rating**: 8.0" or "**Rating**: [8.0]"
            if val is None:
                match = re.search(rf'\*?\*{dim}\*?\*[\s]*[:=][\s]*[\[\(]?(\d+(?:\.\d+)?)[\]\)]?', text, re.IGNORECASE)
                if match:
                    val = float(match.group(1))
            # Pattern 3: loose match on the line
            if val is None:
                match = re.search(rf'^{dim}[\s]*[:=]?[\s]*[\[\(]?(\d+(?:\.\d+)?)[\]\)]?', text, re.IGNORECASE | re.MULTILINE)
                if match:
                    val = float(match.group(1))

            if val is not None:
                scores[dim] = round_to_step(val)

        # Extract decision
        dec_match = re.search(r'Decision[\s]*[:=][\s]*(Accept|Reject)', text, re.IGNORECASE)
        if dec_match:
            scores["decision"] = dec_match.group(1).lower()
        else:
            # Fallback: use rating threshold
            if scores.get("rating") is not None:
                scores["decision"] = "accept" if scores["rating"] >= 6.5 else "reject"
            else:
                scores["decision"] = "reject"

        # Clamp values
        if scores["soundness"] is not None:
            scores["soundness"] = max(1.0, min(4.0, scores["soundness"]))
        if scores["presentation"] is not None:
            scores["presentation"] = max(1.0, min(4.0, scores["presentation"]))
        if scores["contribution"] is not None:
            scores["contribution"] = max(1.0, min(4.0, scores["contribution"]))
        if scores["rating"] is not None:
            scores["rating"] = max(1.0, min(10.0, scores["rating"]))

        return scores
