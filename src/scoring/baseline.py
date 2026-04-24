"""
Baseline scoring implementations aligned with DeepReviewer official modes.

Baseline 1: Fast Mode - pure LLM end-to-end scoring
Baseline 2: Standard Mode - multi-reviewer simulation + self-verification
Baseline 3: Best Mode - retrieval-enhanced scoring (with local fallback)

Reference:
- Research/ai_researcher/deep_reviewer.py
- Research/ai_researcher/cycle_reviewer.py
"""

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.scoring.aggregation import aggregate_reviews
from src.utils.llm_wrapper import LLMWrapper
from src.utils.logger import get_logger
from src.utils.parser import parse_deepreviewer_output, round_to_step

logger = get_logger(__name__)


class BaseScorer(ABC):
    """Abstract base class for all scorers."""

    def __init__(self, llm: LLMWrapper | None = None):
        self.llm = llm or LLMWrapper()

    @abstractmethod
    def score(self, paper_context: str, title: str = "") -> dict[str, Any]:
        """Score a single paper and return structured results."""
        pass

    def score_batch(
        self,
        samples: list[dict[str, Any]],
        output_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Score a batch of papers.

        Args:
            samples: List of samples with 'paper_context' and 'title'.
            output_path: Optional path to save incremental results.

        Returns:
            List of results dicts.
        """
        results = []
        for i, sample in enumerate(samples):
            logger.info(f"Scoring sample {i + 1}/{len(samples)}: {sample.get('id', 'unknown')}")
            try:
                result = self.score(
                    paper_context=sample.get("paper_context", ""),
                    title=sample.get("title", ""),
                )
                result["sample_id"] = sample.get("id", "")
                results.append(result)

                # Incremental save
                if output_path and (i + 1) % 5 == 0:
                    self._save_results(results, output_path)

            except Exception as e:
                logger.error(f"Failed to score sample {sample.get('id')}: {e}")
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
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


class FastModeScorer(BaseScorer):
    """Baseline 1: Pure LLM end-to-end scoring (Fast Mode).

    Aligns with DeepReviewer Fast Mode:
    - Single direct review output
    - \boxed_review{} format
    - Sections: Summary, Soundness, Presentation, Contribution, Strengths,
                Weaknesses, Suggestions, Questions, Rating, Confidence, Decision
    """

    def score(self, paper_context: str, title: str = "") -> dict[str, Any]:
        system_prompt = self.llm.get_scoring_prompt(mode="fast")
        user_prompt = self._build_user_prompt(paper_context, title)

        raw_output = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=7000,
        )

        parsed = parse_deepreviewer_output(raw_output)
        meta = parsed.get("meta_review", {})

        # Fallback: if boxed_review is missing dimensions, try extracting from raw text
        soundness = self._extract_number(meta.get("soundness"))
        presentation = self._extract_number(meta.get("presentation"))
        contribution = self._extract_number(meta.get("contribution"))
        rating = meta.get("rating")
        decision = parsed.get("decision", "")

        if soundness is None:
            soundness = self._extract_from_raw(raw_output, "soundness")
        if presentation is None:
            presentation = self._extract_from_raw(raw_output, "presentation")
        if contribution is None:
            contribution = self._extract_from_raw(raw_output, "contribution")
        if rating is None:
            rating = self._extract_from_raw(raw_output, "rating")
        if not decision:
            dec_m = re.search(r'Decision[:\s]+(Accept|Reject)', raw_output, re.IGNORECASE)
            if dec_m:
                decision = dec_m.group(1).lower()

        # Strong fallback: if still no rating, ask LLM again with simplified prompt
        if rating is None:
            retry_prompt = (
                f"Based on your review above, please now output ONLY the numerical scores in this exact format:\n\n"
                f"Soundness: [1-4]\n"
                f"Presentation: [1-4]\n"
                f"Contribution: [1-4]\n"
                f"Rating: [1-10]\n"
                f"Decision: [Accept or Reject]"
            )
            retry_output = self.llm.generate(
                prompt=retry_prompt,
                system_prompt=None,
                max_tokens=500,
            )
            raw_output += "\n\n" + retry_output
            # Parse retry output
            for dim in ["soundness", "presentation", "contribution", "rating"]:
                val = locals().get(dim)
                if val is None:
                    m = re.search(rf'{dim.capitalize()}[:\s]+(\d+(?:\.\d+)?)', retry_output, re.IGNORECASE)
                    if m:
                        locals()[dim] = round(float(m.group(1)), 2)
            dec_m = re.search(r'Decision[:\s]+(Accept|Reject)', retry_output, re.IGNORECASE)
            if dec_m:
                decision = dec_m.group(1).lower()

        # Build scores dict
        scores = {
            "rating": rating,
            "soundness": soundness,
            "presentation": presentation,
            "contribution": contribution,
            "decision": decision.lower() if decision else "reject",
        }

        # Enforce decision consistency with rating threshold
        if scores.get("rating") is not None and scores["rating"] < 6.5:
            scores["decision"] = "reject"

        return {
            "mode": "fast",
            "raw_output": raw_output,
            "parsed": parsed,
            "scores": scores,
        }

    @staticmethod
    def _extract_from_raw(text: str, dimension: str) -> float | None:
        """Extract score for a dimension from raw text when boxed_review misses it."""
        if not text:
            return None
        # Look for patterns like "## Soundness: 3" or "Soundness: 3 good" or "**Soundness**: 3"
        patterns = [
            rf'##\s*{dimension.capitalize()}[:\s]+(\d+(?:\.\d+)?)',
            rf'\*\*\s*{dimension.capitalize()}\s*\*\*[:\s]+(\d+(?:\.\d+)?)',
            rf'{dimension.capitalize()}[:\s]+(\d+(?:\.\d+)?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return round_to_step(float(match.group(1)))
        return None

    @staticmethod
    def _build_user_prompt(paper_context: str, title: str) -> str:
        header = f"Title: {title}\n\n" if title else ""
        return (
            f"{header}Please review the following research paper.\n\n"
            f"{paper_context}\n\n"
<<<<<<< HEAD
            f"=== SCORING INSTRUCTIONS ===\n"
            f"You are reviewing for a NeurIPS/ICML-level conference. "
            f"Score this paper by comparing it against typical submissions.\n\n"
            f"DATASET CALIBRATION (human reviewer statistics):\n"
            f"- Rating (1-10): typical papers score 5.5 (range 4-7)\n"
            f"  9.0+: Exceptional  |  7.5: Strong  |  6.0: Above avg  |  5.0: Average  |  3.5: Below avg  |  2.0: Poor\n"
            f"- Soundness/Presentation/Contribution (1-4): typical papers score 2.5-3.0\n"
            f"  4.0: Outstanding  |  3.25: Good  |  2.75: Average  |  2.0: Weak  |  1.25: Poor\n\n"
            f"RULES:\n"
            f"1. Do NOT start from max and deduct. Judge absolute quality directly.\n"
            f"2. Most papers cluster near the typical range. Only exceptional work deserves top scores.\n"
            f"3. Be specific in your weaknesses, but score based on overall quality, not flaw count.\n"
            f"4. Use 0.05 precision for all scores (e.g. 3.25, 5.50, 7.15).\n\n"
=======
            f"=== CRITICAL SCORING INSTRUCTIONS ===\n"
            f"You MUST use DEDUCTION-BASED scoring to avoid inflation:\n"
            f"1. For each dimension, start from the MAXIMUM possible score.\n"
            f"2. Identify specific flaws, weaknesses, or limitations.\n"
            f"3. Deduct points for each flaw:\n"
            f"   - Rating (1-10): minor -0.5, moderate -1.0, major -2.0\n"
            f"   - Sub-dims (1-4): minor -0.25, moderate -0.5, major -1.0\n"
            f"4. Final score = max_score - total_deductions.\n\n"
            f"SCORE ANCHORS (based on NeurIPS/ICML standards):\n"
            f"- Rating 8.5-10: Truly exceptional, landmark contribution. Almost no flaws.\n"
            f"- Rating 7.0-8.0: Good paper with minor issues. Worth accepting.\n"
            f"- Rating 5.0-6.9: Fair but noticeable weaknesses. Marginal.\n"
            f"- Rating 3.0-4.9: Significant flaws. Below acceptance threshold.\n"
            f"- Rating 1.0-2.9: Major methodological errors or insufficient evidence.\n"
            f"- Soundness/Contribution/Presentation 3.5-4.0: Outstanding.\n"
            f"- 2.5-3.5: Good but flawed (most papers fall here).\n"
            f"- 1.5-2.5: Noticeable problems.\n"
            f"- 1.0-1.5: Serious deficiencies.\n\n"
            f"MOST papers are in the 5-7 (Rating) and 2.0-3.0 (sub-dims) range.\n"
            f"You MUST find at least 3 weaknesses. If you cannot, you are not being critical enough.\n\n"
>>>>>>> newb
            f"You MUST output your review in the following exact format:\n\n"
            f"\\boxed_review{{\n"
            f"## Summary:\n[Your summary]\n\n"
            f"## Soundness:\n[Score 1-4 with precision 0.05]\n\n"
            f"## Presentation:\n[Score 1-4 with precision 0.05]\n\n"
            f"## Contribution:\n[Score 1-4 with precision 0.05]\n\n"
            f"## Strengths:\n[Your strengths - be sparing with praise]\n\n"
<<<<<<< HEAD
            f"## Weaknesses:\n[Your weaknesses - be specific but score holistically]\n\n"
=======
            f"## Weaknesses:\n[Your weaknesses - point out at least 3 real flaws]\n\n"
>>>>>>> newb
            f"## Suggestions:\n[Your suggestions]\n\n"
            f"## Questions:\n[Your questions]\n\n"
            f"## Rating:\n[Overall score 1-10 with precision 0.05]\n\n"
            f"## Confidence:\n[Confidence 1-5]\n\n"
            f"## Decision:\n[Accept or Reject - Accept ONLY if Rating >= 6.5]\n"
            f"}}"
        )

    @staticmethod
    def _extract_number(text: str | None) -> float | None:
        if not text:
            return None
        match = re.search(r'(\d+(?:\.\d+)?)', str(text))
        return round_to_step(float(match.group(1))) if match else None


class StandardModeScorer(BaseScorer):
    """Baseline 2: Multi-reviewer simulation (Standard Mode).

    Aligns with DeepReviewer Standard Mode:
    - Simulate N different reviewers
    - Self-verification to double-check deficiencies
    - \boxed_review{} + \boxed_simreviewers{} format
    - Average scores across simulated reviewers
    """

    def __init__(self, llm: LLMWrapper | None = None, reviewer_num: int = 4):
        super().__init__(llm)
        self.reviewer_num = reviewer_num

    def score(self, paper_context: str, title: str = "") -> dict[str, Any]:
        system_prompt = self.llm.get_scoring_prompt(
            mode="standard", reviewer_num=self.reviewer_num
        )
        user_prompt = self._build_user_prompt(paper_context, title)

        raw_output = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=35000,
        )

        parsed = parse_deepreviewer_output(raw_output)
        sim_reviews = parsed.get("simulated_reviews", [])

        if sim_reviews:
            aggregated = aggregate_reviews(sim_reviews, method="mean")
        else:
            # Fallback to meta review
            meta = parsed.get("meta_review", {})
            aggregated = {
                "rating": meta.get("rating"),
                "soundness": self._extract_number(meta.get("soundness")),
                "presentation": self._extract_number(meta.get("presentation")),
                "contribution": self._extract_number(meta.get("contribution")),
                "decision": parsed.get("decision", "reject"),
            }

        # Enforce decision consistency with rating threshold
        if aggregated.get("rating") is not None and aggregated["rating"] < 6.5:
            aggregated["decision"] = "reject"

        return {
            "mode": "standard",
            "raw_output": raw_output,
            "parsed": parsed,
            "scores": aggregated,
            "num_simulated_reviewers": len(sim_reviews),
        }

    @staticmethod
    def _build_user_prompt(paper_context: str, title: str) -> str:
        header = f"Title: {title}\n\n" if title else ""
        return (
            f"{header}Please review the following research paper. Simulate multiple reviewers and verify your assessment:\n\n"
            f"{paper_context}\n\n"
<<<<<<< HEAD
            f"=== SCORING INSTRUCTIONS ===\n"
            f"You are reviewing for a NeurIPS/ICML-level conference. "
            f"Each reviewer should score by comparing against typical submissions.\n\n"
            f"DATASET CALIBRATION (human reviewer statistics):\n"
            f"- Rating (1-10): typical papers score 5.5 (range 4-7)\n"
            f"  9.0+: Exceptional  |  7.5: Strong  |  6.0: Above avg  |  5.0: Average  |  3.5: Below avg  |  2.0: Poor\n"
            f"- Soundness/Presentation/Contribution (1-4): typical papers score 2.5-3.0\n"
            f"  4.0: Outstanding  |  3.25: Good  |  2.75: Average  |  2.0: Weak  |  1.25: Poor\n\n"
            f"RULES:\n"
            f"1. Do NOT start from max and deduct. Judge absolute quality directly.\n"
            f"2. Most papers cluster near the typical range. Only exceptional work deserves top scores.\n"
            f"3. Be specific in weaknesses, but score based on overall quality, not flaw count.\n"
            f"4. Different reviewers may disagree - that is natural and expected.\n"
            f"5. Use 0.05 precision for all scores (e.g. 3.25, 5.50, 7.15).\n\n"
=======
            f"=== CRITICAL SCORING INSTRUCTIONS ===\n"
            f"Each reviewer MUST use DEDUCTION-BASED scoring to avoid inflation:\n"
            f"1. Start from MAXIMUM score for each dimension.\n"
            f"2. Identify specific flaws and deduct points:\n"
            f"   - Rating (1-10): minor -0.5, moderate -1.0, major -2.0\n"
            f"   - Sub-dims (1-4): minor -0.25, moderate -0.5, major -1.0\n"
            f"3. Final score = max_score - total_deductions.\n\n"
            f"SCORE ANCHORS (NeurIPS/ICML standards):\n"
            f"- Rating 8.5-10: Truly exceptional. Almost no flaws.\n"
            f"- Rating 7.0-8.0: Good with minor issues. Accept.\n"
            f"- Rating 5.0-6.9: Fair but flawed. Marginal.\n"
            f"- Rating 3.0-4.9: Significant flaws. Reject.\n"
            f"- Rating 1.0-2.9: Major errors or insufficient evidence.\n"
            f"- Sub-dims 3.5-4.0: Outstanding. 2.5-3.5: Good but flawed. 1.0-2.5: Problems.\n\n"
            f"MOST papers should score 5-7 (Rating) and 2.0-3.0 (sub-dims).\n"
            f"Each reviewer MUST find at least 3 weaknesses.\n\n"
>>>>>>> newb
            f"You MUST wrap all simulated reviewers in the following exact format:\n\n"
            f"\\boxed_simreviewers{{\n"
            f"## Reviewer 1\n"
            f"### Summary:\n...\n"
<<<<<<< HEAD
            f"### Soundness:\n[Score 1-4 with precision 0.05]\n"
            f"### Presentation:\n[Score 1-4 with precision 0.05]\n"
            f"### Contribution:\n[Score 1-4 with precision 0.05]\n"
            f"### Rating:\n[Score 1-10 with precision 0.05]\n"
=======
            f"### Soundness:\n[Score 1-4 with precision 0.01]\n"
            f"### Presentation:\n[Score 1-4 with precision 0.01]\n"
            f"### Contribution:\n[Score 1-4 with precision 0.01]\n"
            f"### Rating:\n[Score 1-10 with precision 0.01]\n"
>>>>>>> newb
            f"### Decision:\n[Accept or Reject - Accept ONLY if Rating >= 6.5]\n\n"
            f"## Reviewer 2\n...\n"
            f"}}"
        )

    @staticmethod
    def _extract_number(text: str | None) -> float | None:
        if not text:
            return None
        match = re.search(r'(\d+(?:\.\d+)?)', str(text))
        return round_to_step(float(match.group(1))) if match else None


class BestModeScorer(BaseScorer):
    """Baseline 3: Retrieval-enhanced scoring (Best Mode).

    Aligns with DeepReviewer Best Mode:
    1. LLM proposes 3 background knowledge questions
    2. Retrieve answers (OpenScholar API or local fallback)
    3. Feed retrieved info back to LLM
    4. Generate final review with simulated reviewers

    Note: If OpenScholar is unavailable, uses local fallback (LLM self-answer).
    """

    def __init__(
        self,
        llm: LLMWrapper | None = None,
        reviewer_num: int = 4,
        use_openscholar: bool = False,
        openscholar_url: str = "http://127.0.0.1:38015/batch_ask",
    ):
        super().__init__(llm)
        self.reviewer_num = reviewer_num
        self.use_openscholar = use_openscholar
        self.openscholar_url = openscholar_url

    def score(self, paper_context: str, title: str = "") -> dict[str, Any]:
        # Step 1: Generate initial review + questions
        step1_prompt = self._build_step1_prompt(paper_context, title)
        system_prompt = self.llm.get_scoring_prompt(
            mode="best", reviewer_num=self.reviewer_num
        )

        step1_output = self.llm.generate(
            prompt=step1_prompt,
            system_prompt=system_prompt,
            max_tokens=35000,
        )

        # Step 2: Extract questions
        questions = self._extract_questions(step1_output)
        logger.info(f"Extracted {len(questions)} questions for retrieval")

        # Step 3: Retrieve information
        if questions:
            retrieved = self._retrieve_information(questions)
        else:
            retrieved = []

        # Step 4: Generate final review with retrieved info
        step2_prompt = self._build_step2_prompt(
            paper_context, title, step1_output, questions, retrieved
        )

        final_output = self.llm.generate(
            prompt=step2_prompt,
            system_prompt=system_prompt,
            max_tokens=35000,
        )

        parsed = parse_deepreviewer_output(final_output)
        sim_reviews = parsed.get("simulated_reviews", [])

        if sim_reviews:
            aggregated = aggregate_reviews(sim_reviews, method="mean")
        else:
            meta = parsed.get("meta_review", {})
            aggregated = {
                "rating": meta.get("rating"),
                "soundness": self._extract_number(meta.get("soundness")),
                "presentation": self._extract_number(meta.get("presentation")),
                "contribution": self._extract_number(meta.get("contribution")),
                "decision": parsed.get("decision", "reject"),
            }

        # Enforce decision consistency with rating threshold
        if aggregated.get("rating") is not None and aggregated["rating"] < 6.5:
            aggregated["decision"] = "reject"

        return {
            "mode": "best",
            "raw_output": final_output,
            "step1_output": step1_output,
            "questions": questions,
            "retrieved": retrieved,
            "parsed": parsed,
            "scores": aggregated,
        }

    def _build_step1_prompt(self, paper_context: str, title: str) -> str:
        header = f"Title: {title}\n\n" if title else ""
        return (
            f"{header}Please review the following research paper. "
            f"First, provide three specific background knowledge questions that would help you evaluate this paper more accurately. "
            f"Then provide a preliminary review.\n\n"
            f"{paper_context}\n\n"
            f"Format your questions clearly, one per line."
        )

    def _build_step2_prompt(
        self,
        paper_context: str,
        title: str,
        step1_output: str,
        questions: list[str],
        retrieved: list[dict[str, Any]],
    ) -> str:
        header = f"Title: {title}\n\n" if title else ""
        qa_text = "\n\n".join(
            f"Question {i + 1}: {q}\nAnswer: {retrieved[i].get('answer', 'No answer available.') if i < len(retrieved) else 'No answer available.'}"
            for i, q in enumerate(questions)
        )

        return (
            f"{header}Here is a research paper and some background information:\n\n"
            f"## Paper:\n{paper_context}\n\n"
            f"## Your preliminary thoughts:\n{step1_output}\n\n"
            f"## Retrieved information:\n{qa_text}\n\n"
            f"Now, please provide the final comprehensive review by simulating {self.reviewer_num} different reviewers. "
            f"Use self-verification to double-check any paper deficiencies identified.\n\n"
<<<<<<< HEAD
            f"=== SCORING INSTRUCTIONS ===\n"
            f"You are reviewing for a NeurIPS/ICML-level conference. "
            f"Each reviewer should score by comparing against typical submissions.\n\n"
            f"DATASET CALIBRATION (human reviewer statistics):\n"
            f"- Rating (1-10): typical papers score 5.5 (range 4-7)\n"
            f"  9.0+: Exceptional  |  7.5: Strong  |  6.0: Above avg  |  5.0: Average  |  3.5: Below avg  |  2.0: Poor\n"
            f"- Soundness/Presentation/Contribution (1-4): typical papers score 2.5-3.0\n"
            f"  4.0: Outstanding  |  3.25: Good  |  2.75: Average  |  2.0: Weak  |  1.25: Poor\n\n"
            f"RULES:\n"
            f"1. Do NOT start from max and deduct. Judge absolute quality directly.\n"
            f"2. Most papers cluster near the typical range. Only exceptional work deserves top scores.\n"
            f"3. Be specific in weaknesses, but score based on overall quality, not flaw count.\n"
            f"4. Different reviewers may disagree - that is natural and expected.\n"
            f"5. Use 0.05 precision for all scores (e.g. 3.25, 5.50, 7.15).\n\n"
=======
            f"=== CRITICAL SCORING INSTRUCTIONS ===\n"
            f"Each reviewer MUST use DEDUCTION-BASED scoring:\n"
            f"1. Start from MAXIMUM score for each dimension.\n"
            f"2. Identify specific flaws and deduct points:\n"
            f"   - Rating (1-10): minor -0.5, moderate -1.0, major -2.0\n"
            f"   - Sub-dims (1-4): minor -0.25, moderate -0.5, major -1.0\n"
            f"3. Final score = max_score - total_deductions.\n\n"
            f"SCORE ANCHORS (NeurIPS/ICML standards):\n"
            f"- Rating 8.5-10: Truly exceptional. Almost no flaws.\n"
            f"- Rating 7.0-8.0: Good with minor issues. Accept.\n"
            f"- Rating 5.0-6.9: Fair but flawed. Marginal.\n"
            f"- Rating 3.0-4.9: Significant flaws. Reject.\n"
            f"- Rating 1.0-2.9: Major errors or insufficient evidence.\n"
            f"- Sub-dims 3.5-4.0: Outstanding. 2.5-3.5: Good but flawed. 1.0-2.5: Problems.\n\n"
            f"MOST papers should score 5-7 (Rating) and 2.0-3.0 (sub-dims).\n"
            f"Each reviewer MUST find at least 3 weaknesses.\n\n"
>>>>>>> newb
            f"You MUST wrap all simulated reviewers in the following exact format:\n\n"
            f"\\boxed_simreviewers{{\n"
            f"## Reviewer 1\n"
            f"### Summary:\n...\n"
<<<<<<< HEAD
            f"### Soundness:\n[Score 1-4 with precision 0.05]\n"
            f"### Presentation:\n[Score 1-4 with precision 0.05]\n"
            f"### Contribution:\n[Score 1-4 with precision 0.05]\n"
            f"### Rating:\n[Score 1-10 with precision 0.05]\n"
=======
            f"### Soundness:\n[Score 1-4 with precision 0.01]\n"
            f"### Presentation:\n[Score 1-4 with precision 0.01]\n"
            f"### Contribution:\n[Score 1-4 with precision 0.01]\n"
            f"### Rating:\n[Score 1-10 with precision 0.01]\n"
>>>>>>> newb
            f"### Decision:\n[Accept or Reject - Accept ONLY if Rating >= 6.5]\n\n"
            f"## Reviewer 2\n...\n"
            f"}}"
        )

    @staticmethod
    def _extract_questions(text: str) -> list[str]:
        """Extract questions from LLM output."""
        questions = []

        # Try boxed_questions format
        boxed_match = re.search(r'\\boxed_questions\{(.*?)\}', text, re.DOTALL)
        if boxed_match:
            lines = [l.strip() for l in boxed_match.group(1).split('\n') if l.strip()]
            for line in lines:
                cleaned = re.sub(r'^\d+\.\s*', '', line).strip()
                if cleaned and cleaned != '}':
                    questions.append(cleaned)
            return questions

        # Fallback: look for lines ending with ? or starting with numbers
        for line in text.split('\n'):
            line = line.strip()
            if line.endswith('?') or 'question' in line.lower():
                cleaned = re.sub(r'^\d+\.\s*[-\*]?\s*', '', line).strip()
                if cleaned and len(cleaned) > 10:
                    questions.append(cleaned)

        # Deduplicate
        return list(dict.fromkeys(questions))[:3]

    def _retrieve_information(self, questions: list[str]) -> list[dict[str, Any]]:
        """Retrieve information for questions."""
        if not questions:
            return []

        if self.use_openscholar:
            return self._retrieve_openscholar(questions)
        else:
            # Local fallback: use LLM to self-answer
            return self._retrieve_local(questions)

    def _retrieve_openscholar(self, questions: list[str]) -> list[dict[str, Any]]:
        """Call OpenScholar API."""
        import requests

        try:
            response = requests.post(
                self.openscholar_url,
                json={"questions": questions},
                timeout=600,
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                return [
                    {
                        "question": questions[i] if i < len(questions) else "",
                        "answer": r.get("output", ""),
                        "passages": r.get("final_passages", ""),
                    }
                    for i, r in enumerate(results)
                ]
        except Exception as e:
            logger.warning(f"OpenScholar retrieval failed: {e}. Falling back to local.")

        return self._retrieve_local(questions)

    def _retrieve_local(self, questions: list[str]) -> list[dict[str, Any]]:
        """Local fallback: ask LLM to answer based on its knowledge."""
        results = []
        for question in questions:
            prompt = f"Please answer this research question based on your knowledge:\n\n{question}\n\nProvide a concise, factual answer."
            try:
                answer = self.llm.generate(prompt, max_tokens=1000)
                results.append({
                    "question": question,
                    "answer": answer,
                    "passages": "",
                    "source": "local_llm_fallback",
                })
            except Exception as e:
                logger.warning(f"Local retrieval failed for question: {e}")
                results.append({
                    "question": question,
                    "answer": "Retrieval failed.",
                    "passages": "",
                    "source": "failed",
                })
        return results

    @staticmethod
    def _extract_number(text: str | None) -> float | None:
        if not text:
            return None
        match = re.search(r'(\d+(?:\.\d+)?)', str(text))
        return round_to_step(float(match.group(1))) if match else None


def run_baseline(
    samples: list[dict[str, Any]],
    mode: str = "fast",
    output_path: str = "experiments/deepreview_baseline/baseline_results.json",
    llm_config: str = "configs/llm.yaml",
) -> list[dict[str, Any]]:
    """Run a baseline scorer on samples.

    Args:
        samples: List of samples to score.
        mode: 'fast', 'standard', or 'best'.
        output_path: Path to save results.
        llm_config: Path to LLM config.

    Returns:
        List of scoring results.
    """
    llm = LLMWrapper(llm_config)
    llm.set_model("cloud_default") 
    
    if mode == "fast":
        scorer = FastModeScorer(llm)
    elif mode == "standard":
        scorer = StandardModeScorer(llm, reviewer_num=4)
    elif mode == "best":
        scorer = BestModeScorer(llm, reviewer_num=4, use_openscholar=False)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    logger.info(f"Running Baseline ({mode} mode) on {len(samples)} samples...")
    results = scorer.score_batch(samples, output_path=output_path)
    logger.info(f"Results saved to {output_path}")

    return results
