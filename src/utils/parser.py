"""
Parser for DeepReviewer output formats.

Aligns with official implementation:
- Research/ai_researcher/utils.py:get_reviewer_score()
- Research/ai_researcher/deep_reviewer.py:_parse_review()

Supports parsing of:
- \boxed_review{...} (meta review)
- \boxed_simreviewers{...} (simulated multiple reviewers)
"""

import re
from typing import Any


def extract_number_from_text(text: str) -> float | None:
    """Extract the first number from text (e.g. '3 good' -> 3.0)."""
    if not text:
        return None
    match = re.search(r'(\d+(?:\.\d+)?)', str(text))
    if match:
        return float(match.group(1))
    return None


def parse_review_sections(text: str, level: str = "##") -> dict[str, str]:
    """Parse review text into sections.

    Args:
        text: Raw review text containing sections like '## Summary: ...'
        level: Header level, '##' for single review, '###' for simulated reviewers.

    Returns:
        Dictionary mapping section names to their content.
    """
    sections = {}
    if not text:
        return sections

    # Split by section headers
    # Support both "## Summary:" and "### Summary" formats
    pattern = rf'{level}\s+([A-Za-z\s]+):?\s*\n'
    parts = re.split(pattern, text)

    # parts[0] is text before first header, then alternating name, content
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            section_name = parts[i].strip()
            section_content = parts[i + 1].strip()
            # Stop at next header level or separator
            section_content = re.split(rf'\n{level}\s+', section_content)[0].strip()
            sections[section_name.lower()] = section_content

    return sections


def parse_boxed_review(text: str) -> dict[str, Any]:
    """Parse \boxed_review{...} block from generated text.

    Aligns with deep_reviewer.py:_parse_review() meta_review extraction.
    """
    result = {
        "raw_text": text,
        "summary": "",
        "soundness": "",
        "presentation": "",
        "contribution": "",
        "strengths": "",
        "weaknesses": "",
        "suggestions": "",
        "questions": "",
        "rating": None,
        "confidence": "",
        "decision": "",
    }

    if not text:
        return result

    # Extract boxed_review content
    match = re.search(r'\\boxed_review\{(.*?)\n\}', text, re.DOTALL)
    if not match:
        # Fallback: try without closing brace on new line
        match = re.search(r'\\boxed_review\{(.*?)\}', text, re.DOTALL)

    if match:
        content = match.group(1).strip()
        sections = parse_review_sections(content, level="##")

        for key in ["summary", "soundness", "presentation", "contribution",
                    "strengths", "weaknesses", "suggestions", "questions",
                    "confidence", "decision"]:
            if key.capitalize() in sections:
                result[key] = sections[key.capitalize()]
            elif key in sections:
                result[key] = sections[key]

        # Extract rating number
        rating_text = sections.get("rating", "")
        if rating_text:
            result["rating"] = extract_number_from_text(rating_text)

    return result


def parse_boxed_simreviewers(text: str) -> list[dict[str, Any]]:
    """Parse \boxed_simreviewers{...} block containing multiple reviewers.

    Aligns with deep_reviewer.py:_parse_review() simulated reviewers extraction.
    """
    reviewers = []
    if not text:
        return reviewers

    match = re.search(r'\\boxed_simreviewers\{(.*?)\n\}', text, re.DOTALL)
    if not match:
        match = re.search(r'\\boxed_simreviewers\{(.*?)\}', text, re.DOTALL)

    if not match:
        return reviewers

    sim_text = match.group(1).strip()

    # Split by reviewer headers
    # Pattern: ## Reviewer 1, ## Reviewer 2, etc.
    reviewer_sections = re.split(r'##\s+Reviewer\s+\d+', sim_text)

    # Skip first empty section
    if reviewer_sections and not reviewer_sections[0].strip():
        reviewer_sections = reviewer_sections[1:]

    for i, section in enumerate(reviewer_sections):
        review = {
            "reviewer_id": i + 1,
            "text": section.strip(),
            "summary": "",
            "soundness": "",
            "presentation": "",
            "contribution": "",
            "strengths": "",
            "weaknesses": "",
            "suggestions": "",
            "questions": "",
            "rating": None,
            "confidence": "",
        }

        sections = parse_review_sections(section, level="###")

        for key in ["summary", "soundness", "presentation", "contribution",
                    "strengths", "weaknesses", "suggestions", "questions",
                    "confidence"]:
            cap_key = key.capitalize()
            if cap_key in sections:
                review[key] = sections[cap_key]
            elif key in sections:
                review[key] = sections[key]

        rating_text = sections.get("rating", "")
        if rating_text:
            review["rating"] = extract_number_from_text(rating_text)

        reviewers.append(review)

    return reviewers


def parse_deepreviewer_output(text: str) -> dict[str, Any]:
    """Parse complete DeepReviewer output (both boxed_review and boxed_simreviewers).

    This is the main entry point for parsing any DeepReviewer mode output.
    """
    result = {
        "raw_text": text,
        "meta_review": parse_boxed_review(text),
        "simulated_reviews": parse_boxed_simreviewers(text),
        "decision": "",
    }

    # Extract decision from raw text if present
    decision_match = re.search(r'##\s*Decision:\s*\n?\s*(\w+)', text, re.IGNORECASE)
    if decision_match:
        decision = decision_match.group(1).strip().lower()
        if "accept" in decision:
            result["decision"] = "accept"
        else:
            result["decision"] = "reject"

    # If meta_review has decision, prefer it
    if result["meta_review"].get("decision"):
        meta_decision = result["meta_review"]["decision"].lower()
        if "accept" in meta_decision:
            result["decision"] = "accept"
        elif "reject" in meta_decision:
            result["decision"] = "reject"

    return result


def get_average_scores(reviews: list[dict[str, Any]]) -> dict[str, float | None]:
    """Compute average scores across multiple reviews.

    Aligns with evalate.py logic: rates.mean(), soundness.mean(), etc.
    """
    if not reviews:
        return {"rating": None, "soundness": None, "presentation": None, "contribution": None}

    scores = {"rating": [], "soundness": [], "presentation": [], "contribution": []}

    for review in reviews:
        for dim in scores:
            val = review.get(dim)
            if val is not None:
                try:
                    scores[dim].append(float(val))
                except (ValueError, TypeError):
                    continue

    return {
        dim: sum(vals) / len(vals) if vals else None
        for dim, vals in scores.items()
    }


def normalize_decision(decision_text: str) -> str:
    """Normalize decision text to 'accept' or 'reject'.

    Aligns with evalate.py logic:
    - If contains 'accept', return 'accept'
    - Otherwise, return 'reject'
    """
    if not decision_text:
        return "reject"
    if "accept" in decision_text.lower():
        return "accept"
    return "reject"
