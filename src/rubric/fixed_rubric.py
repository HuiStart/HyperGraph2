"""
Fixed rubric for DeepReview dataset.

DeepReview has exactly 4 dimensions with fixed scales:
- Rating: 1-10
- Soundness: 1-5
- Presentation: 1-5
- Contribution: 1-5

This aligns with official evalate.py dimensions.
"""

from typing import Any

import yaml

from src.rubric.base import BaseRubric


class FixedRubric(BaseRubric):
    """Fixed rubric with predefined dimensions (DeepReview style)."""

    def __init__(self, config_path: str = "configs/deepreview.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.deepreview_config = config
        super().__init__(config)

    def _load_dimensions(self) -> None:
        """Load dimensions from deepreview.yaml config."""
        dims = self.config.get("dimensions", [])
        self.dimensions = []
        for dim in dims:
            self.dimensions.append({
                "name": dim["name"],
                "key": dim.get("key", dim["name"].lower()),
                "description": dim.get("description", ""),
                "scale": dim.get("scale", [1, 5]),
            })

    def get_dimension(self, name: str) -> dict[str, Any] | None:
        """Get dimension by name or key."""
        name_lower = name.lower()
        for dim in self.dimensions:
            if dim["name"].lower() == name_lower or dim["key"] == name_lower:
                return dim
        return None

    def get_all_dimensions(self) -> list[dict[str, Any]]:
        """Return all dimension definitions."""
        return self.dimensions

    def validate_score(self, dimension: str, score: float) -> bool:
        """Check if score is within valid range."""
        dim = self.get_dimension(dimension)
        if not dim or "scale" not in dim:
            return False
        min_val, max_val = dim["scale"]
        return min_val <= score <= max_val

    def get_dimension_names(self) -> list[str]:
        """Get list of dimension keys."""
        return [dim["key"] for dim in self.dimensions]

    def get_scoring_dimensions(self) -> list[str]:
        """Get dimensions that are scored (excluding overall Rating if handled separately)."""
        # All 4 dimensions are scored independently in DeepReview
        return self.get_dimension_names()
