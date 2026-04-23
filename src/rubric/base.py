"""
Base rubric interface.

Provides abstract base for both fixed rubrics (DeepReview) and hierarchical rubrics (future datasets).
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseRubric(ABC):
    """Abstract base class for rubric definitions."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.dimensions = []
        self._load_dimensions()

    @abstractmethod
    def _load_dimensions(self) -> None:
        """Load dimension definitions from config."""
        pass

    @abstractmethod
    def get_dimension(self, name: str) -> dict[str, Any] | None:
        """Get a specific dimension by name."""
        pass

    @abstractmethod
    def get_all_dimensions(self) -> list[dict[str, Any]]:
        """Get all dimension definitions."""
        pass

    @abstractmethod
    def validate_score(self, dimension: str, score: float) -> bool:
        """Check if a score is valid for the given dimension."""
        pass

    def get_scale(self, dimension: str) -> tuple[int, int] | None:
        """Get the valid score range for a dimension."""
        dim = self.get_dimension(dimension)
        if dim and "scale" in dim:
            return tuple(dim["scale"])
        return None

    def get_description(self, dimension: str) -> str:
        """Get the description for a dimension."""
        dim = self.get_dimension(dimension)
        if dim:
            return dim.get("description", "")
        return ""
