"""
LaTeX paper parser for extracting structured sections from paper_context.

DeepReview paper_context is full LaTeX text. We need to extract:
- abstract
- introduction
- methods / methodology / approach
- experiments / results
- related work
- conclusion

This supports both rule-based extraction (regex) and LLM-based extraction.
"""

import re
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LaTeXParser:
    """Parse LaTeX paper text into structured sections."""

    # Common section name variants in academic papers
    SECTION_PATTERNS = {
        "abstract": [r"\\begin\{abstract\}", r"\\section\{Abstract\}"],
        "introduction": [r"\\section\{Introduction\}", r"\\section\{.*[Ii]ntroduction.*\}"],
        "related_work": [
            r"\\section\{Related [Ww]ork\}",
            r"\\section\{[Bb]ackground\}",
            r"\\section\{[Ll]iterature [Rr]eview\}",
        ],
        "methods": [
            r"\\section\{[Mm]ethods?\}",
            r"\\section\{[Mm]ethodology\}",
            r"\\section\{[Aa]pproach\}",
            r"\\section\{[Mm]odel\}",
            r"\\section\{[Aa]lgorithm\}",
        ],
        "experiments": [
            r"\\section\{[Ee]xperiments?\}",
            r"\\section\{[Ee]xperimental [Ss]etup\}",
            r"\\section\{[Ee]valuation\}",
            r"\\section\{[Rr]esults\}",
        ],
        "discussion": [r"\\section\{[Dd]iscussion\}"],
        "conclusion": [
            r"\\section\{[Cc]onclusion\}",
            r"\\section\{[Cc]onclusions\}",
            r"\\section\{[Ff]uture [Ww]ork\}",
        ],
    }

    def __init__(self):
        self.sections = {}

    def parse(self, latex_text: str) -> dict[str, str]:
        """Parse LaTeX text into sections.

        Args:
            latex_text: Full LaTeX paper content.

        Returns:
            Dictionary mapping section names to their text content.
        """
        self.sections = {}
        if not latex_text:
            return self.sections

        # Extract abstract first (special handling)
        self.sections["abstract"] = self._extract_abstract(latex_text)

        # Extract title
        self.sections["title"] = self._extract_title(latex_text)

        # Extract all \section{...} blocks
        section_blocks = self._extract_sections(latex_text)
        self.sections.update(section_blocks)

        # Extract figures/tables info
        self.sections["figures"] = self._extract_figures(latex_text)
        self.sections["tables"] = self._extract_tables(latex_text)

        # Extract references
        self.sections["references"] = self._extract_references(latex_text)

        return self.sections

    def _extract_abstract(self, text: str) -> str:
        """Extract abstract content."""
        match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', text, re.DOTALL)
        if match:
            return self._clean_latex(match.group(1))
        return ""

    def _extract_title(self, text: str) -> str:
        """Extract paper title."""
        match = re.search(r'\\title\{(.*?)\}', text, re.DOTALL)
        if match:
            return self._clean_latex(match.group(1))
        return ""

    def _extract_sections(self, text: str) -> dict[str, str]:
        """Extract all section contents."""
        sections = {}

        # Find all section boundaries
        section_headers = list(re.finditer(r'\\section\{([^}]+)\}', text))
        section_headers += list(re.finditer(r'\\section\*\{([^}]+)\}', text))

        # Sort by position
        section_headers.sort(key=lambda m: m.start())

        for i, match in enumerate(section_headers):
            section_name = match.group(1).strip().lower()
            start = match.end()
            end = section_headers[i + 1].start() if i + 1 < len(section_headers) else len(text)
            content = text[start:end]

            # Clean up
            content = self._clean_latex(content)
            sections[section_name] = content

            # Map to standard names
            for std_name, patterns in self.SECTION_PATTERNS.items():
                if std_name == "abstract":
                    continue
                for pattern in patterns:
                    # Remove regex escapes for comparison
                    plain_pattern = pattern.replace("\\", "").replace("{", "").replace("}", "")
                    if re.search(pattern, match.group(0)) or plain_pattern.lower() in section_name:
                        sections[std_name] = content
                        break

        return sections

    def _extract_figures(self, text: str) -> list[dict[str, Any]]:
        """Extract figure environments."""
        figures = []
        for match in re.finditer(r'\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}', text, re.DOTALL):
            figures.append({
                "raw": match.group(1),
                "caption": self._extract_caption(match.group(1)),
            })
        return figures

    def _extract_tables(self, text: str) -> list[dict[str, Any]]:
        """Extract table environments."""
        tables = []
        for match in re.finditer(r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}', text, re.DOTALL):
            tables.append({
                "raw": match.group(1),
                "caption": self._extract_caption(match.group(1)),
            })
        return tables

    def _extract_caption(self, text: str) -> str:
        """Extract caption from figure/table environment."""
        match = re.search(r'\\caption\{(.*?)\}', text, re.DOTALL)
        if match:
            return self._clean_latex(match.group(1))
        return ""

    def _extract_references(self, text: str) -> list[str]:
        """Extract citation keys."""
        cites = re.findall(r'\\cite\{([^}]+)\}', text)
        refs = []
        for cite in cites:
            refs.extend([c.strip() for c in cite.split(",")])
        return list(dict.fromkeys(refs))  # Deduplicate preserve order

    @staticmethod
    def _clean_latex(text: str) -> str:
        """Remove common LaTeX commands for plain text extraction."""
        if not text:
            return ""

        # Remove comments
        text = re.sub(r'(?<!\\)%.*?\n', '\n', text)

        # Remove common commands
        text = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?', '', text)

        # Clean up braces
        text = text.replace("{", "").replace("}", "")

        # Clean up math
        text = re.sub(r'\$+.*?\$+', ' ', text, flags=re.DOTALL)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def get_section(self, name: str) -> str:
        """Get a specific section by name."""
        return self.sections.get(name, "")

    def get_full_text(self) -> str:
        """Get cleaned full paper text."""
        parts = []
        for key in ["abstract", "introduction", "related_work", "methods",
                    "experiments", "discussion", "conclusion"]:
            if key in self.sections and self.sections[key]:
                parts.append(self.sections[key])
        return "\n\n".join(parts)


def parse_paper(latex_text: str) -> dict[str, Any]:
    """Convenience function to parse LaTeX paper text."""
    parser = LaTeXParser()
    return parser.parse(latex_text)


if __name__ == "__main__":
    import json
    import sys

    # Quick test with sample data
    sample_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.json"
    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data:
        paper = data[0]
        parser = LaTeXParser()
        sections = parser.parse(paper["paper_context"])
        print(f"Title: {sections.get('title', 'N/A')}")
        print(f"Sections found: {list(sections.keys())}")
        print(f"Abstract length: {len(sections.get('abstract', ''))}")
        print(f"Num figures: {len(sections.get('figures', []))}")
        print(f"Num tables: {len(sections.get('tables', []))}")
        print(f"Num references: {len(sections.get('references', []))}")
