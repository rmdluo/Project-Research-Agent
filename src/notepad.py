"""Shared notepad: a single markdown file used by all agents."""

from pathlib import Path

DEFAULT_NOTEPAD = "notepad.md"

# Ordered section headers
SECTIONS = (
    "Project Spec",
    "Research Findings",
    "Open Questions",
    "Decisions",
    "Progress",
    "Final Report",
)


class Notepad:
    """Manage a single shared markdown file with named sections."""

    def __init__(self, path: str = DEFAULT_NOTEPAD) -> None:
        self.path = Path(path)
        if not self.path.exists():
            self._initialize()

    def _initialize(self) -> None:
        """Create the notepad with all section headers."""
        parts = [f"# Project Agent Notepad\n\n"]
        for name in SECTIONS:
            parts.append(f"## {name}\n\n")
        self.path.write_text("".join(parts))

    def _header_for(self, section: str) -> str:
        return f"## {section}"

    def _find_section_range(self, content: str, section: str) -> tuple[int, int]:
        """Return (start_idx, end_idx) for a section's text content."""
        header = self._header_for(section)
        header_idx = content.find(header)
        if header_idx == -1:
            return -1, -1

        header_end = header_idx + len(header)

        # Find next section header
        next_idx = len(content)
        for other in SECTIONS:
            if other != section:
                idx = content.find(self._header_for(other), header_end)
                if 0 < idx < next_idx:
                    next_idx = idx

        return header_idx, next_idx

    def read_section(self, section: str) -> str:
        """Return the text content of a section."""
        content = self.path.read_text()
        start, end = self._find_section_range(content, section)
        if start == -1:
            return ""
        text = content[start + len(self._header_for(section)):end].strip()
        return text if text else ""

    def set_section(self, section: str, content: str) -> None:
        """Replace the entire content of a section."""
        text = self.path.read_text()
        start, end = self._find_section_range(text, section)

        if start == -1:
            # Section doesn't exist; append before Final Report or at end
            final_idx = text.find(self._header_for("Final Report"))
            insert_point = final_idx if final_idx != -1 else len(text)
            insert_text = f"\n\n## {section}\n\n{content}\n\n"
            if final_idx != -1:
                text = text[:insert_point] + insert_text + text[insert_point:]
            else:
                text = text.rstrip() + "\n\n" + insert_text
            self.path.write_text(text)
        else:
            text = text[:start + len(self._header_for(section))] + f"\n\n{content}\n\n" + text[end:]
            self.path.write_text(text)

    def append_section(self, section: str, text: str) -> None:
        """Append text to a section."""
        existing = self.read_section(section)
        new_content = f"{existing}\n\n{text}".strip() if existing else text
        self.set_section(section, new_content)

    def get_all_content(self) -> str:
        """Return the full notepad content."""
        return self.path.read_text()

    def update_progress(self, message: str) -> None:
        """Append a progress entry."""
        self.append_section("Progress", f"- {message}")

    def save_report(self, report: str) -> None:
        """Save the final report."""
        self.set_section("Final Report", report)
