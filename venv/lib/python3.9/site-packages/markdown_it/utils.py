from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


class OptionsDict(dict):
    """A dictionary, with attribute access to core markdownit configuration options."""

    @property
    def maxNesting(self) -> int:
        """Internal protection, recursion limit."""
        return self["maxNesting"]

    @maxNesting.setter
    def maxNesting(self, value: int):
        self["maxNesting"] = value

    @property
    def html(self) -> bool:
        """Enable HTML tags in source."""
        return self["html"]

    @html.setter
    def html(self, value: bool):
        self["html"] = value

    @property
    def linkify(self) -> bool:
        """Enable autoconversion of URL-like texts to links."""
        return self["linkify"]

    @linkify.setter
    def linkify(self, value: bool):
        self["linkify"] = value

    @property
    def typographer(self) -> bool:
        """Enable smartquotes and replacements."""
        return self["typographer"]

    @typographer.setter
    def typographer(self, value: bool):
        self["typographer"] = value

    @property
    def quotes(self) -> str:
        """Quote characters."""
        return self["quotes"]

    @quotes.setter
    def quotes(self, value: str):
        self["quotes"] = value

    @property
    def xhtmlOut(self) -> bool:
        """Use '/' to close single tags (<br />)."""
        return self["xhtmlOut"]

    @xhtmlOut.setter
    def xhtmlOut(self, value: bool):
        self["xhtmlOut"] = value

    @property
    def breaks(self) -> bool:
        """Convert newlines in paragraphs into <br>."""
        return self["breaks"]

    @breaks.setter
    def breaks(self, value: bool):
        self["breaks"] = value

    @property
    def langPrefix(self) -> str:
        """CSS language prefix for fenced blocks."""
        return self["langPrefix"]

    @langPrefix.setter
    def langPrefix(self, value: str):
        self["langPrefix"] = value

    @property
    def highlight(self) -> Callable[[str, str, str], str] | None:
        """Highlighter function: (content, langName, langAttrs) -> escaped HTML."""
        return self["highlight"]

    @highlight.setter
    def highlight(self, value: Callable[[str, str, str], str] | None):
        self["highlight"] = value


def read_fixture_file(path: str | Path) -> list[list]:
    text = Path(path).read_text(encoding="utf-8")
    tests = []
    section = 0
    last_pos = 0
    lines = text.splitlines(keepends=True)
    for i in range(len(lines)):
        if lines[i].rstrip() == ".":
            if section == 0:
                tests.append([i, lines[i - 1].strip()])
                section = 1
            elif section == 1:
                tests[-1].append("".join(lines[last_pos + 1 : i]))
                section = 2
            elif section == 2:
                tests[-1].append("".join(lines[last_pos + 1 : i]))
                section = 0

            last_pos = i
    return tests


def _removesuffix(string: str, suffix: str) -> str:
    """Remove a suffix from a string.

    Replace this with str.removesuffix() from stdlib when minimum Python
    version is 3.9.
    """
    if suffix and string.endswith(suffix):
        return string[: -len(suffix)]
    return string
