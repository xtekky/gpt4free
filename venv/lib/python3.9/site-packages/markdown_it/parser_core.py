"""
 * class Core
 *
 * Top-level rules executor. Glues block/inline parsers and does intermediate
 * transformations.
"""
from __future__ import annotations

from .ruler import RuleFunc, Ruler
from .rules_core import block, inline, linkify, normalize, replace, smartquotes
from .rules_core.state_core import StateCore

_rules: list[tuple[str, RuleFunc]] = [
    ("normalize", normalize),
    ("block", block),
    ("inline", inline),
    ("linkify", linkify),
    ("replacements", replace),
    ("smartquotes", smartquotes),
]


class ParserCore:
    def __init__(self):
        self.ruler = Ruler()
        for name, rule in _rules:
            self.ruler.push(name, rule)

    def process(self, state: StateCore) -> None:
        """Executes core chain rules."""
        for rule in self.ruler.getRules(""):
            rule(state)
