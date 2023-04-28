from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING

from ..ruler import StateBase
from ..token import Token

if TYPE_CHECKING:
    from markdown_it import MarkdownIt


class StateCore(StateBase):
    def __init__(
        self,
        src: str,
        md: MarkdownIt,
        env: MutableMapping,
        tokens: list[Token] | None = None,
    ):
        self.src = src
        self.md = md  # link to parser instance
        self.env = env
        self.tokens: list[Token] = tokens or []
        self.inlineMode = False
