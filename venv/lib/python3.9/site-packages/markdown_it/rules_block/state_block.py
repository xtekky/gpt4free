from __future__ import annotations

from typing import TYPE_CHECKING

from ..common.utils import isSpace
from ..ruler import StateBase
from ..token import Token

if TYPE_CHECKING:
    from markdown_it.main import MarkdownIt


class StateBlock(StateBase):
    def __init__(
        self,
        src: str,
        md: MarkdownIt,
        env,
        tokens: list[Token],
        srcCharCode: tuple[int, ...] | None = None,
    ):
        if srcCharCode is not None:
            self._src = src
            self.srcCharCode = srcCharCode
        else:
            self.src = src

        # link to parser instance
        self.md = md

        self.env = env

        #
        # Internal state variables
        #

        self.tokens = tokens

        self.bMarks = []  # line begin offsets for fast jumps
        self.eMarks = []  # line end offsets for fast jumps
        # offsets of the first non-space characters (tabs not expanded)
        self.tShift = []
        self.sCount = []  # indents for each line (tabs expanded)

        # An amount of virtual spaces (tabs expanded) between beginning
        # of each line (bMarks) and real beginning of that line.
        #
        # It exists only as a hack because blockquotes override bMarks
        # losing information in the process.
        #
        # It's used only when expanding tabs, you can think about it as
        # an initial tab length, e.g. bsCount=21 applied to string `\t123`
        # means first tab should be expanded to 4-21%4 === 3 spaces.
        #
        self.bsCount = []

        # block parser variables
        self.blkIndent = 0  # required block content indent (for example, if we are
        # inside a list, it would be positioned after list marker)
        self.line = 0  # line index in src
        self.lineMax = 0  # lines count
        self.tight = False  # loose/tight mode for lists
        self.ddIndent = -1  # indent of the current dd block (-1 if there isn't any)
        self.listIndent = -1  # indent of the current list block (-1 if there isn't any)

        # can be 'blockquote', 'list', 'root', 'paragraph' or 'reference'
        # used in lists to determine if they interrupt a paragraph
        self.parentType = "root"

        self.level = 0

        # renderer
        self.result = ""

        # Create caches
        # Generate markers.
        indent_found = False

        start = pos = indent = offset = 0
        length = len(self.src)

        for pos, character in enumerate(self.srcCharCode):
            if not indent_found:
                if isSpace(character):
                    indent += 1

                    if character == 0x09:
                        offset += 4 - offset % 4
                    else:
                        offset += 1
                    continue
                else:
                    indent_found = True

            if character == 0x0A or pos == length - 1:
                if character != 0x0A:
                    pos += 1
                self.bMarks.append(start)
                self.eMarks.append(pos)
                self.tShift.append(indent)
                self.sCount.append(offset)
                self.bsCount.append(0)

                indent_found = False
                indent = 0
                offset = 0
                start = pos + 1

        # Push fake entry to simplify cache bounds checks
        self.bMarks.append(length)
        self.eMarks.append(length)
        self.tShift.append(0)
        self.sCount.append(0)
        self.bsCount.append(0)

        self.lineMax = len(self.bMarks) - 1  # don't count last fake line

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(line={self.line},level={self.level},tokens={len(self.tokens)})"
        )

    def push(self, ttype: str, tag: str, nesting: int) -> Token:
        """Push new token to "stream"."""
        token = Token(ttype, tag, nesting)
        token.block = True
        if nesting < 0:
            self.level -= 1  # closing tag
        token.level = self.level
        if nesting > 0:
            self.level += 1  # opening tag
        self.tokens.append(token)
        return token

    def isEmpty(self, line: int) -> bool:
        """."""
        return (self.bMarks[line] + self.tShift[line]) >= self.eMarks[line]

    def skipEmptyLines(self, from_pos: int) -> int:
        """."""
        while from_pos < self.lineMax:
            try:
                if (self.bMarks[from_pos] + self.tShift[from_pos]) < self.eMarks[
                    from_pos
                ]:
                    break
            except IndexError:
                pass
            from_pos += 1
        return from_pos

    def skipSpaces(self, pos: int) -> int:
        """Skip spaces from given position."""
        while pos < len(self.src):
            if not isSpace(self.srcCharCode[pos]):
                break
            pos += 1
        return pos

    def skipSpacesBack(self, pos: int, minimum: int) -> int:
        """Skip spaces from given position in reverse."""
        if pos <= minimum:
            return pos
        while pos > minimum:
            pos -= 1
            if not isSpace(self.srcCharCode[pos]):
                return pos + 1
        return pos

    def skipChars(self, pos: int, code: int) -> int:
        """Skip char codes from given position."""
        while pos < len(self.src):
            if self.srcCharCode[pos] != code:
                break
            pos += 1
        return pos

    def skipCharsBack(self, pos: int, code: int, minimum: int) -> int:
        """Skip char codes reverse from given position - 1."""
        if pos <= minimum:
            return pos
        while pos > minimum:
            pos -= 1
            if code != self.srcCharCode[pos]:
                return pos + 1
        return pos

    def getLines(self, begin: int, end: int, indent: int, keepLastLF: bool) -> str:
        """Cut lines range from source."""
        line = begin
        if begin >= end:
            return ""

        queue = [""] * (end - begin)

        i = 1
        while line < end:
            lineIndent = 0
            lineStart = first = self.bMarks[line]
            if line + 1 < end or keepLastLF:
                last = self.eMarks[line] + 1
            else:
                last = self.eMarks[line]

            while (first < last) and (lineIndent < indent):
                ch = self.srcCharCode[first]
                if isSpace(ch):
                    if ch == 0x09:
                        lineIndent += 4 - (lineIndent + self.bsCount[line]) % 4
                    else:
                        lineIndent += 1
                elif first - lineStart < self.tShift[line]:
                    lineIndent += 1
                else:
                    break
                first += 1

            if lineIndent > indent:
                # partially expanding tabs in code blocks, e.g '\t\tfoobar'
                # with indent=2 becomes '  \tfoobar'
                queue[i - 1] = (" " * (lineIndent - indent)) + self.src[first:last]
            else:
                queue[i - 1] = self.src[first:last]

            line += 1
            i += 1

        return "".join(queue)
