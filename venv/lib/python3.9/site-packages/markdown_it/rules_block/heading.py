""" Atex heading (#, ##, ...) """
from __future__ import annotations

import logging

from ..common.utils import isSpace
from .state_block import StateBlock

LOGGER = logging.getLogger(__name__)


def heading(state: StateBlock, startLine: int, endLine: int, silent: bool):
    LOGGER.debug("entering heading: %s, %s, %s, %s", state, startLine, endLine, silent)

    pos = state.bMarks[startLine] + state.tShift[startLine]
    maximum = state.eMarks[startLine]

    # if it's indented more than 3 spaces, it should be a code block
    if state.sCount[startLine] - state.blkIndent >= 4:
        return False

    ch: int | None = state.srcCharCode[pos]

    # /* # */
    if ch != 0x23 or pos >= maximum:
        return False

    # count heading level
    level = 1
    pos += 1
    try:
        ch = state.srcCharCode[pos]
    except IndexError:
        ch = None
    # /* # */
    while ch == 0x23 and pos < maximum and level <= 6:
        level += 1
        pos += 1
        try:
            ch = state.srcCharCode[pos]
        except IndexError:
            ch = None

    if level > 6 or (pos < maximum and not isSpace(ch)):
        return False

    if silent:
        return True

    # Let's cut tails like '    ###  ' from the end of string

    maximum = state.skipSpacesBack(maximum, pos)
    tmp = state.skipCharsBack(maximum, 0x23, pos)  # #
    if tmp > pos and isSpace(state.srcCharCode[tmp - 1]):
        maximum = tmp

    state.line = startLine + 1

    token = state.push("heading_open", "h" + str(level), 1)
    token.markup = "########"[:level]
    token.map = [startLine, state.line]

    token = state.push("inline", "", 0)
    token.content = state.src[pos:maximum].strip()
    token.map = [startLine, state.line]
    token.children = []

    token = state.push("heading_close", "h" + str(level), -1)
    token.markup = "########"[:level]

    return True
