"""Horizontal rule

At least 3 of these characters on a line * - _
"""
import logging

from ..common.utils import isSpace
from .state_block import StateBlock

LOGGER = logging.getLogger(__name__)


def hr(state: StateBlock, startLine: int, endLine: int, silent: bool):
    LOGGER.debug("entering hr: %s, %s, %s, %s", state, startLine, endLine, silent)

    pos = state.bMarks[startLine] + state.tShift[startLine]
    maximum = state.eMarks[startLine]

    # if it's indented more than 3 spaces, it should be a code block
    if state.sCount[startLine] - state.blkIndent >= 4:
        return False

    marker = state.srcCharCode[pos]
    pos += 1

    # Check hr marker: /* * */ /* - */ /* _ */
    if marker != 0x2A and marker != 0x2D and marker != 0x5F:
        return False

    # markers can be mixed with spaces, but there should be at least 3 of them

    cnt = 1
    while pos < maximum:
        ch = state.srcCharCode[pos]
        pos += 1
        if ch != marker and not isSpace(ch):
            return False
        if ch == marker:
            cnt += 1

    if cnt < 3:
        return False

    if silent:
        return True

    state.line = startLine + 1

    token = state.push("hr", "hr", 0)
    token.map = [startLine, state.line]
    token.markup = chr(marker) * (cnt + 1)

    return True
