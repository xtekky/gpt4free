# lheading (---, ==)
import logging

from ..ruler import Ruler
from .state_block import StateBlock

LOGGER = logging.getLogger(__name__)


def lheading(state: StateBlock, startLine: int, endLine: int, silent: bool):
    LOGGER.debug("entering lheading: %s, %s, %s, %s", state, startLine, endLine, silent)

    level = None
    nextLine = startLine + 1
    ruler: Ruler = state.md.block.ruler
    terminatorRules = ruler.getRules("paragraph")

    # if it's indented more than 3 spaces, it should be a code block
    if state.sCount[startLine] - state.blkIndent >= 4:
        return False

    oldParentType = state.parentType
    state.parentType = "paragraph"  # use paragraph to match terminatorRules

    # jump line-by-line until empty one or EOF
    while nextLine < endLine and not state.isEmpty(nextLine):
        # this would be a code block normally, but after paragraph
        # it's considered a lazy continuation regardless of what's there
        if state.sCount[nextLine] - state.blkIndent > 3:
            nextLine += 1
            continue

        # Check for underline in setext header
        if state.sCount[nextLine] >= state.blkIndent:
            pos = state.bMarks[nextLine] + state.tShift[nextLine]
            maximum = state.eMarks[nextLine]

            if pos < maximum:
                marker = state.srcCharCode[pos]

                # /* - */  /* = */
                if marker == 0x2D or marker == 0x3D:
                    pos = state.skipChars(pos, marker)
                    pos = state.skipSpaces(pos)

                    # /* = */
                    if pos >= maximum:
                        level = 1 if marker == 0x3D else 2
                        break

        # quirk for blockquotes, this line should already be checked by that rule
        if state.sCount[nextLine] < 0:
            nextLine += 1
            continue

        # Some tags can terminate paragraph without empty line.
        terminate = False
        for terminatorRule in terminatorRules:
            if terminatorRule(state, nextLine, endLine, True):
                terminate = True
                break
        if terminate:
            break

        nextLine += 1

    if not level:
        # Didn't find valid underline
        return False

    content = state.getLines(startLine, nextLine, state.blkIndent, False).strip()

    state.line = nextLine + 1

    token = state.push("heading_open", "h" + str(level), 1)
    token.markup = chr(marker)
    token.map = [startLine, state.line]

    token = state.push("inline", "", 0)
    token.content = content
    token.map = [startLine, state.line - 1]
    token.children = []

    token = state.push("heading_close", "h" + str(level), -1)
    token.markup = chr(marker)

    state.parentType = oldParentType

    return True
