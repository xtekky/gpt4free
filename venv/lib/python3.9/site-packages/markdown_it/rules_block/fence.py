# fences (``` lang, ~~~ lang)
import logging

from .state_block import StateBlock

LOGGER = logging.getLogger(__name__)


def fence(state: StateBlock, startLine: int, endLine: int, silent: bool):
    LOGGER.debug("entering fence: %s, %s, %s, %s", state, startLine, endLine, silent)

    haveEndMarker = False
    pos = state.bMarks[startLine] + state.tShift[startLine]
    maximum = state.eMarks[startLine]

    # if it's indented more than 3 spaces, it should be a code block
    if state.sCount[startLine] - state.blkIndent >= 4:
        return False

    if pos + 3 > maximum:
        return False

    marker = state.srcCharCode[pos]

    # /* ~ */  /* ` */
    if marker != 0x7E and marker != 0x60:
        return False

    # scan marker length
    mem = pos
    pos = state.skipChars(pos, marker)

    length = pos - mem

    if length < 3:
        return False

    markup = state.src[mem:pos]
    params = state.src[pos:maximum]

    # /* ` */
    if marker == 0x60:
        if chr(marker) in params:
            return False

    # Since start is found, we can report success here in validation mode
    if silent:
        return True

    # search end of block
    nextLine = startLine

    while True:
        nextLine += 1
        if nextLine >= endLine:
            # unclosed block should be autoclosed by end of document.
            # also block seems to be autoclosed by end of parent
            break

        pos = mem = state.bMarks[nextLine] + state.tShift[nextLine]
        maximum = state.eMarks[nextLine]

        if pos < maximum and state.sCount[nextLine] < state.blkIndent:
            # non-empty line with negative indent should stop the list:
            # - ```
            #  test
            break

        if state.srcCharCode[pos] != marker:
            continue

        if state.sCount[nextLine] - state.blkIndent >= 4:
            # closing fence should be indented less than 4 spaces
            continue

        pos = state.skipChars(pos, marker)

        # closing code fence must be at least as long as the opening one
        if pos - mem < length:
            continue

        # make sure tail has spaces only
        pos = state.skipSpaces(pos)

        if pos < maximum:
            continue

        haveEndMarker = True
        # found!
        break

    # If a fence has heading spaces, they should be removed from its inner block
    length = state.sCount[startLine]

    state.line = nextLine + (1 if haveEndMarker else 0)

    token = state.push("fence", "code", 0)
    token.info = params
    token.content = state.getLines(startLine + 1, nextLine, length, True)
    token.markup = markup
    token.map = [startLine, state.line]

    return True
