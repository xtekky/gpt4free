# Process html tags
from ..common.html_re import HTML_TAG_RE
from .state_inline import StateInline


def isLetter(ch: int):
    lc = ch | 0x20  # to lower case
    # /* a */ and /* z */
    return (lc >= 0x61) and (lc <= 0x7A)


def html_inline(state: StateInline, silent: bool):
    pos = state.pos

    if not state.md.options.get("html", None):
        return False

    # Check start
    maximum = state.posMax
    if state.srcCharCode[pos] != 0x3C or pos + 2 >= maximum:  # /* < */
        return False

    # Quick fail on second char
    ch = state.srcCharCode[pos + 1]
    if (
        ch != 0x21
        and ch != 0x3F  # /* ! */
        and ch != 0x2F  # /* ? */
        and not isLetter(ch)  # /* / */
    ):
        return False

    match = HTML_TAG_RE.search(state.src[pos:])
    if not match:
        return False

    if not silent:
        token = state.push("html_inline", "", 0)
        token.content = state.src[pos : pos + len(match.group(0))]

    state.pos += len(match.group(0))
    return True
