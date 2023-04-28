# Proceess '\n'
import re

from ..common.utils import charCodeAt, isSpace
from .state_inline import StateInline

endSpace = re.compile(r" +$")


def newline(state: StateInline, silent: bool):
    pos = state.pos

    # /* \n */
    if state.srcCharCode[pos] != 0x0A:
        return False

    pmax = len(state.pending) - 1
    maximum = state.posMax

    # '  \n' -> hardbreak
    # Lookup in pending chars is bad practice! Don't copy to other rules!
    # Pending string is stored in concat mode, indexed lookups will cause
    # conversion to flat mode.
    if not silent:
        if pmax >= 0 and charCodeAt(state.pending, pmax) == 0x20:
            if pmax >= 1 and charCodeAt(state.pending, pmax - 1) == 0x20:
                state.pending = endSpace.sub("", state.pending)
                state.push("hardbreak", "br", 0)
            else:
                state.pending = state.pending[:-1]
                state.push("softbreak", "br", 0)

        else:
            state.push("softbreak", "br", 0)

    pos += 1

    # skip heading spaces for next line
    while pos < maximum and isSpace(state.srcCharCode[pos]):
        pos += 1

    state.pos = pos
    return True
