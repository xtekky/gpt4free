"""
Process escaped chars and hardbreaks
"""
from ..common.utils import isSpace
from .state_inline import StateInline

ESCAPED = [0 for _ in range(256)]
for ch in "\\!\"#$%&'()*+,./:;<=>?@[]^_`{|}~-":
    ESCAPED[ord(ch)] = 1


def escape(state: StateInline, silent: bool):
    pos = state.pos
    maximum = state.posMax

    # /* \ */
    if state.srcCharCode[pos] != 0x5C:
        return False

    pos += 1

    if pos < maximum:
        ch = state.srcCharCode[pos]

        if ch < 256 and ESCAPED[ch] != 0:
            if not silent:
                state.pending += state.src[pos]
            state.pos += 2
            return True

        if ch == 0x0A:
            if not silent:
                state.push("hardbreak", "br", 0)

            pos += 1
            # skip leading whitespaces from next line
            while pos < maximum:
                ch = state.srcCharCode[pos]
                if not isSpace(ch):
                    break
                pos += 1

            state.pos = pos
            return True

    if not silent:
        state.pending += "\\"
    state.pos += 1
    return True
