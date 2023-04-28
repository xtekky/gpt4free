# Skip text characters for text token, place those to pending buffer
# and increment current pos

from .state_inline import StateInline

# Rule to skip pure text
# '{}$%@~+=:' reserved for extensions

# !, ", #, $, %, &, ', (, ), *, +, ,, -, ., /, :, ;, <, =, >, ?, @, [, \, ], ^, _, `, {, |, }, or ~

# !!!! Don't confuse with "Markdown ASCII Punctuation" chars
# http://spec.commonmark.org/0.15/#ascii-punctuation-character


def isTerminatorChar(ch):
    return ch in {
        0x0A,  # /* \n */:
        0x21,  # /* ! */:
        0x23,  # /* # */:
        0x24,  # /* $ */:
        0x25,  # /* % */:
        0x26,  # /* & */:
        0x2A,  # /* * */:
        0x2B,  # /* + */:
        0x2D,  # /* - */:
        0x3A,  # /* : */:
        0x3C,  # /* < */:
        0x3D,  # /* = */:
        0x3E,  # /* > */:
        0x40,  # /* @ */:
        0x5B,  # /* [ */:
        0x5C,  # /* \ */:
        0x5D,  # /* ] */:
        0x5E,  # /* ^ */:
        0x5F,  # /* _ */:
        0x60,  # /* ` */:
        0x7B,  # /* { */:
        0x7D,  # /* } */:
        0x7E,  # /* ~ */:
    }


def text(state: StateInline, silent: bool, **args):
    pos = state.pos
    posMax = state.posMax
    while (pos < posMax) and not isTerminatorChar(state.srcCharCode[pos]):
        pos += 1

    if pos == state.pos:
        return False

    if not silent:
        state.pending += state.src[state.pos : pos]

    state.pos = pos

    return True
