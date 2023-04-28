# For each opening emphasis-like marker find a matching closing one
#
from .state_inline import StateInline


def processDelimiters(state: StateInline, delimiters, *args):
    openersBottom = {}
    maximum = len(delimiters)

    closerIdx = 0
    while closerIdx < maximum:
        closer = delimiters[closerIdx]

        # Length is only used for emphasis-specific "rule of 3",
        # if it's not defined (in strikethrough or 3rd party plugins),
        # we can default it to 0 to disable those checks.
        #
        closer.length = closer.length or 0

        if not closer.close:
            closerIdx += 1
            continue

        # Previously calculated lower bounds (previous fails)
        # for each marker, each delimiter length modulo 3,
        # and for whether this closer can be an opener;
        # https://github.com/commonmark/cmark/commit/34250e12ccebdc6372b8b49c44fab57c72443460
        if closer.marker not in openersBottom:
            openersBottom[closer.marker] = [-1, -1, -1, -1, -1, -1]

        minOpenerIdx = openersBottom[closer.marker][
            (3 if closer.open else 0) + (closer.length % 3)
        ]

        openerIdx = closerIdx - closer.jump - 1

        # avoid crash if `closer.jump` is pointing outside of the array,
        # e.g. for strikethrough
        if openerIdx < -1:
            openerIdx = -1

        newMinOpenerIdx = openerIdx

        while openerIdx > minOpenerIdx:
            opener = delimiters[openerIdx]

            if opener.marker != closer.marker:
                openerIdx -= opener.jump + 1
                continue

            if opener.open and opener.end < 0:
                isOddMatch = False

                # from spec:
                #
                # If one of the delimiters can both open and close emphasis, then the
                # sum of the lengths of the delimiter runs containing the opening and
                # closing delimiters must not be a multiple of 3 unless both lengths
                # are multiples of 3.
                #
                if opener.close or closer.open:
                    if (opener.length + closer.length) % 3 == 0:
                        if opener.length % 3 != 0 or closer.length % 3 != 0:
                            isOddMatch = True

                if not isOddMatch:
                    # If previous delimiter cannot be an opener, we can safely skip
                    # the entire sequence in future checks. This is required to make
                    # sure algorithm has linear complexity (see *_*_*_*_*_... case).
                    #
                    if openerIdx > 0 and not delimiters[openerIdx - 1].open:
                        lastJump = delimiters[openerIdx - 1].jump + 1
                    else:
                        lastJump = 0

                    closer.jump = closerIdx - openerIdx + lastJump
                    closer.open = False
                    opener.end = closerIdx
                    opener.jump = lastJump
                    opener.close = False
                    newMinOpenerIdx = -1
                    break

            openerIdx -= opener.jump + 1

        if newMinOpenerIdx != -1:
            # If match for this delimiter run failed, we want to set lower bound for
            # future lookups. This is required to make sure algorithm has linear
            # complexity.
            #
            # See details here:
            # https:#github.com/commonmark/cmark/issues/178#issuecomment-270417442
            #
            openersBottom[closer.marker][
                (3 if closer.open else 0) + ((closer.length or 0) % 3)
            ] = newMinOpenerIdx

        closerIdx += 1


def link_pairs(state: StateInline) -> None:
    tokens_meta = state.tokens_meta
    maximum = len(state.tokens_meta)

    processDelimiters(state, state.delimiters)

    curr = 0
    while curr < maximum:
        curr_meta = tokens_meta[curr]
        if curr_meta and "delimiters" in curr_meta:
            processDelimiters(state, curr_meta["delimiters"])
        curr += 1
