# Process [link](<to> "stuff")

from ..common.utils import isSpace, normalizeReference
from .state_inline import StateInline


def link(state: StateInline, silent: bool):
    href = ""
    title = ""
    label = None
    oldPos = state.pos
    maximum = state.posMax
    start = state.pos
    parseReference = True

    if state.srcCharCode[state.pos] != 0x5B:  # /* [ */
        return False

    labelStart = state.pos + 1
    labelEnd = state.md.helpers.parseLinkLabel(state, state.pos, True)

    # parser failed to find ']', so it's not a valid link
    if labelEnd < 0:
        return False

    pos = labelEnd + 1

    if pos < maximum and state.srcCharCode[pos] == 0x28:  # /* ( */
        #
        # Inline link
        #

        # might have found a valid shortcut link, disable reference parsing
        parseReference = False

        # [link](  <href>  "title"  )
        #        ^^ skipping these spaces
        pos += 1
        while pos < maximum:
            code = state.srcCharCode[pos]
            if not isSpace(code) and code != 0x0A:
                break
            pos += 1

        if pos >= maximum:
            return False

        # [link](  <href>  "title"  )
        #          ^^^^^^ parsing link destination
        start = pos
        res = state.md.helpers.parseLinkDestination(state.src, pos, state.posMax)
        if res.ok:
            href = state.md.normalizeLink(res.str)
            if state.md.validateLink(href):
                pos = res.pos
            else:
                href = ""

            # [link](  <href>  "title"  )
            #                ^^ skipping these spaces
            start = pos
            while pos < maximum:
                code = state.srcCharCode[pos]
                if not isSpace(code) and code != 0x0A:
                    break
                pos += 1

            # [link](  <href>  "title"  )
            #                  ^^^^^^^ parsing link title
            res = state.md.helpers.parseLinkTitle(state.src, pos, state.posMax)
            if pos < maximum and start != pos and res.ok:
                title = res.str
                pos = res.pos

                # [link](  <href>  "title"  )
                #                         ^^ skipping these spaces
                while pos < maximum:
                    code = state.srcCharCode[pos]
                    if not isSpace(code) and code != 0x0A:
                        break
                    pos += 1

        if pos >= maximum or state.srcCharCode[pos] != 0x29:  # /* ) */
            # parsing a valid shortcut link failed, fallback to reference
            parseReference = True

        pos += 1

    if parseReference:
        #
        # Link reference
        #
        if "references" not in state.env:
            return False

        if pos < maximum and state.srcCharCode[pos] == 0x5B:  # /* [ */
            start = pos + 1
            pos = state.md.helpers.parseLinkLabel(state, pos)
            if pos >= 0:
                label = state.src[start:pos]
                pos += 1
            else:
                pos = labelEnd + 1

        else:
            pos = labelEnd + 1

        # covers label == '' and label == undefined
        # (collapsed reference link and shortcut reference link respectively)
        if not label:
            label = state.src[labelStart:labelEnd]

        label = normalizeReference(label)

        ref = (
            state.env["references"][label] if label in state.env["references"] else None
        )
        if not ref:
            state.pos = oldPos
            return False

        href = ref["href"]
        title = ref["title"]

    #
    # We found the end of the link, and know for a fact it's a valid link
    # so all that's left to do is to call tokenizer.
    #
    if not silent:
        state.pos = labelStart
        state.posMax = labelEnd

        token = state.push("link_open", "a", 1)
        token.attrs = {"href": href}

        if title:
            token.attrSet("title", title)

        # note, this is not part of markdown-it JS, but is useful for renderers
        if label and state.md.options.get("store_labels", False):
            token.meta["label"] = label

        state.md.inline.tokenize(state)

        token = state.push("link_close", "a", -1)

    state.pos = pos
    state.posMax = maximum
    return True
