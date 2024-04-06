import re

def read_json(text: str) -> dict:
    """
    Parses JSON code block from a string.

    Args:
        text (str): A string containing a JSON code block.

    Returns:
        dict: A dictionary parsed from the JSON code block.
    """
    match = re.search(r"```(json|)\n(?P<code>[\S\s]+?)\n```", text)
    if match:
        return match.group("code")
    return text

def find_stop(stop, content: str, chunk: str):
    first = -1
    word = None
    if stop is not None:
        for word in list(stop):
            first = content.find(word)
            if first != -1:
                content = content[:first]
                break
        if stream and first != -1:
            first = chunk.find(word)
            if first != -1:
                chunk = chunk[:first]
            else:
                first = 0
    return first, content, chunk
