from typing import Awaitable

class AsyncDocumentConverterResult:
    """The result of converting a document to Markdown."""

    def __init__(
        self,
        text_content: Awaitable[str],
    ):
        self.text_content = text_content