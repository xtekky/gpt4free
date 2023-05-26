from dataclasses import dataclass


@dataclass
class AiColorsResponse:
    background: str
    primary: str
    accent: str
    text: str
