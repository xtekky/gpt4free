import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import g4f

response = g4f.ChatCompletion.create(
    model=g4f.models.gpt_35_turbo,
    messages=[{"role": "user", "content": "hello, are you GPT 4?"}]
)
print(response)