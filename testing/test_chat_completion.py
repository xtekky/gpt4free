import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import g4f

stream = False
response = g4f.ChatCompletion.create(
    model="gpt-3.5-turbo",
    provider=g4f.Provider.Ails,
    messages=[{"role": "user", "content": "hello"}],
    stream=stream,
    active_server=5,
)

print(response)
