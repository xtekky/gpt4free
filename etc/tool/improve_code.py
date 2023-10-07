
import sys, re
from pathlib import Path
from os import path

sys.path.append(str(Path(__file__).parent.parent.parent))

import g4f

def read_code(text):
    match = re.search(r"```(python|py|)\n(?P<code>[\S\s]+?)\n```", text)
    if match:
        return match.group("code")
    
path = input("Path: ")

with open(path, "r") as file:
    code = file.read()

prompt = f"""
Improve the code in this file:
```py
{code}
```
Don't remove anything.
Add typehints if possible.
Don't add any typehints to kwargs.
Don't remove license comments.
"""

print("Create code...")
response = []
for chunk in g4f.ChatCompletion.create(
    model=g4f.models.gpt_35_long,
    messages=[{"role": "user", "content": prompt}],
    timeout=300,
    stream=True
):
    response.append(chunk)
    print(chunk, end="", flush=True)
print()
response = "".join(response)

code = read_code(response)
if code:
    with open(path, "w") as file:
        file.write(code)