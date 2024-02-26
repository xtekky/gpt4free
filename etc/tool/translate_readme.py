
import sys
from pathlib import Path
import asyncio

sys.path.append(str(Path(__file__).parent.parent.parent))

import g4f
g4f.debug.logging = True
from g4f.debug import access_token
provider = g4f.Provider.OpenaiChat

iso = "GE"
language = "german"
translate_prompt = f"""
Translate this markdown document to {language}.
Don't translate or change inline code examples.
```md
"""
keep_note = "Keep this: [!Note] as [!Note].\n"
blocklist = [
    '## ©️ Copyright',
    '## 🚀 Providers and Models',
    '## 🔗 Related GPT4Free Projects'
]
allowlist = [
    "### Other",
    "### Models"
]

def read_text(text):
    start = end = 0
    new = text.strip().split('\n')
    for i, line in enumerate(new):
        if line.startswith('```'):
            if not start:
                start = i + 1
            end = i
    return '\n'.join(new[start:end]).strip()

async def translate(text):
    prompt = translate_prompt + text.strip() + '\n```'
    if "[!Note]" in text:
        prompt = keep_note + prompt
    result = read_text(await provider.create_async(
        model="",
        messages=[{"role": "user", "content": prompt}],
        access_token=access_token
    ))
    if text.endswith("```") and not result.endswith("```"):
        result += "\n```"
    return result

async def translate_part(part, i):
    blocklisted = False
    for headline in blocklist:
        if headline in part:
            blocklisted = True
    if blocklisted:
        lines = part.split('\n')
        lines[0] = await translate(lines[0])
        part = '\n'.join(lines)
        for trans in allowlist:
            if trans in part:
                part = part.replace(trans, await translate(trans))
    else:
        part = await translate(part)
    print(f"[{i}] translated")
    return part

async def translate_readme(readme) -> str:
    parts = readme.split('\n## ')
    print(f"{len(parts)} parts...")
    parts = await asyncio.gather(
        *[translate_part("## " + part, i) for i, part in enumerate(parts)]
    )
    return "\n\n".join(parts)

with open("README.md", "r") as fp:
    readme = fp.read()

print("Translate readme...")
readme = asyncio.run(translate_readme(readme))

file = f"README-{iso}.md"
with open(file, "w") as fp:
    fp.write(readme)
print(f'"{file}" saved')