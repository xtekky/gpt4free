import json

from g4f.api import create_app

app = create_app()

with open("openapi.json", "w") as f:
    data = json.dumps(app.openapi())
    f.write(data)

print(f"openapi.json - {round(len(data)/1024, 2)} kbytes")