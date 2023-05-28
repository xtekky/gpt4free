from gpt4free import aicolors

prompt = "Light green color"
req = aicolors.Completion.create(prompt=prompt)

print(req)
