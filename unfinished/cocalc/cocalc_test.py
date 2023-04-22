import cocalc


response = cocalc.Completion.create(
    prompt = 'hello world'
)

print(response)