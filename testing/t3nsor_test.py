import t3nsor

for response in t3nsor.StreamCompletion.create(prompt='write python code to reverse a string', messages=[]):
    print(response.completion.choices[0].text)
