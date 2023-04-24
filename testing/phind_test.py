import phind

# set cf_clearance cookie ( not needed at the moment)
phind.cf_clearance = ''
phind.user_agent   = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'

prompt = 'hello world'

# normal completion
result = phind.Completion.create(
    model  = 'gpt-4',
    prompt = prompt,
    results     = phind.Search.create(prompt, actualSearch = False), # create search (set actualSearch to False to disable internet)
    creative    = False,
    detailed    = False,
    codeContext = '') # up to 3000 chars of code

print(result.completion.choices[0].text)

prompt = 'who won the quatar world cup'

# help needed: not getting newlines from the stream, please submit a PR if you know how to fix this
# stream completion
for result in phind.StreamingCompletion.create(
    model  = 'gpt-4',
    prompt = prompt,
    results     = phind.Search.create(prompt, actualSearch = True), # create search (set actualSearch to False to disable internet)
    creative    = False,
    detailed    = False,
    codeContext = ''):  # up to 3000 chars of code

    print(result.completion.choices[0].text, end='', flush=True)