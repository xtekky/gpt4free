### Example: `phind` (use like openai pypi package) <a name="example-phind"></a>

```python
import phind

# set cf_clearance cookie
phind.cf_clearance = 'xx.xx-1682166681-0-160'

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

# normal completion
result = phind.Completion.create(
    model  = 'gpt-4',
    prompt = prompt,
    results     = phind.Search.create(prompt, actualSearch = True), # create search (set actualSearch to False to disable internet)
    creative    = False,
    detailed    = False,
    codeContext = '') # up to 3000 chars of code

print(result.completion.choices[0].text)
```
