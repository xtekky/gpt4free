# OraAI
Written by [hp_mzx](https://github.com/hpsj).

## Examples:
### Completion:
```python
chunk = oraai.Completion.create("who are you")
print(chunk)
```

### Chat Completion:
Support context
```python
obj = oraai.Completion()
whilt True:
    prompt = input("Please enter a question:")
	chunk = obj.create(prompt)
    print(chunk)
print()
```