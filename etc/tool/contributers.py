import requests

url = "https://api.github.com/repos/xtekky/gpt4free/contributors?per_page=100"

for user in requests.get(url).json():
    print(f'<a href="https://github.com/{user["login"]}" target="_blank"><img src="{user["avatar_url"]}&s=45" width="45" title="{user["login"]}"></a>')