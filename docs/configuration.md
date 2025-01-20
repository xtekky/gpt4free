
### G4F - Configuration


## Table of Contents
- [Authentication](#authentication)
- [Cookies Configuration](#cookies-configuration)
- [HAR and Cookie Files](#har-and-cookie-files) 
- [Debug Mode](#debug-mode)
- [Proxy Configuration](#proxy-configuration)


#### Authentication

Refer to the [G4F Authentication Setup Guide](authentication.md) for detailed instructions on setting up authentication.

### Cookies Configuration
Cookies are essential for using Meta AI and Microsoft Designer to create images.
Additionally, cookies are required for the Google Gemini and WhiteRabbitNeo Provider.
From Bing, ensure you have the "\_U" cookie, and from Google, all cookies starting with "\_\_Secure-1PSID" are needed.

**You can pass these cookies directly to the create function or set them using the `set_cookies` method before running G4F:**
```python
from g4f.cookies import set_cookies

set_cookies(".bing.com", {
  "_U": "cookie value"
})

set_cookies(".google.com", {
  "__Secure-1PSID": "cookie value"
})
```
---
### HAR and Cookie Files
**Using .har and Cookie Files**
You can place `.har` and cookie files `.json` in the default `./har_and_cookies` directory. To export a cookie file, use the [EditThisCookie Extension](https://chromewebstore.google.com/detail/editthiscookie/fngmhnnpilhplaeedifhccceomclgfbg) available on the Chrome Web Store.

**Creating .har Files to Capture Cookies**
To capture cookies, you can also create `.har` files. For more details, refer to the next section.

### Changing the Cookies Directory and Loading Cookie Files in Python
**You can change the cookies directory and load cookie files in your Python environment. To set the cookies directory relative to your Python file, use the following code:**
```python
import os.path
from g4f.cookies import set_cookies_dir, read_cookie_files

import g4f.debug
g4f.debug.logging = True

cookies_dir = os.path.join(os.path.dirname(__file__), "har_and_cookies")
set_cookies_dir(cookies_dir)
read_cookie_files(cookies_dir)
```

### Debug Mode
**If you enable debug mode, you will see logs similar to the following:**

```
Read .har file: ./har_and_cookies/you.com.har
Cookies added: 10 from .you.com
Read cookie file: ./har_and_cookies/google.json
Cookies added: 16 from .google.com
```

#### .HAR File for OpenaiChat Provider

##### Generating a .HAR File

**To utilize the OpenaiChat provider, a .har file is required from https://chatgpt.com/. Follow the steps below to create a valid .har file:**
1. Navigate to https://chatgpt.com/ using your preferred web browser and log in with your credentials.
2. Access the Developer Tools in your browser. This can typically be done by right-clicking the page and selecting "Inspect," or by pressing F12 or Ctrl+Shift+I (Cmd+Option+I on a Mac).
3. With the Developer Tools open, switch to the "Network" tab.
4. Reload the website to capture the loading process within the Network tab.
5. Initiate an action in the chat which can be captured in the .har file.
6. Right-click any of the network activities listed and select "Save all as HAR with content" to export the .har file.

##### Storing the .HAR File

- Place the exported .har file in the `./har_and_cookies` directory if you are using Docker. Alternatively, if you are using Python from a terminal, you can store it in a `./har_and_cookies` directory within your current working directory.

> **Note:** Ensure that your .har file is stored securely, as it may contain sensitive information.

### Proxy Configuration
**If you want to hide or change your IP address for the providers, you can set a proxy globally via an environment variable:**

**- On macOS and Linux:**
```bash
export G4F_PROXY="http://host:port"
```

**- On Windows:**
```bash
set G4F_PROXY=http://host:port
```
